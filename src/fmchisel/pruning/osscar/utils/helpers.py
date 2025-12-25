import json
import os
import shutil
from typing import List

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DOWN_PROJ_KEYWORDS = ["down_proj", "w2"]
UP_GATE_PROJ_KEYWORDS = ["up_proj", "gate_proj", "w1", "w3"]
KV_PROJ_KEYWORDS = ["k_proj", "v_proj"]
Q_PROJ_KEYWORDS = ["q_proj"]
O_PROJ_KEYWORDS = ["o_proj"]


def is_keyword_layer(name: str, keywords: List[str]) -> bool:
    for key in keywords:
        if key in name:
            return True
    return False


def copy_weights(layer: torch.nn.Module, new_weight: torch.Tensor, key_to_load: str = "weight"):

    state_dict = layer.state_dict()
    state_dict[key_to_load] = new_weight.detach().clone()
    layer.load_state_dict(state_dict)


def pack(
    model_path: str,
    num_drop_mlp_neuron: int,
    num_drop_attn_group: int,
    original_model_path: str = None,
):
    """
    Create a smaller model from an initial model that has pruned mlp layers.

    :param model_path: initial pruned model
    :param num_drop_mlp_neuron: how many MLP intermediate neurons to drop
    :param num_drop_attn_group: how many attention groups to drop
    """

    initial_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    if initial_model.dtype == torch.bfloat16:
        bf16 = True
    else:
        bf16 = False
    model_config = initial_model.config
    model_config.intermediate_size = int(model_config.intermediate_size - num_drop_mlp_neuron)
    if num_drop_attn_group > 0:

        orig_num_key_value_heads = model_config.num_key_value_heads
        try:
            head_dim = model_config.head_dim  # Does not change
        except AttributeError:
            raise ValueError(
                "Transformers version does not support attention head pruning. Please upgrade transformers."
            )
        hidden_size = model_config.hidden_size  # Does not change

        kv_group_size = model_config.num_attention_heads // model_config.num_key_value_heads  # Does not change
        model_config.num_key_value_heads = int(model_config.num_key_value_heads - num_drop_attn_group)
        model_config.num_attention_heads = int(model_config.num_attention_heads - num_drop_attn_group * kv_group_size)

    copy_lm_head = False
    if original_model_path is not None:

        orig_config = AutoConfig.from_pretrained(original_model_path)

        if hasattr(orig_config, "tie_word_embeddings") and orig_config.tie_word_embeddings:
            model_config.tie_word_embeddings = True
        else:
            copy_lm_head = True
    else:
        copy_lm_head = True

    new_model = AutoModelForCausalLM.from_config(model_config)
    nnz_mlp_indices = {}
    nnz_attn_indices = {}

    """
    The idea is that the MLP down projection (attention output projection) matrices are sparse, and we can
    remove some of their input channels. To keep things consistent, we need to
    remove the corresponding output channels from the up/gate projection (qkv projection) matrices.
    So we have to first chop off the down_proj (o_proj), find which channels (kv groups) are kept,
    and then chop off up/gate proj (qkv proj).

    The logic below is as follows:
    1- Do a pass over the model. Copy any weights that are not sparse
    (embedding/normalization, also: mlp if not pruning mlp, attn if not prunning attn)
    2- If pruning MLP, update down_proj matrices and save the indices of remaining channels.
    3- If pruning attn, update o_proj matrices and save the indices of remaining kv groups.
    4- If pruning MLP, do another pass over the model and update up/gate proj matrices,
       using the indices that have been saved from the first pass.
    5- If pruning attn, do another pass over the model and update qkv proj matrices,
       using the indices of active kv groups that have been saved from the first pass.
    """

    for (name_new, module_new), (name_initial, module_initial) in zip(
        new_model.model.named_modules(), initial_model.model.named_modules()
    ):

        if not hasattr(module_new, "weight"):  # not a weight, skip
            continue
        ##
        is_up_proj = is_keyword_layer(name_new, UP_GATE_PROJ_KEYWORDS)
        is_down_proj = is_keyword_layer(name_new, DOWN_PROJ_KEYWORDS)
        is_o_proj = is_keyword_layer(name_new, O_PROJ_KEYWORDS)
        is_qkv_proj = False
        if is_keyword_layer(name_new, Q_PROJ_KEYWORDS) or is_keyword_layer(name_new, KV_PROJ_KEYWORDS):
            is_qkv_proj = True
        ##
        if is_qkv_proj and num_drop_attn_group > 0:  # if qkv_proj and pruning attn heads, skip in the first pass.
            continue
        if is_up_proj and num_drop_mlp_neuron > 0:  # if up_proj and pruning mlp, skip in the first pass.
            continue
        """
        The following possibilities, for all of which we just copy the weights:
        1- normalization/emb layer
        2- qkv attn layer but we are not pruning attention heads
        3- up_proj layer but we are not pruning MLP
        """
        if not is_down_proj and not is_o_proj:  # emb/normalization layers
            copy_weights(
                module_new,
                module_initial.weight,
            )
            continue
        if (
            is_o_proj and num_drop_attn_group == 0
        ):  # attn o_proj layer, but we are not pruning attn heads, so just copy the weights
            copy_weights(
                module_new,
                module_initial.weight,
            )
            continue
        if (
            is_down_proj and num_drop_mlp_neuron == 0
        ):  # mlp down proj layer but we are not pruning, so we just copy the weights
            copy_weights(
                module_new,
                module_initial.weight,
            )
            continue

        if is_down_proj:
            """
            MLP down_proj layer and we are pruning MLP
            Take care of down project layers.
            Find the nonzero columns and save their indices
            The indices are then used to update the up project layers
            """

            down_proj_layer = module_initial
            idx = torch.norm(down_proj_layer.weight, dim=0).nonzero(as_tuple=True)[0]  # Active mlp channels
            nnz_mlp_indices[name_initial.rsplit(".", 1)[0]] = idx

            weight_down = down_proj_layer.weight[:, idx]
            copy_weights(module_new, weight_down)

        if is_o_proj:
            """
            Attention o_proj layer and we are pruning attention
            Take care of o project layers.
            Find the nonzero attention groups and save their indices.
            The indices are then used to update the qkv project layers
            """

            o_proj_layer = module_initial  # weights: (hidden_size) * (num_heads * head_dim) =  (hidden_size) * (num_key_value_heads * kv_group_size * head_dim)
            reshaped_weight = (
                o_proj_layer.weight.clone()
                .t()
                .detach()
                .reshape(orig_num_key_value_heads, hidden_size * kv_group_size * head_dim)
            )  # reshaped_weights: (num_key_value_heads) * (hidden_size * kv_group_size * head_dim)
            idx = torch.norm(reshaped_weight, dim=1).nonzero(as_tuple=True)[0]  # Active kv groups channels
            nnz_attn_indices[name_initial.rsplit(".", 1)[0]] = idx
            weight_o = reshaped_weight[
                idx, :
            ]  # size: (new_num_key_value_heads) * (hidden_size * kv_group_size * head_dim)
            weight_o = weight_o.reshape(
                (orig_num_key_value_heads - num_drop_attn_group) * kv_group_size * head_dim,
                hidden_size,
            )
            copy_weights(module_new, weight_o.t())

    if num_drop_mlp_neuron > 0:  # Pruning MLP
        for (name_new, module_new), (name_initial, module_initial) in zip(
            new_model.model.named_modules(), initial_model.model.named_modules()
        ):

            if not hasattr(module_new, "weight"):
                continue
            is_up_proj = is_keyword_layer(name_new, UP_GATE_PROJ_KEYWORDS)
            if not is_up_proj:
                continue
            # Copy up/gate projection weights
            up_proj_layer = module_initial
            idx = nnz_mlp_indices[name_initial.rsplit(".", 1)[0]]
            weight_up = up_proj_layer.weight[idx, :]
            copy_weights(module_new, weight_up)

    if num_drop_attn_group > 0:
        for (name_new, module_new), (name_initial, module_initial) in zip(
            new_model.model.named_modules(), initial_model.model.named_modules()
        ):

            if not hasattr(module_new, "weight"):
                continue
            is_kv_proj = is_keyword_layer(name_new, KV_PROJ_KEYWORDS)
            is_q_proj = is_keyword_layer(name_new, Q_PROJ_KEYWORDS)

            if not is_q_proj and not is_kv_proj:
                continue

            idx = nnz_attn_indices[name_initial.rsplit(".", 1)[0]]

            if is_kv_proj:  # can be either k_proj or v_proj
                kv_proj_layer = module_initial  # (num_key_value_heads *  head_dim) * (hidden_size)
                weight_kv = (
                    kv_proj_layer.weight.detach()
                    .clone()
                    .reshape(orig_num_key_value_heads, head_dim * hidden_size)[idx, :]
                )
                weight_kv = weight_kv.reshape(
                    (orig_num_key_value_heads - num_drop_attn_group) * head_dim,
                    hidden_size,
                )
                copy_weights(module_new, weight_kv)

                if hasattr(module_initial, "bias") and module_initial.bias is not None:
                    pruned_bias = (
                        module_initial.bias.detach().clone().reshape(orig_num_key_value_heads, head_dim)[idx, :]
                    )
                    copy_weights(module_new, pruned_bias[:].squeeze(), "bias")

            if is_q_proj:
                q_proj_layer = module_initial  # (num_key_value_heads * kv_group_size * head_dim) * (hidden_size)
                reshaped_q_weight = (
                    q_proj_layer.weight.detach()
                    .clone()
                    .reshape(orig_num_key_value_heads, hidden_size * kv_group_size * head_dim)
                )
                weight_q = reshaped_q_weight[idx, :].reshape(
                    (orig_num_key_value_heads - num_drop_attn_group) * kv_group_size * head_dim,
                    hidden_size,
                )
                copy_weights(module_new, weight_q)

                if hasattr(module_initial, "bias") and module_initial.bias is not None:
                    pruned_bias = (
                        module_initial.bias.detach()
                        .clone()
                        .reshape(orig_num_key_value_heads, kv_group_size * head_dim)[idx, :]
                    )
                    copy_weights(module_new, pruned_bias[:].squeeze(), "bias")

    # Copy bias, if exists
    for (name_new, module_new), (name_initial, module_initial) in zip(
        new_model.model.named_modules(), initial_model.model.named_modules()
    ):

        if not hasattr(module_new, "bias") or module_new.bias is None:
            continue
        # If pruning attention, the bias is copied already above.
        if is_keyword_layer(name_new, Q_PROJ_KEYWORDS) or is_keyword_layer(name_new, KV_PROJ_KEYWORDS):
            if num_drop_attn_group > 0:
                continue
        copy_weights(module_new, module_initial.bias.detach().clone(), "bias")

    # lm_head is not a part of the model, copy separately
    if copy_lm_head and hasattr(new_model, "lm_head"):  # Do we need this if condition (for emb models)?
        copy_weights(new_model.lm_head, initial_model.lm_head.weight)

    if bf16:
        new_model.to(torch.bfloat16)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:  # noqa: E722
        tokenizer = None
    os.system(f"rm -rf {model_path}-uncompressed")
    shutil.move(model_path, f"{model_path}-uncompressed")

    if num_drop_mlp_neuron + num_drop_attn_group == 0:  # no need to keep the uncompressed copy
        os.system(f"rm -rf {model_path}-uncompressed")
    new_model.save_pretrained(model_path)
    if tokenizer is not None:
        tokenizer.save_pretrained(model_path)


def cleanup_after_prune(output_dir: str):
    # Clean up what remains from llmcompressor
    recipe_path = os.path.join(output_dir, "recipe.yaml")
    if os.path.isfile(recipe_path):
        os.remove(recipe_path)

    config_json_path = os.path.join(output_dir, "config.json")
    with open(config_json_path, "r") as file:
        config = json.load(file)

    if "compression_config" in config:
        config.pop("compression_config")
    with open(config_json_path, "w") as file:
        json.dump(config, file, indent=4)

    # If the model has tied word embeddings and contains multiple checkpoint files,
    # sometimes an additional `lm_head.weight` entry is inserted into model.safetensors.index.json.
    # As the model does not have lm_head weights, this causes error with sglang.
    # Below if model.safetensors.index.json exists (multiple checkpoint files),
    # and if the model has tied word embeddings, we remove `lm_head.weight` from model.safetensors.index.json.
    safetensors_index_path = os.path.join(output_dir, "model.safetensors.index.json")
    if os.path.isfile(safetensors_index_path):
        model_config = AutoConfig.from_pretrained(output_dir)
        if model_config.tie_word_embeddings:
            with open(safetensors_index_path, "r") as file:
                safetensors_index = json.load(file)
            weight_map = safetensors_index["weight_map"]
            if "lm_head.weight" in weight_map.keys():
                weight_map.pop("lm_head.weight")

            safetensors_index["weight_map"] = weight_map
            with open(safetensors_index_path, "w") as file:
                json.dump(safetensors_index, file, indent=4)
