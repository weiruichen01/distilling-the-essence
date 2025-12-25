import json
import os
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

from fmchisel.pruning.osscar.utils.helpers import cleanup_after_prune

PATH = "./test_model_cleanup"


@pytest.mark.parametrize(
    "intermediate_size",
    [
        20,
        30,
    ],
)
@pytest.mark.parametrize(
    "num_hidden_layers",
    [2, 4],
)
@pytest.mark.parametrize(
    "tie_word_embeddings",
    [True, False],
)
@pytest.mark.parametrize(
    "multiple_ckpt",
    [True, False],
)
@pytest.mark.parametrize(
    "inject_lm_head_weight",
    [True, False],
)
def test_cleanup_llama(
    intermediate_size,
    num_hidden_layers,
    tie_word_embeddings,
    multiple_ckpt,
    inject_lm_head_weight,
):

    torch.manual_seed(0)
    if os.path.isdir(PATH):
        shutil.rmtree(PATH)

    group_size = 2

    config = LlamaConfig(
        vocab_size=10,
        hidden_size=15,
        intermediate_size=intermediate_size,
        head_dim=20,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=group_size * 8,
        num_key_value_heads=8,
        tie_word_embeddings=tie_word_embeddings,
    )

    # Create the base model
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(PATH)

    if multiple_ckpt:  # Should the model have multiple checkpoint files?
        ckpt_size = os.path.getsize(os.path.join(PATH, "model.safetensors"))
        shutil.rmtree(PATH)
        model.save_pretrained(
            PATH, max_shard_size=str(ckpt_size // 3000) + "KB"
        )  # Change shard size to ensure segmentation

        assert os.path.isfile(os.path.join(PATH, "model.safetensors.index.json"))  # Check the ckeckpoint is segmented
    else:
        assert not os.path.isfile(
            os.path.join(PATH, "model.safetensors.index.json")
        )  # Check the ckeckpoint is not segmented

    # should we add `lm_head_weight` to model.safetensors.index.json, if the checkpoint is segmented and the model has
    # tied word embeddings, when it should not normally have `lm_head_weight` in model.safetensors.index.json.
    if inject_lm_head_weight and tie_word_embeddings and multiple_ckpt:
        with open(os.path.join(PATH, "model.safetensors.index.json"), "r") as file:
            safetensors_index = json.load(file)
        weight_map = safetensors_index["weight_map"]
        weight_map["lm_head.weight"] = "model-00001-of-00004.safetensors"  # Fake lm_head key
        safetensors_index["weight_map"] = weight_map
        with open(os.path.join(PATH, "model.safetensors.index.json"), "w") as file:
            json.dump(safetensors_index, file, indent=4)
        with open(os.path.join(PATH, "model.safetensors.index.json"), "r") as file:
            safetensors_index = json.load(file)
        weight_map = safetensors_index["weight_map"]
        assert (
            "lm_head.weight" in weight_map.keys()
        )  # Check lm_head.weight is added to model.safetensors.index.json, which can cause error in sglang.

    # Clean up the model config
    cleanup_after_prune(PATH)

    new_model = AutoModelForCausalLM.from_pretrained(PATH)

    assert torch.allclose(model.lm_head.weight, new_model.lm_head.weight)  # lm_head should not change

    if multiple_ckpt and tie_word_embeddings:
        with open(os.path.join(PATH, "model.safetensors.index.json"), "r") as file:
            safetensors_index = json.load(file)
        weight_map = safetensors_index["weight_map"]
        assert (
            "lm_head.weight" not in weight_map.keys()
        )  # There should not be a `lm_head.weight` key in model.safetensors.index.json for segmented ckpts.
