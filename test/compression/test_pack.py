import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mixtral.configuration_mixtral import MixtralConfig

from fmchisel.pruning.osscar.utils.helpers import pack

registry = {"Llama": LlamaConfig, "Mistral": MistralConfig}

PATH = "./test_model"


@pytest.mark.parametrize(
    "model_family",
    [
        "Llama",
        "Mistral",
    ],
)
@pytest.mark.parametrize(
    "hidden_size",
    [
        10,
        15,
    ],
)
@pytest.mark.parametrize(
    "intermediate_size",
    [
        20,
        30,
    ],
)
@pytest.mark.parametrize(
    "head_dim",
    [
        4,
        8,
    ],
)
@pytest.mark.parametrize(
    "num_key_value_heads",
    [
        4,
        8,
    ],
)
@pytest.mark.parametrize(
    "num_drop_mlp_neuron",
    [0, 2, 4],
)
@pytest.mark.parametrize(
    "num_drop_attn_group",
    [0, 2],
)
def test_pack_llama(
    model_family,
    hidden_size,
    intermediate_size,
    head_dim,
    num_key_value_heads,
    num_drop_mlp_neuron,
    num_drop_attn_group,
):

    torch.manual_seed(0)

    group_size = 2

    ConfigClass = registry[model_family]

    config = ConfigClass(
        vocab_size=10,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        head_dim=head_dim,
        num_hidden_layers=1,
        num_attention_heads=group_size * num_key_value_heads,
        num_key_value_heads=num_key_value_heads,
    )
    model = AutoModelForCausalLM.from_config(config)
    # Make MLP Down_proj sparse
    down_proj_layer = model.model.layers[0].mlp.down_proj
    down_proj_weight = down_proj_layer.weight.detach().clone()
    down_proj_weight[:, :num_drop_mlp_neuron] = torch.zeros_like(down_proj_weight[:, :num_drop_mlp_neuron])
    state_dict = down_proj_layer.state_dict()
    state_dict["weight"] = down_proj_weight.detach().clone()
    down_proj_layer.load_state_dict(state_dict)

    self_attn = model.model.layers[0].self_attn
    # Make Attn O_proj sparse
    o_proj_layer = self_attn.o_proj
    o_proj_weight = o_proj_layer.weight.detach().clone()
    k = num_drop_attn_group * group_size * head_dim
    o_proj_weight[:, :k] = torch.zeros_like(o_proj_weight[:, :k])
    state_dict = o_proj_layer.state_dict()
    state_dict["weight"] = o_proj_weight.detach().clone()
    o_proj_layer.load_state_dict(state_dict)

    model.save_pretrained(PATH)

    pack(
        model_path=PATH,
        num_drop_mlp_neuron=num_drop_mlp_neuron,
        num_drop_attn_group=num_drop_attn_group,
    )

    model_packed = AutoModelForCausalLM.from_pretrained(PATH)

    down_orig = torch.linalg.norm(model.model.layers[0].mlp.down_proj.weight)
    down_packed = torch.linalg.norm(model_packed.model.layers[0].mlp.down_proj.weight)
    attn_orig = model.model.layers[0].self_attn.o_proj.weight
    attn_packed = model_packed.model.layers[0].self_attn.o_proj.weight
    head_orig = model.lm_head.weight.detach()
    head_packed = model_packed.lm_head.weight.detach()

    assert torch.allclose(down_orig, down_packed)  # Check MLP down_proj is packed correctly.
    assert torch.allclose(
        torch.linalg.norm(attn_orig), torch.linalg.norm(attn_packed)
    )  # Check Atnn O_proj is packed correctly.
    assert torch.allclose(head_orig, head_packed)  # Lm head should not change.
    assert (
        model_packed.config.intermediate_size == intermediate_size - num_drop_mlp_neuron
    )  # Check the intermediate size is correct after packing.
    assert model_packed.config.head_dim == model.config.head_dim  # head_dim should not change.
    assert (
        model_packed.config.num_key_value_heads == num_key_value_heads - num_drop_attn_group
    )  # Check the number of kv heads is correct after packing.


@pytest.mark.parametrize(
    "hidden_size",
    [
        10,
        15,
    ],
)
@pytest.mark.parametrize(
    "intermediate_size",
    [
        20,
        30,
    ],
)
@pytest.mark.parametrize(
    "num_local_experts",
    [
        2,
        3,
    ],
)
@pytest.mark.parametrize(
    "num_drop_mlp_neuron",
    [2, 4],
)
def test_pack_mixtral(hidden_size, intermediate_size, num_local_experts, num_drop_mlp_neuron):

    torch.manual_seed(0)

    config = MixtralConfig(
        vocab_size=10,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_local_experts=num_local_experts,
    )
    model = AutoModelForCausalLM.from_config(config)

    experts = model.model.layers[0].block_sparse_moe.experts
    for i in range(num_local_experts):
        down_proj_layer = getattr(experts, str(i)).w2
        down_proj_weight = down_proj_layer.weight.detach().clone()
        down_proj_weight[:, :num_drop_mlp_neuron] = torch.zeros_like(down_proj_weight[:, :num_drop_mlp_neuron])
        state_dict = down_proj_layer.state_dict()
        state_dict["weight"] = down_proj_weight.detach().clone()
        down_proj_layer.load_state_dict(state_dict)

    model.save_pretrained(PATH)

    pack(model_path=PATH, num_drop_mlp_neuron=num_drop_mlp_neuron, num_drop_attn_group=0)

    model_packed = AutoModelForCausalLM.from_pretrained(PATH)

    down_orig = []
    down_packed = []

    experts_orig = model.model.layers[0].block_sparse_moe.experts
    experts_packed = model_packed.model.layers[0].block_sparse_moe.experts
    for i in range(num_local_experts):
        down_orig.append(torch.linalg.norm(getattr(experts_orig, str(i)).w2.weight.detach()))
        down_packed.append(torch.linalg.norm(getattr(experts_packed, str(i)).w2.weight.detach()))

    attn_orig = model.model.layers[0].self_attn.q_proj.weight.detach()
    attn_packed = model_packed.model.layers[0].self_attn.q_proj.weight.detach()
    head_orig = model.lm_head.weight.detach()
    head_packed = model_packed.lm_head.weight.detach()

    assert torch.allclose(
        torch.tensor(down_orig), torch.tensor(down_packed)
    )  # Check MLP down_proj is packed correctly.
    assert torch.allclose(attn_orig, attn_packed)  # attn should not change.
    assert torch.allclose(head_orig, head_packed)  # Lm head should not change.
    assert (
        model_packed.config.intermediate_size == intermediate_size - num_drop_mlp_neuron
    )  # Check the intermediate size is correct after packing.
