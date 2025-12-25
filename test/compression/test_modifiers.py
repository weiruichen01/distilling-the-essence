from llmcompressor.modifiers.factory import ModifierFactory

from fmchisel.pruning.alps.base import ALPSModifier
from fmchisel.pruning.osscar.base import OSSCARModifier
from fmchisel.quantization.quantease.base import QuantEaseModifier


def test_alps_modifier():
    # Test if ALPS modifier is registered.
    kwargs = dict(
        sparsity=0.5,
        targets="__ALL_PRUNABLE__",
    )
    type_ = ModifierFactory.create(
        type_="ALPSModifier",
        allow_experimental=True,
        allow_registered=True,
        **kwargs,
    )

    assert isinstance(type_, ALPSModifier)


def test_osscar_modifier():
    # Test if OSSCAR modifier is registered.
    kwargs = dict(
        num_drop_mlp_neuron=1,
        num_drop_attn_group=1,
    )
    type_ = ModifierFactory.create(
        type_="OSSCARModifier",
        allow_experimental=True,
        allow_registered=True,
        **kwargs,
    )

    assert isinstance(type_, OSSCARModifier)


def test_quantease_modifier():
    # Test if QuantEase modifier is registered.
    kwargs = dict(
        scheme="W4A16",
    )
    type_ = ModifierFactory.create(
        type_="QuantEaseModifier",
        allow_experimental=True,
        allow_registered=True,
        **kwargs,
    )

    assert isinstance(type_, QuantEaseModifier)
