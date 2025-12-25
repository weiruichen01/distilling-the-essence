import pytest
import torch
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.quant_args import (
    QuantizationStrategy,
    QuantizationType,
)
from utils import DummyNetwork, get_all_quantization_parameters, get_quant_args

from fmchisel.quantization.quantease.utils.quantease_quantize import quantize_module

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize(
    "dim",
    [
        5,
        10,
    ],
)
@pytest.mark.parametrize(
    "num_bits",
    [
        8,
        4,
    ],
)
@pytest.mark.parametrize(
    "type",
    [
        QuantizationType.INT,
        QuantizationType.FLOAT,
    ],
)
def test_quantease_weights(dim, num_bits, type):
    """
    This tests if the weights of QuantEase are correctly quantized.
    This is done by considering a case where X^T*X = I, so the
    layerwise quantization solution is the same as RTN.
    """
    if num_bits == 4 and type == QuantizationType.FLOAT:  # FP4 not implemented
        return

    torch.manual_seed(0)

    layer = DummyNetwork(
        p=dim,
    )
    layer.to(DEV)

    quant_args = get_quant_args(num_bits=num_bits, type=type, strategy=QuantizationStrategy.TENSOR)

    W0 = layer.weight.clone()

    all_qauntization_args = get_all_quantization_parameters(quant_args, W0)  # Will give scale and zero for quantization

    W2 = fake_quantize(
        W0,
        all_qauntization_args.scale,
        all_qauntization_args.zero_point,
        all_qauntization_args.quant_args,
    )  # RTN solution

    _, W1, _, _, _ = quantize_module(
        module=layer,
        quant_args=quant_args,
        nsamples=1,
        hessians_dict={layer: torch.eye(dim, device=DEV)},  # Identity Hessian
        do_gptq=False,
    )

    assert torch.allclose(W1.cpu().detach(), W2.cpu().detach())


@pytest.mark.parametrize(
    "dim",
    [
        5,
        10,
    ],
)
@pytest.mark.parametrize(
    "num_bits",
    [
        8,
        4,
    ],
)
@pytest.mark.parametrize(
    "sparsity",
    [
        0.6,
        0.7,
    ],
)
def test_quantease_preserve_sparsity(dim, num_bits, sparsity):
    """
    This tests if QuantEase correctly preserves the sparse weights
    """
    torch.manual_seed(0)

    layer = DummyNetwork(
        p=dim,
        sparsity=sparsity,  # Make the weights sparse
    )
    layer.to(DEV)

    quant_args = get_quant_args(num_bits=num_bits, type=QuantizationType.INT, strategy=QuantizationStrategy.TENSOR)

    W0 = layer.weight.clone()

    indices0 = torch.nonzero(W0.view(-1) == 0, as_tuple=False).squeeze().tolist()
    assert len(indices0) / dim**2 >= sparsity
    _, W1, _, _, _ = quantize_module(
        module=layer,
        quant_args=quant_args,
        nsamples=1,
        hessians_dict={layer: torch.eye(dim, device=DEV)},  # Identity Hessian
        do_gptq=False,
    )

    indices1 = torch.nonzero(W1.view(-1) == 0, as_tuple=False).squeeze().tolist()

    assert set(indices0).issubset(set(indices1))  # Check the sparsity pattern was preserved
