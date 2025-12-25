import torch
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from llmcompressor.observers.base import Observer
from torch.nn import Linear

from fmchisel.quantization.quantease.utils.quantease_quantize import AllQuantizationArgs


# This returns quantization args
def get_quant_args(num_bits: int, type: QuantizationType, strategy: QuantizationStrategy) -> QuantizationArgs:
    quant_args = QuantizationArgs(
        num_bits=num_bits,
        type=type,
        strategy=strategy,
        observer="minmax",
    )
    return quant_args


# This returns all quantization parameters, including zero point, scale, etc. Does not support actorder
def get_all_quantization_parameters(quant_args: QuantizationArgs, W: torch.Tensor) -> AllQuantizationArgs:

    actorder = quant_args.actorder

    assert (actorder is None) or (not actorder), "actorder is not supported here."
    strategy = quant_args.strategy

    observer = Observer.load_from_registry(
        quant_args.observer,
        quantization_args=quant_args,
        averaging_constant=1.0,  # ignore moving average
    )
    scale, zero_point = observer(W, g_idx=None)

    all_args = AllQuantizationArgs(
        strategy=strategy,
        scale=scale,
        zero_point=zero_point,
        quant_args=quant_args,
        g_idx=None,
        preserve_zeros=False,
        W_nz_mask=None,
    )

    return all_args


class DummyNetwork(Linear):

    def __init__(
        self,
        p: int,
        sparsity: float = 0.0,
    ):

        super().__init__(p, p, bias=False)
        assert sparsity >= 0 and sparsity <= 1, "sparsity must be in [0,1]"
        self.weight.requires_grad = False
        if sparsity > 0:
            num_zeros = int(torch.ceil(torch.tensor([sparsity * (p**2)])).item())
            mask = torch.ones_like(self.weight).flatten()
            mask[torch.randperm(p**2)[:num_zeros]] = 0
            self.weight *= mask.reshape((p, p))
