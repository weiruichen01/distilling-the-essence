from typing import Dict, List, Optional, Union

import torch
from compressed_tensors.quantization.quant_args import ActivationOrdering
from compressed_tensors.utils import (
    align_module_device,
    getattr_chain,
    update_offload_parameter,
)
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from llmcompressor.utils.metric_logging import CompressionLogger
from loguru import logger
from pydantic import PrivateAttr

from fmchisel.quantization.quantease.utils.quantease_quantize import quantize_module

__all__ = ["QuantEaseModifier"]


class QuantEaseModifier(GPTQModifier):
    """
    Modifier for applying the one-shot QuantEase algorithm to a model

    Lifecycle:
        - on_initialize
            - apply config to model
        - on_start
            - add activation calibration hooks
            - add gptq weight calibration hooks
        - on_sequential_epoch_end
            - quantize_module
        - on_finalize
            - remove_hooks()
            - model.apply(freeze_module_quantization)
    | Sample yaml:
    | test_stage:
    |      QuantEaseModifier:
    |          sequential_update: True
    |          dampening_frac: 0.001
    |          block_size: 128
    |          config_groups:
    |            group_0:
    |                targets:
    |                  - "Linear"
    |                input_activations: null
    |                output_activations: null
    |                weights:
    |                    num_bits: 8
    |                    type: "int"
    |                    symmetric: true
    |                    strategy: "tensor"
    |                    group_size: 128



    :param block_size: Used to determine number of columns to compress in one pass for GPTQ.
                       Only used with GPTQ initialization
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
            diagonal norm
    :param do_gptq: Use GPTQ initialization
    :param num_iter: Number of QuantEase iterations
    """

    block_size: int = 128
    dampening_frac: Optional[float] = 0.01
    do_gptq: bool = True
    num_iter: int = 3
    sequential_targets: Union[str, List[str], None] = None
    actorder: Optional[ActivationOrdering] = None
    offload_hessians: bool = False

    _module_names: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _hessians: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)
    _num_samples: Dict[torch.nn.Module, int] = PrivateAttr(default_factory=dict)

    def on_sequential_batch_end(self):
        """
        Quantize modules which have been calibrated
        """
        for module in list(self._num_samples.keys()):
            name = self._module_names[module]
            num_samples = self._num_samples[module]
            quant_args = getattr_chain(module, "quantization_scheme.weights")

            logger.info(f"Quantizing {name} using {num_samples} samples")
            with torch.no_grad(), align_module_device(module), self._maybe_onload_hessian(module), CompressionLogger(
                module
            ) as comp_logger:
                loss, quantized_weight, scale, zero_point, g_idx = quantize_module(
                    module=module,
                    quant_args=quant_args,
                    hessians_dict=self._hessians,
                    nsamples=num_samples,
                    blocksize=self.block_size,
                    percdamp=self.dampening_frac,
                    do_gptq=self.do_gptq,
                    num_iter=self.num_iter,
                )
                comp_logger.set_loss(loss)

            update_offload_parameter(module, "weight", quantized_weight)
            update_offload_parameter(module, "weight_scale", scale)
            update_offload_parameter(module, "weight_zero_point", zero_point)
            if g_idx is not None:
                update_offload_parameter(module, "weight_g_idx", g_idx)

            # self._hessians[module] already deleted by quantize_weight
            del self._num_samples[module]
