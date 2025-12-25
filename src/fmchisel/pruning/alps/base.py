from typing import Dict, Optional, Tuple

import torch
from compressed_tensors.utils import (
    align_module_device,
    get_execution_device,
    update_offload_parameter,
)
from llmcompressor.modifiers.obcq import SparseGPTModifier
from llmcompressor.modifiers.obcq.sgpt_sparsify import make_empty_hessian
from llmcompressor.utils.metric_logging import CompressionLogger
from loguru import logger
from pydantic import PrivateAttr

from fmchisel.pruning.alps.utils.alps_sparsify import sparsify_weight
from fmchisel.pruning.osscar.utils.osscar_sparsify import accumulate_hessian

__all__ = ["ALPSModifier"]


class ALPSModifier(SparseGPTModifier):
    """
    Modifier for applying the one-shot ALPS algorithm to a model

    Lifecycle:
        - on_initialize
            - register_hook(module, calibrate_module, "forward")
            - run_sequential / run_layer_sequential / run_basic
                - make_empty_hessian
                - accumulate_hessian
        - on_sequential_batch_end
            - sparsify_weight
        - on_finalize
            - remove_hooks()

    | Sample yaml:
    |   test_stage:
    |           ALPSModifier:
    |               sparsity: 0.5
    |               mask_structure: "2:4"
    |               sequential_update: True
    |               dampening_frac: 0.001

    :param sparsity: Sparsity to compress model to
    :param mask_structure: String to define the structure of the mask to apply.
        Must be of the form N:M where N, M are integers that define a custom block
        shape. Defaults to 0:0 which represents an unstructured mask.
    :param targets: list of layer names to compress during OBCQ, or '__ALL__'
        to compress every layer in the model
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    """

    # modifier arguments
    dampening_frac: Optional[float] = 0.01
    preserve_sparsity_mask: Optional[bool] = False
    verbose: Optional[bool] = False

    # private variables
    _num_samples: Dict[torch.nn.Module, int] = PrivateAttr(default_factory=dict)
    _hessians: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)

    def calibrate_module(
        self,
        module: torch.nn.Module,
        args: Tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ):
        # Assume that the first argument is the input
        inp = args[0]

        # Initialize hessian if not present
        if module not in self._num_samples:
            device = get_execution_device(module)
            self._hessians[module] = make_empty_hessian(module, device=device)
            self._num_samples[module] = 0

        # Accumulate hessian with input with optional offloading
        with self._maybe_onload_hessian(module):
            self._hessians[module], self._num_samples[module] = accumulate_hessian(
                inp,
                module,
                self._hessians[module],
                self._num_samples[module],
            )

    def on_sequential_batch_end(self):

        for module in list(self._num_samples.keys()):
            name = self._module_names[module]
            sparsity = self._module_sparsities[module]
            num_samples = self._num_samples[module]

            logger.info(f"Sparsifying {name} using {num_samples} samples")
            with torch.no_grad(), align_module_device(module), CompressionLogger(module) as comp_logger:
                loss, sparsified_weight = sparsify_weight(
                    module=module,
                    hessians_dict=self._hessians,
                    sparsity=sparsity,
                    prunen=self._prune_n,
                    prunem=self._prune_m,
                    percdamp=self.dampening_frac,
                    verbose=self.verbose,
                    preserve_sparsity_mask=self.preserve_sparsity_mask,
                )
                comp_logger.set_loss(loss)

            update_offload_parameter(module, "weight", sparsified_weight)

            # self._hessians[module] already deleted by sparsify_weight
            del self._num_samples[module]
