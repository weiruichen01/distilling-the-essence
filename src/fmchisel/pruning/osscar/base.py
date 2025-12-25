import warnings
from typing import Dict, Optional, Tuple

import torch
from compressed_tensors.utils import (
    align_module_device,
    get_execution_device,
    update_offload_parameter,
)
from llmcompressor.core import State
from llmcompressor.modifiers.obcq.base import SparseGPTModifier
from llmcompressor.modifiers.obcq.sgpt_sparsify import make_empty_hessian
from llmcompressor.pipelines.basic import run_pipeline as run_basic
from llmcompressor.pipelines.layer_sequential import (
    run_pipeline as run_layer_sequential,
)
from llmcompressor.pipelines.sequential import run_pipeline as run_sequential
from llmcompressor.utils.metric_logging import CompressionLogger
from llmcompressor.utils.pytorch.module import (
    get_layers,
    get_no_split_params,
    get_prunable_layers,
    match_targets,
)
from loguru import logger
from pydantic import PrivateAttr

from fmchisel.pruning.osscar.utils.helpers import (
    DOWN_PROJ_KEYWORDS,
    O_PROJ_KEYWORDS,
    is_keyword_layer,
)
from fmchisel.pruning.osscar.utils.osscar_sparsify import (
    accumulate_hessian,
    sparsify_weight,
)

__all__ = ["OSSCARModifier"]


class OSSCARModifier(SparseGPTModifier):
    """
    Modifier for applying the one-shot OSSCAR algorithm to a model


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

    :param num_drop_mlp_neuron: Number of MLP intermediate neurons to drop.
    :param num_drop_attn_group: How many groups of attention heads to prune.
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    """

    # modifier arguments
    num_drop_mlp_neuron: int = 128
    num_drop_attn_group: int = 0
    update_iter_mlp: Optional[int] = 1
    update_iter_attn: Optional[int] = 1
    dampening_frac: Optional[float] = 0.01

    # dummy, only used to bypass pydantic validation
    sparsity: Optional[float] = 0

    # private variables
    _num_samples: Dict[torch.nn.Module, int] = PrivateAttr(default_factory=dict)
    _hessians: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)
    _module_names: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _module_num_cin: Dict[torch.nn.Module, int] = PrivateAttr(default_factory=dict)
    _module_num_keep: Dict[torch.nn.Module, int] = PrivateAttr(default_factory=dict)
    _module_update_iter: Dict[torch.nn.Module, int] = PrivateAttr(default_factory=dict)
    _intermediate_size: int = PrivateAttr(default_factory=int)
    _num_key_value_heads: int = PrivateAttr(default_factory=int)

    def get_targets(self):
        self.targets = []
        # kbehdin: The following will not work for MoE models.
        if self.num_drop_mlp_neuron > 0:
            self.targets.append("re:.*down_proj.weight")
        if self.num_drop_attn_group > 0:
            self.targets.append("re:.*o_proj.weight")

    def on_initialize(self, state: "State", **kwargs) -> bool:
        """
        Initialize and run the OBCQ algorithm on the current state

        :param state: session state storing input model and calibration data
        """
        model: torch.nn.Module = state.model
        dataloader: torch.utils.data.DataLoader = state.data.calib

        # infer module and sequential targets
        self.sequential_targets = get_no_split_params(model)
        self.get_targets()
        target_layers = get_layers(self.targets, model)  # layers containing targets

        # model configs
        self._intermediate_size = state.model.config.intermediate_size
        assert (
            self._intermediate_size > self.num_drop_mlp_neuron
        ), "The number of MLP intermediate neurons to be dropped must be smaller than the intermediate size."
        self._num_key_value_heads = state.model.config.num_key_value_heads
        assert (
            self._num_key_value_heads > self.num_drop_attn_group
        ), "The number of attention groups to be dropped must be smaller than the number of KV heads."

        # register hooks
        for index, (layer_name, layer) in enumerate(target_layers.items()):

            for name, module in get_prunable_layers(layer).items():
                name = f"{layer_name}.{name}"

                if match_targets(name, self.ignore)[0]:
                    continue

                # HACK: previously, embeddings were not quantized because they were not
                # accessible by the layer compressor. For now, we manually ignore it,
                # but in the FUTURE this should be ignored by the user
                if isinstance(module, torch.nn.Embedding):
                    continue

                if name.endswith("lm_head"):
                    logger.warning(
                        "`lm_head` was previously auto-ignored by SparseGPT and Wanda "
                        "modifiers and is not advised. Please add `re:.*lm_head` to "
                        "your ignore list if this was unintentional"
                    )
                if is_keyword_layer(name, DOWN_PROJ_KEYWORDS):
                    num_keep = int(self._intermediate_size - self.num_drop_mlp_neuron)
                    num_cin = self._intermediate_size
                    update_iter = self.update_iter_mlp
                elif is_keyword_layer(name, O_PROJ_KEYWORDS):
                    num_keep = int(self._num_key_value_heads - self.num_drop_attn_group)
                    num_cin = self._num_key_value_heads
                    update_iter = self.update_iter_attn
                self._module_names[module] = name
                self._module_num_keep[module] = num_keep
                self._module_num_cin[module] = num_cin
                self._module_update_iter[module] = update_iter
                self.register_hook(module, self.calibrate_module, "forward")

        # infer and run pipeline
        model_name = state.model.__class__.__name__
        input_names = dataloader.dataset.column_names
        unfixable_errors = (torch.OutOfMemoryError, torch._C._LinAlgError)
        try:
            run_sequential(
                state.model,
                state.data.calib,
                self.sequential_targets,
                self.ignore,
                self,
            )
            return True

        except Exception as exception:
            if isinstance(exception, torch.fx.proxy.TraceError):
                warnings.warn(f"Failed to trace {model_name} with inputs {input_names}")
            if isinstance(exception, unfixable_errors):
                raise exception

            warnings.warn("Falling back to layer_sequential pipeline")
            try:
                run_layer_sequential(
                    state.model,
                    state.data.calib,
                    self.sequential_targets,
                    self,
                )
                return True

            except Exception as exception:
                if isinstance(exception, TypeError):
                    warnings.warn(f"{model_name} fails layer-wise assumptions")
                if isinstance(exception, unfixable_errors):
                    raise exception

                warnings.warn(
                    "Falling back to basic pipeline, which requires extra memory and "
                    "may result in decreased accuracy"
                )
                run_basic(state.model, state.data.calib, self)
                return True

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
            num_cin = self._module_num_cin[module]
            num_keep = self._module_num_keep[module]
            update_iter = self._module_update_iter[module]
            num_samples = self._num_samples[module]

            logger.info(f"Sparsifying {name} using {num_samples} samples")
            with torch.no_grad(), align_module_device(module), CompressionLogger(module) as comp_logger:
                loss, sparsified_weight = sparsify_weight(
                    module=module,
                    hessians_dict=self._hessians,
                    num_cin=num_cin,
                    num_keep=num_keep,
                    update_iter=update_iter,
                    percdamp=self.dampening_frac,
                )
                comp_logger.set_loss(loss)

            update_offload_parameter(module, "weight", sparsified_weight)

            # self._hessians[module] already deleted by sparsify_weight
            del self._num_samples[module]

    def on_finalize(self, state: State, **kwargs) -> bool:
        self.remove_hooks()
        self._hessians = dict()
        self._num_samples = dict()
        self._module_names = dict()
        self._module_num_cin = dict()
        self._module_num_keep = dict()
        self._module_update_iter = dict()

        return True
