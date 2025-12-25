import time
from copy import copy
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import torch
import transformers
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    fake_quantize,
)
from llmcompressor.modifiers.quantization.gptq.gptq_quantize import (
    GPTQ_PRECISION,
    _apply_activation_ordering,
)
from llmcompressor.modifiers.utils import SPARSITY_THRESHOLD
from llmcompressor.observers.base import Observer
from llmcompressor.pytorch.utils.helpers import tensor_sparsity
from loguru import logger

__all__ = ["quantize_module"]


@dataclass
class AllQuantizationArgs:
    strategy: QuantizationStrategy
    scale: torch.Tensor
    zero_point: torch.Tensor
    quant_args: QuantizationArgs
    g_idx: torch.Tensor
    preserve_zeros: bool
    W_nz_mask: Union[torch.Tensor, None]


def quantize_module(
    module: torch.nn.Module,
    quant_args: QuantizationArgs,
    hessians_dict: Dict[torch.nn.Module, torch.Tensor],
    nsamples: int,
    blocksize: int = 128,
    percdamp: float = 0.01,
    do_gptq: bool = True,
    num_iter: int = 3,
    calculate_error: bool = True,
) -> Tuple[float, torch.Tensor, torch.Tensor, Union[torch.Tensor, None], torch.Tensor]:
    """
    Quantize a module weight according to the GPTQ algorithm

    :param module: module with weight being quantized
    :param quant_args: quantization arguments used to find quantization parameters
    :param hessian_dict: dictionary containing preaccumulated hessian for quantization
    :param blocksize: chunk size of quantization updates
    :param percdamp: dampening factor on hessian diagonal
    :param do_gptq: Use GPTQ initialization
    :param num_iter: Number of QuantEase iterations
    :param calculate_error: Calculate and log the quantization error?
    :return: loss, quantized_weight, scale, zero_point, g_idx
    """
    strategy = quant_args.strategy
    actorder = quant_args.actorder
    final_shape = module.weight.shape
    final_dtype = module.weight.dtype
    W = module.weight.clone()
    H = hessians_dict[module]  # unfortunately python does not have a `move` keyword
    del hessians_dict[module]  # so we have to delete the original reference manually

    # create observer for calculating quantization parameters
    observer = Observer.load_from_registry(
        quant_args.observer,
        quantization_args=quant_args,
        averaging_constant=1.0,  # ignore moving average
    )

    # standardize shape and dtype
    # TODO: Check conv layers
    if isinstance(module, torch.nn.Conv2d):
        W = W.flatten(1)
    elif isinstance(module, transformers.Conv1D):
        W.transpose_(0, 1)
    W = W.to(dtype=GPTQ_PRECISION)
    num_columns = W.shape[1]

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(num_columns, device=H.device)
    H[diag, diag] += damp
    g_idx = None
    if strategy == QuantizationStrategy.GROUP:
        # mapping from column index to group index
        g_idx = torch.arange(num_columns, device=W.device, dtype=torch.int) // quant_args.group_size

        if actorder == ActivationOrdering.GROUP:
            # permute by activation order first, then update groups
            W, H, perm = _apply_activation_ordering(W, H)
            scale, zero_point = observer(W, g_idx=None)

            # use identity g_idx (invert permutation later)

        elif actorder == ActivationOrdering.WEIGHT:
            # update groups first, then permute by activation order
            scale, zero_point = observer(W, g_idx=None)
            W, H, perm = _apply_activation_ordering(W, H)

            # permute g_idx to maintain identity mapping after unpermutation
            g_idx = g_idx[perm]

        else:
            scale, zero_point = observer(W, g_idx=None)
    else:
        scale, zero_point = observer(W, g_idx=None)

    # sparsity mask
    sparsity = tensor_sparsity(W)
    preserve_zeros = sparsity >= SPARSITY_THRESHOLD
    W_nz_mask = (~torch.isclose(W, torch.zeros(1, device=W.device).float())).float() if preserve_zeros else None

    all_args = AllQuantizationArgs(
        strategy=strategy,
        scale=scale,
        zero_point=zero_point,
        quant_args=quant_args,
        g_idx=g_idx,
        preserve_zeros=preserve_zeros,
        W_nz_mask=W_nz_mask,
    )

    W_orig = W.clone()
    tick = time.time()

    if calculate_error:
        Res0 = torch.diag(torch.matmul(torch.matmul(W_orig, H), W_orig.t()))
        Res0 = torch.sum(Res0).item()

    if do_gptq:
        W = gptq(
            H,
            W,
            blocksize,
            all_args,
        )
        logger.info("GPTQ time %.2f" % (time.time() - tick))
        if calculate_error:
            diff_W = W - W_orig
            Res = torch.diag(torch.matmul(torch.matmul(diff_W, H), diff_W.t()))
            Res = torch.sum(Res).item()
            error = Res / Res0
            logger.info("GPTQ error %.4f" % error)

    W = quantease(
        W,
        W_orig,
        H,
        num_iter,
        nsamples,
        all_args,
    )
    logger.info("QuantEase time %.2f" % (time.time() - tick))
    if calculate_error:
        diff_W = W - W_orig
        Res = torch.diag(torch.matmul(torch.matmul(diff_W, H), diff_W.t()))
        Res = torch.sum(Res).item()
        error = Res / Res0
        logger.info("QuantEase error %.4f" % error)
    has_gidx = False
    if strategy == QuantizationStrategy.GROUP:
        if actorder == ActivationOrdering.WEIGHT:
            # restore original permutation
            invperm = torch.argsort(perm)
            W = W[:, invperm]

        elif actorder == ActivationOrdering.GROUP:
            # restore original permutation
            invperm = torch.argsort(perm)
            W = W[:, invperm]
            g_idx = g_idx[invperm]

            # only save g_idx if mapping is not identity
            has_gidx = True

    if not has_gidx:
        g_idx = None

    if isinstance(module, transformers.Conv1D):
        W.transpose_(0, 1)
    W = W.reshape(final_shape).to(final_dtype)

    return (
        error if calculate_error else -1,
        W,
        scale.to(dtype=final_dtype),
        zero_point.to(dtype=quant_args.pytorch_dtype()),
        g_idx,
    )


def quantize_single_weight(
    q: torch.Tensor,
    column_idx: int,
    all_args: AllQuantizationArgs,
) -> torch.Tensor:

    if all_args.strategy == QuantizationStrategy.TENSOR:
        q = fake_quantize(
            q,
            all_args.scale,
            all_args.zero_point,
            all_args.quant_args,
        )
    elif all_args.strategy == QuantizationStrategy.CHANNEL:
        q = fake_quantize(
            q,
            all_args.scale[:, 0],
            all_args.zero_point[:, 0],
            all_args.quant_args,
        )
    elif all_args.strategy == QuantizationStrategy.GROUP:
        # get the group index for the current column
        group_index = all_args.g_idx[column_idx]

        # Since we're only applying quantization to a slice, this
        # ends up being a channelwise application
        altered_qargs = copy(all_args.quant_args)
        altered_qargs.strategy = QuantizationStrategy.CHANNEL
        q = fake_quantize(
            q,
            all_args.scale[:, group_index],
            all_args.zero_point[:, group_index],
            altered_qargs,
        )
    else:
        raise ValueError("Quantization strategy is not supported: " f"{all_args.strategy}")
    return q


def gptq(
    H: torch.Tensor,
    W: torch.Tensor,
    blocksize: int,
    all_args: AllQuantizationArgs,
) -> torch.Tensor:  # Run GPTQ initialization. The code taken from llmcompressor

    Hinv = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Hinv)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    H = None
    del H
    columns = W.shape[1]
    # See section 3.4 of https://arxiv.org/abs/2203.07259
    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        if all_args.preserve_zeros:
            W1_nz_mask = all_args.W_nz_mask[:, i1:i2]

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = w.clone()

            q = quantize_single_weight(q, i1 + i, all_args)

            Q1[:, i] = q

            err1 = (w - q) / d
            w1_err = err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            if all_args.preserve_zeros:
                W1[:, i:] -= w1_err * W1_nz_mask[:, i:]
            else:
                W1[:, i:] -= w1_err
            Err1[:, i] = err1

        W[:, i1:i2] = Q1

        w_err = Err1.matmul(Hinv[i1:i2, i2:])
        if all_args.preserve_zeros:
            W[:, i2:] -= w_err * all_args.W_nz_mask[:, i2:]
        else:
            W[:, i2:] -= w_err

    Hinv = None

    return W


def quantease(
    W_hat: torch.Tensor, W: torch.Tensor, H: torch.Tensor, num_iter: int, nsamples: int, all_args: AllQuantizationArgs
) -> torch.Tensor:  # Run QuantEase

    num_columns = W.shape[1]
    diag_Sigma = torch.diagonal(H, 0)

    norm_Sigma = torch.div(H, diag_Sigma)

    P = torch.matmul(W, norm_Sigma)
    # equivalent to (diagonal - 1) to help absorb -W matrix into XtXB for reducing computation of updating W[:, i]
    norm_Sigma.fill_diagonal_(0)

    norm_Sigma = norm_Sigma.t()
    for _ in range(num_iter):  # Go over columns
        delta_W_hat = W_hat.clone().t()
        P_hat = torch.matmul(norm_Sigma, delta_W_hat).t()  # Update entire XtXB before each iteration

        for j in range(num_columns):

            u = P[:, j] - P_hat[:, j]
            if j > 0:
                # Update single row i of XtXB, this is slightly different from the paper algorithm as we transpose
                # both norm_Sigma and delta_W_hat and do the matmul by switching their order, this is faster and
                # more memory efficient with the current torch matmul implementation based on CUDA GEMM kernel
                # though does not have theoretical matmul complexity improvement
                u += torch.matmul(norm_Sigma[j, :j], delta_W_hat[:j, :])

            u = quantize_single_weight(u, j, all_args)

            if all_args.preserve_zeros:
                u *= all_args.W_nz_mask[:, j]

            W_hat[:, j] = u
            delta_W_hat[j, :] -= u

    XtX = None
    del XtX
    return W_hat
