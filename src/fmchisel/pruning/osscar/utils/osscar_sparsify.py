from typing import Dict, Tuple

import torch
import transformers
from llmcompressor.modifiers.obcq.sgpt_sparsify import SGPT_PRECISION


def accumulate_hessian(
    inp: torch.Tensor,
    module: torch.nn.Module,
    H: torch.Tensor,
    num_samples: int,
) -> Tuple[torch.Tensor, int]:
    inp = inp.to(device=H.device)
    if len(inp.shape) == 2:
        inp = inp.unsqueeze(0)

    num_added = inp.shape[0]  # note this is the number of dataset samples, not
    # multiplied by the sequence length

    if isinstance(module, (torch.nn.Linear, transformers.Conv1D)):
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
    # TODO: Check conv layers
    if isinstance(module, torch.nn.Conv2d):
        unfold = torch.nn.Unfold(
            module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        )
        inp = unfold(inp)
        inp = inp.permute([1, 0, 2])
        inp = inp.flatten(1)

    num_samples += num_added

    inp = inp.to(dtype=SGPT_PRECISION)
    H += inp.matmul(inp.t())

    return H, num_samples


def sparsify_weight(
    module: torch.nn.Module,
    hessians_dict: Dict[torch.nn.Module, torch.Tensor],
    num_keep: int,
    update_iter: int,
    num_cin: int,
    percdamp: float = 0.01,
    verbose=False,
):
    """
    Run pruning  on the layer up to the target sparsity value.

    :param num_keep: How many groups to be kept
    :param update_iter: How often to upate support.
        More often update leads to better quality
        but will be slower.
    :param num_cin: Number of input channels
    :param percdamp: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    """
    final_shape = module.weight.shape
    final_dtype = module.weight.dtype
    W = module.weight.clone()
    XtX = hessians_dict[module]  # unfortunately python does not have a `move` keyword
    del hessians_dict[module]  # so we have to delete the original reference manually

    # TODO: Test Conv layers
    if isinstance(module, torch.nn.Conv2d):
        W = W.flatten(1)
    elif isinstance(module, transformers.Conv1D):
        W.transpose_(0, 1)
    W = W.to(dtype=SGPT_PRECISION)

    dead = torch.diag(XtX) == 0
    B = W.t()
    B[dead, :] = 0

    XtX += torch.eye(B.shape[0]).to(B.device) * percdamp * torch.mean(torch.diag(XtX))
    XtY = XtX @ B
    if verbose:
        print("num of dead is", torch.sum(dead).item())

    B_sol, B_obj = OSSCAR_fastprune(B.clone(), XtX, XtY, num_cin, num_keep, verbose, update_iter)
    if isinstance(module, transformers.Conv1D):
        B_sol.transpose_(0, 1)
    B_sol = B_sol.reshape(final_shape).to(final_dtype)
    if verbose:
        print(f"error: {B_obj}")

    return B_obj, B_sol


def OSSCAR_fastprune(W, XTX, XTY, num_cin, num_sp, verbose, update_iter=1):

    if verbose:
        W_orig = W.clone()
    DEV = W.device
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)

    XTX_inv = torch.linalg.inv(XTX)

    num_prune = torch.sum(torch.abs(torch.sum(torch.sum(W.reshape(num_cin, ksize, num_cout), axis=2), axis=1)) <= 1e-12)
    prune_list = torch.abs(torch.sum(torch.sum(W.reshape(num_cin, ksize, num_cout), axis=2), axis=1)) <= 1e-12

    if num_prune:
        upd_idx = torch.cat([torch.arange(i * ksize, (i + 1) * ksize) for i in range(num_cin) if prune_list[i]])
        XTX_inv[upd_idx, :] = 0
        XTX_inv[:, upd_idx] = 0

    W = XTX_inv @ XTY

    if int(num_cin - num_sp - num_prune) <= 0:
        upd_it = 0
    else:
        upd_it = int((num_cin - num_sp - num_prune) / update_iter)
        if upd_it == 0:
            upd_it = 1
        quo, rem = divmod(int(num_cin - num_sp - num_prune), int(upd_it))
        update_ten = torch.full((upd_it,), quo, dtype=torch.int).to(DEV)
        update_ten[:rem] += 1

    for i1 in range(upd_it):

        obj_mat = torch.zeros_like(W)
        if ksize > 1:
            for i2 in range(num_cin):
                if prune_list[i2]:
                    continue
                obj_mat[i2 * ksize : (i2 + 1) * ksize, :] = (  # noqa: E203
                    torch.linalg.inv(
                        XTX_inv[
                            i2 * ksize : (i2 + 1) * ksize,  # noqa: E203
                            i2 * ksize : (i2 + 1) * ksize,  # noqa: E203
                        ]
                    )
                    @ W[i2 * ksize : (i2 + 1) * ksize, :]  # noqa: E203
                    / 2
                )
        else:
            obj_mat = (1 / (prune_list + torch.diag(XTX_inv)))[:, None] * W / 2

        obj_cha = W * obj_mat
        obj_cha = obj_cha.reshape(num_cin, ksize, num_cout)
        obj_sum = torch.sum(torch.sum(obj_cha, axis=2), axis=1)

        idx = torch.argsort(obj_sum + 1e20 * (prune_list))

        upd_idx = torch.cat([torch.arange(idx[i] * ksize, (idx[i] + 1) * ksize) for i in range(update_ten[i1])])

        Xinv_tmp = torch.linalg.inv(XTX_inv[upd_idx[:, None], upd_idx])

        W -= XTX_inv[:, upd_idx] @ Xinv_tmp @ W[upd_idx, :]
        W = W.reshape(num_cin, ksize, num_cout)

        W[idx[: update_ten[i1]], :, :] = 0
        W = W.reshape(totp, num_cout)
        XTX_inv -= XTX_inv[:, upd_idx] @ Xinv_tmp @ XTX_inv[upd_idx, :]
        XTX_inv[upd_idx, :] = 0
        XTX_inv[:, upd_idx] = 0

        prune_list[idx[: update_ten[i1]]] = True

    W_sol = torch.zeros_like(W)
    nzi = torch.nonzero(W[:, 0], as_tuple=True)[0]
    W_sol[nzi, :] = torch.linalg.inv(XTX[nzi[:, None], nzi]) @ XTY[nzi, :]
    if verbose:
        return W_sol, torch.sum(torch.diag((W_orig - W_sol).t() @ XTX @ (W_orig - W_sol))) / torch.sum(
            torch.diag((W_orig).t() @ XTX @ (W_orig))
        )
    else:
        return W_sol.t(), None
