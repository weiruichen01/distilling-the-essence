import logging
import time
from typing import Dict

import numpy as np
import torch
import transformers
from llmcompressor.modifiers.obcq.sgpt_sparsify import SGPT_PRECISION

logger = logging.getLogger(__name__)


def get_residue(B, B_orig, XtX, YtX):
    Res = torch.matmul(XtX, B) - YtX.T
    Res = torch.diag(torch.matmul((B - B_orig).t(), Res))
    return torch.sum(Res).item()


def L0_proj(
    B: torch.Tensor,
    k_spar: int,
    prunen: int,
    prunem: int,
    preserve_sparsity_mask: bool,
    init_mask: torch.Tensor,
) -> torch.Tensor:
    totp, num_cout = B.shape
    if prunen == 0:
        if not preserve_sparsity_mask:
            D = B.reshape(-1)
        else:
            D = (B * init_mask).reshape(-1)
        _, loss_idx = torch.topk(-(D**2), totp * num_cout - k_spar)
        D[loss_idx] = 0
        D = D.reshape(totp, num_cout)
    else:
        new_dim = totp * num_cout / prunem
        new_dim = int(new_dim)
        k_spar = totp * num_cout * prunen / prunem

        if not preserve_sparsity_mask:
            D = B.t().reshape((new_dim, prunem))
        else:
            D = (B * init_mask).t().reshape((new_dim, prunem))
        _, loss_idx = torch.topk(-(D**2), prunem - prunen, dim=1)
        D = D.scatter(
            src=torch.zeros((new_dim, prunem - prunen)).to(B.device),
            dim=1,
            index=loss_idx,
        )
        D = D.reshape(num_cout, totp).t()
    return D


def sparsify_weight(
    module: torch.nn.Module,
    hessians_dict: Dict[torch.nn.Module, torch.Tensor],
    sparsity: float,
    prunen: int = 0,
    prunem: int = 0,
    percdamp: float = 0.01,
    rho: float = 0.1,
    max_iter: int = 300,
    update_iter: int = 3,
    switch_iter: int = 30,
    preserve_sparsity_mask: bool = False,
    verbose: bool = False,
):
    """
    Run pruning on the layer up to the target
    sparsity value.

    :param sparsity: target sparsity to reach for layer
    :param prunen: N for N:M pruning
    :param prunem: M for N:M pruning
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

    init_mask = None
    if preserve_sparsity_mask:
        # compute existing sparsity mask
        init_mask = torch.where(
            W == 0,
            torch.tensor(1, dtype=torch.bool),
            torch.tensor(0, dtype=torch.bool),
        ).t()
        current_sparsity = init_mask.sum() / W.numel()
        if current_sparsity > sparsity:
            raise ValueError(
                "The target sparsity is lower than the sparsity "
                "of the base model. Please retry "
                "after turning preserve_sparsity_mask=False"
            )

    damp1 = percdamp * torch.mean(torch.diag(XtX)).item()
    diag_ind = torch.arange(XtX.shape[0], device=XtX.device)
    XtX[diag_ind, diag_ind] += damp1

    X_norm = torch.diag(XtX).sqrt() + 1e-8
    XtX = XtX / X_norm
    XtX = (XtX.T / X_norm).T

    YtX = torch.zeros_like(W)
    YtX = torch.matmul(W * X_norm, XtX)

    admm_st = time.time()

    XTX_inv = torch.zeros_like(XtX).to(dtype=SGPT_PRECISION)
    B = (W * X_norm).t().clone()
    W = None
    if verbose:
        B_orig = B.clone()
    V = torch.zeros_like(B)
    D = torch.zeros_like(B)
    D_support_old = torch.zeros_like(B)
    D_support = torch.zeros_like(B)

    totp, num_cout = B.shape

    L, Q = torch.linalg.eigh(XtX.double())

    XTX_inv = (Q @ ((1 / (L + (rho))) * Q).T).to(dtype=SGPT_PRECISION)

    init_rho = False

    if verbose:
        Residue0 = get_residue(torch.zeros_like(B_orig), B_orig, XtX, YtX)

    params = B.shape[0] * B.shape[1]
    k_spar = int(np.round((1 - sparsity) * params))

    D = L0_proj(
        B.clone(),
        k_spar,
        prunen,
        prunem,
        preserve_sparsity_mask,
        init_mask,
    )

    D_support_old = (D == 0).to(dtype=SGPT_PRECISION).reshape(-1)
    D_init = D.clone()

    for i_admm in range(max_iter):
        B = XTX_inv @ (YtX.T - V + rho * D)

        D = (V + rho * B) / rho

        D = L0_proj(
            D,
            k_spar,
            prunen,
            prunem,
            preserve_sparsity_mask,
            init_mask,
        )

        V = V + rho * (B - D)

        if (i_admm + 1) % update_iter == 0:
            D_support = (D.reshape(-1) == 0).to(dtype=SGPT_PRECISION)
            supp_change = torch.sum((D_support - D_support_old) ** 2)

            if supp_change / k_spar > 0.1:
                init_rho = True
                rho *= 1.3
            elif supp_change / k_spar > 0.005:
                init_rho = True
                rho *= 1.2
            elif supp_change > 0.5:
                if init_rho:
                    rho *= 1.1
                else:
                    rho /= 5
                    B = B_orig.clone()
                    D = D_init.clone()
                    V = torch.zeros_like(B)
            else:
                if init_rho:
                    break
                else:
                    rho /= 5

            D_support_old = (D_support).clone()
            if rho > 1e6:
                rho = 1e6

            XTX_inv = (Q @ ((1 / (L + (rho))) * Q).T).to(dtype=SGPT_PRECISION)

            if i_admm >= switch_iter and supp_change / k_spar < 0.0003:
                break

    B = L0_proj(
        B,
        k_spar,
        prunen,
        prunem,
        preserve_sparsity_mask,
        init_mask,
    )

    V = None
    D = None

    if verbose:

        Residue = get_residue(B, B_orig, XtX, YtX)
        error = Residue / Residue0

        logger.info("Before backsolve, error is {}".format(error))
    admm_time = time.time() - admm_st
    back_st = time.time()
    B = cg_batch(
        XtX,
        YtX.T,
        (B != 0).to(dtype=SGPT_PRECISION),
        M_bmm=None,
        X0=B,
        rtol=1e-4,
        atol=0.0,
        maxiter=10,
        verbose=verbose,
    )
    back_time = time.time() - back_st
    if verbose:
        Residue = get_residue(B, B_orig, XtX, YtX)
        error = Residue / Residue0

        logger.info("Number of iter is {}".format(i_admm))
        logger.info("Final Error is {}".format(error))
        logger.info("Time is admm: {} back:{}".format(admm_time, back_time))
    else:
        error = None
    # TODO: Run tests with Conv layers
    if isinstance(module, transformers.Conv1D):
        W = (B / X_norm).reshape(final_shape).to(final_dtype)
    else:
        W = (B.t() / X_norm).reshape(final_shape).to(final_dtype)

    return error, W


def cg_batch(
    A,
    B,
    A_supp,
    M_bmm=None,
    X0=None,
    rtol=1e-3,
    atol=0.0,
    maxiter=None,
    verbose=False,
):
    """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.
    This function solves matrix linear systems of the form
    A X = B,where A is a n x n positive definite matrix and B is
    a n x m matrix, and X is the n x m matrix representing the solution for the ith system.
    Args:
        A_bmm: A callable that performs a batch matrix multiply of A and a n x m matrix.
        B: A n x m matrix representing the right hand sides.
        M_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
        matrices M and a n x m matrix. (default=identity matrix)
        X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
        rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
        atol: (optional) Absolute tolerance for norm of residual. (default=0)
        maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
        verbose: (optional) Whether or not to print status messages. (default=False)
    """
    n, m = B.shape

    if M_bmm is None:
        M_bmm = lambda x: x
    if X0 is None:
        X0 = M_bmm(B)
    if maxiter is None:
        maxiter = 5 * n

    assert B.shape == (n, m)
    assert X0.shape == (n, m)
    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)

    X_k = X0
    R_k = B - A @ X_k
    R_k = R_k * A_supp
    Z_k = M_bmm(R_k)
    P_k = torch.zeros_like(Z_k)

    P_k1 = P_k
    R_k1 = R_k
    R_k2 = R_k
    X_k1 = X0
    Z_k1 = Z_k
    Z_k2 = Z_k

    B_norm = torch.norm(B, dim=1)
    stopping_matrix = torch.max(rtol * B_norm, atol * torch.ones_like(B_norm))

    if verbose:
        logger.info("%03s | %010s %06s" % ("it", "dist", "it/s"))

    optimal = False
    start = time.perf_counter()
    for k in range(1, maxiter + 1):
        Z_k = M_bmm(R_k)

        if k == 1:
            P_k = Z_k
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
        else:
            R_k2 = R_k1
            Z_k2 = Z_k1
            P_k1 = P_k
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            denominator = (R_k2 * Z_k2).sum(0)
            denominator[denominator == 0] = 1e-8
            beta = (R_k1 * Z_k1).sum(0) / denominator
            P_k = Z_k1 + beta.unsqueeze(0) * P_k1

        denominator = (P_k * (A @ P_k)).sum(0)
        denominator[denominator == 0] = 1e-8
        alpha = (R_k1 * Z_k1).sum(0) / denominator
        X_k = X_k1 + alpha.unsqueeze(0) * P_k
        R_k = R_k1 - alpha.unsqueeze(0) * (A @ P_k)
        R_k = R_k * A_supp
        residual_norm = torch.norm(A @ X_k - B, dim=1)

        if verbose:
            logger.info("%03d | %8.4e" % (k, torch.max(residual_norm / B_norm)))

        if (residual_norm <= stopping_matrix).all():
            optimal = True
            break
    end = time.perf_counter()
    if verbose:
        if optimal:
            logger.info("Terminated in %d steps (optimal). Took %.3f ms." % (k, (end - start) * 1000))
        else:
            logger.info("Terminated in %d steps (reached maxiter). Took %.3f ms." % (k, (end - start) * 1000))
    return X_k
