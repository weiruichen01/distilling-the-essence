import pytest
import torch
from utils import DummyNetwork

from fmchisel.pruning.alps.utils.alps_sparsify import sparsify_weight

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def _project_nm(W: torch.Tensor, n: int, m: int) -> torch.Tensor:
    p, q = W.shape
    W = W.reshape(p * q // m, m)
    _, idx = torch.topk(-(W**2), m - n, dim=1)
    D = W.clone()
    D = D.scatter(
        src=torch.zeros((p * q // m, m - n)).to(W.device),
        dim=1,
        index=idx,
    )
    return D.reshape(p, q)


@pytest.mark.parametrize(
    "dim",
    [
        165,
        321,
        640,
        1000,
    ],
)
@pytest.mark.parametrize(
    "init_sparsity",
    [0.03, 0.12, 0.2],
)
@pytest.mark.parametrize(
    "goal_sparsity",
    [0.3, 0.41, 0.52],
)
def test_alps_preserve_sparsity(dim, init_sparsity, goal_sparsity):
    """
    This tests if ALPS properly preserves the sparsity mask.
    """

    torch.manual_seed(0)

    layer = DummyNetwork(
        p=dim,
        sparsity=init_sparsity,
    )
    layer.to(DEV)

    W0 = layer.weight.clone().reshape(-1)
    # The initial layer is sparse.
    indices0 = torch.nonzero(W0.view(-1) == 0, as_tuple=False).squeeze().tolist()
    assert len(indices0) / dim**2 >= init_sparsity

    _, W1 = sparsify_weight(
        module=layer,
        hessians_dict={layer: torch.eye(dim, device=DEV)},
        sparsity=goal_sparsity,
        prunen=0,
        prunem=0,
        preserve_sparsity_mask=True,
    )  # ALPS weights

    W1 = W1.reshape(-1)

    # Initial sparsity mask
    idx = torch.where(torch.abs(W0) == 0)[0]
    # Sparsity mask must be preserved.
    assert torch.linalg.norm(W1[idx]) < 1e-6

    # ALPS solution should be sparse.
    indices1 = torch.nonzero(W1 == 0, as_tuple=False).squeeze().tolist()
    assert len(indices1) / dim**2 >= init_sparsity

    idx = torch.where(torch.abs(W1) > 0)
    # The non-zeros of ALPS solution should be the same as the original weights.
    # This is because X^T*X=I so pcg backsolve must return the original weights.
    assert torch.allclose(W1[idx], W0[idx])


@pytest.mark.parametrize(
    "dim",
    [
        160,
        320,
        640,
        1000,
    ],
)
@pytest.mark.parametrize(
    "nm_n",
    [1, 2, 4],
)
def test_alps_semi_structured(dim, nm_n):
    """
    This tests if the weights of ALPS are correctly pruned for N:M
    semi-structured sparsity.
    This is done by considering a case where X^T*X = I.
    """
    nm_m = 2 * nm_n
    torch.manual_seed(0)

    layer = DummyNetwork(
        p=dim,
    )
    layer.to(DEV)

    W0 = layer.weight.clone().reshape(-1)

    _, W1 = sparsify_weight(
        module=layer,
        hessians_dict={layer: torch.eye(dim, device=DEV)},
        sparsity=0.5,
        prunen=nm_n,
        prunem=nm_m,
    )  # ALPS weights

    # Check ALPS mask has N:M structure
    assert torch.allclose(W1, _project_nm(W1, nm_n, nm_m))

    W0 = W0.reshape(-1)
    W1 = W1.reshape(-1)
    idx = torch.where(torch.abs(W1) > 0)[0]
    # The non-zeros of ALPS solution should be the same as the original weights.
    # This is because X^T*X=I so pcg backsolve must return the original weights.
    assert torch.allclose(W1[idx], W0[idx])
