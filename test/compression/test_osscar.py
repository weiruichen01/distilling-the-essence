import pytest
import torch
from utils import DummyNetwork

from fmchisel.pruning.osscar.utils.osscar_sparsify import sparsify_weight

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize(
    "dim",
    [
        5,
        10,
        15,
    ],
)
@pytest.mark.parametrize(
    "num_drop",
    [1, 2, 3],
)
def test_osscar_weights(dim, num_drop):
    """
    This tests if the weights of OSSCAR are correctly pruned.
    This is done by considering a case where X^T*X = I, so the
    layerwise pruning solution is the same as MP.
    """

    torch.manual_seed(0)

    layer = DummyNetwork(
        p=dim,
    )
    layer.to(DEV)

    W0 = layer.weight.clone()

    _, W1 = sparsify_weight(
        module=layer,
        hessians_dict={layer: torch.eye(dim, device=DEV)},
        num_keep=dim - num_drop,
        update_iter=1,
        num_cin=dim,
    )  # OSSCAR weights

    _, idx = torch.topk(torch.linalg.norm(W0, dim=0), num_drop, largest=False)

    W2 = W0.clone()
    W2[:, idx] = torch.zeros_like(W2[:, idx]).to(DEV)  # MP weights

    assert torch.allclose(W1.cpu().detach(), W2.cpu().detach())
