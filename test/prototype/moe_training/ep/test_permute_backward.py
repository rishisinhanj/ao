import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("Test requires CUDA", allow_module_level=True)

from torchao.prototype.moe_training.ep.kernels import generate_permute_indices
from torchao.prototype.moe_training.ep.permute import (
    _DispatchGatherBF16,
    _PermuteBF16FwdBF16Bwd,
    _UnpermuteBF16,
    dispatch_gather,
    permute_and_pad,
    unpermute_and_unpad,
)

DEVICE = "cuda"
ALIGNMENT = 16


def _setup(M, D, num_experts, ep_degree):
    """Generate test fixtures: input tensor, permuted indices, and shapes."""
    torch.manual_seed(42)
    num_local = num_experts // ep_degree
    weights = torch.rand(ep_degree * num_local, device=DEVICE)
    tokens_per_expert = (weights / weights.sum() * M).long()
    tokens_per_expert[0] += M - tokens_per_expert.sum()

    padded_max = ((M + num_local * ALIGNMENT + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    with torch.no_grad():
        indices, sizes, offsets = generate_permute_indices(
            tokens_per_expert, num_local, ep_degree, padded_max, ALIGNMENT
        )
    x = torch.randn(M, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    return x, indices, tokens_per_expert, num_local


@pytest.mark.parametrize("M,D,ne,ep", [(64, 16, 4, 2), (4096, 128, 32, 8)])
def test_permute_backward_matches_pytorch(M, D, ne, ep):
    x, indices, _, _ = _setup(M, D, ne, ep)

    # Reference: plain PyTorch autograd
    x_pad = torch.vstack((x, x.new_zeros((1, D))))
    y_ref = x_pad[indices, :]
    y_ref.sum().backward()
    grad_ref = x.grad.clone()

    # Custom Triton backward
    x2 = x.data.clone().requires_grad_(True)
    y2 = _PermuteBF16FwdBF16Bwd.apply(x2, indices)
    y2.sum().backward()

    assert torch.equal(y_ref, y2)
    assert torch.equal(grad_ref, x2.grad)


@pytest.mark.parametrize("M,D,ne,ep", [(64, 16, 4, 2), (4096, 128, 32, 8)])
def test_unpermute_backward_matches_pytorch(M, D, ne, ep):
    _, indices, _, _ = _setup(M, D, ne, ep)
    routed = torch.randn(
        indices.shape[0], D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True
    )

    # Reference
    out = routed.new_empty((M + 1, D))
    out[indices, :] = routed
    out[:-1].sum().backward()
    grad_ref = routed.grad.clone()

    # Custom Triton backward
    r2 = routed.data.clone().requires_grad_(True)
    _UnpermuteBF16.apply(r2, indices, M + 1, D).sum().backward()

    assert torch.equal(grad_ref, r2.grad)


@pytest.mark.parametrize("M,D,ne,ep", [(64, 16, 4, 2), (4096, 128, 32, 8)])
def test_round_trip_gradient(M, D, ne, ep):
    x, _, tpe, num_local = _setup(M, D, ne, ep)

    shape, x_perm, indices, _, _ = permute_and_pad(x, tpe, ne // num_local, num_local, ALIGNMENT)
    unpermute_and_unpad(x_perm * 2.0, shape, indices).sum().backward()

    assert torch.equal(x.grad, torch.full_like(x.grad, 2.0))


@pytest.mark.parametrize("M,D,top_k", [(64, 16, 1), (64, 16, 8), (4096, 128, 8)])
def test_dispatch_gather_backward_matches_pytorch(M, D, top_k):
    """dispatch_gather forward and backward match plain x[indices]."""
    torch.manual_seed(42)
    # Each token index appears top_k times (multiple experts per token)
    indices = torch.randint(0, M, (M * top_k,), device=DEVICE, dtype=torch.int64)

    x_ref = torch.randn(M, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    y_ref = x_ref[indices]
    y_ref.sum().backward()
    grad_ref = x_ref.grad.clone()

    x2 = x_ref.data.clone().requires_grad_(True)
    y2 = _DispatchGatherBF16.apply(x2, indices)
    y2.sum().backward()

    assert torch.equal(y_ref, y2), "dispatch_gather forward mismatch"
    # Backward uses atomic_add, so order may differ slightly from PyTorch's
    # reference, but the result should be very close (BF16 atomics).
    assert torch.allclose(grad_ref, x2.grad, atol=1e-2, rtol=1e-2), (
        f"dispatch_gather backward mismatch: max diff "
        f"{(grad_ref - x2.grad).abs().max().item()}"
    )


def test_zero_token_experts():
    D = 64
    tpe = torch.tensor([10, 0, 5, 0, 8, 0, 3, 0], device=DEVICE, dtype=torch.int64)
    M = int(tpe.sum())
    padded_max = ((M + 4 * ALIGNMENT + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    with torch.no_grad():
        indices, _, _ = generate_permute_indices(tpe, 4, 2, padded_max, ALIGNMENT)

    x = torch.randn(M, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    _PermuteBF16FwdBF16Bwd.apply(x, indices).sum().backward()
    assert not x.grad.isnan().any()

    r = torch.randn(indices.shape[0], D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    _UnpermuteBF16.apply(r, indices, M + 1, D).sum().backward()
    assert not r.grad.isnan().any()
