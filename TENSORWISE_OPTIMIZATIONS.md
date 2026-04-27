# FP8 Tensorwise Grouped GEMM — Optimization Log

Tracks optimizations in `ao_tensorwise_fork/` (branch `tensorwise-fp8-optimized`)
relative to Alex's original tensorwise code on `alex-minooka/ao:tensorwise-fp8ggemm`.

Base: `pytorch/ao` main at commit `67a78e581`.

## Verification

Both agents passed:
- **Correctness reviewer**: No critical bugs. One minor note: the Triton fast path
  in per-group tensorwise does not `nan_to_num` before the kernel, but this is
  consistent with the existing rowwise/colwise kernels in `jagged_float8_scales.py`.
- **Style reviewer**: Several minor deviations noted (underscore convention on custom_op
  names, `quantize_2d` vs `scale_and_cast` naming). All low severity and consistent
  with Alex's original code. The `tl.minimum(tl.maximum(...))` pattern instead of
  `tl.clamp()` is intentional — `tl.clamp` produces incorrect FP8 values on AMD ROCm.

## Commit 1: Port + Initial Optimizations

**`13d0d5523`** — Port tensorwise files from Alex's branch with 3 optimizations:

### Change 1: Cache B_t FP8 from forward
- Forward saves `(A, B_t_fp8, B_t_scales, offs)` instead of `(A, B_t, offs)`
- Backward reuses FP8 weights directly — zero GPU work
- **Saves: ~13 kernel launches, ~1.3ms per backward, ~469MB memory**

### Change 2: Fuse scale math into per-group kernel
- Quantize kernel computes `scale = fp8_max / max(amax, EPS)` inline in registers
- Eliminates 7 PyTorch kernel launches on 32-element vectors between Triton passes
- **Saves: ~14 kernel launches per backward (2 calls x 7)**

### Change 3: Switch to 2D tiled per-group kernel
- Uses `triton_fp8_per_group_tensorwise_scales` from `jagged_float8_scales.py`
- Grid: `(ceil(K/BLOCK_SIZE), num_groups)` — no atomics, no offset scanning
- **Eliminates atomic_max contention + 32-iteration offset scan per block**

## Commit 2: Fused 3D Tensorwise Kernel

**`203f2d662`** — New file `kernels/fp8_tensorwise_3d.py`

Replaces the ~14-kernel PyTorch fallback in `_fp8_tensorwise_quantize_3d` with:
- Triton kernel 1: per-(expert, N-block) partial amax
- PyTorch: `partial_amax.max(dim=1).values` — tiny (E, n_blocks) reduce
- Triton kernel 2: inline scale + scale+clamp+FP8 cast

Grid: `(E, ceil(N/BLOCK_SIZE_N))`. Handles col-major `(E, K, N)` strides natively.
- **Saves: ~14 -> 3 kernel launches in forward**

## Commit 3: Col-Major Output + Dual Kernel

**`4cef14e2b`** — Three new variants in `jagged_float8_scales.py`

### Optimization 5: Col-major output
- `triton_fp8_per_group_tensorwise_scales_col_major` writes FP8 output directly
  in column-major layout by passing col-major strides to the quantize kernel
- Eliminates `.copy_()` strided transpose of 68M and 238M FP8 elements
- **Saves: 2 kernel launches + 306M element copies**

### Optimization 6: Dual per-group quantize
- `triton_fp8_per_group_tensorwise_dual_col_major` processes both grad_output
  and A in a single pair of kernel launches (amax + quantize)
- Row iteration loops merged so each row is visited once per tensor pair
- Both outputs written directly in col-major layout
- **Saves: halves kernel launches for grad_B path**

## Summary

| Phase | Original launches | After optimizations |
|-------|:-----------------:|:-------------------:|
| Forward: quant A (2D) | 3 | 3 |
| Forward: quant B_t (3D fallback) | ~14 | **3** (fused 3D) |
| Forward: GEMM + reciprocals | 3 | 3 |
| Backward grad_A: quant grad_out | 3 | 3 |
| Backward grad_A: re-quant B_t | ~14 | **0** (cached) |
| Backward grad_A: transpose + GEMM | ~6 | ~6 |
| Backward grad_B: per-group quant x2 | ~20 | **~3** (dual) |
| Backward grad_B: FP8 transposes | 2 | **0** (col-major) |
| Backward grad_B: GEMM + reciprocals | 3 | 3 |
| **TOTAL** | **~68** | **~24** |

Estimated time savings: ~4-5ms per MoE linear op on MI325X (from ~8-10ms overhead
down to ~3-4ms), primarily from eliminating kernel launch dispatch overhead at ~45us each.
