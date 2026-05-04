# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Fused Triton kernel for per-expert tensorwise FP8 quantization of 3D
column-major tensors (E, K, N) with strides (K*N, 1, K).

Unlike the colwise kernel in float8_rowwise.py which computes per-(expert, n)
scales, this kernel computes one scale per expert (tensorwise scaling) by
reducing over both the K and N dimensions.

Two-pass approach per (expert, N-block):
  - Pass 1: iterate over K blocks, accumulate per-column absmax, then reduce
    across columns to get a per-block scalar, written to partial_amax buffer.
  - Python: reduce partial_amax across N-blocks to get per-expert amax.
  - Pass 2: load per-expert amax, compute scale inline, iterate over K blocks,
    apply scale + clamp + FP8 cast.
"""

from typing import Tuple

import torch
from torch.utils._triton import has_triton

from torchao.utils import torch_version_at_least

if torch_version_at_least("2.7.0") and has_triton():
    import triton
    import triton.language as tl

    EPS = 1e-12

    FP8_DTYPE_MAP = {
        torch.float8_e4m3fn: tl.float8e4nv,
        torch.float8_e4m3fnuz: tl.float8e4b8,
        torch.float8_e5m2: tl.float8e5,
        torch.float8_e5m2fnuz: tl.float8e5b16,
    }

    if torch.version.hip is not None:
        _tensorwise_3d_configs = [
            triton.Config(
                {"BLOCK_SIZE_K": bk, "BLOCK_SIZE_N": bn},
                num_warps=warps,
                num_stages=2,
            )
            for bk in [128, 256]
            for bn in [64, 128]
            for warps in [4, 8]
        ]
    else:
        _tensorwise_3d_configs = [
            triton.Config(
                {"BLOCK_SIZE_K": bk, "BLOCK_SIZE_N": bn},
                num_warps=warps,
                num_stages=4,
            )
            for bk in [128, 256]
            for bn in [64, 128]
            for warps in [4, 8]
        ]

    # ── Pass 1: per-(expert, n_block) partial amax ──────────────────────────
    @triton.autotune(configs=_tensorwise_3d_configs, key=["K", "N"])
    @triton.jit
    def _fp8_tensorwise_3d_amax_kernel(
        input_ptr,
        stride_input_e: tl.int64,
        stride_input_k,
        stride_input_n,
        partial_amax_ptr,  # (E, n_n_blocks) float32
        stride_pa_e: tl.int64,
        stride_pa_nb,
        E: int,
        K: int,
        N: int,
        INPUT_DTYPE_MAX: tl.constexpr,  # max finite value of the input dtype
        BLOCK_SIZE_K: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        EPS: tl.constexpr,
    ):
        expert_idx = tl.program_id(0)
        n_block_idx = tl.program_id(1)

        n_offs = n_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offs < N

        col_amax = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)

        for k_start in range(0, K, BLOCK_SIZE_K):
            k_offs = k_start + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offs < K

            input_offs = (
                expert_idx * stride_input_e
                + k_offs[:, None] * stride_input_k
                + n_offs[None, :] * stride_input_n
            )
            mask = k_mask[:, None] & n_mask[None, :]
            vals = tl.load(input_ptr + input_offs, mask=mask, other=0.0).to(
                tl.float32
            )

            # nan_to_num: NaN → 0, ±inf → ±INPUT_DTYPE_MAX (matches torch.nan_to_num).
            # Must be applied before the abs+max so the resulting amax stays
            # finite even in the presence of NaN/inf inputs (early-training
            # loss spikes / LR overshoot). Pass 2 must apply the identical
            # transformation so scaled values are consistent with the amax.
            vals = tl.where(vals != vals, 0.0, vals)
            vals = tl.where(vals > INPUT_DTYPE_MAX, INPUT_DTYPE_MAX, vals)
            vals = tl.where(vals < -INPUT_DTYPE_MAX, -INPUT_DTYPE_MAX, vals)

            block_amax = tl.max(tl.abs(vals), axis=0)
            col_amax = tl.maximum(col_amax, block_amax)

        # Reduce across N columns to get a single scalar for this (expert, n_block).
        block_scalar_amax = tl.max(col_amax, axis=0).to(tl.float32)

        tl.store(
            partial_amax_ptr
            + expert_idx * stride_pa_e
            + n_block_idx * stride_pa_nb,
            block_scalar_amax,
        )

    # ── Pass 2: quantize with per-expert scale (inline from amax) ───────────
    @triton.autotune(configs=_tensorwise_3d_configs, key=["K", "N"])
    @triton.jit
    def _fp8_tensorwise_3d_quantize_kernel(
        input_ptr,
        stride_input_e: tl.int64,
        stride_input_k,
        stride_input_n,
        output_ptr,
        stride_output_e: tl.int64,
        stride_output_k,
        stride_output_n,
        expert_amax_ptr,  # (E,) float32
        scales_out_ptr,   # (E,) float32 computed scales
        E: int,
        K: int,
        N: int,
        fp8_dtype_min: tl.constexpr,
        fp8_dtype_max: tl.constexpr,
        output_dtype: tl.constexpr,
        ROUND_POW2: tl.constexpr,
        INPUT_DTYPE_MAX: tl.constexpr,  # max finite value of the input dtype
        BLOCK_SIZE_K: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        EPS: tl.constexpr,
    ):
        expert_idx = tl.program_id(0)
        n_block_idx = tl.program_id(1)

        n_offs = n_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offs < N

        amax = tl.load(expert_amax_ptr + expert_idx).to(tl.float32)
        scale = fp8_dtype_max / tl.maximum(amax, EPS)
        if ROUND_POW2:
            scale = tl.exp2(tl.floor(tl.log2(scale)))

        # First N-block writes the scale for the caller.
        if n_block_idx == 0:
            tl.store(scales_out_ptr + expert_idx, scale)

        for k_start in range(0, K, BLOCK_SIZE_K):
            k_offs = k_start + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offs < K

            input_offs = (
                expert_idx * stride_input_e
                + k_offs[:, None] * stride_input_k
                + n_offs[None, :] * stride_input_n
            )
            mask = k_mask[:, None] & n_mask[None, :]
            vals = tl.load(input_ptr + input_offs, mask=mask, other=0.0).to(
                tl.float32
            )

            # nan_to_num: must match pass 1 so scaled values are consistent
            # with the amax (and so a NaN/inf input cannot poison the FP8 cast).
            vals = tl.where(vals != vals, 0.0, vals)
            vals = tl.where(vals > INPUT_DTYPE_MAX, INPUT_DTYPE_MAX, vals)
            vals = tl.where(vals < -INPUT_DTYPE_MAX, -INPUT_DTYPE_MAX, vals)

            scaled_vals = vals * scale
            clamped_vals = tl.minimum(
                tl.maximum(scaled_vals, fp8_dtype_min), fp8_dtype_max
            ).to(output_dtype)

            output_offs = (
                expert_idx * stride_output_e
                + k_offs[:, None] * stride_output_k
                + n_offs[None, :] * stride_output_n
            )
            tl.store(output_ptr + output_offs, clamped_vals, mask=mask)

    # ── Python wrapper ──────────────────────────────────────────────────────
    @torch.library.custom_op(
        "torchao::triton_fp8_tensorwise_3d_scale_and_cast", mutates_args={}
    )
    def triton_fp8_tensorwise_3d_scale_and_cast(
        hp_tensor: torch.Tensor,
        output_dtype: torch.dtype = torch.float8_e4m3fn,
        round_scales_to_power_of_2: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fused per-expert tensorwise scale computation and FP8 cast for 3D
        column-major tensors.

        Replaces the ~14-kernel PyTorch fallback in _fp8_tensorwise_quantize_3d
        with 2 Triton kernels + 1 tiny PyTorch reduce:
            Kernel 1: per-(expert, N-block) partial amax
            PyTorch:  reduce partial amax across N-blocks -> per-expert amax
            Kernel 2: compute scale inline, apply scale + clamp + FP8 cast

        Args:
            hp_tensor: Input tensor of shape (E, K, N) in column-major layout
                (strides: K*N, 1, K). Must be float32 or bfloat16.
            output_dtype: Target FP8 dtype. Defaults to torch.float8_e4m3fn.
            round_scales_to_power_of_2: Whether to round scales to nearest power of 2.

        Returns:
            Tuple of (fp8_data, scales):
                - fp8_data: shape (E, K, N) in output_dtype, column-major layout.
                - scales: shape (E,) in float32 (forward scales: FP8_MAX / amax).
        """
        assert hp_tensor.ndim == 3, "input tensor must be 3D"

        tl_output_dtype = FP8_DTYPE_MAP[output_dtype]

        fp8_dtype_min = torch.finfo(output_dtype).min
        fp8_dtype_max = torch.finfo(output_dtype).max
        input_dtype_max = torch.finfo(hp_tensor.dtype).max

        e, k, n = hp_tensor.shape

        output_buffer = torch.empty(
            (e, k, n), dtype=output_dtype, device=hp_tensor.device
        ).as_strided((e, k, n), (k * n, 1, k))

        # Partial amax: (E, n_n_blocks). Each block reduces its (K x BLOCK_SIZE_N)
        # chunk to a single scalar. Then PyTorch reduces across n_blocks per expert.
        min_bn = min(c.kwargs["BLOCK_SIZE_N"] for c in _tensorwise_3d_configs)
        n_n_blocks = triton.cdiv(n, min_bn)
        partial_amax = torch.zeros(
            (e, n_n_blocks), dtype=torch.float32, device=hp_tensor.device
        )

        grid = lambda meta: (e, triton.cdiv(n, meta["BLOCK_SIZE_N"]))

        _fp8_tensorwise_3d_amax_kernel[grid](
            hp_tensor,
            hp_tensor.stride(0),
            hp_tensor.stride(1),
            hp_tensor.stride(2),
            partial_amax,
            partial_amax.stride(0),
            partial_amax.stride(1),
            e,
            k,
            n,
            INPUT_DTYPE_MAX=input_dtype_max,
            EPS=EPS,
        )

        expert_amax = partial_amax.max(dim=1).values  # (E,)

        scales = torch.empty(e, dtype=torch.float32, device=hp_tensor.device)

        _fp8_tensorwise_3d_quantize_kernel[grid](
            hp_tensor,
            hp_tensor.stride(0),
            hp_tensor.stride(1),
            hp_tensor.stride(2),
            output_buffer,
            output_buffer.stride(0),
            output_buffer.stride(1),
            output_buffer.stride(2),
            expert_amax,
            scales,
            e,
            k,
            n,
            fp8_dtype_min,
            fp8_dtype_max,
            tl_output_dtype,
            round_scales_to_power_of_2,
            INPUT_DTYPE_MAX=input_dtype_max,
            EPS=EPS,
        )

        return output_buffer, scales

    @triton_fp8_tensorwise_3d_scale_and_cast.register_fake
    def _fake_triton_fp8_tensorwise_3d_scale_and_cast(
        hp_tensor: torch.Tensor,
        output_dtype: torch.dtype = torch.float8_e4m3fn,
        round_scales_to_power_of_2: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert hp_tensor.ndim == 3, "input tensor must be 3D"
        e, k, n = hp_tensor.shape
        fp8_data = torch.empty(
            (e, k, n), dtype=output_dtype, device=hp_tensor.device
        ).as_strided((e, k, n), (k * n, 1, k))
        scales = torch.empty(e, dtype=torch.float32, device=hp_tensor.device)
        return fp8_data, scales

else:

    def triton_fp8_tensorwise_3d_scale_and_cast(
        hp_tensor: torch.Tensor,
        output_dtype: torch.dtype = torch.float8_e4m3fn,
        round_scales_to_power_of_2: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "triton_fp8_tensorwise_3d_scale_and_cast requires torch 2.7.0+ and triton installed"
        )
