import math
import operator

import torch
import torch.nn as nn
import triton
import triton.language as tl
from ops_utils import ensure_contiguous, calculate_settings, torch_to_triton_dtype
from triton.language.extra.libdevice import rsqrt

@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_cols,
    eps,
    offset,
    BLOCK_SIZE: tl.constexpr,
):
    """
    y_i = (x_i / (RMS)) * (offset + wi), RMS = sqrt(sum(x_i^2) / N)

    Reference:
    1. https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    2. https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22
    3. https://arxiv.org/pdf/1910.07467
    """

    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    X_row_dtype = X_row.dtype
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    eps = eps.to(X_row_dtype)
    offset = offset.to(X_row_dtype)

    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)

    # We can save time by caching rms with minimal memory overhead
    # because rms is much smaller compared to X_row, as rms is for each row.
    # However, on the computation side, it can save 4 operations (*, sum, /, sqrt).
    tl.store(RSTD_ptr, rstd)

    X_row = X_row * rstd


    Y_row = X_row * (offset + W_row)

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)

@triton.jit
def _block_rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows,
    n_cols,
    eps,
    offset,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    """
    y_i = (x_i / (RMS)) * (offset + wi), RMS = sqrt(sum(x_i^2) / N)

    Reference:
    1. https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    2. https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22
    3. https://arxiv.org/pdf/1910.07467
    """

    row_idx = tl.program_id(0) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = row_idx < n_rows
    col_mask = col_offsets < n_cols

    X_row = tl.load(
        X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=0,
    )
    X_row_dtype = X_row.dtype
    W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0)

    eps = eps.to(X_row_dtype)
    offset = offset.to(X_row_dtype)

    mean_square = tl.sum(X_row * X_row, axis=1) / n_cols
    rstd = rsqrt(mean_square + eps)

    # We can save time by caching rms with minimal memory overhead
    # because rms is much smaller compared to X_row, as rms is for each row.
    # However, on the computation side, it can save 4 operations (*, sum, /, sqrt).
    tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd, row_mask)

    X_row = X_row * rstd[:, None]


    Y_row = X_row * (offset + W_row)[None, :]

    tl.store(
        Y_ptr + row_idx[:, None] * Y_row_stride + col_offsets[None, :],
        Y_row,
        mask=row_mask[:, None] & col_mask[None, :],
    )


def rms_norm_forward(X, W, eps, offset, row_mode):
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    # RSTD is to cache rstd for each row
    # RSTD is always computed/stored in fp32 if we are using Llama or Gemma casting mode
    rstd_dtype = X.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)

    # Check constraints.
    assert X.shape[1] == W.shape[0], "Incompatible hidden size dimension between tensor1.shape[1] and tensor2.shape[0]"

    kernel_args = {}
    if BLOCK_SIZE > 256 or n_rows < 4096 * 8 or row_mode:
        _rms_norm_forward_kernel[(n_rows,)](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            W,
            W.stride(0),
            RSTD,
            RSTD.stride(0),
            n_cols,
            eps,
            offset,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            **kernel_args,  # XPU-specific optimization
        )
    else:
        BLOCK_ROW = 16
        kernel_args["BLOCK_ROW"] = BLOCK_ROW
        _block_rms_norm_forward_kernel[(triton.cdiv(n_rows, BLOCK_ROW),)](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            W,
            W.stride(0),
            RSTD,
            RSTD.stride(0),
            n_rows,
            n_cols,
            eps,
            offset,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            **kernel_args,  # XPU-specific optimization
        )
    return Y.view(*shape), X, RSTD, BLOCK_SIZE, num_warps


@triton.jit
def _block_rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    rows_per_program: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    """
    dx = (1 / RMS) * [dy * (w + offset - (1 / N) * (1 / RMS^2) * ((dy * (w + offset)) dot x) * x]. * means element-wise multiplication, whileas dot means dot product
    dw = sum(dy * (x / RMS)). summation over BxT dimension
    """

    pid = tl.program_id(0).cast(tl.int64)
    NUM_SMS = tl.num_programs(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols

    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    W_row = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0)
    W_row = W_row + offset

    for start in range(pid * BLOCK_ROW, n_rows, NUM_SMS * BLOCK_ROW):
        row_idx = start + tl.arange(0, BLOCK_ROW)
        row_mask = row_idx < n_rows
        dY_row = tl.load(
            dY_ptr + row_idx[:, None] * dY_row_stride + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        X_row = tl.load(
            X_ptr + row_idx[:, None] * X_row_stride + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )

        # Get cached rms
        rstd_row = tl.load(RSTD_ptr + row_idx * RSTD_row_stride, row_mask)

        X_row = X_row.to(tl.float32)


        m = dY_row * W_row[None, :]

        dX_row = rstd_row[:, None] * m

        dX_row += (rstd_row[:, None]) * (
            -(1 / n_cols) * (rstd_row * rstd_row * tl.sum(m * X_row, axis=1))[:, None] * X_row
        )


        # here X_row is already in fp32 (see previous if block)
        dW_row += tl.sum(dY_row * (X_row * rstd_row[:, None]), 0)

        tl.store(
            dX_ptr + row_idx[:, None] * dX_row_stride + col_offsets[None, :],
            dX_row,
            mask=row_mask[:, None] & col_mask[None, :],
        )

    tl.store(dW_ptr + pid * dW_row_stride + col_offsets, dW_row, mask=col_mask)


@triton.jit
def _rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    rows_per_program: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    dx = (1 / RMS) * [dy * (w + offset - (1 / N) * (1 / RMS^2) * ((dy * (w + offset)) dot x) * x]. * means element-wise multiplication, whileas dot means dot product
    dw = sum(dy * (x / RMS)). summation over BxT dimension
    """

    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    dY_ptr += row_start * dY_row_stride
    dX_ptr += row_start * dX_row_stride

    X_ptr += row_start * X_row_stride
    RSTD_ptr += row_start

    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    W_row = W_row + offset

    for _ in range(row_start, row_end):
        dY_row = tl.load(dY_ptr + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)

        # Get cached rms
        rstd_row = tl.load(RSTD_ptr)

        X_row = X_row.to(tl.float32)

        m = dY_row * W_row

        dX_row = rstd_row * m

        dX_row += (rstd_row) * (-(1 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row, axis=0) * X_row)


        dW_row += dY_row * (X_row * rstd_row)

        tl.store(dX_ptr + col_offsets, dX_row.to(X_dtype), mask=mask)

        dY_ptr += dY_row_stride
        dX_ptr += dX_row_stride
        X_ptr += X_row_stride
        RSTD_ptr += RSTD_row_stride

    tl.store(dW_ptr + row_block_id * dW_row_stride + col_offsets, dW_row, mask=mask)


def rms_norm_backward(dY, X, W, RSTD, offset, BLOCK_SIZE, num_warps, in_place, row_mode):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count

    # fp32 for numerical stability especially.
    _dW = torch.empty((sm_count, n_cols), dtype=torch.float32, device=W.device)

    if n_cols > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    rows_per_program = math.ceil(n_rows / sm_count)
    grid = (sm_count,)

    if in_place is True:
        dX = dY
    else:
        dX = torch.zeros_like(dY)

    # XPU-specific optimization
    kernel_args = {}

    if BLOCK_SIZE > 256 or n_rows < 4096 * 8 or row_mode:
        _rms_norm_backward_kernel[grid](
            dY,
            dY.stride(0),
            dX,
            dX.stride(0),
            X,
            X.stride(0),
            torch_to_triton_dtype[X.dtype],
            W,
            W.stride(0),
            RSTD,
            RSTD.stride(0),
            _dW,
            _dW.stride(0),
            n_rows,
            n_cols,
            offset,
            rows_per_program,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            **kernel_args,  # XPU-specific optimization
        )
    else:
        BLOCK_ROW = 16
        kernel_args["BLOCK_ROW"] = BLOCK_ROW
        _block_rms_norm_backward_kernel[grid](
            dY,
            dY.stride(0),
            dX,
            dX.stride(0),
            X,
            X.stride(0),
            torch_to_triton_dtype[X.dtype],
            W,
            W.stride(0),
            RSTD,
            RSTD.stride(0),
            _dW,
            _dW.stride(0),
            n_rows,
            n_cols,
            offset,
            rows_per_program,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            **kernel_args,  # XPU-specific optimization
        )
    dX = dX.view(*shape)
    dW = _dW.sum(dim=0).to(W.dtype)

    return dX, dW
class RMSNormFunction(torch.autograd.Function):
    """
    Performs RMSNorm (Root Mean Square Normalization), which normalizes the input tensor `X` using the
    weight tensor `W`, with an optional offset and casting mode.

    Some models use an 'offset' to shift the weight tensor `W` by a constant value. For example, Gemma
    uses an offset of 1.0, so the computation becomes `(X / RMS(X)) * (W + 1.0)` instead of the usual
    `(X / RMS(X)) * W`. You can pass the offset value as an argument to the forward function.

    In addition, different models cast their inputs at different places during RMSNorm computation. For
    example, Gemma casts everything to fp32 nefore starting the computation, while Llama casts only the
    inverse RMS to fp32. You can specify the casting mode using the `casting_mode` argument. We currently
    support the following casting modes (they match HuggingFace Transformers' implementations):
    - 'llama': matches the Llama implementation, where only the inverse RMS is computed on fp32.
    - 'gemma': matches the Gemma implementation, where everything is cast to fp32, then computed, then cast back to the original dtype.
    - 'none': no casting is done. The computation is done in the original dtype. This saves memory and is slightly faster, but has more error w.r.t. the original implementation.

    `in_place` option means whether to in_place modify dY to store dX. This is default to `True` to save memory. However, under certain cases, it can produce incorrect inputs.
        For example, gemma2 uses two rmsnorm sequentially with residual in between. The resesidual part needs dY so it cannot be modified in-place.
        Therefore, for the patching of RMSNorm in gemma2, we set `in_place` to `False`
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, eps, offset=0.0, in_place=True, row_mode=None):
        """
        X: (B, T, H) or (BxT, H)
        W: (H,)
        """
        Y, X, RSTD, BLOCK_SIZE, num_warps = rms_norm_forward(X, W, eps, offset, row_mode)
        ctx.offset = offset
        ctx.in_place = in_place
        ctx.row_mode = row_mode
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(X, W, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        """
        Y: (B, T, H) or (BxT, H)
        """
        X, W, RSTD = ctx.saved_tensors
        dX, dW = rms_norm_backward(
            dY, X, W, RSTD, ctx.offset, ctx.BLOCK_SIZE, ctx.num_warps, ctx.in_place, ctx.row_mode
        )
        return dX, dW, None, None, None, None, None
    
class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size,
        eps=1e-6,
        offset=0.0,
        init_fn="ones",
        in_place=True,
        row_mode=None,
    ):
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.weight = nn.Parameter(torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size))
        self.variance_epsilon, self.offset, self.in_place, self.row_mode = (
            eps,
            offset,
            in_place,
            row_mode,
        )

    def forward(self, hidden_states):
        return RMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            self.offset,
            self.in_place,
            self.row_mode,
        )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, offset={self.offset}, in_place={self.in_place}, row_mode={self.row_mode}"

if __name__ == "__main__":
    # Example usage
    x = torch.randn(2, 3, 4).to("cuda")
    w = torch.randn(4).to("cuda")
    rms_norm = RMSNorm(hidden_size=4, eps=1e-6, offset=0.0, in_place=True, row_mode=False).to("cuda")
    y = rms_norm(x)
    print(y.shape)  # Should be (2, 3, 4)