"""
gpt-oss style SwiGLU Feed-Forward Network

Reference SwiGLU implementation:
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings, ensure_contiguous


@triton.jit
def silu(x, alpha):
    return x * tl.sigmoid(alpha * x)


@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, alpha, limit, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    a_row = tl.clamp(a_row, min=-float('inf'), max=limit)
    b_row = tl.clamp(b_row, min=-limit, max=limit)
    c_row = (b_row + 1.0) * silu(a_row, alpha).cast(b_row.dtype)
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


@triton.jit
def _swiglu_backward_kernel(dc_ptr, a_ptr, b_ptr, alpha, limit, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)

    a_row = tl.clamp(a_row, min=-float('inf'), max=limit)
    b_row = tl.clamp(b_row, min=-limit, max=limit)
    sig_a = tl.sigmoid(a_row * alpha).cast(b_row.dtype)
    silu_a = (a_row * sig_a)
    da_row = dc_row * (b_row + 1.0) * (silu_a * (1.0 - sig_a) * alpha + sig_a)
    db_row = (dc_row * silu_a)

    tl.store(a_ptr + col_offsets, da_row, mask=mask)
    tl.store(b_ptr + col_offsets, db_row, mask=mask)
    tl.store(dc_ptr + col_offsets, (b_row + 1.0) * silu_a, mask=mask)


def swiglu_forward(a, b, alpha, limit):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        alpha,
        limit,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return c.view(*ori_shape)


def swiglu_backward(a, b, dc, alpha, limit):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_backward_kernel[(n_rows,)](
        dc,
        a,
        b,
        alpha,
        limit,
        dc.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a.view(*ori_shape), b.view(*ori_shape), dc.view(*ori_shape)

@torch.compile
def sum_compute(gate_grad, w_gate, up_grad, w_up):
    
    return gate_grad @ w_gate + up_grad @ w_up

@torch.compile
def multi(mat1, mat2, x):
    b, s, h1 = mat1.shape
    return mat1.view(-1, h1).transpose(-1, -2) @ x.view(b * s, -1), mat2.view(-1, h1).transpose(-1, -2) @ x.view(b * s, -1)


    
class MemoryEfficientSwiGLUMLP(torch.autograd.Function):
    """
    Memory-optimized SwiGLU MLP with selective recomputation.
    """
    
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, x, w_gate, w_up, w_down, alpha, limit, sum_compute, multi):
        gate = x @ w_gate.T
        up = x @ w_up.T

        # TODO: Replace with fused swiglu_forward kernel
        activation_out = swiglu_forward(gate, up, alpha, limit)

        # TODO: Save tensors for backward
        ctx.save_for_backward(x, gate, up, w_gate, w_up, w_down)
        ctx.alpha = alpha
        ctx.limit = limit
        ctx.sum_compute = sum_compute
        ctx.multi = multi

        return activation_out @ w_down.T
    
    
    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        x, gate, up, w_gate, w_up, w_down = ctx.saved_tensors
        alpha = ctx.alpha
        limit = ctx.limit
        sum_compute = ctx.sum_compute
        multi = ctx.multi

        with torch.no_grad():
            gate_grad, up_grad, activation_out = swiglu_backward(gate, up, grad_output @ w_down, alpha, limit)
            b, s, h = grad_output.shape
            b, s, h1 = gate_grad.shape
            grad_w_down = grad_output.view(b * s, -1).transpose(-1, -2) @ activation_out.view(b * s, -1)
            
            del gate, up, grad_output, activation_out
            
            return (
                sum_compute(gate_grad, w_gate, up_grad, w_up),
                *multi(gate_grad, up_grad, x.detach()),
                grad_w_down,
                None,
                None,
                None,
                None
            )


class SwiGLUFeedForward(nn.Module):
    """
    gpt-oss style SwiGLU.
    
    output = W_down @ ((up + 1) * gate * sigmoid(gate * alpha))
    """
    
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.alpha = 1.702
        self.limit = 7.0

        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        
        self.fn = sum_compute
        self.multi = multi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return MemoryEfficientSwiGLUMLP.apply(
            x, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight, self.alpha, self.limit, sum_compute, multi
        )
