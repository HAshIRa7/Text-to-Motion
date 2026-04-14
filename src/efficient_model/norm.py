"""
Zero-Centered RMSNorm
"""

import torch
import torch.nn as nn

@torch.compile
def rmsnorm_forward(x, weight, eps):
    """Zero-Centered RMSNorm forward."""
    input_dtype = x.dtype
    x = x.float()
    x_squared = x * x
    mean_squared = x_squared.mean(dim=-1, keepdim=True)
    mean_squared_eps = mean_squared + eps
    rsqrt = torch.rsqrt(mean_squared_eps)
    normalized = x * rsqrt
    scale = 1.0 + weight.float()
    output = normalized * scale
    return output.to(input_dtype)

@torch.compile
def rmsnorm_backward(grad_output, x, weight, eps):
    """Zero-Centered RMSNorm backward."""
    with torch.no_grad():
        x = x.float()
        x_squared = x * x
        mean_squared = x_squared.mean(dim=-1, keepdim=True)
        mean_squared_eps = mean_squared + eps
        rsqrt = torch.rsqrt(mean_squared_eps)
        normalized = x * rsqrt

        x_grad = rsqrt * grad_output - (rsqrt**3) * x * (x * grad_output).mean(dim=-1, keepdim=True)
        x_grad *= (1 + weight.float())

    return x_grad, grad_output * normalized, None

class RMSNormFunction(torch.autograd.Function):
    """
    Template for memory-efficient and fused Zero-Centered RMSNorm autograd function.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        # TODO: Replace with fused implementation
        output = rmsnorm_forward(x, weight, eps)
        # TODO: Save tensors for backward (make it memory-efficient)
        ctx.save_for_backward(x, weight)  # TODO: Fill this
        ctx._saved_eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        eps = ctx._saved_eps
        return rmsnorm_backward(grad_output, x, weight, eps)


class RMSNorm(nn.Module):
    """
    Zero-Centered RMSNorm: y = x/rms(x) * (1 + weight), weight init to zeros.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RMSNormFunction.apply(x, self.weight, self.eps)
