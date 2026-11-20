"""Root Mean Square Layer Normalization (RMSNorm).

Reference: Zhang & Sennrich (2019) - Root Mean Square Layer Normalization.
Mathematical definition:
    RMS(x) = sqrt( (1 / d) * sum_{i=1}^d x_i^2 + eps )
    RMSNorm(x) = (x / RMS(x)) * gamma

Key benefits over standard LayerNorm:
1. No mean centering: saves 1 reduction pass across the hidden dimension.
2. 10% to 50% faster on modern GPUs.
3. Scale-invariant: RMSNorm(c * x) == RMSNorm(x) for any positive scalar c.
4. Upcasts internal variance accumulation to FP32 to prevent FP16 overflow at 65,504.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # Learnable affine scale parameter gamma (initialized to 1.0)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with FP32 variance precision preservation.

        Args:
            x: Input tensor of shape (..., dim)
        Returns:
            Normalized tensor with identical shape and input dtype.
        """
        orig_dtype = x.dtype
        # Upcast to FP32 to prevent numerical overflow in sum of squares
        x_fp32 = x.to(torch.float32)

        # Variance: mean of squares along the last dimension
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)

        # Scale by reciprocal square root
        x_norm = x_fp32 * torch.rsqrt(variance + self.eps)

        # Cast back to original dtype and scale by learnable weight gamma
        return (x_norm.to(orig_dtype)) * self.weight
