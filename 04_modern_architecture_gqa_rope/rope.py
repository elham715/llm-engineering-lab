"""Rotary Position Embeddings (RoPE) from First Principles.

Reference: Su et al. (2021) - RoFormer: Enhanced Transformer with Rotary Position Embedding.
Mathematical property:
    <R_m q, R_n k> = g(q, k, m - n)
The inner product between query at position m and key at position n depends
strictly on the relative displacement (m - n), encoding relative geometry directly
into the attention dot product.
"""

from typing import Tuple
import torch
import torch.nn as nn


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputes cosine and sine frequency matrices for RoPE.

    Args:
        dim: Dimension of each attention head (must be even).
        max_seq_len: Maximum sequence length supported.
        theta: Base frequency hyperparameter (default: 10000.0).
        device: Target torch device.

    Returns:
        cos: Tensor of shape (max_seq_len, dim)
        sin: Tensor of shape (max_seq_len, dim)
    """
    assert dim % 2 == 0, f"Head dimension must be even for RoPE, got {dim}"

    # Frequencies theta_i = 1.0 / (theta ** (2i / dim)) for i in [0, 1, ..., dim/2 - 1]
    # In log-space for numerical stability:
    freq_exponents = torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
    inv_freq = 1.0 / (theta ** freq_exponents)  # Shape: (dim / 2,)

    # Time indices t = [0, 1, ..., max_seq_len - 1]
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)  # Shape: (max_seq_len,)

    # Outer product to form angle matrix: (max_seq_len, dim / 2)
    freqs = torch.outer(t, inv_freq)

    # Repeat frequencies for real and imaginary pairs: (max_seq_len, dim)
    # [theta_0, theta_1, ..., theta_{d/2-1}, theta_0, theta_1, ..., theta_{d/2-1}]
    freqs = torch.cat([freqs, freqs], dim=-1)

    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates the upper and lower halves of each head dimension.

    Splits x into [x1, x2] along the last dimension and returns [-x2, x1].
    Shape: (..., dim) -> (..., dim)
    """
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embeddings to query and key tensors.

    Equation:
        x_rotated = (x * cos) + (rotate_half(x) * sin)

    Args:
        xq: Query tensor of shape (batch, n_heads, seq_len, head_dim)
        xk: Key tensor of shape (batch, n_kv_heads, seq_len, head_dim)
        cos: Precomputed cosine tensor of shape (max_seq_len, head_dim)
        sin: Precomputed sine tensor of shape (max_seq_len, head_dim)
        position_ids: Optional 1D or 2D tensor of token positions.

    Returns:
        xq_out: RoPE-encoded query tensor with identical shape and dtype.
        xk_out: RoPE-encoded key tensor with identical shape and dtype.
    """
    seq_len = xq.shape[2]

    if position_ids is None:
        # Default sequential positions: [0, 1, ..., seq_len - 1]
        cos_slice = cos[:seq_len, :]  # (seq_len, head_dim)
        sin_slice = sin[:seq_len, :]  # (seq_len, head_dim)
        # Reshape for broadcasting: (1, 1, seq_len, head_dim)
        cos_slice = cos_slice.unsqueeze(0).unsqueeze(0)
        sin_slice = sin_slice.unsqueeze(0).unsqueeze(0)
    else:
        # Gather from arbitrary position IDs (e.g. KV cache decode step)
        # position_ids shape: (batch, seq_len)
        cos_slice = cos[position_ids].unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        sin_slice = sin[position_ids].unsqueeze(1)  # (batch, 1, seq_len, head_dim)

    cos_slice = cos_slice.to(dtype=xq.dtype, device=xq.device)
    sin_slice = sin_slice.to(dtype=xq.dtype, device=xq.device)

    xq_out = (xq * cos_slice) + (rotate_half(xq) * sin_slice)
    xk_out = (xk * cos_slice) + (rotate_half(xk) * sin_slice)

    return xq_out, xk_out


class RotaryEmbedding(nn.Module):
    """Encapsulated RoPE module for attention layers."""

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        cos, sin = precompute_freqs_cis(dim, max_seq_len, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        position_ids: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return apply_rotary_emb(xq, xk, self.cos, self.sin, position_ids)
