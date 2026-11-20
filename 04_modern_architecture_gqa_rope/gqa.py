"""Grouped-Query Attention (GQA) with RoPE and Zero-Copy Broadcasting.

Reference: Ainslie et al. (2023) - GQA: Training Generalized Multi-Query Transformer Models
from Multi-Head Checkpoints.

Silicon Memory Architecture Principle:
    In standard Multi-Head Attention (MHA), every query head has a dedicated (K, V) head.
    During autoregressive token generation (decode mode), loading (K, V) from High Bandwidth
    Memory (VRAM) is bandwidth-bound (1 FLOP per byte transferred).
    
    GQA groups N_q query heads into N_kv key-value heads.
    For a 7B model with N_q = 32, N_kv = 8:
        KV cache memory transfer is reduced by 4x!
        Memory bandwidth saturation is eliminated!
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RotaryEmbedding, apply_rotary_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expands Key or Value tensor along head dimension to match Query heads.

    Zero-copy broadcasting using expand and reshape:
        Input:  (batch, n_kv_heads, seq_len, head_dim)
        Output: (batch, n_q_heads,  seq_len, head_dim) where n_q_heads = n_kv_heads * n_rep
    """
    if n_rep == 1:
        return x

    bs, n_kv_heads, seq_len, head_dim = x.shape
    # (batch, n_kv_heads, 1, seq_len, head_dim)
    # -> expand to (batch, n_kv_heads, n_rep, seq_len, head_dim)
    # -> reshape to (batch, n_kv_heads * n_rep, seq_len, head_dim)
    return (
        x.unsqueeze(2)
        .expand(bs, n_kv_heads, n_rep, seq_len, head_dim)
        .reshape(bs, n_kv_heads * n_rep, seq_len, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention Module with RoPE and Causal Masking."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = 8,
        head_dim: Optional[int] = None,
        max_seq_len: int = 4096,
        theta: float = 10000.0,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        assert n_heads % self.n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        self.num_groups = n_heads // self.n_kv_heads

        if head_dim is None:
            assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"
            self.head_dim = dim // n_heads
        else:
            self.head_dim = head_dim

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=bias)

        # Rotary position embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim, max_seq_len=max_seq_len, theta=theta
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for Grouped-Query Attention.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            position_ids: Token position indices for RoPE
            mask: Optional attention mask (batch, 1, seq_len, seq_len)
            kv_cache: Optional past (K, V) tuple for fast autoregressive generation

        Returns:
            out: Output tensor of shape (batch, seq_len, dim)
            new_kv: Updated (K, V) tuple
        """
        bs, seq_len, _ = x.shape

        # 1. Project Q, K, V
        # Q: (batch, seq_len, n_heads * head_dim) -> (batch, n_heads, seq_len, head_dim)
        xq = self.q_proj(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # K, V: (batch, n_kv_heads, seq_len, head_dim)
        xk = self.k_proj(x).view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = self.v_proj(x).view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 2. Apply Rotary Position Embeddings
        xq, xk = self.rotary_emb(xq, xk, position_ids=position_ids)

        # 3. KV Cache update (if provided)
        if kv_cache is not None:
            past_k, past_v = kv_cache
            xk = torch.cat([past_k, xk], dim=-2)
            xv = torch.cat([past_v, xv], dim=-2)
        new_kv = (xk, xv)

        # Total key/value sequence length (including history)
        kv_seq_len = xk.shape[-2]

        # 4. Repeat K and V along head dimension for grouped broadcasting
        keys = repeat_kv(xk, self.num_groups)  # (batch, n_heads, kv_seq_len, head_dim)
        values = repeat_kv(xv, self.num_groups)  # (batch, n_heads, kv_seq_len, head_dim)

        # 5. Scaled Dot-Product Attention: (Q @ K.T) * scale
        scores = torch.matmul(xq, keys.transpose(-2, -1)) * self.scale
        # Shape: (batch, n_heads, seq_len, kv_seq_len)

        # 6. Apply Causal Mask if seq_len > 1
        if mask is not None:
            scores = scores + mask
        elif seq_len > 1 and kv_cache is None:
            # Build lower-triangular causal mask
            causal_mask = torch.triu(
                torch.full((seq_len, kv_seq_len), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # 7. Softmax and Value aggregation
        attn_probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=x.dtype)
        # Output: (batch, n_heads, seq_len, head_dim)
        context = torch.matmul(attn_probs, values)

        # 8. Concatenate heads and project out
        # (batch, n_heads, seq_len, head_dim) -> (batch, seq_len, n_heads * head_dim)
        context = context.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        out = self.out_proj(context)

        return out, new_kv
