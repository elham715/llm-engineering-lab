"""Modern Transformer Block (LLaMA-3 / Mistral Backbone).

Architecture:
    x -> RMSNorm -> GroupedQueryAttention (with RoPE) -> + x -> Residual Stream
      -> RMSNorm -> SwiGLU Gated Feed-Forward         -> +   -> Residual Stream

Key properties:
- Pre-normalization design: stable training up to 100+ layers without gradient explosion.
- SwiGLU activations: higher non-linear capacity per FLOP.
- GQA: reduces KV cache footprint by 4x to 8x.
- Rotary embeddings: relative position encoding directly in attention space.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from rmsnorm import RMSNorm
from swiglu import SwiGLU
from gqa import GroupedQueryAttention


class ModernTransformerBlock(nn.Module):
    """Full Modern Transformer Block integrating RMSNorm, GQA, RoPE, and SwiGLU."""

    def __init__(
        self,
        dim: int = 512,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        head_dim: Optional[int] = None,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        ffn_multiple_of: int = 256,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim

        # Pre-attention normalization
        self.attn_norm = RMSNorm(dim=dim, eps=eps)

        # Grouped-Query Attention with Rotary Embeddings
        self.attn = GroupedQueryAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            theta=theta,
        )

        # Pre-FFN normalization
        self.ffn_norm = RMSNorm(dim=dim, eps=eps)

        # SwiGLU Gated Feed-Forward Network
        self.ffn = SwiGLU(dim=dim, multiple_of=ffn_multiple_of)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through block.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            position_ids: Token indices for RoPE
            mask: Optional attention mask
            kv_cache: Optional past (K, V) tuple

        Returns:
            out: Processed tensor of shape (batch, seq_len, dim)
            new_kv: Updated (K, V) cache tuple
        """
        # Sub-layer 1: RMSNorm + GQA + Residual
        normed_attn = self.attn_norm(x)
        attn_out, new_kv = self.attn(
            normed_attn, position_ids=position_ids, mask=mask, kv_cache=kv_cache
        )
        h = x + attn_out

        # Sub-layer 2: RMSNorm + SwiGLU + Residual
        normed_ffn = self.ffn_norm(h)
        ffn_out = self.ffn(normed_ffn)
        out = h + ffn_out

        return out, new_kv
