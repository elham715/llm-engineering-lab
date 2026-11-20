"""
Causal Multi-Head Self-Attention from Scratch in Pure PyTorch.
Implements the core attention engine of GPT-2 (Radford et al., 2019)
and Vaswani et al. (2017).

Formula:
    Attention(Q, K, V) = Softmax( (Q K^T) / sqrt(d_k) + M ) V
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention module.
    
    Args:
        d_model (int): Hidden embedding dimension (e.g., 768 for GPT-2 Small).
        n_heads (int): Number of attention heads (e.g., 12 for GPT-2 Small).
        max_seq_len (int): Maximum sequence context length (e.g., 1024).
        dropout (float): Dropout probability for attention weights and output.
    """
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        max_seq_len: int = 1024,
        dropout: float = 0.0
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head: 768 // 12 = 64
        self.dropout_p = dropout
        
        # 1. Combined Q, K, V projection in a single linear layer for maximum GPU parallelism
        # Input: (B, T, d_model) -> Output: (B, T, 3 * d_model)
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        
        # 2. Output projection layer W_O
        # Input: (B, T, d_model) -> Output: (B, T, d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        
        # 3. Regularization dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # 4. Causal Mask: Lower-triangular matrix buffer (1s in lower triangle, 0s in upper)
        # Registered as buffer so it moves with the model across CPU/GPU but is NOT a learnable parameter.
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False,
        past_kv=None,
        use_cache: bool = False,
    ):
        """
        Forward pass for Causal Self-Attention with optional KV-Cache.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C)
            return_attention_weights (bool): Whether to return raw attention matrix for inspection
            past_kv (tuple[Tensor, Tensor] | None): Cached (k, v) of shape (B, n_heads, T_past, d_k)
            use_cache (bool): If True, return updated (k, v) cache alongside output
        """
        B, T, C = x.size()
        assert C == self.d_model, f"Expected hidden dimension {self.d_model}, got {C}"

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(2)
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        present_kv = (k, v) if use_cache else None
        T_total = k.size(2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
        att = att.masked_fill(self.bias[:, :, past_len : past_len + T, :T_total] == 0, float('-inf'))

        att_weights = F.softmax(att, dim=-1)
        att_probs = self.attn_dropout(att_weights)

        y = att_probs @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(y))

        if return_attention_weights and use_cache:
            return out, att_weights, present_kv
        if return_attention_weights:
            return out, att_weights
        if use_cache:
            return out, present_kv
        return out

