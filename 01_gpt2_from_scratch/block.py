"""
Transformer Block Implementation from Scratch in Pure PyTorch.
Implements the Pre-LayerNorm Transformer Block of GPT-2 (Radford et al., 2019)
featuring Causal Self-Attention, MLP with GELU activation, and uninterrupted
Additive Residual Stream highways.

Equations:
    1. Attention Highway: x = x + Attention(LayerNorm_1(x))
    2. MLP Highway:       x = x + MLP(LayerNorm_2(x))
    where MLP(h) = Dropout(W_2 * GELU(W_1 * h + b_1) + b_2)
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import CausalSelfAttention


@dataclass
class GPT2Config:
    """Configuration hyperparameters for GPT-2 Architecture."""
    block_size: int = 1024       # Maximum context length (T)
    vocab_size: int = 50257      # Vocabulary size (50,000 BPE tokens + 256 bytes + 1 special)
    n_layer: int = 12            # Number of Transformer blocks (depth L)
    n_head: int = 12             # Number of attention heads
    n_embd: int = 768            # Embedding dimension (d_model)
    dropout: float = 0.1         # Dropout probability
    bias: bool = True            # True: use bias in Linear layers and LayerNorms, like GPT-2
    layer_norm_epsilon: float = 1e-5


class NewGELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) with polynomial approximation.
    Reference: Section 2 of GPT-2 paper (Radford et al., 2019)
               and Hendrycks & Gimpel (2016).
    Formula:
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """
    Layer Normalization from first principles with learnable affine parameters (gamma, beta).
    Normalizes across the last dimension (channels / d_model) independently for each token.
    Formula:
        LN(x) = ((x - mean) / sqrt(var + eps)) * gamma + beta
    """
    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean and variance along feature channel dimension (last dim)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Standardize to zero mean and unit variance
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Affine transformation
        if self.bias is not None:
            return self.weight * x_norm + self.bias
        return self.weight * x_norm


class MLP(nn.Module):
    """
    Feed-Forward Network (MLP) sublayer.
    Acts as an associative key-value memory bank (Geva et al., 2021).
    Expands channel dimension by 4x (768 -> 3072 -> 768).
    
    Structure:
        x -> Linear(d, 4d) -> GELU -> Linear(4d, d) -> Dropout -> out
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        # 1. First projection (Keys): 768 -> 3072
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # 2. Smooth non-linear activation
        self.gelu = NewGELU()
        # 3. Second projection (Values): 3072 -> 768
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        # 4. Residual dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    GPT-2 Pre-LayerNorm Transformer Block.
    
    Contains two sublayers:
        1. Causal Self-Attention
        2. Feed-Forward MLP
    Both sublayers are wrapped with Pre-LayerNorm and Additive Residual Connections.
    
    Data Flow:
        Input x
           ├──► LayerNorm_1 ──► CausalSelfAttention ──► (⊕)
           │                                             ▲
           └─────────────────────────────────────────────┘  Residual Shortcut 1
           │
           ├──► LayerNorm_2 ──► MLP ──────────────────► (⊕)
           │                                             ▲
           └─────────────────────────────────────────────┘  Residual Shortcut 2
           │
        Output x
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        
        # Sublayer 1: Attention branch
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(
            d_model=config.n_embd,
            n_heads=config.n_head,
            max_seq_len=config.block_size,
            dropout=config.dropout
        )
        
        # Sublayer 2: MLP branch
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False):
        """Forward pass with Pre-LayerNorm, Residual Connections, and optional KV-Cache."""
        attn_res = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        if use_cache:
            attn_out, present_kv = attn_res
        else:
            attn_out, present_kv = attn_res, None

        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))

        if use_cache:
            return x, present_kv
        return x

