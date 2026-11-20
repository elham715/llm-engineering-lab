from pydantic_core.core_schema import dataclass_schema
import math
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F


from attention import CausalSelfAttention

@dataclass
class GPT2Config:
    block_size: int =1024
    vocab_size:int= 50257
    n_layers:int=12
    n_head:int=12
    n_embd:int=768
    dropout:float=0.1
    bisa:bool=True
    layer_norm_epsilon:float=1e-5


class NewGELU(nn.Module):
    def forward(self,x: torch.Tensor)->torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self,ndim:int,bias:bool=True,eps:float=1e-5):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(ndim))
        self.bias=nn.Parameter(torch.zeros(ndim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.bias is not None:
            return self.weight * x_norm + self.bias
        return self.weight * x_norm


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = NewGELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):

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

