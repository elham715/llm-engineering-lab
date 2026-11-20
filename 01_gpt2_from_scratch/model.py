"""
Full GPT-2 (124M) Decoder-Only Model Assembly in Pure PyTorch.
Implements Chapters 03-04:
    - Token Embeddings (wte) + Learned Positional Embeddings (wpe)
    - 12 Stacked Pre-LN TransformerBlocks (nn.ModuleList)
    - Final LayerNorm (ln_f) + Unembedding Head (lm_head)
    - Weight Tying (wte.weight == lm_head.weight) -> 124,439,808 unique parameters
    - OpenAI / HuggingFace Checkpoint Transfer with Conv1D transposition (.t())
    - Autoregressive Generation Loop (Greedy, Temperature, Top-K)
"""

import math
from typing import Callable, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from block import GPT2Config, LayerNorm, TransformerBlock


GPT2_SPECS = {
    "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),   # 124M
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
    "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
    "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
}

# OpenAI TensorFlow Conv1D stores [in, out]; PyTorch nn.Linear stores [out, in].
CONV1D_TRANSPOSE_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


class GPT2Model(nn.Module):
    """Complete GPT-2 Causal Language Model with Weight Tying."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias, eps=config.layer_norm_epsilon),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying: bind input embedding and output unembedding to the same tensor in RAM
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights and apply GPT-2 scaled init to residual projections
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self, unique_only: bool = True) -> int:
        """Count model parameters; deduplicates tied weights via data_ptr() when unique_only=True."""
        if not unique_only:
            return sum(p.numel() for p in self.parameters())
        seen_ptrs = set()
        total = 0
        for p in self.parameters():
            ptr = p.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                total += p.numel()
        return total

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """
        Forward pass from token IDs (B, T) to vocabulary logits (B, T, vocab_size).
        Supports optional KV-Cache for fast single-token autoregressive decoding.
        """
        device = idx.device
        _, t = idx.size()
        past_len = past_key_values[0][0].size(2) if past_key_values is not None else 0
        assert past_len + t <= self.config.block_size, (
            f"Sequence length {past_len + t} exceeds block_size {self.config.block_size}"
        )

        pos = torch.arange(past_len, past_len + t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        presents = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            layer_past = past_key_values[i] if past_key_values is not None else None
            if use_cache:
                x, present = block(x, past_kv=layer_past, use_cache=True)
                presents.append(present)
            else:
                x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        if use_cache:
            return logits, loss, presents
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str = "gpt2") -> "GPT2Model":
        """Load official OpenAI / HuggingFace GPT-2 weights with Conv1D transposition."""
        if model_type not in GPT2_SPECS:
            raise ValueError(f"Unsupported model_type '{model_type}'. Choose from {list(GPT2_SPECS)}")

        from transformers import GPT2LMHeadModel

        spec = GPT2_SPECS[model_type]
        config = GPT2Config(
            vocab_size=50257,
            block_size=1024,
            bias=True,
            dropout=0.0,
            **spec,
        )
        model = cls(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        hf_sd = hf_model.state_dict()
        hf_keys = [
            k for k in hf_sd.keys()
            if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")
        ]

        assert len(sd_keys) == len(hf_keys), (
            f"State-dict key count mismatch: local={len(sd_keys)} vs hf={len(hf_keys)}"
        )

        for key in hf_keys:
            src = hf_sd[key]
            if key.endswith(CONV1D_TRANSPOSE_SUFFIXES):
                assert src.shape[::-1] == sd[key].shape, (
                    f"Transpose shape mismatch for {key}: {src.shape[::-1]} vs {sd[key].shape}"
                )
                with torch.no_grad():
                    sd[key].copy_(src.t())
            else:
                assert src.shape == sd[key].shape, (
                    f"Shape mismatch for {key}: {src.shape} vs {sd[key].shape}"
                )
                with torch.no_grad():
                    sd[key].copy_(src)

        return model

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        on_token: Optional[Callable[[int], None]] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressively sample max_new_tokens using KV-Cache + Temperature + Top-K."""
        past_kv = None
        for step in range(max_new_tokens):
            if use_cache and past_kv is not None:
                if past_kv[0][0].size(2) >= self.config.block_size:
                    keep = self.config.block_size - 1
                    past_kv = [(k[:, :, -keep:, :], v[:, :, -keep:, :]) for k, v in past_kv]
                logits, _, past_kv = self(idx[:, -1:], past_key_values=past_kv, use_cache=True)
            else:
                idx_cond = (
                    idx if idx.size(1) <= self.config.block_size
                    else idx[:, -self.config.block_size:]
                )
                if use_cache:
                    logits, _, past_kv = self(idx_cond, use_cache=True)
                else:
                    logits, _ = self(idx_cond)

            logits = logits[:, -1, :]  # (B, vocab_size)

            if temperature <= 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, k)
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
            if on_token is not None and idx_next.size(0) == 1:
                on_token(int(idx_next.item()))

        return idx

