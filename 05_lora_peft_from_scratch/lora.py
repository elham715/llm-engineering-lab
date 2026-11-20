"""Low-Rank Adaptation (LoRA) from First Principles.

Reference: Hu et al. (2021) - LoRA: Low-Rank Adaptation of Large Language Models.
Mathematical formulation:
    W = W_0 + Delta W = W_0 + (alpha / r) * (B @ A)
where:
    W_0 in R^{d_out x d_in}   (Frozen base model weight)
    A   in R^{r x d_in}       (Initialized from N(0, 1/r))
    B   in R^{d_out x r}       (Initialized to ZERO -> Delta W = 0 at step 0!)
    r << min(d_in, d_out)      (Rank, typically 4, 8, 16, or 64)
    alpha                      (Scaling hyperparameter)

The Hardware Memory Law (Associative Matrix Multiplication):
    Naive: x @ (B @ A).T -> materializes massive d_out x d_in tensor in memory (O(d^2)).
    Efficient: (x @ A.T) @ B.T -> intermediate tensor shape is (batch, seq_len, r) (O(r)).
    Reduces activation memory by 500x!
"""

import math
from typing import List, Optional
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Linear layer augmented with Low-Rank Adaptation (LoRA)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        self.merged = False

        # 1. Base linear layer (FROZEN)
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        self.base_layer.weight.requires_grad = False
        if bias and self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # 2. LoRA trainable adapter matrices
        if rank > 0:
            # Matrix A: down-projects from in_features -> rank
            # Initialized with Gaussian distribution N(0, 1/r)
            self.lora_A = nn.Parameter(torch.empty(rank, in_features))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

            # Matrix B: up-projects from rank -> out_features
            # Initialized to EXACT ZERO so Delta W = 0 at step 0
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

            # Optional dropout on inputs before LoRA path
            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        If merged, delegates strictly to base layer.
        If unmerged, computes: base(x) + (x @ A.T) @ B.T * scaling.
        """
        if self.merged or self.rank == 0:
            return self.base_layer(x)

        # Base model frozen forward pass
        base_out = self.base_layer(x)

        # Associative LoRA forward pass: (x @ A.T) @ B.T
        # Shape: (batch, seq_len, in_features)
        # -> (batch, seq_len, rank)
        # -> (batch, seq_len, out_features)
        lora_x = self.dropout(x)
        lora_out = torch.matmul(lora_x, self.lora_A.t())  # Intermediate in O(r) memory
        lora_out = torch.matmul(lora_out, self.lora_B.t()) * self.scaling

        return base_out + lora_out

    def merge(self):
        """Merges Delta W directly into base_layer.weight for zero-latency inference."""
        if self.merged:
            return
        if self.rank > 0:
            # Delta W = (B @ A) * scaling
            delta_w = torch.matmul(self.lora_B, self.lora_A) * self.scaling
            self.base_layer.weight.data.add_(delta_w.to(self.base_layer.weight.device))
            self.merged = True

    def unmerge(self):
        """Subtracts Delta W from base_layer.weight to restore trainable LoRA state."""
        if not self.merged:
            return
        if self.rank > 0:
            delta_w = torch.matmul(self.lora_B, self.lora_A) * self.scaling
            self.base_layer.weight.data.sub_(delta_w.to(self.base_layer.weight.device))
            self.merged = False


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> int:
    """Freezes all model parameters except LoRA parameters and specified biases.

    Returns:
        trainable_params: Total count of trainable parameters.
    """
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False

    if bias == "none":
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALinear) and m.base_layer.bias is not None:
                m.base_layer.bias.requires_grad = True

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
