"""SwiGLU Gated Feed-Forward Network.

Reference: Noam Shazeer (2020) - GLU Variants Improve Transformer.
Used in: LLaMA (1, 2, 3), Mistral, Gemma, DeepSeek, PaLM.

Mathematical formulation:
    SwiGLU(x) = ( Swish_1(x W_gate) * (x W_up) ) W_down
where Swish_1(z) = z * sigmoid(z) = SiLU(z).

Hidden dimension formula:
    d_ffn = floor( (8/3) * d_model ) rounded up to multiple of 256.
This preserves parameter parity with standard 4 * d_model MLPs while drastically
increasing gradient expressiveness.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def find_multiple(n: int, k: int = 256) -> int:
    """Rounds n up to the nearest multiple of k."""
    return ((n + k - 1) // k) * k


class SwiGLU(nn.Module):
    """Gated Feed-Forward Network with SiLU activation."""

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        multiple_of: int = 256,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim

        if hidden_dim is None:
            # Standard LLaMA heuristic: 8/3 * dim rounded to multiple of 256
            raw_hidden = int(2 * (4 * dim) / 3)
            hidden_dim = find_multiple(raw_hidden, multiple_of)

        self.hidden_dim = hidden_dim

        # Gate projection: computes gating activation z
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        # Up projection: computes information signal
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        # Down projection: projects gated representation back to d_model
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass:
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        # Element-wise modulation and projection back to model dimension
        return self.w_down(gate * up)
