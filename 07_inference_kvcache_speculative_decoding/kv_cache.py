"""Dynamic Key-Value (KV) Cache from First Principles.

Silicon Architecture Context:
    During autoregressive token generation (decode mode), calculating self-attention
    naively recomputes K and V projections for ALL past tokens at every step (O(T^2) FLOPs
    and massive redundant memory allocation).
    
    A KV Cache stores past Key and Value representations in GPU memory.
    At step t, the model only computes Q, K, V for the single new token:
        K_new in R^{batch x n_kv x 1 x head_dim}
        V_new in R^{batch x n_kv x 1 x head_dim}
    and appends them to the cache. FLOPs drop from O(T^2) to O(T)!
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class DynamicKVCache:
    """Pre-allocated, zero-copy dynamic KV Cache for autoregressive decoding."""

    def __init__(
        self,
        batch_size: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.batch_size = batch_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        # Preallocate static buffers in memory to prevent dynamic GPU re-allocation
        self.key_cache = torch.zeros(
            (batch_size, n_kv_heads, max_seq_len, head_dim),
            device=device,
            dtype=dtype,
        )
        self.value_cache = torch.zeros(
            (batch_size, n_kv_heads, max_seq_len, head_dim),
            device=device,
            dtype=dtype,
        )
        self.current_len = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Appends new incoming key and value states into the pre-allocated cache.

        Args:
            key_states: (batch, n_kv_heads, seq_len, head_dim)
            value_states: (batch, n_kv_heads, seq_len, head_dim)

        Returns:
            keys: Cached keys up to the current sequence length (batch, n_kv_heads, total_len, head_dim)
            values: Cached values up to the current sequence length (batch, n_kv_heads, total_len, head_dim)
        """
        incoming_len = key_states.shape[-2]
        new_total_len = self.current_len + incoming_len

        if new_total_len > self.max_seq_len:
            raise ValueError(
                f"KV cache capacity exceeded: total {new_total_len} > max {self.max_seq_len}"
            )

        # Slice-assignment into pre-allocated memory buffer (zero memory reallocation!)
        self.key_cache[:, :, self.current_len : new_total_len, :] = key_states
        self.value_cache[:, :, self.current_len : new_total_len, :] = value_states

        self.current_len = new_total_len

        # Return view up to active length
        active_keys = self.key_cache[:, :, :self.current_len, :]
        active_values = self.value_cache[:, :, :self.current_len, :]
        return active_keys, active_values

    def reset(self):
        """Resets the active sequence length pointer."""
        self.current_len = 0

    @property
    def seq_len(self) -> int:
        return self.current_len
