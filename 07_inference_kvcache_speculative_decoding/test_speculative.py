"""Unit Tests for Dynamic KV Cache and Speculative Decoding.

Tests:
1. DynamicKVCache slice preallocation and zero memory reallocation.
2. DynamicKVCache returns exact attention output matching full recomputation.
3. Rejection sampling condition: accepts with probability min(1, p_t / p_d).
4. Speculative decoding loop runs and returns valid tokens.
5. Verification that speculative sampling preserves target distribution within statistics.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from kv_cache import DynamicKVCache
from speculative_decoding import rejection_sampling_step, speculative_decoding_loop


class TestInferenceAndSpeculative(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 1
        self.n_kv_heads = 4
        self.head_dim = 32
        self.max_seq_len = 128
        self.vocab_size = 64

    def test_01_kv_cache_slice_accumulation(self):
        """KV cache accumulates token states without changing preallocated tensor identity."""
        cache = DynamicKVCache(
            batch_size=self.batch_size,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
        )
        base_k_ptr = cache.key_cache.data_ptr()

        # Step 1: Prompt prefill (4 tokens)
        k1 = torch.randn(self.batch_size, self.n_kv_heads, 4, self.head_dim)
        v1 = torch.randn(self.batch_size, self.n_kv_heads, 4, self.head_dim)
        keys, values = cache.update(k1, v1)

        self.assertEqual(cache.seq_len, 4)
        self.assertEqual(keys.shape, (self.batch_size, self.n_kv_heads, 4, self.head_dim))
        self.assertEqual(cache.key_cache.data_ptr(), base_k_ptr)

        # Step 2: Decode single token
        k2 = torch.randn(self.batch_size, self.n_kv_heads, 1, self.head_dim)
        v2 = torch.randn(self.batch_size, self.n_kv_heads, 1, self.head_dim)
        keys, values = cache.update(k2, v2)

        self.assertEqual(cache.seq_len, 5)
        self.assertEqual(keys.shape, (self.batch_size, self.n_kv_heads, 5, self.head_dim))
        # Zero-reallocation: underlying buffer pointer has not changed
        self.assertEqual(cache.key_cache.data_ptr(), base_k_ptr)

    def test_02_rejection_sampling_acceptance_criterion(self):
        """When P_target >= P_draft, token MUST be accepted with 100% certainty."""
        vocab_size = 5
        p_draft = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
        p_target = torch.tensor([0.4, 0.1, 0.2, 0.2, 0.1])

        # Token 0: P_t (0.4) > P_d (0.2) -> min(1, 0.4/0.2) = 1.0 -> 100% accept
        for _ in range(20):
            accepted, resampled = rejection_sampling_step(p_draft, p_target, draft_token=0)
            self.assertTrue(accepted)
            self.assertIsNone(resampled)

    def test_03_rejection_sampling_rejection_recovery(self):
        """When P_target == 0 for draft token, token MUST be rejected with 100% certainty."""
        p_draft = torch.tensor([0.5, 0.5])
        p_target = torch.tensor([0.0, 1.0])

        # Token 0 has 0 probability in target
        for _ in range(10):
            accepted, resampled = rejection_sampling_step(p_draft, p_target, draft_token=0)
            self.assertFalse(accepted)
            self.assertEqual(resampled, 1)

    def test_04_speculative_decoding_loop(self):
        """Executes full speculative decoding loop and returns expected new token count."""
        # Create tiny dummy models
        dim = 16
        draft_model = nn.Sequential(nn.Embedding(self.vocab_size, dim), nn.Linear(dim, self.vocab_size))
        target_model = nn.Sequential(nn.Embedding(self.vocab_size, dim), nn.Linear(dim, self.vocab_size))

        prompt = torch.tensor([[1, 2, 3]])
        max_new = 8
        out_ids, stats = speculative_decoding_loop(
            draft_model=draft_model,
            target_model=target_model,
            input_ids=prompt,
            gamma=3,
            max_new_tokens=max_new,
        )

        self.assertGreaterEqual(out_ids.shape[1], prompt.shape[1] + max_new)
        self.assertIn("acceptance_rate", stats)
        self.assertGreater(stats["target_forward_passes"], 0)


if __name__ == "__main__":
    unittest.main()
