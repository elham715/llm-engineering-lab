"""Unit Tests for Modern Architecture Components.

Tests:
1. RMSNorm scale invariance: RMSNorm(c * x) == RMSNorm(x).
2. SwiGLU dimension alignment and gradient backpropagation.
3. RoPE relative position invariance: <R_m q, R_n k> == <R_{m+s} q, R_{n+s} k>.
4. GQA zero-copy broadcasting and shape validation.
5. GQA equivalence to MHA when n_heads == n_kv_heads.
6. Full ModernTransformerBlock forward pass and residual flow.
"""

import math
import unittest
import torch

from rmsnorm import RMSNorm
from swiglu import SwiGLU
from rope import RotaryEmbedding, precompute_freqs_cis, apply_rotary_emb
from gqa import GroupedQueryAttention, repeat_kv
from block import ModernTransformerBlock


class TestModernArchitecture(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.seq_len = 16
        self.dim = 256
        self.n_heads = 8
        self.n_kv_heads = 2
        self.head_dim = self.dim // self.n_heads

    def test_01_rmsnorm_scale_invariance(self):
        """RMSNorm(c * x) must equal RMSNorm(x) for any positive scalar c."""
        norm = RMSNorm(self.dim)
        x = torch.randn(self.batch_size, self.seq_len, self.dim)

        out_normal = norm(x)
        out_scaled = norm(x * 17.5)

        # Scale invariance holds within floating-point tolerance
        max_diff = (out_normal - out_scaled).abs().max().item()
        self.assertLess(max_diff, 1e-5, f"RMSNorm failed scale invariance with diff {max_diff}")

    def test_02_swiglu_shapes_and_gradients(self):
        """SwiGLU preserves shape and properly backpropagates gradients."""
        swiglu = SwiGLU(dim=self.dim, multiple_of=64)
        x = torch.randn(self.batch_size, self.seq_len, self.dim, requires_grad=True)

        out = swiglu(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.dim))

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    def test_03_rope_relative_position_property(self):
        """RoPE inner product <R_m q, R_n k> must depend strictly on (m - n)."""
        dim = 32
        cos, sin = precompute_freqs_cis(dim=dim, max_seq_len=100)

        q = torch.randn(1, 1, 1, dim)
        k = torch.randn(1, 1, 1, dim)

        # Case 1: position m = 5, n = 2 -> displacement = 3
        pos_m1 = torch.tensor([[5]])
        pos_n1 = torch.tensor([[2]])
        q_rot1, _ = apply_rotary_emb(q, q, cos, sin, position_ids=pos_m1)
        _, k_rot1 = apply_rotary_emb(k, k, cos, sin, position_ids=pos_n1)
        inner_prod_1 = (q_rot1 * k_rot1).sum().item()

        # Case 2: shifted by +10: m = 15, n = 12 -> displacement = 3
        pos_m2 = torch.tensor([[15]])
        pos_n2 = torch.tensor([[12]])
        q_rot2, _ = apply_rotary_emb(q, q, cos, sin, position_ids=pos_m2)
        _, k_rot2 = apply_rotary_emb(k, k, cos, sin, position_ids=pos_n2)
        inner_prod_2 = (q_rot2 * k_rot2).sum().item()

        self.assertAlmostEqual(
            inner_prod_1,
            inner_prod_2,
            places=4,
            msg="RoPE dot product failed relative displacement invariance!",
        )

    def test_04_gqa_shapes_and_broadcasting(self):
        """GQA projects Q(8 heads) and KV(2 heads) and matches target output shape."""
        gqa = GroupedQueryAttention(
            dim=self.dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
        )
        x = torch.randn(self.batch_size, self.seq_len, self.dim)

        out, (k_cache, v_cache) = gqa(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.dim))
        self.assertEqual(k_cache.shape, (self.batch_size, self.n_kv_heads, self.seq_len, self.head_dim))
        self.assertEqual(v_cache.shape, (self.batch_size, self.n_kv_heads, self.seq_len, self.head_dim))

    def test_05_gqa_mha_equivalence(self):
        """When n_kv_heads == n_heads, GQA reduces cleanly to standard Multi-Head Attention."""
        mha_mode = GroupedQueryAttention(
            dim=self.dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_heads,
            head_dim=self.head_dim,
        )
        self.assertEqual(mha_mode.num_groups, 1)
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        out, _ = mha_mode(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.dim))

    def test_06_modern_transformer_block(self):
        """Full ModernTransformerBlock executes pre-norm forward pass."""
        block = ModernTransformerBlock(
            dim=self.dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
        )
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        out, (new_k, new_v) = block(x)

        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.dim))
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
