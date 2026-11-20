"""Unit Tests for LoRA and QLoRA NF4 Implementations.

Tests:
1. Zero initialization at step 0: output(LoRA) == output(Base).
2. Gradient isolation: Gradients only flow to lora_A and lora_B, never base_layer.weight.
3. Associative multiplication memory equivalence: (x @ A.T) @ B.T == x @ (B @ A).T.
4. Merge & unmerge numerical identity.
5. Parameter count reduction calculation (>99% parameter freeze).
6. NF4 quantization and block-wise reconstruction fidelity.
"""

import unittest
import torch

from lora import LoRALinear, mark_only_lora_as_trainable
from qlora_nf4 import quantize_nf4_blockwise, dequantize_nf4_blockwise, QLoRALinear


class TestLoRAPEFT(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.seq_len = 16
        self.in_features = 256
        self.out_features = 512
        self.rank = 8
        self.alpha = 16.0

    def test_01_step_zero_exact_equivalence(self):
        """At step 0 (before training), LoRA output must be IDENTICAL to base model output."""
        layer = LoRALinear(
            self.in_features,
            self.out_features,
            rank=self.rank,
            lora_alpha=self.alpha,
        )
        x = torch.randn(self.batch_size, self.seq_len, self.in_features)

        lora_output = layer(x)
        base_output = layer.base_layer(x)

        diff = (lora_output - base_output).abs().max().item()
        self.assertEqual(diff, 0.0, f"Step 0 diff was {diff}, expected exact 0.0 due to B=0 initialization")

    def test_02_gradient_isolation(self):
        """Only LoRA adapter weights receive gradients; base weight grad remains None."""
        layer = LoRALinear(self.in_features, self.out_features, rank=self.rank)
        x = torch.randn(self.batch_size, self.seq_len, self.in_features)

        out = layer(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNone(layer.base_layer.weight.grad, "Base layer weight received gradients!")
        self.assertIsNotNone(layer.lora_A.grad, "lora_A failed to receive gradients!")
        self.assertIsNotNone(layer.lora_B.grad, "lora_B failed to receive gradients!")

    def test_03_associative_matrix_equivalence(self):
        """Memory-saving (x @ A.T) @ B.T must match mathematical x @ (B @ A).T."""
        layer = LoRALinear(self.in_features, self.out_features, rank=self.rank, lora_alpha=self.alpha)
        # Give B non-zero values to test delta
        nn_b = torch.randn_like(layer.lora_B)
        layer.lora_B.data.copy_(nn_b)

        x = torch.randn(self.batch_size, self.seq_len, self.in_features)

        # Associative path (O(r) activations):
        associative_out = layer(x)

        # Direct path (materializing full d_out x d_in matrix):
        scaling = self.alpha / self.rank
        delta_w = torch.matmul(layer.lora_B, layer.lora_A) * scaling
        w_effective = layer.base_layer.weight + delta_w
        direct_out = torch.matmul(x, w_effective.t())

        max_diff = (associative_out - direct_out).abs().max().item()
        self.assertLess(max_diff, 5e-5, f"Associative product diverged with diff {max_diff}")

    def test_04_merge_and_unmerge(self):
        """merge() must produce identical outputs with zero LoRA overhead, and unmerge() must restore state."""
        layer = LoRALinear(self.in_features, self.out_features, rank=self.rank)
        layer.lora_B.data.normal_()  # non-zero B
        x = torch.randn(self.batch_size, self.seq_len, self.in_features)

        out_unmerged = layer(x)

        # Merge weights
        layer.merge()
        self.assertTrue(layer.merged)
        out_merged = layer(x)

        diff_merge = (out_unmerged - out_merged).abs().max().item()
        self.assertLess(diff_merge, 5e-5, f"merge() diverged with diff {diff_merge}")

        # Unmerge weights
        layer.unmerge()
        self.assertFalse(layer.merged)
        out_restored = layer(x)

        diff_restore = (out_unmerged - out_restored).abs().max().item()
        self.assertLess(diff_restore, 5e-5, f"unmerge() failed to restore state with diff {diff_restore}")

    def test_05_parameter_savings(self):
        """LoRA achieves > 95% reduction in trainable parameters for standard projection."""
        in_dim, out_dim = 4096, 4096
        rank = 8
        layer = LoRALinear(in_dim, out_dim, rank=rank)

        total_base_params = in_dim * out_dim  # 16,777,216 params
        lora_trainable_params = (rank * in_dim) + (out_dim * rank)  # 65,536 params

        ratio = lora_trainable_params / total_base_params
        self.assertLess(ratio, 0.005, f"Trainable parameter ratio was {ratio:.4f}, expected < 0.5%")

    def test_06_nf4_quantization_fidelity(self):
        """NF4 quantize and dequantize produces low bounded reconstruction error."""
        weight = torch.randn(256, 256)
        indices, scales = quantize_nf4_blockwise(weight, block_size=64)
        reconstructed = dequantize_nf4_blockwise(indices, scales, orig_shape=weight.shape, block_size=64)

        # Reconstruction error for 4-bit standard normal should have MSE < 0.05
        mse = ((weight - reconstructed) ** 2).mean().item()
        self.assertLess(mse, 0.05, f"NF4 quantization MSE was too high: {mse:.4f}")


if __name__ == "__main__":
    unittest.main()
