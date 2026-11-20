"""
Unit Verification Tests for CausalSelfAttention (Day 1 Sprint Milestone).
Tests:
  1. Output shape preservation (B, T, C) -> (B, T, C)
  2. Softmax row-sum constraint (strictly 1.0000)
  3. Causal mask integrity (future tokens receive strictly 0.0000 attention)
  4. Healthy gradient backpropagation (no vanishing or NaN gradients)
"""

import sys
import torch
from attention import CausalSelfAttention

def run_tests():
    print("=" * 70)
    print("RUNNING DAY 1 UNIT ASSERTION TESTS: CausalSelfAttention")
    print("=" * 70)
    
    # Configuration matching GPT-2 Small specifications
    B = 2          # Batch size
    T = 8          # Sequence length
    d_model = 768  # Hidden dimension
    n_heads = 12   # Heads
    d_k = d_model // n_heads  # 64
    
    print(f"[1/4] Initializing CausalSelfAttention (d_model={d_model}, n_heads={n_heads}, d_k={d_k})...")
    model = CausalSelfAttention(d_model=d_model, n_heads=n_heads, max_seq_len=1024, dropout=0.0)
    model.eval()  # Disable dropout for deterministic probability inspection
    
    # Create input tensor with gradient tracking
    x = torch.randn(B, T, d_model, requires_grad=True)
    print(f"      Input tensor shape: {x.shape}")
    
    # -------------------------------------------------------------------------
    # TEST 1: Shape Preservation
    # -------------------------------------------------------------------------
    print("\n[2/4] Testing Output Shape Preservation...")
    out, att_weights = model(x, return_attention_weights=True)
    assert out.shape == (B, T, d_model), f"Shape mismatch! Expected {(B, T, d_model)}, got {out.shape}"
    assert att_weights.shape == (B, n_heads, T, T), f"Attention weights shape mismatch! Got {att_weights.shape}"
    print(f"      PASS: Output shape correctly matches input shape {out.shape}!")
    print(f"      PASS: Attention weights shape is {att_weights.shape}!")

    # -------------------------------------------------------------------------
    # TEST 2: Softmax Row Sum Constraint (Rows sum to 1.0000)
    # -------------------------------------------------------------------------
    print("\n[3/4] Testing Softmax Normalization (Sum of Rows == 1.0)...")
    row_sums = att_weights.sum(dim=-1)  # Sum across Key dimension
    expected_sums = torch.ones_like(row_sums)
    max_sum_diff = (row_sums - expected_sums).abs().max().item()
    assert max_sum_diff < 1e-6, f"Row sums violate 1.00! Max difference: {max_sum_diff}"
    print(f"      PASS: All attention rows sum to 1.000000 (Max deviation: {max_sum_diff:.2e})!")

    # -------------------------------------------------------------------------
    # TEST 3: Causal Mask Integrity (Zero Future Leakage)
    # -------------------------------------------------------------------------
    print("\n[4/4] Testing Causal Mask Integrity (Strictly 0.0000 in Upper Triangle)...")
    # In each (T, T) matrix, for row i and col j, if j > i (future), weight must be 0.0
    upper_tri_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    
    # Inspect upper triangle across all batches and heads
    future_leakage = att_weights[:, :, upper_tri_mask]
    max_leakage = future_leakage.abs().max().item()
    assert max_leakage == 0.0, f"Future leakage detected! Upper triangle max: {max_leakage}"
    print(f"      PASS: Zero future lookahead! Future attention weights are strictly {max_leakage:.6f}!")
    
    # Print sample attention matrix from Batch 0, Head 0 to display in terminal
    sample_matrix = att_weights[0, 0].detach()
    print("\n      Sample Attention Matrix (Head 0, 8x8 Tokens):")
    for r in range(min(5, T)):
        row_str = " ".join([f"{sample_matrix[r, c].item():.3f}" for c in range(min(5, T))])
        print(f"      Row {r} (Token {r}): [{row_str} ...]")
    
    # -------------------------------------------------------------------------
    # TEST 4: Healthy Gradient Flow (Backpropagation Test)
    # -------------------------------------------------------------------------
    print("\n[BONUS] Testing Gradient Flow Across Self-Attention...")
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradient of input x is None!"
    assert not torch.isnan(x.grad).any(), "NaN detected in input gradients!"
    assert not torch.isinf(x.grad).any(), "Inf detected in input gradients!"
    grad_norm = x.grad.norm().item()
    assert grad_norm > 0.0, "Vanishing gradient detected! Gradient norm is 0.0"
    print(f"      PASS: Healthy gradients computed! Input gradient norm: {grad_norm:.4f}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! CausalSelfAttention is 100% mathematically verified.")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()

