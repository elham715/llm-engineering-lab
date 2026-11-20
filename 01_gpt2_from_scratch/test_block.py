"""
Unit Test Suite for TransformerBlock and Subcomponents.
Verifies:
    1. GELU numerical approximation accuracy against PyTorch reference.
    2. LayerNorm standardization (zero mean, unit variance).
    3. MLP dimension expansion (768 -> 3072 -> 768).
    4. TransformerBlock shape preservation: (B, T, C) -> (B, T, C).
    5. Residual Shortcut Additive Integrity: output == input when sublayers are zeroed.
    6. Backward Gradient Flow: non-zero, healthy, unvanished input gradient.
"""

import sys
import torch
import torch.nn as nn

from block import GPT2Config, NewGELU, LayerNorm, MLP, TransformerBlock


def test_gelu_accuracy():
    """Verify NewGELU polynomial approximation against PyTorch official GELU(approximate='tanh')."""
    custom_gelu = NewGELU()
    torch_gelu = nn.GELU(approximate='tanh')
    
    x = torch.linspace(-5.0, 5.0, 1000)
    out_custom = custom_gelu(x)
    out_torch = torch_gelu(x)
    
    max_diff = torch.max(torch.abs(out_custom - out_torch)).item()
    assert max_diff < 1e-5, f"GELU deviation too large: {max_diff}"
    print(f"✅ [PASS] GELU Polynomial Accuracy: Max deviation = {max_diff:.8f} (< 1e-5)")


def test_layernorm_properties():
    """Verify LayerNorm standardizes channel dimension to mean=0 and var=1."""
    C = 768
    ln = LayerNorm(C)
    
    # Random tensor with high mean and variance
    x = torch.randn(4, 16, C) * 50.0 + 25.0
    out = ln(x)
    
    mean = out.mean(dim=-1)
    var = out.var(dim=-1, unbiased=False)
    
    max_mean_error = torch.max(torch.abs(mean)).item()
    max_var_error = torch.max(torch.abs(var - 1.0)).item()
    
    assert max_mean_error < 1e-4, f"LayerNorm mean deviation too large: {max_mean_error}"
    assert max_var_error < 1e-3, f"LayerNorm variance deviation too large: {max_var_error}"
    print(f"✅ [PASS] LayerNorm Channel Standardization: Mean dev = {max_mean_error:.6f}, Var dev = {max_var_error:.6f}")


def test_mlp_dimensions():
    """Verify MLP 4x expansion and shape preservation."""
    config = GPT2Config(n_embd=768)
    mlp = MLP(config)
    
    x = torch.randn(2, 8, 768)
    out = mlp(x)
    
    assert out.shape == (2, 8, 768), f"Expected shape (2, 8, 768), got {out.shape}"
    assert mlp.c_fc.weight.shape == (3072, 768), f"Expected W1 shape (3072, 768), got {mlp.c_fc.weight.shape}"
    assert mlp.c_proj.weight.shape == (768, 3072), f"Expected W2 shape (768, 3072), got {mlp.c_proj.weight.shape}"
    print(f"✅ [PASS] MLP 4x Dimension Expansion: (768 -> 3072 -> 768) verified")


def test_block_shape_preservation():
    """Verify TransformerBlock preserves input tensor shapes."""
    config = GPT2Config(n_embd=768, n_head=12, block_size=1024, dropout=0.0)
    block = TransformerBlock(config)
    block.eval()
    
    B, T, C = 2, 16, 768
    x = torch.randn(B, T, C)
    out = block(x)
    
    assert out.shape == (B, T, C), f"Expected output shape ({B}, {T}, {C}), got {out.shape}"
    print(f"✅ [PASS] TransformerBlock Shape Preservation: ({B}, {T}, {C}) -> {tuple(out.shape)}")


def test_residual_identity_integrity():
    """Verify that when sublayer outputs are zeroed, output == input exactly."""
    config = GPT2Config(n_embd=768, n_head=12, block_size=1024, dropout=0.0)
    block = TransformerBlock(config)
    block.eval()
    
    # Zero out the final projection weights of both Attention and MLP
    with torch.no_grad():
        block.attn.c_proj.weight.zero_()
        block.attn.c_proj.bias.zero_()
        block.mlp.c_proj.weight.zero_()
        block.mlp.c_proj.bias.zero_()
        
    x = torch.randn(2, 8, 768)
    out = block(x)
    
    # When sublayers output 0, x_out = x + 0 + 0 = x
    diff = torch.max(torch.abs(out - x)).item()
    assert diff == 0.0, f"Identity bypass corrupted! Max difference: {diff}"
    print(f"✅ [PASS] Residual Identity Shortcut Integrity: Max diff when sublayers zeroed = {diff:.8f}")


def test_backward_gradient_flow():
    """Verify unattenuated backpropagation gradient flow through the residual highway."""
    config = GPT2Config(n_embd=768, n_head=12, block_size=1024, dropout=0.0)
    block = TransformerBlock(config)
    block.train()
    
    x = torch.randn(2, 8, 768, requires_grad=True)
    out = block(x)
    
    loss = (out ** 2).sum()
    loss.backward()
    
    assert x.grad is not None, "Input gradient is None!"
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaN values!"
    assert not torch.isinf(x.grad).any(), "Input gradient contains Inf values!"
    
    grad_norm = x.grad.norm().item()
    assert grad_norm > 0.0, f"Gradient vanished to zero: {grad_norm}"
    print(f"✅ [PASS] Backpropagation Gradient Flow: Input grad norm = {grad_norm:.4f} (Healthy & Finite)")


if __name__ == '__main__':
    print("================================================================================")
    print("RUNNING DAY 02 VERIFICATION SUITE: TransformerBlock & Residual Stream Mechanics")
    print("================================================================================")
    test_gelu_accuracy()
    test_layernorm_properties()
    test_mlp_dimensions()
    test_block_shape_preservation()
    test_residual_identity_integrity()
    test_backward_gradient_flow()
    print("================================================================================")
    print("ALL 6 TESTS PASSED CLEANLY! DAY 02 OBJECTIVES 100% ACHIEVED.")
    print("================================================================================")

