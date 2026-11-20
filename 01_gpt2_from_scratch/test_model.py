"""
Unit Test Suite for Full GPT2Model (Chapters 03-04).
Verifies:
    1. Exact unique parameter count == 124,439,808.
    2. Weight tying memory identity (wte.weight is lm_head.weight & same data_ptr).
    3. Forward pass shapes: (B, T) -> logits (B, T, 50257) and scalar cross-entropy loss.
    4. Context-window assertion when T > block_size.
    5. Autoregressive generation (greedy, temperature + top_k, left-cropping, streaming callback).
    6. Gradient accumulation into tied wte/lm_head weights and wpe.
"""

import torch

from block import GPT2Config
from model import GPT2Model


def test_parameter_count_and_weight_tying():
    config = GPT2Config()
    model = GPT2Model(config)

    assert model.transformer.wte.weight is model.lm_head.weight, "Weights are not tied!"
    assert model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()

    unique_params = model.count_parameters(unique_only=True)
    raw_params = model.count_parameters(unique_only=False)
    assert unique_params == 124_439_808, f"Expected 124,439,808, got {unique_params:,}"
    assert raw_params == 124_439_808, "PyTorch parameters() should deduplicate shared Parameter"
    print(f"✅ [PASS] Parameter Count & Weight Tying: {unique_params:,} unique parameters")


def test_forward_shapes_and_loss():
    config = GPT2Config(vocab_size=200, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)
    model = GPT2Model(config)
    model.eval()

    idx = torch.randint(0, 200, (2, 10))
    targets = torch.randint(0, 200, (2, 10))
    logits, loss = model(idx, targets=targets)

    assert logits.shape == (2, 10, 200), f"Unexpected logits shape: {logits.shape}"
    assert loss is not None and loss.ndim == 0 and loss.item() > 0
    print(f"✅ [PASS] Forward Shapes & Cross-Entropy Loss: logits={tuple(logits.shape)}, loss={loss.item():.4f}")


def test_block_size_limit():
    config = GPT2Config(vocab_size=100, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = GPT2Model(config)
    try:
        model(torch.randint(0, 100, (1, 9)))
        raise AssertionError("Expected assertion error when T > block_size")
    except AssertionError as exc:
        assert "exceeds block_size" in str(exc)
    print("✅ [PASS] Context Window Guard: T > block_size blocked cleanly")


def test_generate_and_cropping():
    torch.manual_seed(42)
    config = GPT2Config(vocab_size=100, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    model = GPT2Model(config)
    model.eval()

    # Start at T=6 and generate 5 tokens -> crosses block_size=8 to test left-cropping
    prompt = torch.randint(0, 100, (1, 6))
    streamed = []
    out = model.generate(prompt, max_new_tokens=5, temperature=0.8, top_k=10, on_token=streamed.append)
    assert out.shape == (1, 11), f"Expected (1, 11), got {out.shape}"
    assert len(streamed) == 5

    # Greedy decoding must be deterministic across repeated calls and identical with/without KV-cache
    g1 = model.generate(prompt, max_new_tokens=4, temperature=0.0, use_cache=True)
    g2 = model.generate(prompt, max_new_tokens=4, temperature=0.0, use_cache=False)
    assert torch.equal(g1, g2), "KV-cache greedy decoding diverged from uncached!"
    print("✅ [PASS] Autoregressive Generation, Left-Cropping & KV-Cache Exact Match")


def test_gradient_flow_with_tied_weights():
    config = GPT2Config(vocab_size=100, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    model = GPT2Model(config)
    idx = torch.randint(0, 100, (2, 6))
    targets = torch.randint(0, 100, (2, 6))

    _, loss = model(idx, targets=targets)
    loss.backward()

    assert model.transformer.wte.weight.grad is not None
    assert model.transformer.wte.weight.grad.norm().item() > 0
    assert model.transformer.wpe.weight.grad is not None
    assert model.transformer.wpe.weight.grad.norm().item() > 0
    print("✅ [PASS] Backward Gradient Flow through Tied Embeddings & wpe")


if __name__ == "__main__":
    print("=== Running Full GPT2Model Unit Tests ===")
    test_parameter_count_and_weight_tying()
    test_forward_shapes_and_loss()
    test_block_size_limit()
    test_generate_and_cropping()
    test_gradient_flow_with_tied_weights()
    print("=== ALL MODEL TESTS PASSED ===")

