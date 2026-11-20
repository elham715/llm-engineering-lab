# Modern Transformer Architecture: RoPE, RMSNorm, SwiGLU & GQA

This directory contains the production-grade modern LLM backbone used in state-of-the-art models like **LLaMA-3, Mistral 7B, Gemma 2, and DeepSeek-V3**.

---

## 1. Architectural Highlights

### A. Rotary Position Embeddings (RoPE) — `rope.py`
- Replaces absolute sinusoidal embeddings with **relative complex rotations**.
- Proves $\langle R_m q, R_n k \rangle = g(q, k, m - n)$.
- Outer-product precomputation of $\cos$ and $\sin$ frequencies $\theta_i = 10000^{-2i/d}$.
- Half-split rotation: `[-x2, x1]` applied directly to head dimensions.

### B. Root Mean Square Layer Normalization (RMSNorm) — `rmsnorm.py`
- Eliminates the mean-centering step of standard LayerNorm, saving 1 memory reduction pass.
- Mathematically proven **scale-invariant**: $\text{RMSNorm}(c \cdot x) = \text{RMSNorm}(x)$.
- Upcasts internal variance accumulation to `FP32` to prevent `FP16` overflow at $65,504$.

### C. SwiGLU Gated Feed-Forward Network — `swiglu.py`
- Equation: $\text{SwiGLU}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot x W_{\text{up}}) W_{\text{down}}$.
- Hidden dimension: $d_{\text{ffn}} = \lfloor \frac{8}{3} d_{\text{model}} \rfloor$ rounded to multiple of 256.
- Preserves parameter parity with standard $4 \times d$ MLPs while boosting gradient expressiveness.

### D. Grouped-Query Attention (GQA) — `gqa.py`
- Solves the memory-bandwidth bottleneck during autoregressive decoding.
- Groups $N_q = 32$ query heads into $N_{kv} = 8$ key-value heads.
- Reduces KV cache memory footprint by **4x to 8x** with zero-copy broadcasting (`expand` + `reshape`).

---

## 2. Running Unit Tests

```bash
/opt/anaconda3/bin/python test_modern_architecture.py
```
Verified passing 6/6 tests covering scale invariance, RoPE relative position invariance, GQA broadcasting, and full block execution.
