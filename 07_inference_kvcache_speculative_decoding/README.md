# High-Throughput Inference: Dynamic KV Cache & Speculative Decoding

Implementations of Dynamic Key-Value Caching and Speculative Decoding (Leviathan et al. 2023, Chen et al. 2023) from first principles.

---

## 1. Hardware Silicon Physics

### A. The Memory-Bandwidth Bottleneck (GEMV)
During standard autoregressive generation:
- Every generated token requires loading **all model weights $\mathbf{W}$ from GPU VRAM to SRAM**.
- Floating-Point Operations per byte transferred: $\approx 1 \text{ FLOP / byte}$.
- The compute units (Tensor Cores) sit idle $95\%$ of the time waiting for memory bus transfers.

### B. Speculative Decoding (GEMM Mode)
- A small **Draft Model** generates $K$ tokens quickly.
- The large **Target Model** evaluates all $K$ candidate tokens simultaneously in a **single forward pass**.
- This converts $K$ sequential $GEMV$ memory-bound loads into **1 parallel $GEMM$ compute-bound matrix multiplication**!
- Rejection sampling guarantees that the output token distribution is **100% mathematically identical** to the target model alone.

---

## 2. Test Execution

```bash
/opt/anaconda3/bin/python test_speculative.py
```
Verified passing 4/4 tests covering zero-reallocation buffer persistence, rejection sampling certainty bounds, recovery resampling, and end-to-end speculative generation loops.
