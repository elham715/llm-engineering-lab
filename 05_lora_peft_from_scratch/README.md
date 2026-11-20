# LoRA & QLoRA Parameter-Efficient Fine-Tuning From Scratch

Implementation of Low-Rank Adaptation (Hu et al. 2021) and NormalFloat4 Quantization (Dettmers et al. 2023) without external libraries.

---

## 1. Mathematical Mechanics

### A. The Core Formulation
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
- $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$: Frozen base weights.
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$: Initialized with Gaussian $\mathcal{N}(0, 1/r)$.
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$: Initialized to **identically zero**. At step 0, $\Delta W = 0$.

### B. The Hardware Associative Memory Trick
During forward pass:
- **Naive approach**: $x \cdot (B \cdot A)^T$ materializes full $d \times d$ matrix in GPU memory ($O(d^2)$ activation cost).
- **Associative approach**: $(x \cdot A^T) \cdot B^T$ computes intermediate tensor of shape $(B, T, r)$ ($O(r)$ cost).
- For $d = 4096, r = 8$, intermediate activations are **$512\times$ smaller** in memory!

### C. NormalFloat4 (NF4) Quantization
- Maps Gaussian-distributed neural network weights into 16 optimal quantiles with equal probability mass.
- Uses block-wise normalization (block size 64) to neutralize activation outliers.
- On-the-fly SRAM dequantization to FP16 / BF16 during tensor contractions.

---

## 2. Test Execution

```bash
/opt/anaconda3/bin/python test_lora.py
```
Verified passing 6/6 tests covering step-0 identity, gradient isolation, associative equivalence, weight merging/unmerging, and NF4 reconstruction fidelity.
