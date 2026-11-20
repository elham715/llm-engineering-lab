# LLM Engineering Lab — From-Scratch Transformers, Steering & Post-Training

> **Author**: Elham (`@elham715`)  
> **Companion Paper-Reproduction Repo**: [`first-principles-ai-lab`](https://github.com/elham715/first-principles-ai-lab)

A hands-on systems repository building modern Large Language Model components and glass-box interpretability tools **from scratch in PyTorch** — starting with a 124M GPT-2 engine and an interactive mechanistic steering studio.

---

## Projects

| # | Project | Status | Highlights |
|---|---|---|---|
| **01** | [`01_gpt2_from_scratch/`](01_gpt2_from_scratch/) | **Completed & Verified** 🟢 | Full GPT-2 (124M) in raw PyTorch (`attention.py`, `block.py`, `model.py`, `generate.py`), `Conv1D` OpenAI weight transfer, weight tying, KV-cache slicing, Temperature / Top-K / Top-p sampling. |
| **02** | [`02_chronosteer_studio/`](02_chronosteer_studio/) | **Completed & Verified** 🟢 | **ChronoSteer Studio** — real-time glass-box steering & time-travel playground (`torch.bfloat16` CPU + Starlette WebSockets + speculative prefetch): scroll-bounded KV-cache rewind, Layer-6 contrastive residual steering, pre-softmax Q·K attention biasing, live LED attention rail, 3×3 multiverse tree, and a 360° concentric future-word wheel. |
| **03** | [`03_bpe_tokenizer/`](03_bpe_tokenizer/) | **Completed & Verified** 🟢 | Pure Python Byte-Level BPE (`256`-byte UTF-8 base, `train`, `encode`, `decode`, JSON serialization, zero `<unk>`, verified round-trip on English, Bengali, biochemistry, and emojis). |
| **04** | [`04_modern_architecture_gqa_rope/`](04_modern_architecture_gqa_rope/) | **Completed & Verified** 🟢 | LLaMA-3/Mistral modern backbone: RoPE 2D/3D rotary embeddings, RMSNorm (scale invariant), SwiGLU gating ($\lfloor \frac{8}{3}d \rfloor$), and Grouped-Query Attention (`repeat_kv` zero-copy broadcasting). |
| **05** | [`05_lora_peft_from_scratch/`](05_lora_peft_from_scratch/) | **Completed & Verified** 🟢 | Low-Rank Adaptation (`LoRA`) from first principles ($B=0, A \sim \mathcal{N}$), associative memory multiplication `(x @ A.T) @ B.T` ($O(r)$ memory), weight merge/unmerge, and NormalFloat4 (`NF4`) block-wise quantization. |
| **06** | [`06_preference_optimization_dpo/`](06_preference_optimization_dpo/) | **Completed & Verified** 🟢 | Direct Preference Optimization (`DPO`) closed-form loss, analytical $Z(x)$ cancellation, autoregressive target shifting (`:-1` vs `1:`), prompt masking (`-100`), and self-contained preference training simulation. |
| **07** | [`07_inference_kvcache_speculative_decoding/`](07_inference_kvcache_speculative_decoding/) | **Completed & Verified** 🟢 | Dynamic pre-allocated zero-reallocation KV-cache and Speculative Decoding engine (Leviathan/Chen et al.) with rejection sampling and target distribution preservation. |

---

## Quick Start (ChronoSteer Studio)

```bash
cd 02_chronosteer_studio
pip install torch starlette uvicorn tiktoken transformers
python server.py
```

Open **`http://127.0.0.1:8765`** in your browser. For full architecture details and benchmarks, see [`02_chronosteer_studio/PROJECT_PROGRESS_AND_HANDOFF.md`](02_chronosteer_studio/PROJECT_PROGRESS_AND_HANDOFF.md).
