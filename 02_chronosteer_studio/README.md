# Project 02: ChronoSteer Studio — Real-Time Token Scrubber, Attention Rail & Multiverse Tree

> **Directory**: `projects/LLM related Projects while learning/02_chronosteer_studio/`  
> **Engine**: Built directly on top of [`01_gpt2_from_scratch/`](../01_gpt2_from_scratch/) (`GPT2Model` + KV-Cache)  
> **Inspiration**: Interactive real-time LLM steering (`2026-11-17 23.54.50.mp4`) + Branching Multiverse Word Tree  

---

## 1. Vision & Core Idea

Instead of treating an LLM like a black-box slot machine where you hit Enter and passively watch text scroll by, **ChronoSteer Studio** gives you **full spatial and temporal control over every token**:

1. **Time-Travel Scrubber Cursor**: Drag a glowing cursor bar **upward** to rewind/erase tokens in time, or **downward** to stream fresh continuations live using your KV-Cached `GPT2Model`.
2. **Left Attention Rail & Steering Boxes (Chapter 1)**:
   - Displays live per-line attention weights (`att_weights`) showing exactly which earlier lines the current token is looking at.
   - Drag manual **Attention Steering Boxes** onto earlier sentences to boost their attention bias (`+boost` before `F.softmax(att, dim=-1)`), forcing future text to stay anchored to those concepts.
3. **Radial Steering Wheel & Vector Dials (Chapter 2)**:
   - Click the wheel at the cursor to open a radial dial of contextual steering vectors (`tone`, `tension`, `focus`, `pace`, `surprise`, `distance`) with adjustable strength (`%`).
   - Injects residual-stream steering vectors ($x = x + \alpha \cdot v_{\text{steer}}$) into a middle `TransformerBlock` and highlights output sentences that realized each vector.
4. **Top-K, Logit & Entropy Inspector (Chapter 4)**:
   - Click or scroll at any token to inspect the exact **Top-K** next-token candidates, live **Temperature ($T$)** probabilities, raw logits ($z_i$), and Shannon **Entropy** ($H = -\sum p \log p$).
5. **Branching Multiverse Word Tree ($3 \times 3 \times \dots$ Explorer)**:
   - Clicking the wheel can also expand a **live branching tree** of candidate futures (e.g., Top-3 next words $\to$ 3 child words each $\to$ 3 grandchild words each).
   - Click any node in the tree to jump down that timeline, or jump **back in time** to any ancestor node, tweak steering/weights, and rewrite the future down a brand-new branch.

---

## 2. Visual Interface Blueprints (`visual_guides/`)

| Preview Image | What It Shows |
|---|---|
| [`visual_guides/01_main_workspace_scrubber_and_attention_rail.png`](visual_guides/01_main_workspace_scrubber_and_attention_rail.png) | Full dark studio layout: Left Attention Rail + orange Steering Box, live streaming text, glowing horizontal Time-Travel Scrubber, and KV-Cache telemetry bar |
| [`visual_guides/02_radial_steering_wheel_and_topk_inspector.png`](visual_guides/02_radial_steering_wheel_and_topk_inspector.png) | Cursor Radial Steering Wheel (`tone`, `tension`, `focus`, `pace`) paired with the live Top-K Probability, Logit ($z_i$), and Entropy Inspector |
| [`visual_guides/03_multiverse_3x3_branching_word_tree.png`](visual_guides/03_multiverse_3x3_branching_word_tree.png) | Interactive $3 \times 3 \times 3$ Branching Multiverse Word Tree showing active timeline, parallel forks, probabilities, and click-to-rewind ancestor nodes |
| [`visual_guides/04_multi_word_attention_tether.png`](visual_guides/04_multi_word_attention_tether.png) | Multi-word Cursor Pinning: click the wheel and select multiple earlier words to draw attention tethers (`att[..., pinned_idx] += boost`) forcing future text to weave them in |

---

## 3. Planned Module Architecture (When Ready to Implement)

```text
02_chronosteer_studio/
├── README.md                          # Project blueprint & architecture guide
├── visual_guides/
│   ├── render_interface_mockups.py    # Matplotlib/PIL UI architecture & interface renderer
│   ├── 01_main_workspace_scrubber_and_attention_rail.png
│   ├── 02_radial_steering_wheel_and_topk_inspector.png
│   └── 03_multiverse_3x3_branching_word_tree.png
├── steer_engine.py                    # (Planned) KV-Cache snapshot tree + attention bias + residual steering hooks
├── server.py                          # (Planned) Lightweight Python HTTP/SSE backend wrapping 01_gpt2_from_scratch/model.py
└── index.html                         # (Planned) Interactive browser workstation UI
```

### How Each Feature Maps to Chapters 1–4
- **Chapter 1 (`attention.py`):** `return_attention_weights=True` feeds the Left Attention Rail; adding `+steer_mask` before `F.softmax` implements Attention Steering Boxes.
- **Chapter 2 (`block.py`):** Residual Stream `x = x + ...` allows clean contrastive steering vector injection at Layer 6.
- **Chapter 3–4 (`model.py`):** `logits[:, -1, :]`, Temperature scaling, Top-K truncation, and KV-Cache slicing (`past_k[:, :, :cursor_pos, :]`) power instant time-travel rewinding and 3-way tree expansion on your 16 GB Mac (`~19.6 tok/s`).

