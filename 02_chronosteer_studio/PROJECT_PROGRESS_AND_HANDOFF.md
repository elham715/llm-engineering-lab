# Project 02 — ChronoSteer Studio: Progress, Architecture & Continuation Handoff

> **Project Directory**: `projects/LLM related Projects while learning/02_chronosteer_studio/`  
> **Live Local Server URL**: `http://127.0.0.1:8765`  
> **Last Updated**: 2026-11-20  
> **Status**: Working interactive prototype (real-time vanishing rewind, 3-level 360° polar wheel with growing concentric speculative word-tree rings and release-to-commit staging, Layer-6 residual steering, Q·K attention biasing, 3×3×preview Multiverse tree, BF16 CPU optimization, speculative prefetch, and scroll/drag-bounded progression implemented and verified).

---

## 1. Executive Summary & Quick Start

**ChronoSteer Studio** transforms autoregressive text generation from a passive one-way stream into a direct manipulation instrument built directly on top of Project 01's [`01_gpt2_from_scratch/`](../01_gpt2_from_scratch/) (`GPT2Model`, `load_tokenizer`, `pick_device`), without modifying a single source file inside Project 01.

### How to Launch & Resume Work

```bash
cd "/Users/admin/Documents/Elhams Universe/Artificial Intelligence/llm Intelligence/projects/LLM related Projects while learning/02_chronosteer_studio"
/opt/anaconda3/bin/python server.py
```

Once `[ChronoSteer] Ready: gpt2 (124,439,808 params) | Steer Layer=6 | Axes=8` and `ChronoSteer Studio live at: http://127.0.0.1:8765` appear in stdout (takes ~15–25s on cold startup for model load + vector bank precomputation), open **`http://127.0.0.1:8765`** in a browser.

### Project 02 Complete File Inventory

| File | Size / Lines | Responsibility |
|---|---:|---|
| [`steer_engine.py`](steer_engine.py) | 842 lines | `BaseSteerableEngine` ABC + `GPT2SteerableEngine` (BF16 CPU inference, Q·K pre-softmax bias hooks, Layer-6 contrastive residual injection, O(K) Top-K + Top-p nucleus sampling, `fast_mode`, KV-cached 3×3 multiverse tree). |
| [`server.py`](server.py) | 387 lines | Starlette + Uvicorn async HTTP/WebSocket backend, `PREFIX_CACHE` KV prefix reuse, per-token `ENGINE_LOCK` cooperative preemption, `run_stream` (`stream_id`-tagged), `run_prefetch` (`gen_id`-tagged + deterministic `prefetch_done`). |
| [`index.html`](index.html) | 3,258 lines | Self-contained 5-column monochrome dark UI (`#000000`, `IBM Plex Mono`), inline crosshair reticle `◎`, 3-Level 360° Polar Steering & Concentric Multi-Ring Speculative Word Tree, real-time scroll/drag-bounded progression, O(1) `appendTokenFast`, speculative `prefetchBuffer`, laser-ray pins, LED attention rail, modals, Web Audio detent clicks. |
| [`README.md`](README.md) | 58 lines | Original vision statement and feature mapping to Chapters 1–4. |
| [`visual_guides/render_interface_mockups.py`](visual_guides/render_interface_mockups.py) | 16.5 KB | Reference UI mockup renderer (Matplotlib/PIL). |
| `visual_guides/*.png` | 4 PNGs | Design targets: `01_main_workspace_scrubber_and_attention_rail.png`, `02_radial_steering_wheel_and_topk_inspector.png`, `03_multiverse_3x3_branching_word_tree.png`, `04_multi_word_attention_tether.png`. |
| [`PROJECT_PROGRESS_AND_HANDOFF.md`](PROJECT_PROGRESS_AND_HANDOFF.md) | This file | Full record of implemented architecture, UX contracts, verified benchmarks, resolved bugs, and roadmap for future work. |

> **Note on Project 01**: All files in [`../01_gpt2_from_scratch/`](../01_gpt2_from_scratch/) (`model.py`, `generate.py`, `attention.py`, etc.) are imported cleanly via `sys.path` in [`steer_engine.py`](steer_engine.py#L30-L35) and remain completely untouched.

---

## 2. Locked-In Interaction & UX Design Specification

Every interaction rule below was refined iteratively across design reviews and live tests:

1. **Pure Monochrome Frontier & Vanishing-Future Time Travel**:
   - The inline crosshair reticle **`◎`** (`#inlineReticle`) sits at the active token frontier (`state.cursorPos === state.pieces.length`).
   - Rewinding backward (scrolling **up**, dragging `◎` **up/left**, Shift+clicking a past token, or pressing `ArrowLeft`/`ArrowUp`) immediately calls `scrubTo(...)`: future tokens **strictly vanish** (`state.pieces = state.pieces.slice(0, clamped)`), rather than remaining grayed out.
   - Rewinding prunes any `pinnedTokens` with `index >= clamped` and clips `railBoxes` to existing lines.
   - Up to **25 undo snapshots** (`state.undoStack`) are saved by `saveUndoSnapshot()` before rewinds/drags and restored via `Cmd/Ctrl + Z` (`restoreUndoSnapshot()`).

2. **Exact Real-Time Scroll-Bounded Progression (No Auto-Complete / No Runaway Streaming)**:
   - Scrolling **down** below `◎` (or anywhere while the Wheel is open, or at the bottom of `#stageScroll`) generates **strictly proportional to exact wheel/trackpad scroll travel**:
     - `handleReticleOrWheelScroll(e)` normalizes line-mode scroll (`e.deltaMode === 1 ? e.deltaY * 24 : e.deltaY`), ignores tiny `< 1px` noise, and accumulates downward pixels into `state.scrollPixelDebt`.
     - **Ratio**: `PX_PER_TOKEN = 16` (16 px down = 1 token budget added to `state.scrollTokenBudget`, capped at 120).
     - **Instant Pop + Exact Deficit Generation**: Existing tokens in `state.prefetchBuffer` are popped instantaneously (`popPrefetchTokens(state.scrollTokenBudget)`). If `scrollTokenBudget` still has remaining tokens and `!state.isGenerating`, only that exact remaining count (`startGen(toReq)`) is requested—never an open-ended `startGen(60)` burst.
     - Scroll **up** (`dy < 0`) cancels `scrollTokenBudget`, clears `dragTarget`, stops any active stream (`stopGen()`), invalidates prefetch (`invalidatePrefetch()`), and accumulates `state.rewindPixelDebt` with `PX_PER_REWIND = 18` (18 px up = 1 token erased, clamped to max 8 per event).
     - Hold `Shift` while scrolling if native page scrolling inside `#stageScroll` is desired.

3. **Exact Spatial Cursor-Bounded Reticle Drag**:
   - Pressing `mousedown` on **`◎`** and dragging (`distMoved > 5px`):
     - **Dragging forward/down below `◎`** sets `state.dragTarget = { stageX, stageY }` relative to `#proseText` and starts a `requestAnimationFrame` loop (`dragWriteLoop`).
     - Velocity-adaptive intervals (`16 ms` / up to 3 tokens if `dy > 80px`, `22 ms` / up to 2 tokens if `dy > 30px`, `34 ms` / 1 token otherwise) pop prefetched tokens **one at a time** while `isReticleBehindDragTarget()` holds true, and request small bounded slices (`estTokens = Math.max(1, Math.min(10, Math.ceil(Math.max(1, dy) / 24) * 6))`) if the prefetch buffer runs empty.
     - On both each `dragWriteLoop` tick and each incoming WebSocket `"token"` frame, `isReticleBehindDragTarget()` is checked (`retY < stageY - lh * 0.45` or same line `retX < stageX - 10`); the moment `◎` catches up to the pointer (or the user releases `mouseup`), `stopGen()` cancels remaining server generation so text never overshoots the cursor.
     - **Dragging backward/up over existing prose** clears `dragTarget`, stops generation, invalidates prefetch, finds the hovered span via `findTokenIndexAtPoint(clientX, clientY)`, and rewinds directly to that token.

4. **3-Level 360° Polar Steering Wheel & Concentric Multi-Ring Speculative Word Tree**:
   - **Click `◎`** (or press `Space` / click `steer` in bottom HUD) toggles `#dialAnchor`, featuring an interactive free-moving **360° coral pie-wedge orange ball** (`appendPieBall` + `handleDialPointer`) with full-screen red crosshair lines (`appendCrosshairs`) that can be dragged or clicked anywhere across horizontal/vertical space (`maxR = 540`, `.dial-anchor svg` has `pointer-events: none` while `.dial-hit`, `.dial-ball`, `.dial-spoke`, and `.dial-sector` accept events):
     - **Level 1 (`dialLevel = 1`, Axes Wheel, `R = 122`, `166°..14°`)**: 8 radial vector axes (`tension`, `tone`, `distance`, `mode`, `fortune`, `surprise`, `resolution`, `pace`) arranged around the dashed semicircle with perpendicular sector divider ticks and a coral highlight arc. Hovering or rotating the 360° orange ball highlights an axis; clicking a spoke or dragging across its circular sector boundary (`r >= R - 2`) transitions into **Level 2**.
     - **Level 2 (`dialLevel = 2`, Multi-Stop Strength Protractor, `R = 148`, `152°..28°`)**: Faint `25` / `50` / `75` concentric guide arcs, ticked ruler needle (`appendRadialNeedle`), and the outer arc split into **all `M` stops** of the active axis (e.g. 4 stops for `tone`: `wry`, `tender`, `somber`, `wondrous`). Radius `r / R` sets strength (`0%..100%`) dynamically while angle selects the stop, highlighting its sector in glowing coral with a bold badge, debouncing `scheduleDialInspect(180)` to refresh live `next:` word predictions. Dragging outward past the stop word region beyond the arc (`r >= R + 28`) — or clicking `[words ◯]` — enters **Level 3**.
     - **Level 3 (`dialLevel = 3`, Concentric Multi-Ring Speculative Word Tree, `R_k = 104 + k·68`, up to `MAX_WORD_RINGS = 6`)**:
       - Projects the Top-6 next-token predictions (`158°..22°`) onto Ring 0 (`R = 104`).
       - **Grow Outward Without Mutating Prose**: Crossing any word's circular arc (`r >= activeRing.radius`) locks that word on the inner ring (`chosenIdx`, kept highlighted while earlier alternatives dim), shows the ephemeral green inline `#wheelGhostPreview` after `◎`, and calls `fetchWordRingCandidates` (`POST /api/inspect` over `baseIds.concat(stagedIds)`, cached in `state.ringInspectCache`) to bloom a **larger concentric circle** (`R = 172, 240, 308, ...`) with the Transformer's next high-probability words following that choice!
       - **Pull Inward to Undo Ring**: Pulling back inside an inner ring (`r < innerRing.radius - 10`) collapses outer ring(s), unlocks the inner ring so another word can be picked, and updates the preview.
       - **Release Click to Commit (`commitStagedWordChain`)**: Releasing the mouse button (`mouseup`) commits the full staged word chain into the real prose at once via `appendTokenFast` + `saveUndoSnapshot()`, then refreshes `inspectCursor()` for the new frontier. Single-clicking a Ring 0 sector commits that single word immediately.
   - **Embedded Next-Token Row & Click-Outside Dismissal**:
     - `#inlineCandStrip` outside the wheel stays hidden (`display: none`); Level 1 & Level 2 embed `renderWheelPredRow()` (`next: 1.tok p% ... 5.tok p% [T/K/p]`), while `[0%]` (`clearCurrentAxisSteer`), `[← axes]`, `[← steer]`, and `[done]` let you navigate or clear steering cleanly.
     - Global `document.addEventListener("mousedown", ...)` calls `closeDial()` whenever clicking outside `#dialAnchor`, `#inlineReticle`, `#hudSteerBtn`, `.margin-tag`, or `.overlay-backdrop` (dragging the `.dial-ball` anywhere in 360° stays inside `#dialAnchor`). `Esc` closes modals and the wheel.

5. **Direct Glass-Box Attention Biasing & Word Tethers**:
   - **Click any past word** in `#proseText` (`idx < state.cursorPos`) to pin/unpin a laser tether (`boost: +3.0` added to `att[..., idx]` prior to `F.softmax`). Dotted SVG rays connect pinned words directly to `◎`, and newly generated tokens containing pinned substrings are flagged `echoed`.
   - **Click any Left LED Rail line** to toggle a single-line Attention Box (`boost: 3.0`), or **Shift+Click** a second line to span a multi-line Attention Box (`boost: 3.4`), drawn dynamically with SVG brackets and displayed `weight` share in the gutter.

---

## 3. Deep Architecture Reference

### 3.1 Engine Layer — [`steer_engine.py`](steer_engine.py)

#### Model-Agnostic Interface (`BaseSteerableEngine`, lines 186–251)
Abstract base class defining 8 core methods so other Transformer backbones (e.g., Llama-3, Qwen-2.5, Mistral) can replace GPT-2 without rewriting [`server.py`](server.py) or [`index.html`](index.html):
- `encode(text)`, `decode(ids)`, `tokenize_with_pieces(text)`
- `get_axes_metadata()`, `add_custom_axis(axis_id, label, pos_prompt, neg_prompt, color)`
- `inspect_position(token_ids, top_k=8, temperature=0.8, attn_biases=None, steering=None, top_p=0.90)`
- `step_generate(token_ids, past_kv=None, temperature=0.8, top_k=50, attn_biases=None, steering=None, forced_token_id=None, top_p=0.90, fast_mode=False)`
- `expand_multiverse_tree(token_ids, temperature=0.8, attn_biases=None, steering=None, branch_k=3)`

#### Concrete Implementation (`GPT2SteerableEngine`, lines 258–842)
- **CPU Runtime Precision & Threading**: On CPU (`pick_device("auto")`), sets `torch.set_num_threads(4)` and loads `GPT2Model.from_pretrained("gpt2")` in **`torch.bfloat16`** (non-CPU falls back to `torch.float32`).
- **Glass-Box Attention Instrumentation (`_install_glass_box_hooks`, lines 307–405)**:
  - Wraps every block's `attn.forward` via `MethodType` so Q·K scores receive `att = att + b_slice.to(att.dtype).view(1, 1, 1, T_total)` **before** `F.softmax(att, dim=-1)`.
  - Gated by `self._capture_attn`: when enabled, captures head-averaged last-token attention weights from later layers (`l_idx >= engine.n_layer // 2`) cast to `.float()` for the Left LED Rail. Disabled during `fast_mode=True`, `compute_vector`, and `expand_multiverse_tree` to save overhead.
- **Layer-6 Residual Stream Steering (`residual_steer_hook`, lines 387–405)**:
  - Attached via `register_forward_hook` on block `self.steer_layer` (default `6`).
  - Captures frontier hidden state `x[0, -1, :].detach().float()` to compute cosine-similarity realization scores (`vector_scores`) for right-margin pills.
  - Injects `x = x + engine._active_steer_vec.to(x.dtype).view(1, 1, -1)` during forward passes.
  - Contrastive vectors are precomputed in `_build_default_vector_bank` via `compute_vector(pos_prompt, neg_prompt)`: normalized difference `(h_pos - h_neg) / (norm + 1e-8) * 9.0` (empirically scaled to ~9% of Layer-6 residual norm ~97 to steer reliably without syntax collapse).
- **Numerically Safe Sampling & `fast_mode=True` (`step_generate`, lines 649–747)**:
  - Casts final logits `logits[0, -1, :].float()` to FP32 prior to temperature division, softmax, entropy, and sampling.
  - **O(K) Nucleus Sampling**: When `top_k > 0`, runs `torch.topk(scaled, top_k)` first and computes `cumsum` nucleus filtering only over those $K$ logits instead of sorting all 50,257 vocabulary entries (falls back to full-vocab sort only if `top_k <= 0`).
  - When `fast_mode=True`, skips 50,257-element `log2` entropy, Top-10 string decoding, and attention extraction, returning only `token_id`, `token_str`, and `tok_per_sec`.
- **Multiverse Tree (`expand_multiverse_tree`, lines 750–841)**:
  - Runs 1 root forward pass for `kv_root`, selects `branch_k=3` children via `torch.topk`, branches 3 child passes for `kv_b1`, selects `branch_k=3` grandchildren (`tid2`), and performs 2 greedy `argmax` continuation steps (`tid3`, `tid4`) per leaf using cached `kv_b2` for preview text.

#### Built-in Steering Axes (`DEFAULT_AXES_SPEC`, 8 Axes)

| Axis `id` | Color | Stops (`default_stop`) |
|---|---|---|
| `tension` | `#EF4444` | `very relaxed` (`0`), `uneasy`, `very strained` |
| `tone` | `#58A6FF` | `wry`, `tender` (`1`), `somber`, `wondrous` |
| `distance` | `#22C55E` | `outside view`, `inside mind` (`1`) |
| `mode` | `#EF4444` | `description`, `action` (`1`) |
| `fortune` | `#F97316` | `grim`, `fortunate` (`1`) |
| `surprise` | `#EAB308` | `expected`, `very unexpected` (`1`) |
| `resolution` | `#58A6FF` | `open mystery` (`0`), `decisive` |
| `pace` | `#A855F7` | `slow & lyrical`, `somewhat compressed` (`1`) |

#### Preset Prompts (`PRESET_STORIES`)
1. `old_lighthouse` — *Captain Elias & The Lighthouse* (auto-loaded default on startup; paired with demo pins `"figure"`, `"jagged"`, `"resting"`, `"crack"`, two LED rail boxes on lines 4–5 and 7–9, and `mode: action 90%`)
2. `forest_legend` — *Elara & The Forest Legend*
3. `biochem_discovery` — *The Molecular Mystery*
4. `blank_canvas` — *Custom / Blank Prompt* (`"The secret to artificial intelligence lies not in a single equation, but in"`)

---

### 3.2 Async Backend & Cache — [`server.py`](server.py)

#### HTTP Endpoints

| Method & Path | Purpose | Key Payload / Response Fields |
|---|---|---|
| `GET /` | Serve UI | Returns [`index.html`](index.html) via `FileResponse`. |
| `GET /api/status` | Runtime metadata | `model_name`, `param_count`, `device`, `steer_layer`, `block_size`, `axes`, `presets`. |
| `POST /api/tokenize` | Tokenize prompt | Request `{ text }` -> Response `{ pieces: [{ index, id, text }] }`. |
| `POST /api/inspect` | Position telemetry | Request `{ token_ids, temperature, top_k, top_p, attn_biases, steering }` -> Response `{ top_k: [{ rank, id, token, prob, cum_prob, in_nucleus, logit }], entropy, token_attention, vector_scores }`. |
| `POST /api/tree` | Multiverse tree | Request `{ token_ids, temperature, attn_biases, steering }` -> Response 3×3×preview tree from `expand_multiverse_tree`. |
| `POST /api/axis/custom` | Add contrastive axis | Request `{ id, label, pos_prompt, neg_prompt, color }` -> Response `{ axis, axes }`. |

#### Persistent Prefix KV-Cache & Locking
- Global `PREFIX_CACHE = {"ids": [], "past_kv": None, "sig": None}` (`server.py#L38`) stores exact token ID history and `_make_steer_sig(attn_biases, steering)` signature.
- `_get_prefix_kv` computes longest common prefix with cached history under matching `sig` and slices `(k[:, :, :common, :], v[:, :, :common, :])`, so rewinding or branching avoids full prompt recomputation.
- `ENGINE_LOCK = asyncio.Lock()` (`server.py#L35`) is acquired **per single token step** inside both `run_stream` and `run_prefetch`, followed by brief yields (`0.002s` / `0.004s`), allowing `stop`, `cancel_prefetch`, and `/api/inspect` to preempt long bursts within one token (~25–30 ms).

#### WebSocket Protocol (`WS /ws`, lines 179–344)

| Direction | `action` / `type` | Contract & Behavior |
|---|---|---|
| Client -> Server | `generate` / `step` | Cancels active `prefetch_task` & previous `gen_task`; runs `run_stream` with `fast_mode=True`. `step` forces `max_tokens=1`. Carries integer `stream_id`. |
| Client -> Server | `prefetch` | Ignored if `gen_task` is active; otherwise cancels any older `prefetch_task` and runs `run_prefetch` up to `min(count, 96)` tokens with `gen_id`. |
| Client -> Server | `cancel_prefetch` | Sets `prefetch_stop_event` and awaits `prefetch_task` shutdown. |
| Client -> Server | `stop` | Cancels both `prefetch_task` and `gen_task`. |
| Server -> Client | `token` | `{ type: "token", stream_id, cursor_t, token_id, token_str, tok_per_sec }`. |
| Server -> Client | `done` | `{ type: "done", stream_id, reason }` (`"max_tokens"`, `"stopped"`, or `"empty_prompt"`). |
| Server -> Client | `prefetch_token` | `{ type: "prefetch_token", gen_id, token_id, token_str, tok_per_sec }`. |
| Server -> Client | `prefetch_done` | Always emitted in `run_prefetch`'s `finally:` block: `{ type: "prefetch_done", gen_id, reason }` (`"complete"`, `"cancelled"`, or `"empty"`). |
| Server -> Client | `error` | `{ type: "error", message }`. |

---

### 3.3 Frontend Instrument UI — [`index.html`](index.html)

- **5-Column Grid Layout** (`.instrument-grid`: `30px 66px 96px 590px 240px`):
  1. `col-ruler`: Monospace line numbers (`measureVisualLines`).
  2. `col-led-rail`: Per-line normalized attention intensity bars (`renderLedRail`).
  3. `col-brackets`: SVG brackets & summed weight readout for clicked Attention Boxes (`renderSpanBrackets`).
  4. `col-prose`: Token spans (`#proseText`), inline `◎` reticle, anchored wheel `#dialAnchor`, and hidden `#inlineCandStrip`.
  5. `col-margin`: Floating steering pills with realization percentages and right-click removal (`renderRightMarginTags`).
- **O(1) Streaming DOM Pipeline & rAF Coalescing**:
  - `appendTokenFast(tid, tstr, tokPerSec)` appends a single `<span class="tok">` before `#inlineReticle` and updates latency/count badges immediately without rebuilding existing DOM spans (`renderAll()`).
  - `scheduleRailSync()` coalesces line remeasurement, ruler/rail/bracket updates (on line wrap or open wheel), laser ray SVG redraws, and `retEl.scrollIntoView({ block: "nearest" })` once per animation frame (`requestAnimationFrame`).
- **Speculative Prefetch Lifecycle (`state.prefetchBuffer`, `prefetchGenId`, `prefetchInFlight`)**:
  - Automatically warmed on `ws.onopen` (`requestPrefetch(60)`), after `inspectCursor()`, and topped up (`requestPrefetch(48)`) on `popPrefetchTokens` when buffer falls below 18 (or on `prefetch_done` when below 30).
  - `invalidatePrefetch()` increments `state.prefetchGenId`, empties `state.prefetchBuffer`, resets `state.prefetchInFlight = false`, and sends `{ action: "cancel_prefetch" }`. Called cleanly across all state edits: `mountText`, `startGen`, `scrubTo`, `restoreUndoSnapshot`, word tether toggle (`createTokenSpan`), rail box click (`onRailLineClick`), bracket deletion, right-margin steer deletion, Level-2 protractor release (`onUp`), and Temperature/Top-K/Top-p modal changes.
- **Race-Free `streamId` Guarding**:
  - Both `startGen()` and `stopGen()` increment `state.streamId`, and `ws.onmessage` drops any late-arriving `"token"`, `"done"`, or `"error"` frame whose `msg.stream_id !== state.streamId` — eliminating ghost tokens after drag release, upward scroll rewind, or cursor arrival.
- **Modals & HUD**:
  - Bottom HUD: latency (`--ms/t`), stream/stop toggle (`#hudGenBtn`), `auto on/off` (`#hudAutoBtn`), token count + undo indicator (`#hudPromptBtn`), `box`, `steer`, `pin`, `T · K · p` (`#hudTempBtn`), `3×3 tree` (`#hudTreeBtn`), audio detent ticks (`#hudAudioBtn`).
  - Overlays: `#topkModal` (Top-K / Top-p / Temperature / candidate bar table), `#treeModal` (930×510 SVG 3×3×preview tree with `jumpTreeBranch`), `#promptModal` (presets + textarea editor), `#axisModal` (custom contrastive axis builder).
  - Keyboard shortcuts: `Esc` (close modals & dial), `Cmd/Ctrl+Z` (undo rewind), `Space` (toggle dial), `ArrowLeft` (`-1`), `ArrowRight` (`+1` or step 1), `ArrowUp` (`-12`), `ArrowDown` (`+12` or `startGen(15)`).

---

## 4. Benchmarks, Optimization Findings & Resolved Issues

### Measured Performance (Observed on Host Mac)
Hardware context: 8-core Intel CPU (`MPS available: False`, `device: cpu`).

| Benchmark / Check | Measured Result | Notes |
|---|---:|---|
| Raw FP32 forward (baseline) | `40.9 tok/s` (`24.4 ms/tok`) | Superseded by BF16. |
| Raw BF16 (`torch.set_num_threads(4)`) | `44.4 tok/s` (`22.5 ms/tok`) | Active dtype (`torch.bfloat16`). |
| Dynamic INT8 (`fbgemm` / `qnnpack`) | `7.0 tok/s` / `6.3 tok/s` | **Rejected** — ~6× slower on this Intel CPU and degraded logits. |
| `POST /api/inspect` (with Layer-6 steer) | `138.9 – 141.9 ms` | Returns FP32-cast entropy (`5.42`), Top-K (`cum_prob`, `in_nucleus`), and `token_attention` (`len=16`). |
| `POST /api/inspect` (concentric ring chain: Ring 0 -> 1 -> 2) | `92.4 ms` -> `38.8 ms` -> `47.8 ms` | `PREFIX_CACHE` KV reuse makes outer concentric speculative rings ~38–48 ms/ring (`"It was a dark and stormy"` -> `' night, and'`). |
| `WS /ws generate` (60 tokens, fast mode) | `32.59 tok/s` e2e (`1840.8 ms`) / `34.40 tok/s` decode / `38.4` server `tok/s` (first run); `28.77 tok/s` e2e / `30.39 tok/s` decode / `33.0` server `tok/s` (second run after final edits) | Includes full WebSocket JSON framing, Starlette asyncio loop yield (`0.002s`), and Steering Vector hook. |
| `WS /ws` prefetched buffer burst (`popPrefetchTokens`) | Instant (1 frame / ~16–34 ms UI drip) | Zero inference stall while buffered tokens remain ahead of cursor. |
| `WS /ws` bounded scroll burst (`max_tokens=3`) | Exact `3` tokens + `done` | Confirmed `stream_id=8` produces strictly requested token budget. |

### Chronological Engineering & Fix Log

1. **BF16 Dtype Safety Across Attention, Residual & Logits**:
   - Guarded `_active_attn_bias` (`.to(att.dtype)`) and `_active_steer_vec` (`.to(x.dtype)`) casts inside hooks, and cast captured activations + final `logits_last` to `.float()` prior to softmax, Shannon entropy `log2`, cosine similarity, and O(K) sampling so `torch.bfloat16` weights never hit dtype mismatches.
2. **Speculative Lookahead & Prefetch Top-Up Race Fix**:
   - Fixed `requestPrefetch` spamming on rapid scroll/drag by adding `state.prefetchInFlight` guard, incrementing `prefetchGenId`, adding `ws.onopen` initial warmup, and emitting deterministic `prefetch_done` (`reason: "complete" | "cancelled" | "empty"`) inside `run_prefetch`'s `finally:` block.
3. **Eliminated Runaway Auto-Complete (`startGen(60)`) & Stale Frame Leaks**:
   - Replaced fixed `startGen(60)` calls in both `handleReticleOrWheelScroll` and `dragWriteLoop` with exact pixel-debt accounting (`scrollPixelDebt` / `scrollTokenBudget`, `PX_PER_TOKEN = 16`, `PX_PER_REWIND = 18`) and 2D spatial drag bounds (`isReticleBehindDragTarget()`).
   - Tagged WS streams with monotonic `stream_id` across `server.py` and `index.html` so in-flight tokens or `done` frames from cancelled streams are ignored immediately.
4. **Wheel Selector Polish, 360° Polar Upgrade & Concentric Multi-Ring Speculative Staging**:
   - Embedded the clickable next-token predictions row directly under the semicircular wheel (`renderWheelPredRow`), kept `#inlineCandStrip` hidden, and added global `mousedown` dismissal when clicking outside the wheel controls.
   - Upgraded `renderDialOverlay()` into a full 360° polar joystick (`handleDialPointer`, `appendPieBall`, `bindGlobalDialDrag`) with all `M` axis stops shown along Level 2's arc and debounced live `scheduleDialInspect(180)` during dragging.
   - Evolved Level 3 into a growing **Concentric Multi-Ring Speculative Word Tree** (`wordRings`, `ringInspectCache`, `fetchWordRingCandidates`, `updateWheelGhostPreview`, `commitStagedWordChain`): dragging across a word's circular arc spawns a larger outer ring (`R = 104, 172, 240, ...`) showing the Transformer's high-probability continuations with a dashed green inline ghost preview after `◎`, pulling inward collapses rings to switch choices, and releasing the mouse (`mouseup`) commits the full staged phrase into the real text.

---

## 5. Verification & Quick-Resume Playbook

### Syntax Verification (`steer_engine.py`, `server.py`, `index.html`)

```bash
cd "/Users/admin/Documents/Elhams Universe/Artificial Intelligence/llm Intelligence/projects/LLM related Projects while learning/02_chronosteer_studio"
/opt/anaconda3/bin/python -m py_compile server.py steer_engine.py
/opt/anaconda3/bin/python -c '
import re, pathlib
html = pathlib.Path("index.html").read_text()
m = re.findall(r"<script>([\s\S]*?)</script>", html)
assert m
pathlib.Path("/tmp/chronosteer_check.js").write_text("\n".join(m))
' && node --check /tmp/chronosteer_check.js
```

### Automated End-to-End Test Suite (`TestClient` Benchmark)
A complete integration script already exists at:
- `/Users/admin/.gemini/antigravity/brain/63d32711-6f97-4251-ba35-2d681f29c3f4/scratch/verify_chronosteer.py`

Run it from the `02_chronosteer_studio/` directory:

```bash
cd "/Users/admin/Documents/Elhams Universe/Artificial Intelligence/llm Intelligence/projects/LLM related Projects while learning/02_chronosteer_studio"
/opt/anaconda3/bin/python "/Users/admin/.gemini/antigravity/brain/63d32711-6f97-4251-ba35-2d681f29c3f4/scratch/verify_chronosteer.py"
```

Checks verified cleanly on the latest code (exit code `0`):
1. `GET /api/status` (`gpt2`, `device=cpu`, `axes=8`, `torch.bfloat16`)
2. `POST /api/tokenize` (16 prompt tokens)
3. `POST /api/inspect` (`141.9 ms`, `entropy=5.42`, `top1=' glass'`, `cum_prob`, `in_nucleus`, `token_attention`)
4. `WS /ws prefetch` + `cancel_prefetch` (verified `prefetch_token` with `gen_id=1` and clean `prefetch_done` `reason='cancelled'`)
5. `WS /ws generate` (60 tokens, `stream_id=7`, `30.39 tok/s` decode / `33.0` server-reported `tok/s`)
6. `WS /ws generate` bounded burst (`max_tokens=3`, `stream_id=8` -> strictly 3 `token` frames + `done`)

---

## 6. Roadmap & Next Steps for Future Work

Recommended improvements when resuming development on Project 02:

1. **Test Promotion into Repo**: Copy `/Users/admin/.gemini/antigravity/brain/63d32711-6f97-4251-ba35-2d681f29c3f4/scratch/verify_chronosteer.py` into `02_chronosteer_studio/tests/test_server_ws.py` (and add a small Playwright/browser interaction test checking scroll-up vanishing, wheel click-outside close, and drag-to-target termination).
2. **Faster / Stronger Backends via `BaseSteerableEngine`**:
   - Implement adapters for larger local models (e.g., `Qwen2.5` / `Llama-3.2` / HuggingFace quantized weights or MLX on Apple Silicon machines) behind `BaseSteerableEngine` for higher semantic coherence while keeping raw single-token latency < 15 ms.
3. **Configurable Scroll/Trackpad Sensitivity**:
   - Expose `PX_PER_TOKEN` (currently `16`) and `PX_PER_REWIND` (currently `18`) in the UI or settings popover so Apple Magic Trackpad vs discrete mouse-wheel users can tune progression speed directly.
4. **Keyboard `ArrowDown` Prefetch Alignment**:
   - Currently [`index.html`](index.html#L2531-L2535) maps `ArrowDown` at frontier directly to `startGen(15)`. Updating `ArrowDown` to drain `popPrefetchTokens` first (like `handleReticleOrWheelScroll`) will make keyboard stepping instantaneous too.
5. **Persistent Branch History / Time-Travel Graph**:
   - `scrubTo` intentionally makes future tokens vanish on the active prose canvas, while `undoStack` keeps flat linear snapshots. An optional non-destructive branch tree sidebar (recording discarded branches when rewinding & rewriting) would preserve explored storylines without cluttering the main canvas.
6. **Per-Layer / Per-Head Attention Filtering**:
   - Currently `GPT2SteerableEngine` averages later-half layers (`l_idx >= n_layer // 2`) across all heads for the Left LED Rail. Adding a head/layer selector in `inspect_position` would turn the LED Rail into an induction-head / syntactic-head microscope.
7. **Multi-Entry LRU `PREFIX_CACHE`**:
   - [`server.py`](server.py#L38) currently holds a single global `PREFIX_CACHE` entry. Switching to a 4–8 entry LRU dict keyed by steering signature (`sig`) would allow instantaneous back-and-forth A/B comparison across different steering angles or tree branches without invalidating KV caches.
8. **Prompt & Steering Export**:
   - Add JSON / Markdown session export (prompt, generated tokens, pinned word boosts, LED box coordinates, Layer-6 steering axes & strengths) so interesting steered stories can be saved and replayed deterministically.

