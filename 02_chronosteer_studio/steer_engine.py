"""
ChronoSteer Studio — Glass-Box Steerable Transformer Engine.

Wraps custom PyTorch Transformer models (starting with Project 01 GPT2Model,
designed with a clean BaseSteerableEngine interface so Llama-3 / Qwen-2.5 /
Mistral adapters can be plugged in seamlessly later).

Capabilities:
  1. Direct Q·K Attention Biasing before Softmax (Left Rail Boxes & Word Tethers).
  2. Live Per-Token Attention Telemetry Extraction for the Left Attention Rail.
  3. Residual-Stream Contrastive Steering Vector Injection at Layer 6 (Radial Dial).
  4. Fast KV-Cache Prefix Rewind & 3 x 3 x 3 Multiverse Word Tree Expansion.
  5. Full Top-K Candidate, Logit (z), and Shannon Entropy H(p) Inspection.
"""

from __future__ import annotations

import math
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Import Project 01 GPT-2 modules cleanly without modifying Project 01 files
PROJECT_01_DIR = Path(__file__).resolve().parent.parent / "01_gpt2_from_scratch"
if str(PROJECT_01_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_01_DIR))

from model import GPT2Model  # noqa: E402
from generate import load_tokenizer, pick_device  # noqa: E402


# ---------------------------------------------------------------------------
# Color & Steering Axis Definitions (Synced with UI Blueprints)
# ---------------------------------------------------------------------------

DEFAULT_AXES_SPEC: List[Dict[str, Any]] = [
    {
        "id": "tension",
        "label": "tension",
        "color": "#EF4444",
        "stops": [
            {"name": "very relaxed", "pos_prompt": "Calm, peaceful stillness, completely safe and relaxed.", "neg_prompt": "Terrifying immediate danger, frantic panic, high stakes."},
            {"name": "uneasy", "pos_prompt": "Subtle dread, quiet suspicion, faint unease in the dark.", "neg_prompt": "Carefree safety and cheerful certainty."},
            {"name": "very strained", "pos_prompt": "Heart-pounding suspense, sharp fear, unbearable tension and peril.", "neg_prompt": "Calm, relaxed, peaceful rest without worry."},
        ],
        "default_stop": 0,
    },
    {
        "id": "tone",
        "label": "tone",
        "color": "#58A6FF",
        "stops": [
            {"name": "wry", "pos_prompt": "A dry, ironic, wryly understated observation about life.", "neg_prompt": "A deeply earnest, dramatic, emotional outburst."},
            {"name": "tender", "pos_prompt": "Gentle, tender warmth, soft affection, and quiet care.", "neg_prompt": "Harsh, cold, cynical indifference and cruelty."},
            {"name": "somber", "pos_prompt": "Heavy sorrow, quiet mourning, dark solemn melancholy.", "neg_prompt": "Bright cheerful optimism and lighthearted comedy."},
            {"name": "wondrous", "pos_prompt": "Awe-inspiring cosmic wonder, luminous magic, and discovery.", "neg_prompt": "Mundane, boring, ordinary everyday routine."},
        ],
        "default_stop": 1,
    },
    {
        "id": "distance",
        "label": "distance",
        "color": "#22C55E",
        "stops": [
            {"name": "outside view", "pos_prompt": "External camera view observing physical shapes from afar.", "neg_prompt": "Deep internal thoughts, hidden memories, subjective mind."},
            {"name": "inside mind", "pos_prompt": "Intimate inner thoughts, memory, emotion, and private consciousness.", "neg_prompt": "Objective external physical description from a distance."},
        ],
        "default_stop": 1,
    },
    {
        "id": "mode",
        "label": "mode",
        "color": "#EF4444",
        "stops": [
            {"name": "description", "pos_prompt": "Rich atmospheric sensory description of shadows, shapes, and silence.", "neg_prompt": "Fast sudden physical movement, lunging, grabbing, action."},
            {"name": "action", "pos_prompt": "Immediate sudden physical action, lunging, grabbing, tearing, swift motion.", "neg_prompt": "Static quiet scenery description without movement."},
        ],
        "default_stop": 1,
    },
    {
        "id": "fortune",
        "label": "fortune",
        "color": "#F97316",
        "stops": [
            {"name": "grim", "pos_prompt": "Dark misfortune, ominous peril, loss and cold ruin.", "neg_prompt": "Lucky escape, hopeful discovery, warm rescue."},
            {"name": "fortunate", "pos_prompt": "Unexpected blessing, hopeful luck, warm rescue and light.", "neg_prompt": "Dark doom, disaster, tragic loss."},
        ],
        "default_stop": 1,
    },
    {
        "id": "surprise",
        "label": "surprise",
        "color": "#EAB308",
        "stops": [
            {"name": "expected", "pos_prompt": "Logical, expected, realistic continuation of events.", "neg_prompt": "Bizarre surreal twist, shocking impossible apparition."},
            {"name": "very unexpected", "pos_prompt": "A sudden astonishing twist, uncanny revelation, unexpected shift.", "neg_prompt": "Ordinary predictable routine continuation."},
        ],
        "default_stop": 1,
    },
    {
        "id": "resolution",
        "label": "resolution",
        "color": "#58A6FF",
        "stops": [
            {"name": "open mystery", "pos_prompt": "Unresolved mystery deepening into darker unknown questions.", "neg_prompt": "Clear decisive closure and tidy resolution."},
            {"name": "decisive", "pos_prompt": "Clear decisive closure, answers revealed, mysteries solved.", "neg_prompt": "Lingering unanswered mystery and ambiguity."},
        ],
        "default_stop": 0,
    },
    {
        "id": "pace",
        "label": "pace",
        "color": "#A855F7",
        "stops": [
            {"name": "slow & lyrical", "pos_prompt": "Slow, lingering poetic rhythm savoring every quiet detail.", "neg_prompt": "Rapid bullet-like action, sudden sprints, terse sentences."},
            {"name": "somewhat compressed", "pos_prompt": "Crisp, tightly compressed prose moving swiftly forward.", "neg_prompt": "Long, meandering, slow descriptive passages."},
        ],
        "default_stop": 1,
    },
]


PRESET_STORIES: List[Dict[str, str]] = [
    {
        "id": "old_lighthouse",
        "title": "Captain Elias & The Lighthouse",
        "text": (
            "It was a dark and stormy night when the wind howled through the cracked windows "
            "of the old lighthouse, rattling the ancient glass panes like a frantic chorus "
            "of ghostly violins. Inside, a lone figure hunched over a weathered desk, the "
            "candlelight flickering against the maps strewn across it. Captain Elias Marlowe "
            "traced his finger along the jagged coastline, his mind racing with memories of "
            "shipwrecks and whispered legends that haunted these of the cliffs for generations. "
            "He paused, his thumb resting on a faded ink blot that marked a spot where no "
            "chart had ever shown a harbor—only a swirling vortex drawn in erratic strokes. "
            "A sudden crack of thunder rattled the lighthouse's stone and sent a shiver down "
            "the captain's spine. He leaned closer, the candle sputtering as the wind "
            "threatened to snuff it"
        ),
    },
    {
        "id": "forest_legend",
        "title": "Elara & The Forest Legend",
        "text": (
            "There was an old legend about a forest that never slept, where the trees "
            "whispered secrets to anyone brave enough to listen. One night, under a sky "
            "brushed with clouds, a young wanderer named Elara stepped through the tangled "
            "undergrowth, guided only by the faint hum of an ancient lullaby carried on "
            "the wind. As she pushed forward, the forest grew quieter, its mossy floor "
            "pulsing with soft bioluminescent glow. She paused beside a silver pool "
            "where the water's surface rippled with a soft, rhythmic pulse, reflecting "
            "a ring of ancient runes beneath her feet. She knelt, and as her fingertips "
            "brushed the cool liquid,"
        ),
    },
    {
        "id": "biochem_discovery",
        "title": "The Molecular Mystery",
        "text": (
            "Inside the quiet biochemistry laboratory past midnight, the X-ray "
            "crystallography map finally resolved on the monitor. For six months the "
            "enzyme had refused to bind its substrate, baffling every researcher in the "
            "department. Looking closely at the chiral center buried inside the active "
            "site pocket, Elham noticed an unexpected water bridge stabilizing the "
            "transition state. With a sudden realization, he reached for the pipette and"
        ),
    },
    {
        "id": "blank_canvas",
        "title": "Custom / Blank Prompt",
        "text": "The secret to artificial intelligence lies not in a single equation, but in",
    },
]


# ---------------------------------------------------------------------------
# Abstract Adapter Interface (Allows Swapping GPT-2 -> Llama-3 / Qwen Later)
# ---------------------------------------------------------------------------

class BaseSteerableEngine(ABC):
    """Unified contract so frontend/server remain 100% model-agnostic."""

    model_name: str
    param_count: int
    device: str
    steer_layer: int
    block_size: int

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        ...

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        ...

    @abstractmethod
    def tokenize_with_pieces(self, text: str) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def get_axes_metadata(self) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def add_custom_axis(self, axis_id: str, label: str, pos_prompt: str, neg_prompt: str, color: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def inspect_position(
        self,
        token_ids: List[int],
        top_k: int = 8,
        temperature: float = 0.8,
        attn_biases: Optional[Dict[int, float]] = None,
        steering: Optional[List[Dict[str, Any]]] = None,
        top_p: float = 0.90,
    ) -> Dict[str, Any]:
        ...

    @abstractmethod
    def step_generate(
        self,
        token_ids: List[int],
        past_kv: Any = None,
        temperature: float = 0.8,
        top_k: int = 50,
        attn_biases: Optional[Dict[int, float]] = None,
        steering: Optional[List[Dict[str, Any]]] = None,
        forced_token_id: Optional[int] = None,
        top_p: float = 0.90,
        fast_mode: bool = False,
    ) -> Tuple[Dict[str, Any], Any]:
        ...

    @abstractmethod
    def expand_multiverse_tree(
        self,
        token_ids: List[int],
        temperature: float = 0.8,
        attn_biases: Optional[Dict[int, float]] = None,
        steering: Optional[List[Dict[str, Any]]] = None,
        branch_k: int = 3,
    ) -> Dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# Concrete Project 01 GPT-2 Steerable Engine
# ---------------------------------------------------------------------------

class GPT2SteerableEngine(BaseSteerableEngine):
    """
    Wraps 01_gpt2_from_scratch.GPT2Model with:
      - Direct pre-softmax attention score biasing (+attn_bias)
      - Live attention matrix readout for the Left Rail
      - Layer-6 contrastive residual-stream steering vector injection
      - KV-cache snapshots for instant timeline rewinding & tree expansion
    """

    def __init__(self, model_type: str = "gpt2", device: str = "auto", steer_layer: int = 6):
        self.model_name = model_type
        self.device = pick_device(device)
        self._encode_fn, self._decode_fn = load_tokenizer()

        if self.device == "cpu":
            torch.set_num_threads(4)
            self.model_dtype = torch.bfloat16
        else:
            self.model_dtype = torch.float32

        print(f"[ChronoSteer] Loading '{model_type}' onto {self.device} ({self.model_dtype})...")
        self.model = GPT2Model.from_pretrained(model_type)
        self.model.to(device=self.device, dtype=self.model_dtype)
        self.model.eval()

        self.param_count = self.model.count_parameters()
        self.block_size = self.model.config.block_size
        self.n_layer = self.model.config.n_layer
        self.steer_layer = min(steer_layer, self.n_layer - 1)

        # Runtime hook state (set cleanly per forward pass)
        self._active_attn_bias: Optional[torch.Tensor] = None
        self._captured_attns: List[torch.Tensor] = []
        self._capture_attn: bool = True
        self._active_steer_vec: Optional[torch.Tensor] = None
        self._captured_steer_hidden: Optional[torch.Tensor] = None

        self._install_glass_box_hooks()

        # Precompute normalized steering vectors for every axis + stop
        self.axes_spec: List[Dict[str, Any]] = []
        self.vector_bank: Dict[Tuple[str, str], torch.Tensor] = {}
        self._build_default_vector_bank()

        print(
            f"[ChronoSteer] Ready: {self.model_name} ({self.param_count:,} params) | "
            f"Steer Layer={self.steer_layer} | Axes={len(self.axes_spec)}"
        )

    def _install_glass_box_hooks(self) -> None:
        """
        Wrap each block's CausalSelfAttention.forward cleanly on this instance
        (zero modifications to Project 01 source files!) to support:
          1) Adding self._active_attn_bias before softmax
          2) Capturing head-averaged attention weights for the Left Rail
        And attach a forward hook on Layer `self.steer_layer` for residual injection.
        """
        engine = self

        for layer_idx, block in enumerate(self.model.transformer.h):
            attn_mod = block.attn

            def make_steerable_forward(mod, l_idx):
                def steerable_forward(
                    this,
                    x: torch.Tensor,
                    return_attention_weights: bool = False,
                    past_kv=None,
                    use_cache: bool = False,
                ):
                    B, T, C = x.size()
                    qkv = this.c_attn(x)
                    q, k, v = qkv.split(this.d_model, dim=2)

                    q = q.view(B, T, this.n_heads, this.d_k).transpose(1, 2)
                    k = k.view(B, T, this.n_heads, this.d_k).transpose(1, 2)
                    v = v.view(B, T, this.n_heads, this.d_k).transpose(1, 2)

                    past_len = 0
                    if past_kv is not None:
                        past_k, past_v = past_kv
                        past_len = past_k.size(2)
                        k = torch.cat((past_k, k), dim=2)
                        v = torch.cat((past_v, v), dim=2)

                    present_kv = (k, v) if use_cache else None
                    T_total = k.size(2)

                    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(this.d_k))
                    att = att.masked_fill(
                        this.bias[:, :, past_len : past_len + T, :T_total] == 0,
                        float("-inf"),
                    )

                    # Glass-Box Attention Bias Injection (from Left Rail & Word Tethers)
                    if engine._active_attn_bias is not None:
                        bias = engine._active_attn_bias
                        if bias.size(-1) >= T_total:
                            b_slice = bias[..., :T_total]
                        else:
                            b_slice = F.pad(bias, (0, T_total - bias.size(-1)), value=0.0)
                        # Apply across all heads & query rows where key position exists
                        att = att + b_slice.to(att.dtype).view(1, 1, 1, T_total)

                    att_weights = F.softmax(att, dim=-1)

                    # Capture later-half layer attention to last token for UI Left Rail
                    if engine._capture_attn and l_idx >= engine.n_layer // 2:
                        # Average over heads -> (T_total,)
                        engine._captured_attns.append(att_weights[0, :, -1, :].mean(dim=0).detach().float())

                    att_probs = this.attn_dropout(att_weights)
                    y = att_probs @ v
                    y = y.transpose(1, 2).contiguous().view(B, T, C)
                    out = this.resid_dropout(this.c_proj(y))

                    if return_attention_weights and use_cache:
                        return out, att_weights, present_kv
                    if return_attention_weights:
                        return out, att_weights
                    if use_cache:
                        return out, present_kv
                    return out

                return MethodType(steerable_forward, mod)

            attn_mod.forward = make_steerable_forward(attn_mod, layer_idx)

        # Residual Stream Steering Hook at middle layer
        def residual_steer_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                x, present = output
            else:
                x, present = output, None

            # Capture raw residual at frontier position for realization alignment check
            if engine._capture_attn or engine._active_steer_vec is None:
                engine._captured_steer_hidden = x[0, -1, :].detach().float()

            if engine._active_steer_vec is not None:
                # Inject steering vector into the frontier / all newly processed positions
                x = x + engine._active_steer_vec.to(x.dtype).view(1, 1, -1)

            if present is not None:
                return (x, present)
            return x

        self.model.transformer.h[self.steer_layer].register_forward_hook(residual_steer_hook)

    # -----------------------------------------------------------------------
    # Tokenization & Contrastive Steering Vectors
    # -----------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        return list(self._encode_fn(text))

    def decode(self, ids: List[int]) -> str:
        return self._decode_fn(ids)

    def tokenize_with_pieces(self, text: str) -> List[Dict[str, Any]]:
        ids = self.encode(text)
        pieces: List[Dict[str, Any]] = []
        for idx, tid in enumerate(ids):
            pieces.append({
                "index": idx,
                "id": int(tid),
                "text": self.decode([tid]),
            })
        return pieces

    @torch.no_grad()
    def compute_vector(self, pos_prompt: str, neg_prompt: str) -> torch.Tensor:
        """Compute unit-normalized contrastive activation vector at self.steer_layer."""
        self._active_attn_bias = None
        self._active_steer_vec = None
        prev_cap = self._capture_attn
        self._capture_attn = False

        pos_ids = torch.tensor([self.encode(pos_prompt)], dtype=torch.long, device=self.device)
        self._captured_steer_hidden = None
        self.model(pos_ids)
        h_pos = self._captured_steer_hidden.clone()

        neg_ids = torch.tensor([self.encode(neg_prompt)], dtype=torch.long, device=self.device)
        self._captured_steer_hidden = None
        self.model(neg_ids)
        h_neg = self._captured_steer_hidden.clone()
        self._capture_attn = prev_cap

        diff = h_pos - h_neg
        norm = diff.norm(p=2) + 1e-8
        # Gentle Layer-6 activation scale (~9% of residual norm ~97) avoids syntax collapse
        return (diff / norm) * 9.0

    def _build_default_vector_bank(self) -> None:
        self.axes_spec = []
        for axis in DEFAULT_AXES_SPEC:
            axis_copy = {
                "id": axis["id"],
                "label": axis["label"],
                "color": axis["color"],
                "default_stop": axis["default_stop"],
                "stops": [],
            }
            for stop in axis["stops"]:
                vec = self.compute_vector(stop["pos_prompt"], stop["neg_prompt"])
                self.vector_bank[(axis["id"], stop["name"])] = vec
                axis_copy["stops"].append({"name": stop["name"]})
            self.axes_spec.append(axis_copy)

    def get_axes_metadata(self) -> List[Dict[str, Any]]:
        return self.axes_spec

    def add_custom_axis(
        self,
        axis_id: str,
        label: str,
        pos_prompt: str,
        neg_prompt: str,
        color: str = "#58A6FF",
    ) -> Dict[str, Any]:
        clean_id = axis_id.strip().lower().replace(" ", "_") or "custom"
        stop_name = label.strip() or clean_id
        vec = self.compute_vector(pos_prompt, neg_prompt)
        self.vector_bank[(clean_id, stop_name)] = vec

        spec = {
            "id": clean_id,
            "label": clean_id,
            "color": color,
            "default_stop": 0,
            "stops": [{"name": stop_name}],
        }
        # Replace if exists, else append
        self.axes_spec = [a for a in self.axes_spec if a["id"] != clean_id] + [spec]
        return spec

    # -----------------------------------------------------------------------
    # Helpers to Assemble Steering Tensor & Attention Bias
    # -----------------------------------------------------------------------

    def _build_attn_bias_tensor(
        self,
        seq_len: int,
        attn_biases: Optional[Dict[int, float]],
    ) -> Optional[torch.Tensor]:
        if not attn_biases:
            return None
        bias = torch.zeros(seq_len, dtype=torch.float32, device=self.device)
        any_nonzero = False
        for k_idx, boost in attn_biases.items():
            idx_int = int(k_idx)
            if 0 <= idx_int < seq_len and abs(float(boost)) > 1e-6:
                bias[idx_int] = float(boost)
                any_nonzero = True
        return bias if any_nonzero else None

    def _build_steer_vec_tensor(
        self,
        steering: Optional[List[Dict[str, Any]]],
    ) -> Tuple[Optional[torch.Tensor], List[Tuple[str, str, torch.Tensor]]]:
        if not steering:
            return None, []
        total_vec: Optional[torch.Tensor] = None
        active_list: List[Tuple[str, str, torch.Tensor]] = []
        for item in steering:
            axis_id = item.get("axis")
            stop_name = item.get("stop")
            strength = float(item.get("strength", 0.0))
            if abs(strength) < 1e-4:
                continue
            vec = self.vector_bank.get((axis_id, stop_name))
            if vec is None:
                continue
            scaled = vec * strength
            total_vec = scaled if total_vec is None else (total_vec + scaled)
            active_list.append((axis_id, stop_name, vec))
        return total_vec, active_list

    def _extract_metrics(
        self,
        logits_1d: torch.Tensor,
        temperature: float,
        top_k: int,
        active_list: List[Tuple[str, str, torch.Tensor]],
        top_p: float = 0.90,
    ) -> Dict[str, Any]:
        t_eff = max(float(temperature), 1e-3)
        scaled = logits_1d / t_eff
        probs_full = F.softmax(scaled, dim=-1)

        # Shannon Entropy H(p) = - sum p log2 p
        log_probs = torch.log2(probs_full.clamp(min=1e-12))
        entropy = float(-(probs_full * log_probs).sum().item())

        k_inspect = min(max(int(top_k), 10), logits_1d.size(-1))
        top_vals, top_idxs = torch.topk(scaled, k=k_inspect)
        top_probs = probs_full[top_idxs]

        candidates = []
        running_cum = 0.0
        p_thresh = max(0.01, min(1.0, float(top_p)))
        k_limit = int(top_k) if (top_k is not None and int(top_k) > 0) else logits_1d.size(-1)
        for rank in range(min(10, k_inspect)):
            tid = int(top_idxs[rank].item())
            raw_logit = float(logits_1d[tid].item())
            prob_val = float(top_probs[rank].item())
            was_below = running_cum < p_thresh
            running_cum += prob_val
            in_k = (rank + 1) <= k_limit
            in_nucleus = (rank == 0 or was_below) and in_k
            candidates.append({
                "rank": rank + 1,
                "id": tid,
                "token": self.decode([tid]),
                "prob": round(prob_val * 100.0, 2),
                "cum_prob": round(min(1.0, running_cum) * 100.0, 1),
                "in_nucleus": bool(in_nucleus),
                "logit": round(raw_logit, 2),
            })

        # Aggregated attention telemetry across later layers
        if self._captured_attns:
            mean_attn = torch.stack(self._captured_attns, dim=0).mean(dim=0)
            token_attn = [round(float(v), 5) for v in mean_attn.tolist()]
        else:
            token_attn = []

        # Steering realization score (cosine similarity against layer-6 hidden state)
        vector_scores: Dict[str, float] = {}
        if self._captured_steer_hidden is not None and active_list:
            h_norm = F.normalize(self._captured_steer_hidden.unsqueeze(0), dim=-1)
            for axis_id, _stop_name, vec in active_list:
                v_norm = F.normalize(vec.unsqueeze(0), dim=-1)
                sim = float((h_norm * v_norm).sum().item())
                vector_scores[axis_id] = round(sim, 3)

        return {
            "top_k": candidates,
            "entropy": round(entropy, 2),
            "token_attention": token_attn,
            "vector_scores": vector_scores,
        }

    # -----------------------------------------------------------------------
    # Public Inference Operations
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def inspect_position(
        self,
        token_ids: List[int],
        top_k: int = 8,
        temperature: float = 0.8,
        attn_biases: Optional[Dict[int, float]] = None,
        steering: Optional[List[Dict[str, Any]]] = None,
        past_kv: Any = None,
        return_cache: bool = False,
        top_p: float = 0.90,
    ) -> Any:
        if not token_ids:
            empty = {"top_k": [], "entropy": 0.0, "token_attention": [], "vector_scores": {}}
            return (empty, None) if return_cache else empty

        cond = token_ids[-(self.block_size - 1) :]
        T_total = len(cond)

        self._captured_attns = []
        self._capture_attn = True
        self._captured_steer_hidden = None
        self._active_attn_bias = self._build_attn_bias_tensor(T_total, attn_biases)
        self._active_steer_vec, active_list = self._build_steer_vec_tensor(steering)

        if past_kv is not None and 0 < past_kv[0][0].size(2) < T_total:
            delta = past_kv[0][0].size(2)
            idx = torch.tensor([cond[delta:]], dtype=torch.long, device=self.device)
            logits, _, new_past_kv = self.model(idx, past_key_values=past_kv, use_cache=True)
        else:
            idx = torch.tensor([cond], dtype=torch.long, device=self.device)
            logits, _, new_past_kv = self.model(idx, use_cache=True)

        logits_last = logits[0, -1, :].float()
        metrics = self._extract_metrics(logits_last, temperature, top_k, active_list, top_p=top_p)
        # Clear hooks cleanly
        self._active_attn_bias = None
        self._active_steer_vec = None
        if return_cache:
            return metrics, new_past_kv
        return metrics

    @torch.no_grad()
    def step_generate(
        self,
        token_ids: List[int],
        past_kv: Any = None,
        temperature: float = 0.8,
        top_k: int = 50,
        attn_biases: Optional[Dict[int, float]] = None,
        steering: Optional[List[Dict[str, Any]]] = None,
        forced_token_id: Optional[int] = None,
        top_p: float = 0.90,
        fast_mode: bool = False,
    ) -> Tuple[Dict[str, Any], Any]:
        t0 = time.perf_counter()

        # Respect block_size window
        if len(token_ids) >= self.block_size:
            token_ids = token_ids[-(self.block_size - 1) :]
            past_kv = None

        T_total = len(token_ids)
        self._captured_attns = []
        self._capture_attn = not fast_mode
        self._captured_steer_hidden = None
        self._active_attn_bias = self._build_attn_bias_tensor(T_total, attn_biases)
        self._active_steer_vec, active_list = self._build_steer_vec_tensor(steering)

        if past_kv is not None and 0 < past_kv[0][0].size(2) < T_total:
            delta = past_kv[0][0].size(2)
            inp = torch.tensor([token_ids[delta:]], dtype=torch.long, device=self.device)
            logits, _, new_past_kv = self.model(inp, past_key_values=past_kv, use_cache=True)
        else:
            inp = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            logits, _, new_past_kv = self.model(inp, use_cache=True)

        logits_last = logits[0, -1, :].float()

        # Sample next token (or force selected token) using O(K) Top-K + Top-p Nucleus!
        if forced_token_id is not None:
            next_id = int(forced_token_id)
        elif temperature <= 0.0:
            next_id = int(torch.argmax(logits_last).item())
        else:
            scaled = logits_last / temperature
            if top_k is not None and 0 < int(top_k) < scaled.size(-1):
                top_v, top_idx = torch.topk(scaled, int(top_k))
                probs_k = F.softmax(top_v, dim=-1)
                if top_p is not None and 0.0 < float(top_p) < 1.0:
                    cum_p = torch.cumsum(probs_k, dim=-1)
                    keep = cum_p <= float(top_p)
                    keep[0] = True
                    cross_idx = int(keep.sum().item())
                    if cross_idx < probs_k.size(0):
                        keep[cross_idx] = True
                    probs_k = probs_k * keep.to(probs_k.dtype)
                    probs_k = probs_k / probs_k.sum().clamp(min=1e-12)
                choice = int(torch.multinomial(probs_k, num_samples=1).item())
                next_id = int(top_idx[choice].item())
            else:
                if top_p is not None and 0.0 < float(top_p) < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)
                    remove_mask = cum_probs > float(top_p)
                    remove_mask[1:] = remove_mask[:-1].clone()
                    remove_mask[0] = False
                    sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
                    scaled = torch.full_like(scaled, float("-inf")).scatter_(0, sorted_indices, sorted_logits)
                probs = F.softmax(scaled, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

        dt = max(time.perf_counter() - t0, 1e-5)
        next_str = self.decode([next_id])

        if fast_mode:
            self._active_attn_bias = None
            self._active_steer_vec = None
            self._capture_attn = True
            return {
                "token_id": next_id,
                "token_str": next_str,
                "tok_per_sec": round(1.0 / dt, 1),
            }, new_past_kv

        metrics = self._extract_metrics(logits_last, temperature, max(top_k, 10), active_list, top_p=top_p)

        # Clear transient state
        self._active_attn_bias = None
        self._active_steer_vec = None
        self._capture_attn = True

        step_payload = {
            "token_id": next_id,
            "token_str": next_str,
            "prob": round(float(F.softmax(logits_last / max(temperature, 1e-3), dim=-1)[next_id].item()) * 100.0, 2),
            "logit": round(float(logits_last[next_id].item()), 2),
            "tok_per_sec": round(1.0 / dt, 1),
            **metrics,
        }
        return step_payload, new_past_kv

    @torch.no_grad()
    def expand_multiverse_tree(
        self,
        token_ids: List[int],
        temperature: float = 0.8,
        attn_biases: Optional[Dict[int, float]] = None,
        steering: Optional[List[Dict[str, Any]]] = None,
        branch_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Expand a 3 x 3 x preview Multiverse Word Tree from the current cursor
        using KV-Cache branching snapshots for blazing speed!
        """
        if not token_ids:
            return {"root": "...", "children": []}

        cond = token_ids[-(self.block_size - 8) :]
        T_total = len(cond)
        t_eff = max(float(temperature), 1e-3)

        self._captured_attns = []
        self._capture_attn = False
        self._active_attn_bias = self._build_attn_bias_tensor(T_total, attn_biases)
        self._active_steer_vec, _ = self._build_steer_vec_tensor(steering)

        inp = torch.tensor([cond], dtype=torch.long, device=self.device)
        logits_root, _, kv_root = self.model(inp, use_cache=True)
        logits_r_last = logits_root[0, -1, :].float()
        probs_root = F.softmax(logits_r_last / t_eff, dim=-1)
        top_p1, top_i1 = torch.topk(probs_root, k=branch_k)

        root_tail = self.decode(cond[-3:])
        level1_nodes = []

        for b1 in range(branch_k):
            tid1 = int(top_i1[b1].item())
            w1 = self.decode([tid1])
            p1 = round(float(top_p1[b1].item()) * 100.0, 1)
            z1 = round(float(logits_r_last[tid1].item()), 2)

            # Advance 1 step from kv_root for this branch
            inp1 = torch.tensor([[tid1]], dtype=torch.long, device=self.device)
            logits_b1, _, kv_b1 = self.model(inp1, past_key_values=kv_root, use_cache=True)
            logits_b1_last = logits_b1[0, -1, :].float()
            probs_b1 = F.softmax(logits_b1_last / t_eff, dim=-1)
            top_p2, top_i2 = torch.topk(probs_b1, k=branch_k)

            level2_nodes = []
            for b2 in range(branch_k):
                tid2 = int(top_i2[b2].item())
                w2 = self.decode([tid2])
                p2 = round(float(top_p2[b2].item()) * 100.0, 1)
                z2 = round(float(logits_b1_last[tid2].item()), 2)

                # Greedy 2-token continuation tail preview (t+3, t+4)
                inp2 = torch.tensor([[tid2]], dtype=torch.long, device=self.device)
                logits_b2, _, kv_b2 = self.model(inp2, past_key_values=kv_b1, use_cache=True)
                tid3 = int(torch.argmax(logits_b2[0, -1, :].float()).item())
                inp3 = torch.tensor([[tid3]], dtype=torch.long, device=self.device)
                logits_b3, _, _ = self.model(inp3, past_key_values=kv_b2, use_cache=True)
                tid4 = int(torch.argmax(logits_b3[0, -1, :].float()).item())

                tail_ids = [tid3, tid4]
                tail_text = self.decode(tail_ids).replace("\n", " ").strip()

                level2_nodes.append({
                    "id": tid2,
                    "token": w2,
                    "prob": p2,
                    "logit": z2,
                    "path_ids": [tid1, tid2],
                    "preview_ids": [tid1, tid2] + tail_ids,
                    "preview_text": f"{tail_text}..." if tail_text else "...",
                })

            level1_nodes.append({
                "id": tid1,
                "token": w1,
                "prob": p1,
                "logit": z1,
                "path_ids": [tid1],
                "children": level2_nodes,
            })

        self._active_attn_bias = None
        self._active_steer_vec = None
        self._capture_attn = True

        return {
            "cursor_t": len(token_ids),
            "root_label": f"...{root_tail}",
            "children": level1_nodes,
        }

