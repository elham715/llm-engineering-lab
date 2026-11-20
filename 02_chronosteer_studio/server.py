"""
ChronoSteer Studio — Fast Async Backend (Starlette + Uvicorn + WebSockets).

Serves the interactive ChronoSteer Studio UI (`index.html`) and exposes:
  - GET  /api/status       : Active model metadata, steering axes & story presets
  - POST /api/tokenize     : Convert raw prompt text into token pieces
  - POST /api/inspect      : Live Top-K, Entropy H(p), and Attention Rail scores
  - POST /api/tree         : Expand 3 x 3 x 3 Multiverse Probability Tree
  - POST /api/axis/custom  : Compute a new live contrastive steering vector
  - WS   /ws               : Bi-directional real-time token streaming & rewind
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from steer_engine import GPT2SteerableEngine, PRESET_STORIES

STUDIO_DIR = Path(__file__).resolve().parent
INDEX_HTML = STUDIO_DIR / "index.html"

# Global steerable model instance (easily swappable via adapter later!)
ENGINE: Optional[GPT2SteerableEngine] = None
ENGINE_LOCK = asyncio.Lock()

# Persistent Prefix KV-Cache (instant time-travel rewind + 22 tok/s single-token stepping)
PREFIX_CACHE: Dict[str, Any] = {"ids": [], "past_kv": None, "sig": None}


def _make_steer_sig(attn_biases: Dict[int, float], steering: List[Dict[str, Any]]) -> str:
    return json.dumps(
        {
            "b": sorted((int(k), round(float(v), 4)) for k, v in attn_biases.items()),
            "s": sorted(
                (str(x.get("axis")), str(x.get("stop")), round(float(x.get("strength", 0.0)), 4))
                for x in steering
            ),
        },
        sort_keys=True,
    )


def _get_prefix_kv(token_ids: List[int], sig: str, max_len: int) -> Any:
    if PREFIX_CACHE["past_kv"] is None or PREFIX_CACHE["sig"] != sig or len(token_ids) <= 1:
        return None
    cached_ids = PREFIX_CACHE["ids"]
    limit = min(len(cached_ids), len(token_ids) - 1, max_len)
    common = 0
    while common < limit and cached_ids[common] == token_ids[common]:
        common += 1
    if common == 0:
        return None
    return [(k[:, :, :common, :], v[:, :, :common, :]) for k, v in PREFIX_CACHE["past_kv"]]


def _set_prefix_kv(token_ids: List[int], past_kv: Any, sig: str) -> None:
    if past_kv is not None:
        PREFIX_CACHE["ids"] = list(token_ids)
        PREFIX_CACHE["past_kv"] = past_kv
        PREFIX_CACHE["sig"] = sig


def get_engine() -> GPT2SteerableEngine:
    global ENGINE
    if ENGINE is None:
        ENGINE = GPT2SteerableEngine(model_type="gpt2", device="auto", steer_layer=6)
    return ENGINE


# ---------------------------------------------------------------------------
# HTTP Routes
# ---------------------------------------------------------------------------

async def serve_index(request):
    return FileResponse(INDEX_HTML)


async def api_status(request):
    eng = await asyncio.to_thread(get_engine)
    return JSONResponse({
        "model_name": eng.model_name,
        "param_count": eng.param_count,
        "device": eng.device,
        "steer_layer": eng.steer_layer,
        "block_size": eng.block_size,
        "axes": eng.get_axes_metadata(),
        "presets": PRESET_STORIES,
    })


async def api_tokenize(request):
    body = await request.json()
    text = body.get("text", "")
    eng = await asyncio.to_thread(get_engine)
    pieces = await asyncio.to_thread(eng.tokenize_with_pieces, text)
    return JSONResponse({"pieces": pieces})


async def api_inspect(request):
    body = await request.json()
    token_ids = [int(t) for t in body.get("token_ids", [])]
    temperature = float(body.get("temperature", 0.8))
    top_k = int(body.get("top_k", 10))
    top_p = float(body.get("top_p", 0.90))
    attn_biases = {int(k): float(v) for k, v in (body.get("attn_biases") or {}).items()}
    steering = body.get("steering") or []

    eng = await asyncio.to_thread(get_engine)
    sig = _make_steer_sig(attn_biases, steering)
    async with ENGINE_LOCK:
        usable_kv = _get_prefix_kv(token_ids, sig, eng.block_size - 2)
        res, new_kv = await asyncio.to_thread(
            eng.inspect_position,
            token_ids,
            top_k,
            temperature,
            attn_biases,
            steering,
            usable_kv,
            True,
            top_p,
        )
        _set_prefix_kv(token_ids, new_kv, sig)
    return JSONResponse(res)


async def api_tree(request):
    body = await request.json()
    token_ids = [int(t) for t in body.get("token_ids", [])]
    temperature = float(body.get("temperature", 0.8))
    attn_biases = {int(k): float(v) for k, v in (body.get("attn_biases") or {}).items()}
    steering = body.get("steering") or []

    eng = await asyncio.to_thread(get_engine)
    async with ENGINE_LOCK:
        tree = await asyncio.to_thread(
            eng.expand_multiverse_tree,
            token_ids,
            temperature,
            attn_biases,
            steering,
            3,
        )
    return JSONResponse(tree)


async def api_custom_axis(request):
    body = await request.json()
    axis_id = body.get("id", "custom")
    label = body.get("label", "custom")
    pos_prompt = body.get("pos_prompt", "")
    neg_prompt = body.get("neg_prompt", "")
    color = body.get("color", "#58A6FF")

    if not pos_prompt or not neg_prompt:
        return JSONResponse({"error": "Both pos_prompt and neg_prompt are required."}, status_code=400)

    eng = await asyncio.to_thread(get_engine)
    async with ENGINE_LOCK:
        spec = await asyncio.to_thread(eng.add_custom_axis, axis_id, label, pos_prompt, neg_prompt, color)
    return JSONResponse({"axis": spec, "axes": eng.get_axes_metadata()})


# ---------------------------------------------------------------------------
# WebSocket Real-Time Streaming Loop
# ---------------------------------------------------------------------------

async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    eng = await asyncio.to_thread(get_engine)
    stop_event = asyncio.Event()
    prefetch_stop_event = asyncio.Event()
    gen_task: Optional[asyncio.Task] = None
    prefetch_task: Optional[asyncio.Task] = None

    async def cancel_task(task: Optional[asyncio.Task], ev: asyncio.Event):
        ev.set()
        if task and not task.done():
            try:
                await task
            except Exception:
                pass

    async def run_stream(params: Dict[str, Any]):
        stop_event.clear()
        stream_id = int(params.get("stream_id", 0))
        token_ids: List[int] = [int(t) for t in params.get("token_ids", [])]
        max_tokens = int(params.get("max_tokens", 60))
        temperature = float(params.get("temperature", 0.8))
        top_k = int(params.get("top_k", 50))
        top_p = float(params.get("top_p", 0.90))
        attn_biases = {int(k): float(v) for k, v in (params.get("attn_biases") or {}).items()}
        steering = params.get("steering") or []
        forced_id = params.get("forced_token_id", None)

        sig = _make_steer_sig(attn_biases, steering)
        if not token_ids:
            await websocket.send_json({"type": "done", "stream_id": stream_id, "reason": "empty_prompt"})
            return

        past_kv = None
        for step_i in range(max_tokens):
            if stop_event.is_set():
                await websocket.send_json({"type": "done", "stream_id": stream_id, "reason": "stopped"})
                return

            use_forced = int(forced_id) if (step_i == 0 and forced_id is not None) else None
            async with ENGINE_LOCK:
                if stop_event.is_set():
                    await websocket.send_json({"type": "done", "stream_id": stream_id, "reason": "stopped"})
                    return
                if past_kv is None:
                    past_kv = _get_prefix_kv(token_ids, sig, eng.block_size - 2)
                payload, past_kv = await asyncio.to_thread(
                    eng.step_generate,
                    token_ids,
                    past_kv,
                    temperature,
                    top_k,
                    attn_biases,
                    steering,
                    use_forced,
                    top_p,
                    True,
                )
                token_ids.append(payload["token_id"])
                _set_prefix_kv(token_ids[:-1], past_kv, sig)

            await websocket.send_json({
                "type": "token",
                "stream_id": stream_id,
                "cursor_t": len(token_ids),
                **payload,
            })
            # Yield briefly so stop frames & inspect requests preempt immediately
            await asyncio.sleep(0.002)

        await websocket.send_json({"type": "done", "stream_id": stream_id, "reason": "max_tokens"})

    async def run_prefetch(params: Dict[str, Any]):
        prefetch_stop_event.clear()
        gen_id = int(params.get("gen_id", 0))
        count = min(int(params.get("count", 64)), 96)
        token_ids: List[int] = [int(t) for t in params.get("token_ids", [])]
        temperature = float(params.get("temperature", 0.8))
        top_k = int(params.get("top_k", 50))
        top_p = float(params.get("top_p", 0.90))
        attn_biases = {int(k): float(v) for k, v in (params.get("attn_biases") or {}).items()}
        steering = params.get("steering") or []

        reason = "complete"
        try:
            if not token_ids:
                reason = "empty"
                return

            sig = _make_steer_sig(attn_biases, steering)
            past_kv = None

            for _ in range(count):
                if prefetch_stop_event.is_set():
                    reason = "cancelled"
                    return

                async with ENGINE_LOCK:
                    if prefetch_stop_event.is_set():
                        reason = "cancelled"
                        return
                    if past_kv is None:
                        past_kv = _get_prefix_kv(token_ids, sig, eng.block_size - 2)
                    payload, past_kv = await asyncio.to_thread(
                        eng.step_generate,
                        token_ids,
                        past_kv,
                        temperature,
                        top_k,
                        attn_biases,
                        steering,
                        None,
                        top_p,
                        True,
                    )
                    token_ids.append(payload["token_id"])
                    _set_prefix_kv(token_ids[:-1], past_kv, sig)

                if prefetch_stop_event.is_set():
                    reason = "cancelled"
                    return

                await websocket.send_json({
                    "type": "prefetch_token",
                    "gen_id": gen_id,
                    "token_id": payload["token_id"],
                    "token_str": payload["token_str"],
                    "tok_per_sec": payload["tok_per_sec"],
                })
                await asyncio.sleep(0.004)
        finally:
            try:
                await websocket.send_json({
                    "type": "prefetch_done",
                    "gen_id": gen_id,
                    "reason": reason,
                })
            except Exception:
                pass

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            action = msg.get("action")

            if action == "stop":
                await cancel_task(prefetch_task, prefetch_stop_event)
                await cancel_task(gen_task, stop_event)
            elif action == "cancel_prefetch":
                await cancel_task(prefetch_task, prefetch_stop_event)
            elif action == "prefetch":
                if gen_task and not gen_task.done():
                    continue
                await cancel_task(prefetch_task, prefetch_stop_event)
                prefetch_task = asyncio.create_task(run_prefetch(msg))
            elif action in ("generate", "step"):
                await cancel_task(prefetch_task, prefetch_stop_event)
                await cancel_task(gen_task, stop_event)
                if action == "step":
                    msg["max_tokens"] = 1
                gen_task = asyncio.create_task(run_stream(msg))
    except WebSocketDisconnect:
        stop_event.set()
        prefetch_stop_event.set()
    except Exception as exc:
        stop_event.set()
        prefetch_stop_event.set()
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


routes = [
    Route("/", serve_index),
    Route("/api/status", api_status),
    Route("/api/tokenize", api_tokenize, methods=["POST"]),
    Route("/api/inspect", api_inspect, methods=["POST"]),
    Route("/api/tree", api_tree, methods=["POST"]),
    Route("/api/axis/custom", api_custom_axis, methods=["POST"]),
    WebSocketRoute("/ws", websocket_endpoint),
]

app = Starlette(
    routes=routes,
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ],
)


def main():
    # Pre-warm the steerable GPT-2 model before accepting browser connections
    get_engine()
    print("\n========================================================")
    print("  ChronoSteer Studio live at: http://127.0.0.1:8765")
    print("========================================================\n")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")


if __name__ == "__main__":
    main()

