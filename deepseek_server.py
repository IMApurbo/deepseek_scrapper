"""
chat.deepseek.com → Anthropic API Proxy
========================================
Exposes a local HTTP server that speaks the Anthropic Messages API,
but routes every request through chat.deepseek.com via Playwright.

Usage:
    pip install playwright flask rich
    playwright install chromium
    python server.py

Then in another shell:
    export ANTHROPIC_BASE_URL="http://localhost:8765"
    export ANTHROPIC_API_KEY="local-proxy-key"
    claude   # or any tool that uses the Anthropic SDK

Supported endpoints:
    POST /v1/messages   (streaming + non-streaming)
    GET  /v1/models
    GET  /health
"""

import json
import threading
import time
import uuid
from datetime import datetime

from flask import Flask, request, Response, jsonify

# ── reuse the scraper (must be in same folder) ────────────────
from deepseek_scraper import DeepSeekScraper

# ─────────────────────────────────────────────────────────────
# Global scraper instance (one browser, one session)
# ─────────────────────────────────────────────────────────────

_scraper: DeepSeekScraper | None = None
_scraper_lock = threading.Lock()
_headless_flag = False          # set before first use via CLI arg


def get_scraper() -> DeepSeekScraper:
    global _scraper
    with _scraper_lock:
        if _scraper is None:
            _scraper = DeepSeekScraper(headless=_headless_flag)
            _scraper.start()
        return _scraper


# ─────────────────────────────────────────────────────────────
# Message Formatting Helpers
# ─────────────────────────────────────────────────────────────

def _messages_to_prompt(messages: list) -> str:
    """
    Flatten the Anthropic `messages` array into a single plain-text prompt.
    Handles:
      - Simple string content
      - Content block arrays (text / image / tool_result / tool_use)
      - System prompt injected at the top (passed separately)
    """
    parts = []
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "tool_result":
                        result_content = block.get("content", "")
                        if isinstance(result_content, list):
                            for rb in result_content:
                                if isinstance(rb, dict) and rb.get("type") == "text":
                                    text_parts.append(f"[Tool Result]\n{rb.get('text', '')}")
                        else:
                            text_parts.append(f"[Tool Result]\n{result_content}")
                    elif btype == "tool_use":
                        name = block.get("name", "tool")
                        inp  = json.dumps(block.get("input", {}), indent=2)
                        text_parts.append(f"[Tool Call: {name}]\n{inp}")
            text = "\n".join(text_parts)
        else:
            text = str(content)

        if role == "system":
            parts.append(f"[System]\n{text}")
        elif role == "user":
            parts.append(f"Human: {text}")
        elif role == "assistant":
            parts.append(f"Assistant: {text}")
        else:
            parts.append(text)

    parts.append("Assistant:")
    return "\n\n".join(parts)


def _extract_prompt(body: dict) -> str:
    """
    Build the prompt string to send to DeepSeek.

    Simple single-turn with no system prompt → send the last user
    message directly (cleanest experience).
    Everything else → flatten the full history into a structured prompt.
    """
    messages = body.get("messages", [])
    system   = body.get("system", "")

    user_msgs = [m for m in messages if m.get("role") == "user"]

    if len(user_msgs) == 1 and not system:
        # Single-turn: send bare user text
        content = user_msgs[0].get("content", "")
        if isinstance(content, list):
            return " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        return str(content)

    # Multi-turn / system prompt: inject everything
    all_messages = []
    if system:
        all_messages.append({"role": "system", "content": system})
    all_messages.extend(messages)
    return _messages_to_prompt(all_messages)


# ─────────────────────────────────────────────────────────────
# Response Builders
# ─────────────────────────────────────────────────────────────

def _build_response_body(content_text: str, model: str, usage_in: int = 0) -> dict:
    """Build a valid non-streaming Anthropic /v1/messages response."""
    return {
        "id":      f"msg_{uuid.uuid4().hex[:24]}",
        "type":    "message",
        "role":    "assistant",
        "content": [{"type": "text", "text": content_text}],
        "model":   model,
        "stop_reason":    "end_turn",
        "stop_sequence":  None,
        "usage": {
            "input_tokens":  max(usage_in, 1),
            "output_tokens": max(len(content_text.split()), 1),
        },
    }


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _stream_response(content_text: str, model: str):
    """
    Yield SSE events matching the Anthropic streaming wire format:
      message_start → content_block_start → content_block_delta(s)
      → content_block_stop → message_delta → message_stop → [DONE]
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    yield _sse({
        "type": "message_start",
        "message": {
            "id":             msg_id,
            "type":           "message",
            "role":           "assistant",
            "content":        [],
            "model":          model,
            "stop_reason":    None,
            "stop_sequence":  None,
            "usage":          {"input_tokens": 1, "output_tokens": 1},
        },
    })

    yield _sse({"type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""}})

    chunk_size = 40
    for i in range(0, len(content_text), chunk_size):
        yield _sse({
            "type":  "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": content_text[i: i + chunk_size]},
        })

    yield _sse({"type": "content_block_stop", "index": 0})

    yield _sse({
        "type":  "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": max(len(content_text.split()), 1)},
    })

    yield _sse({"type": "message_stop"})
    yield "data: [DONE]\n\n"


# ─────────────────────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.after_request
def _add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Authorization, x-api-key, anthropic-version"
    )
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def _cors_preflight():
    resp = Response("", status=204)
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Authorization, x-api-key, anthropic-version"
    )
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


# ── POST /v1/messages ─────────────────────────────────────────

@app.route("/v1/messages", methods=["POST", "OPTIONS"])
def messages():
    if request.method == "OPTIONS":
        return _cors_preflight()

    try:
        body = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": {"type": "invalid_request_error",
                                  "message": "Invalid JSON body"}}), 400

    model  = body.get("model", "deepseek-chat")
    stream = body.get("stream", False)

    if not body.get("messages"):
        return jsonify({"error": {"type": "invalid_request_error",
                                  "message": "messages required"}}), 400

    prompt = _extract_prompt(body)
    if not prompt.strip():
        return jsonify({"error": {"type": "invalid_request_error",
                                  "message": "Empty prompt"}}), 400

    # ── Send to DeepSeek ──────────────────────────────────────
    try:
        scraper = get_scraper()
        # send_message → (md_response, reasoning_blocks: list[str], elapsed: float)
        md, reasoning_blocks, elapsed = scraper.send_message(prompt)
    except Exception as e:
        return jsonify({"error": {"type": "api_error",
                                  "message": f"DeepSeek scraper error: {e}"}}), 500

    if md.startswith("[Error]"):
        return jsonify({"error": {"type": "api_error", "message": md}}), 500

    # Prepend reasoning blocks as a collapsible think section if present
    if reasoning_blocks:
        think_header = "\n\n---\n*Reasoning:*\n"
        think_body   = "\n\n---\n".join(reasoning_blocks)
        full_text    = f"{think_header}{think_body}\n\n---\n\n{md}"
    else:
        full_text = md

    # ── Return response ───────────────────────────────────────
    if stream:
        return Response(
            _stream_response(full_text, model),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    input_tokens = max(len(prompt.split()), 1)
    return jsonify(_build_response_body(full_text, model, usage_in=input_tokens))


# ── GET /v1/models ────────────────────────────────────────────

@app.route("/v1/models", methods=["GET", "OPTIONS"])
def list_models():
    if request.method == "OPTIONS":
        return _cors_preflight()
    return jsonify({
        "data": [
            {"id": "deepseek-chat",      "object": "model",
             "created": 1720000000, "owned_by": "deepseek"},
            {"id": "deepseek-reasoner",  "object": "model",
             "created": 1720000000, "owned_by": "deepseek"},
            # Aliases so tools that hard-code Claude model names still work
            {"id": "claude-opus-4-5",    "object": "model",
             "created": 1720000000, "owned_by": "anthropic"},
            {"id": "claude-sonnet-4-5",  "object": "model",
             "created": 1720000000, "owned_by": "anthropic"},
            {"id": "claude-haiku-3-5",   "object": "model",
             "created": 1720000000, "owned_by": "anthropic"},
        ]
    })


# ── GET /health ───────────────────────────────────────────────

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    scraper_status = "ready" if _scraper is not None else "not started"
    return jsonify({
        "status":  "ok",
        "proxy":   "chat.deepseek.com → Anthropic API",
        "browser": scraper_status,
        "time":    datetime.now().isoformat(),
    })


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="chat.deepseek.com Anthropic API Proxy")
    ap.add_argument("--host",      default="0.0.0.0",   help="Bind host (default: 0.0.0.0)")
    ap.add_argument("--port",      default=8765, type=int, help="Port (default: 8765)")
    ap.add_argument("--headless",  action="store_true",  help="Run browser headless")
    ap.add_argument("--no-warmup", action="store_true",
                    help="Don't pre-launch browser on startup (lazy init)")
    args = ap.parse_args()

    _headless_flag = args.headless

    print("\n" + "=" * 60)
    print("   chat.deepseek.com  →  Anthropic API Proxy")
    print("=" * 60)
    print(f"  Listening on : http://{args.host}:{args.port}")
    print(f"  Headless     : {args.headless}")
    print()
    print("  Set these in your shell, then run Claude Code:")
    print(f'    export ANTHROPIC_BASE_URL="http://localhost:{args.port}"')
    print(f'    export ANTHROPIC_API_KEY="local-proxy-key"')
    print()
    print("  Supported endpoints:")
    print("    POST /v1/messages   (streaming + non-streaming)")
    print("    GET  /v1/models")
    print("    GET  /health")
    print("=" * 60 + "\n")

    if not args.no_warmup:
        print("[*] Pre-launching browser (--no-warmup to skip) ...")
        get_scraper()
        print("[+] Browser ready. Proxy is live!\n")

    app.run(host=args.host, port=args.port, threaded=False, debug=False)
