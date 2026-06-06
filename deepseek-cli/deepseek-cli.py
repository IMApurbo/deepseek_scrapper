#!/usr/bin/env python3
"""
DeepSeek CLI Chat Client
Uses the same WASM-based SHA-3 PoW solver as the browser.

Requirements:
    pip install requests wasmtime numpy

Setup:
    Place sha3_wasm_bg.wasm in the same directory as this script.
    Download from:
    https://github.com/Fundiman/dskpp/raw/refs/heads/main/wasm/sha3_wasm_bg.7b9ca65ddd.wasm
"""

import argparse
import base64
import json
import os
import sys
import requests
import numpy as np
import wasmtime

BASE_URL = "https://chat.deepseek.com"
CLIENT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.5",
    "X-Client-Platform": "web",
    "X-Client-Version": "2.0.0",
    "X-Client-Locale": "en_US",
    "X-Client-Timezone-Offset": "-14400",
    "X-App-Version": "2.0.0",
    "Origin": "https://chat.deepseek.com",
    "Referer": "https://chat.deepseek.com/",
}

WASM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sha3_wasm_bg.wasm")


# ── WASM PoW Solver ───────────────────────────────────────────────────────────

class DeepSeekHash:
    """WASM-based SHA-3 PoW solver — identical to the browser implementation."""

    def __init__(self, wasm_path: str):
        engine = wasmtime.Engine()
        with open(wasm_path, "rb") as f:
            wasm_bytes = f.read()
        module = wasmtime.Module(engine, wasm_bytes)
        self.store = wasmtime.Store(engine)
        linker = wasmtime.Linker(engine)
        linker.define_wasi()
        self.instance = linker.instantiate(self.store, module)
        self.memory = self.instance.exports(self.store)["memory"]

    def _write(self, text: str) -> tuple[int, int]:
        encoded = text.encode("utf-8")
        length = len(encoded)
        ptr = self.instance.exports(self.store)["__wbindgen_export_0"](self.store, length, 1)
        mem = self.memory.data_ptr(self.store)
        for i, byte in enumerate(encoded):
            mem[ptr + i] = byte
        return ptr, length

    def solve(self, challenge: str, salt: str, difficulty: int, expire_at: int) -> int:
        prefix = f"{salt}_{expire_at}_"
        retptr = self.instance.exports(self.store)["__wbindgen_add_to_stack_pointer"](self.store, -16)
        try:
            c_ptr, c_len = self._write(challenge)
            p_ptr, p_len = self._write(prefix)
            self.instance.exports(self.store)["wasm_solve"](
                self.store, retptr,
                c_ptr, c_len,
                p_ptr, p_len,
                float(difficulty),
            )
            mem = self.memory.data_ptr(self.store)
            status = int.from_bytes(bytes(mem[retptr:retptr + 4]), "little", signed=True)
            if status == 0:
                raise RuntimeError("WASM solver returned no result")
            value = np.frombuffer(bytes(mem[retptr + 8:retptr + 16]), dtype=np.float64)[0]
            return int(value)
        finally:
            self.instance.exports(self.store)["__wbindgen_add_to_stack_pointer"](self.store, 16)


def build_pow_response(challenge_data: dict, answer: int) -> str:
    payload = {
        "algorithm":   challenge_data["algorithm"],
        "challenge":   challenge_data["challenge"],
        "salt":        challenge_data["salt"],
        "answer":      answer,
        "signature":   challenge_data["signature"],
        "target_path": challenge_data["target_path"],
    }
    return base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()


# ── API helpers ───────────────────────────────────────────────────────────────

def make_session(token: str, cookies: dict) -> requests.Session:
    s = requests.Session()
    s.headers.update(CLIENT_HEADERS)
    s.headers["Authorization"] = f"Bearer {token}"
    if cookies:
        s.cookies.update(cookies)
    return s


def parse_cookies(cookie_str: str) -> dict:
    cookies = {}
    for part in cookie_str.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            cookies[k.strip()] = v.strip()
    return cookies


def create_chat_session(s: requests.Session) -> dict:
    resp = s.post(f"{BASE_URL}/api/v0/chat_session/create", json={})
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"create_chat_session failed: {data}")
    return data["data"]["biz_data"]["chat_session"]


def get_pow_challenge(s: requests.Session) -> dict:
    resp = s.post(
        f"{BASE_URL}/api/v0/chat/create_pow_challenge",
        json={"target_path": "/api/v0/chat/completion"},
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"create_pow_challenge failed: {data}")
    return data["data"]["biz_data"]["challenge"]


def stream_completion(
    s: requests.Session,
    session_id: str,
    parent_message_id,
    prompt: str,
    pow_response: str,
    thinking: bool = False,
    search: bool = True,
) -> tuple[str, int]:
    headers = {
        "X-Ds-Pow-Response": pow_response,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    body = {
        "chat_session_id": session_id,
        "parent_message_id": parent_message_id,
        "model_type": "default" if parent_message_id is None else None,
        "prompt": prompt,
        "ref_file_ids": [],
        "thinking_enabled": thinking,
        "search_enabled": search,
        "action": None,
        "preempt": False,
    }

    response_text = ""
    new_parent_id = None

    with s.post(
        f"{BASE_URL}/api/v0/chat/completion",
        headers=headers,
        json=body,
        stream=True,
        timeout=60,
    ) as resp:
        resp.raise_for_status()

        ct = resp.headers.get("content-type", "")
        if "application/json" in ct:
            raise RuntimeError(f"API error: {resp.json()}")

        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            try:
                payload = json.loads(raw_line[5:].strip())
            except json.JSONDecodeError:
                continue

            v = payload.get("v")
            p = payload.get("p", "")
            o = payload.get("o", "")

            # 1. Initial snapshot with first fragment
            if isinstance(v, dict) and "response" in v:
                resp_obj = v["response"]
                new_parent_id = resp_obj.get("message_id")
                for f in resp_obj.get("fragments", []):
                    if f.get("type") == "RESPONSE":
                        chunk = f.get("content", "")
                        response_text += chunk
                        print(chunk, end="", flush=True)

            # 2. APPEND patch on the last fragment's content
            elif isinstance(v, str) and o == "APPEND" and p == "response/fragments/-1/content":
                response_text += v
                print(v, end="", flush=True)

            # 3. Plain token delta (no "p" key)
            elif isinstance(v, str) and "p" not in payload:
                response_text += v
                print(v, end="", flush=True)

    print()
    return response_text, new_parent_id


# ── Main conversation loop ─────────────────────────────────────────────────────

def chat_loop(token: str, cookies: dict, thinking: bool, search: bool):
    if not os.path.exists(WASM_PATH):
        print(
            f"Error: WASM file not found at {WASM_PATH}\n"
            "Download it with:\n"
            "  curl -L 'https://github.com/Fundiman/dskpp/raw/refs/heads/main/wasm/sha3_wasm_bg.7b9ca65ddd.wasm'"
            " -o sha3_wasm_bg.wasm",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Loading WASM solver...", end=" ", flush=True)
    hasher = DeepSeekHash(WASM_PATH)
    print("OK")

    s = make_session(token, cookies)

    print("Creating chat session...", end=" ", flush=True)
    chat = create_chat_session(s)
    session_id = chat["id"]
    print(f"OK (id={session_id})")

    parent_message_id = None
    print("\nDeepSeek CLI — type your message, or 'exit'/'quit' to quit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        print("  [PoW] Requesting challenge...", end=" ", flush=True)
        try:
            challenge_data = get_pow_challenge(s)
        except Exception as e:
            print(f"\nError getting PoW challenge: {e}")
            continue
        print("OK")

        print("  [PoW] Solving...", end=" ", flush=True)
        try:
            answer = hasher.solve(
                challenge_data["challenge"],
                challenge_data["salt"],
                challenge_data["difficulty"],
                challenge_data["expire_at"],
            )
        except Exception as e:
            print(f"\nError solving PoW: {e}")
            continue
        print(f"OK (answer={answer})")

        pow_response = build_pow_response(challenge_data, answer)

        print(f"\nDeepSeek: ", end="", flush=True)
        try:
            _, new_pid = stream_completion(
                s, session_id, parent_message_id, prompt,
                pow_response, thinking=thinking, search=search,
            )
            parent_message_id = new_pid
        except Exception as e:
            print(f"\nError during completion: {e}")
            continue

        print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek CLI — uses browser-identical WASM SHA-3 PoW solver."
    )
    parser.add_argument("--token", default=os.environ.get("DEEPSEEK_TOKEN", ""),
                        help="Bearer token (or set DEEPSEEK_TOKEN)")
    parser.add_argument("--cookies", default=os.environ.get("DEEPSEEK_COOKIES", ""),
                        help="Browser cookies string e.g. 'ds_session_id=abc; smidV2=xyz'")
    parser.add_argument("--thinking", action="store_true", default=False)
    parser.add_argument("--no-search", action="store_true", default=False)
    args = parser.parse_args()

    if not args.token:
        print(
            "Error: No bearer token.\n"
            "  --token TOKEN  or  export DEEPSEEK_TOKEN=...\n\n"
            "  Get it: DevTools → Network → any request → Authorization header\n",
            file=sys.stderr,
        )
        sys.exit(1)

    cookies = parse_cookies(args.cookies) if args.cookies else {}

    try:
        chat_loop(args.token, cookies, thinking=args.thinking, search=not args.no_search)
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
