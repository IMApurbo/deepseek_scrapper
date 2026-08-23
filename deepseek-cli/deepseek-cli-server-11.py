#!/usr/bin/env python3
"""
DeepSeek → Anthropic API Proxy
Exposes /v1/messages that Claude Code can talk to, forwarding to DeepSeek.

- Translates Anthropic tool definitions → XML schema in the prompt
- Parses DeepSeek plain-text responses for <tool_use> blocks
- Re-emits proper Anthropic tool_use / text content blocks
- Handles tool_result turns in conversation history
- Memory-efficient: root-pins after first reply (history stays at 2 msgs)

Requirements:
    pip install requests wasmtime numpy flask

Setup:
    Place sha3_wasm_bg.wasm next to this file.

    # Single token (original behaviour)
    export DEEPSEEK_TOKEN="your-bearer-token"

    # Multiple tokens — automatically rotates on "Server busy" / generation_timeout
    export DEEPSEEK_TOKEN="token1,token2,token3"

    python deepseek_server.py

Then:
    export ANTHROPIC_BASE_URL="http://localhost:8765"
    export ANTHROPIC_API_KEY="local-proxy-key"
    claude
"""

import argparse
import base64
import collections
import concurrent.futures
import ctypes
import hashlib
import json
import os
import re
import sys
import time
import uuid
import threading

import numpy as np
import requests
import wasmtime
from flask import Flask, Response, request, jsonify

# ── CLI flags ─────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSeek → Anthropic API Proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model / feature flags (all optional; defaults shown):
  --model fast      Use model_type="default"  (fast model)   [DEFAULT]
  --model expert    Use model_type=null        (expert model)
  --search          Enable web search (search_enabled=true)   [DEFAULT]
  --no-search       Disable web search
  --think           Enable thinking mode (thinking_enabled=true)
                    (thinking is OFF by default)
        """,
    )
    parser.add_argument(
        "--model",
        choices=["fast", "expert"],
        default="fast",
        help='Model tier: "fast" → model_type="default", "expert" → model_type=null (default: fast)',
    )
    parser.add_argument(
        "--search",
        dest="search",
        action="store_true",
        default=True,
        help="Enable web search (default: on)",
    )
    parser.add_argument(
        "--no-search",
        dest="search",
        action="store_false",
        help="Disable web search",
    )
    parser.add_argument(
        "--think",
        dest="think",
        action="store_true",
        default=False,
        help="Enable thinking mode (default: off)",
    )
    # Allow Flask's reloader to pass extra args without crashing
    args, _ = parser.parse_known_args()
    return args

_CLI_ARGS = _parse_args()

# ── Config ────────────────────────────────────────────────────────────────────

PORT       = int(os.environ.get("PROXY_PORT", 8765))
COOKIE_STR = os.environ.get("DEEPSEEK_COOKIES", "")

# Cap how many distinct DeepSeek chat sessions we keep alive at once (one per
# distinct conversation — see derive_session_key). Oldest is evicted on overflow.
MAX_CACHED_SESSIONS   = int(os.environ.get("PROXY_MAX_SESSIONS", 64))

# Cap how many messages of a single conversation get flattened into the prompt.
# Keeps the original task framing (first message) plus the most recent turns,
# so the tool-call instructions at the top of the prompt aren't drowned out by
# an ever-growing wall of history in long multi-step sessions.
MAX_HISTORY_MESSAGES  = int(os.environ.get("PROXY_MAX_HISTORY_MESSAGES", 40))

# ── Multi-token pool ──────────────────────────────────────────────────────────
# DEEPSEEK_TOKEN may contain multiple bearer tokens separated by commas.
# On "server busy" / generation_timeout the pool rotates to the next token
# so requests continue without interruption.

class TokenPool:
    """Round-robin pool of DeepSeek bearer tokens with busy-rotation support."""

    def __init__(self, token_env: str):
        raw = token_env or ""
        self._tokens = [t.strip() for t in raw.split(",") if t.strip()]
        if not self._tokens:
            self._tokens = [""]          # keep empty-string sentinel so startup check works
        self._index = 0
        self._lock  = threading.Lock()

    def current(self) -> str:
        with self._lock:
            return self._tokens[self._index]

    def rotate(self) -> str:
        """Advance to the next token and return it."""
        with self._lock:
            if len(self._tokens) <= 1:
                print("[token-pool] only one token available — cannot rotate", flush=True)
                return self._tokens[0]
            self._index = (self._index + 1) % len(self._tokens)
            tok = self._tokens[self._index]
            print(f"[token-pool] rotated to token index {self._index} "
                  f"({tok[:8]}…)", flush=True)
            return tok

    def __bool__(self):
        return bool(self._tokens[0])   # falsy when no token was supplied

    def __len__(self):
        return len(self._tokens)


_token_pool = TokenPool(os.environ.get("DEEPSEEK_TOKEN", ""))

# Convenience alias used by the rest of the file where a single TOKEN was used
def TOKEN():
    return _token_pool.current()

BASE_URL   = "https://chat.deepseek.com"
WASM_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sha3_wasm_bg.wasm")

CLIENT_HEADERS = {
    "User-Agent":               "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0",
    "Accept":                   "*/*",
    "Accept-Language":          "en-US,en;q=0.5",
    "X-Client-Platform":        "web",
    "X-Client-Version":         "2.0.0",
    "X-Client-Locale":          "en_US",
    "X-Client-Timezone-Offset": "-14400",
    "X-App-Version":            "2.0.0",
    "Origin":                   "https://chat.deepseek.com",
    "Referer":                  "https://chat.deepseek.com/",
}

# ── WASM PoW Solver ───────────────────────────────────────────────────────────

class DeepSeekHash:
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
        self._lock = threading.Lock()

    def _write(self, text: str):
        encoded = text.encode("utf-8")
        length = len(encoded)
        ptr = self.instance.exports(self.store)["__wbindgen_export_0"](self.store, length, 1)
        mem = self.memory.data_ptr(self.store)
        # Bulk copy via memmove — avoids slow Python byte-by-byte loop.
        # mem is ctypes._Pointer[c_ubyte]; get base address as integer then add ptr offset.
        base = ctypes.cast(mem, ctypes.c_void_p).value
        ctypes.memmove(base + ptr, encoded, length)
        return ptr, length

    def solve(self, challenge: str, salt: str, difficulty: int, expire_at: int) -> int:
        with self._lock:
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

# ── DeepSeek HTTP helpers ─────────────────────────────────────────────────────

def parse_cookies(cookie_str: str) -> dict:
    cookies = {}
    for part in cookie_str.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            cookies[k.strip()] = v.strip()
    return cookies


def make_http_session(token: str, cookies: dict) -> requests.Session:
    s = requests.Session()
    s.headers.update(CLIENT_HEADERS)
    s.headers["Authorization"] = f"Bearer {token}"
    if cookies:
        s.cookies.update(cookies)
    return s


def create_chat_session(http: requests.Session) -> str:
    resp = http.post(f"{BASE_URL}/api/v0/chat_session/create", json={})
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"create_chat_session failed: {data}")
    return data["data"]["biz_data"]["chat_session"]["id"]


def get_pow_challenge(http: requests.Session) -> dict:
    resp = http.post(
        f"{BASE_URL}/api/v0/chat/create_pow_challenge",
        json={"target_path": "/api/v0/chat/completion"},
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"create_pow_challenge failed: {data}")
    return data["data"]["biz_data"]["challenge"]

# ── Session store ─────────────────────────────────────────────────────────────

class SessionStore:
    """
    Keyed by a per-conversation identifier (see derive_session_key) rather
    than one shared key — otherwise concurrent/independent conversations
    (e.g. Claude Code subagents running in parallel) would collide onto the
    same DeepSeek chat session and corrupt each other's context.

    Bounded to MAX_CACHED_SESSIONS via LRU eviction so a long-running proxy
    handling many distinct conversations doesn't accumulate sessions forever.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: "collections.OrderedDict[str, dict]" = collections.OrderedDict()

    def _new_session_dict(self, token: str | None = None) -> dict:
        cookies = parse_cookies(COOKIE_STR) if COOKIE_STR else {}
        http = make_http_session(token or _token_pool.current(), cookies)
        ds_id = create_chat_session(http)
        print(f"[session] new ds_id={ds_id}")
        return {
            "ds_session_id":   ds_id,
            "http":            http,
            "anchor_message_id": None,  # parent_message_id we keep resending against (None = turn 1)
            "last_good_message_id": None,  # most recent successful response's message_id (advance target)
        }

    def get_or_create(self, key: str) -> dict:
        with self._lock:
            if key in self._sessions:
                self._sessions.move_to_end(key)
                return self._sessions[key]
            session = self._new_session_dict()
            self._sessions[key] = session
            if len(self._sessions) > MAX_CACHED_SESSIONS:
                evicted_key, _ = self._sessions.popitem(last=False)
                print(f"[session] evicted LRU session key={evicted_key}", flush=True)
                _evict_session_lock(evicted_key)
            return session

    def reset(self, key: str, token: str | None = None) -> dict:
        """Start a brand new chat session/conversation from scratch."""
        with self._lock:
            session = self._new_session_dict(token=token)
            self._sessions[key] = session
            self._sessions.move_to_end(key)
            return session


def derive_session_key(system: str, msgs: list) -> str:
    """
    Derive a stable per-conversation key from the system prompt + first user
    message, both of which stay constant for the life of one conversation
    thread. This keeps independent conversations (main agent vs. a subagent
    launched via the Task tool, or two separate Claude Code instances) from
    being funneled into the same DeepSeek chat session.
    """
    first_user_text = ""
    for msg in msgs:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                first_user_text = "\n".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                first_user_text = str(content)
            break
    basis = f"{system[:2000]}\n---\n{first_user_text[:2000]}"
    return hashlib.sha256(basis.encode("utf-8", "ignore")).hexdigest()[:16]

# ── Tool schema → prompt helpers ──────────────────────────────────────────────

def tools_to_xml(tools: list) -> str:
    """Render Anthropic tool definitions as XML so DeepSeek understands them."""
    if not tools:
        return ""
    lines = ["<tools>"]
    for t in tools:
        name = t.get("name", "")
        desc = t.get("description", "")
        schema = t.get("input_schema", {})
        props = schema.get("properties", {})
        required = schema.get("required", [])
        lines.append(f"  <tool>")
        lines.append(f"    <name>{name}</name>")
        lines.append(f"    <description>{desc}</description>")
        if props:
            lines.append(f"    <parameters>")
            for pname, pdef in props.items():
                req = " required=\"true\"" if pname in required else ""
                ptype = pdef.get("type", "string")
                pdesc = pdef.get("description", "")
                lines.append(f"      <parameter name=\"{pname}\" type=\"{ptype}\"{req}>{pdesc}</parameter>")
            lines.append(f"    </parameters>")
        lines.append(f"  </tool>")
    lines.append("</tools>")
    return "\n".join(lines)


TOOL_CALL_SYSTEM = """You are Claude, an AI assistant that can use tools.

═══════════════════════════════════════════════════════════════════════════════
CRITICAL TOOL-CALLING RULES (READ CAREFULLY):
═══════════════════════════════════════════════════════════════════════════════

1. IMMEDIATE EXECUTION - NO ANNOUNCEMENTS:
   ❌ WRONG: "Now let's check the files" (then nothing)
   ❌ WRONG: "I'll run ls -la for you" (then nothing)
   ❌ WRONG: "I'll explore everything in this directory in detail." (then nothing)
   ❌ WRONG: "Let me check this." (then nothing)
   These are ALL the same mistake: a sentence describing an action, with no
   <tool_use> block anywhere in the reply. If your reply doesn't need a tool,
   answer normally. If it DOES need a tool, the FIRST thing you output must
   be the <tool_use> block itself — not a sentence promising it.
   ✅ CORRECT: Immediately output the tool_use block with NO preamble

2. EXACT FORMAT REQUIRED:
   The tool call MUST be valid JSON wrapped in <tool_use> tags:

   <tool_use>
   {"name": "TOOL_NAME", "id": "call_UNIQUE_ID", "input": {PARAMETERS}}
   </tool_use>

   ❌ NO markdown backticks: ```json ... ```
   ❌ NO extra text before or after
   ❌ NO malformed JSON (trailing commas, missing braces)

3. ONLY USE LISTED TOOLS:
   • You can ONLY call tools that are defined in the <tools> section
   • If you try to call a non-existent tool, you will get an error
   • Check the tool name spelling EXACTLY

3b. THE "name" FIELD MUST BE A REAL TOOL NAME — NEVER A TAG:
   The "name" value is one of the exact tool names listed in <tools>
   (e.g. "Bash", "run_shell") — it is NEVER the literal text "<tool_call>",
   "<tool_use>", or any other wrapper/tag string. The wrapper tags
   (<tool_use> ... </tool_use>) go AROUND the JSON object — they are not a
   value that goes INSIDE it, and they are never repeated as if they were
   the tool's name.
   ❌ WRONG: <tool_use>{"name": "<tool_call>", "input": {...}}</tool_use>
   ❌ WRONG: <tool_use>{"name": "<tool_use>", "input": {...}}</tool_use>
   ❌ WRONG: <tool_use><tool_use>{"name": "Bash", "input": {...}}</tool_use></tool_use>
   ✅ CORRECT: <tool_use>{"name": "Bash", "id": "call_abc123", "input": {"command": "ls -la"}}</tool_use>
   There is exactly ONE opening tag, ONE JSON object with a real tool name,
   and ONE closing tag — nothing duplicated, nothing nested.

4. WAIT FOR RESULTS:
   • After calling a tool, STOP and wait for the result
   • DO NOT make up or fabricate results
   • DO NOT continue with "The result shows..." before receiving actual output

5. NEVER CLAIM UNVERIFIED WORK:
   • Never say you already checked, verified, ran, tested, or fixed something
     unless a <tool_use> call and its real <tool_result> actually happened
     earlier IN THIS CONVERSATION.
   ❌ WRONG: "I checked the file, it looks correct." (no prior tool_result)
   ❌ WRONG: "Done — all tests pass." (no tool ever ran the tests)
   ✅ CORRECT: If it needs checking, call the tool first and wait for the
      result before making any claim about what it shows.

6. PROPER WORKFLOW:
   Step 1: User asks question
   Step 2: You output <tool_use> block (if needed)
   Step 3: System returns <tool_result>
   Step 4: You read result and respond naturally

7. NEVER USE ```bash / ```sh / ```shell / ```zsh FENCES — FOR ANYTHING:
   The ONLY way to actually run a command is a <tool_use> block. A fenced
   code block is NEVER executed — it is just displayed to the user. Do not
   use a bash/sh/shell/zsh fence even for an example, a suggestion, or a
   command you expect the human to copy and run manually themselves — those
   fences read as commands and must not be used at all.
   ❌ WRONG (illustrative example in a bash fence):
      ```bash
      hydra -l admin -P wordlist.txt ssh://target
      ```
   ✅ CORRECT (same example, shown for the human to read only — still
      renders as highlighted inline code, just not a triple-fenced block):
      Run this yourself if you want to try it: `hydra -l admin -P wordlist.txt ssh://target`
   Always use inline single backticks around example/manual commands,
   including multi-line ones (one pair of backticks per line). Never use a
   triple-fence of any kind for a command. If you actually want the command
   run now, use a real <tool_use> block instead — never a fence.

EXAMPLE - Correct tool call for listing files:
<tool_use>
{"name": "Bash", "id": "call_abc123", "input": {"command": "ls -la"}}
</tool_use>

That's it. Nothing before, nothing after. Just the block.
═══════════════════════════════════════════════════════════════════════════════
"""


def _bound_messages(messages: list) -> list:
    """
    Cap how many turns of a conversation get flattened into the prompt.
    Keeps the first message (original task framing) plus the most recent
    MAX_HISTORY_MESSAGES-1 turns, dropping the middle. Without this, a long
    multi-step session grows the prompt without bound and the tool-call
    instructions at the top get diluted by the time the model reaches the
    end of it.
    """
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    head = messages[:1]
    tail_count = max(0, MAX_HISTORY_MESSAGES - 1)
    # messages[-0:] would (mis)slice as "all messages" rather than "none" —
    # guard the zero case explicitly rather than relying on negative-index slicing.
    tail = messages[-tail_count:] if tail_count else []
    omitted = len(messages) - len(head) - len(tail)
    marker = {
        "role": "user",
        "content": f"[...{omitted} earlier messages omitted for length...]",
    }
    return head + [marker] + tail


TOOL_CALL_REMINDER = (
    "<system-reminder>\n"
    "If your next step requires a tool, output the <tool_use> block immediately.\n"
    "No \"let me...\" / \"I'll...\" announcement first, nothing after it — just the block.\n"
    "Writing a sentence like \"I'll explore this in detail\" or \"Let me check this\"\n"
    "and then stopping is a failure, even if the sentence sounds reasonable.\n"
    "NEVER output a ```bash, ```sh, ```shell, or ```zsh fenced code block — not even\n"
    "for an example or a command meant for the human to run manually. Those fences\n"
    "are banned with no exceptions, no matter the language tag on the fence. For a\n"
    "command you want run now, use a real <tool_use> block. For a command you are\n"
    "only showing/suggesting, use inline single backticks instead — never a\n"
    "triple-fenced block of any kind.\n"
    "The JSON \"name\" field must be an exact tool name from <tools> (e.g. \"Bash\") —\n"
    "never the literal text \"<tool_call>\" or \"<tool_use>\". Those are wrapper tags\n"
    "that go around the JSON, not a value inside it.\n"
    "</system-reminder>"
)


def build_prompt(system: str, messages: list, tools: list) -> str:
    """
    Build a single prompt string from the full conversation history,
    including system prompt, tool definitions, and all turns.
    """
    messages = _bound_messages(messages)
    parts = []

    # System
    combined_system = TOOL_CALL_SYSTEM
    if system:
        combined_system += "\n\n" + system
    parts.append(f"<system>\n{combined_system}\n</system>")

    # Tool schema
    if tools:
        parts.append(tools_to_xml(tools))

    # Conversation turns
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            # content can be a list of blocks (text + tool_result)
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    btype = block.get("type", "")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "tool_result":
                        tool_use_id = block.get("tool_use_id", "")
                        result_content = block.get("content", "")
                        if isinstance(result_content, list):
                            result_content = "\n".join(
                                b.get("text", "") for b in result_content
                                if isinstance(b, dict) and b.get("type") == "text"
                            )
                        text_parts.append(
                            f"<tool_result tool_use_id=\"{tool_use_id}\">\n{result_content}\n</tool_result>"
                        )
                content = "\n".join(text_parts)
            parts.append(f"Human: {content}")

        elif role == "assistant":
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    btype = block.get("type", "")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "tool_use":
                        tool_json = json.dumps({
                            "name":  block.get("name", ""),
                            "id":    block.get("id", ""),
                            "input": block.get("input", {}),
                        }, indent=2)
                        text_parts.append(f"<tool_use>\n{tool_json}\n</tool_use>")
                content = "\n".join(text_parts)
            parts.append(f"Assistant: {content}")

    if tools:
        parts.append(TOOL_CALL_REMINDER)
    parts.append("Assistant:")
    return "\n\n".join(parts)

# ── Response parser: plain text → Anthropic content blocks ───────────────────

# Known DeepSeek native tag names -> Claude Code Anthropic tool names.
# DeepSeek may emit its own XML tags (e.g. <Bash>, <Editor>) which we map
# to the exact tool names Claude Code expects.
DEEPSEEK_TAG_TO_TOOL = {
    # Shell / file ops
    "bash":                        "Bash",
    "read":                        "Read",
    "write":                       "Write",
    "edit":                        "Edit",
    "multiedit":                   "MultiEdit",
    "read_file":                   "Read",
    "write_file":                  "Write",
    # DeepSeek native equivalents
    "python":                      "Bash",
    "editor":                      "Edit",
    "str_replace_based_edit_tool": "Edit",
    # Search
    "grep":                        "Grep",
    "glob":                        "Glob",
    # Background bash management
    "bashoutput":                  "BashOutput",
    "killbash":                    "KillBash",
    # Todo list / slash commands
    "todowrite":                   "TodoWrite",
    "slashcommand":                "SlashCommand",
    # Web
    "webfetch":                    "WebFetch",
    "websearch":                   "WebSearch",
    # Agent / sub-tasks
    "agent":                       "Agent",
    # Task management
    "taskcreate":                  "TaskCreate",
    "taskupdate":                  "TaskUpdate",
    "tasklist":                    "TaskList",
    "taskget":                     "TaskGet",
    # Scheduling / monitoring
    "croncreate":                  "CronCreate",
    "cronlist":                    "CronList",
    "crondelete":                  "CronDelete",
    "monitor":                     "Monitor",
    "schedulewakeup":              "ScheduleWakeup",
    # Notifications
    "pushnotification":            "PushNotification",
    # Notebook
    "notebookedit":                "NotebookEdit",
    "notebookread":                "NotebookRead",
    # Misc Claude Code tools
    "skill":                       "Skill",
    "workflow":                    "Workflow",
    "enterplanmode":               "EnterPlanMode",
    "exitplanmode":                "ExitPlanMode",
    "enterworktree":               "EnterWorktree",
    "exitworktree":                "ExitWorktree",
    "askuserquestion":             "AskUserQuestion",
    "computer":                    "computer",
}

# All known Claude Code tool names (exact casing) used to recognise native
# XML tool calls where DeepSeek already uses the correct name.
CLAUDE_CODE_TOOL_NAMES = set(DEEPSEEK_TAG_TO_TOOL.values()) | {
    "Bash", "Read", "Write", "Edit", "MultiEdit",
    "WebFetch", "WebSearch",
    "Agent",
    "TaskCreate", "TaskUpdate", "TaskList", "TaskGet",
    "CronCreate", "CronList", "CronDelete",
    "Monitor", "ScheduleWakeup",
    "PushNotification",
    "NotebookEdit", "NotebookRead",
    "Skill", "Workflow",
    "EnterPlanMode", "ExitPlanMode",
    "EnterWorktree", "ExitWorktree",
    "AskUserQuestion",
    "computer",
    "str_replace_based_edit_tool",
}

# XML tags that are structural/metadata or HTML — never treat as tool calls.
# Stored as lowercase for case-insensitive lookup.
_SKIP_TAGS = {
    "system", "tools", "tool", "tool_result", "tool_use",
    "parameter", "parameters", "name", "description",
    "thinking", "block", "reason", "decision", "antml",
    # HTML / markdown fragments DeepSeek may emit
    "p", "br", "div", "span", "code", "pre", "ul", "ol", "li",
    "h1", "h2", "h3", "h4", "strong", "em", "b", "i",
    "a", "img", "table", "tr", "td", "th", "thead", "tbody",
    "form", "input", "button", "select", "option", "textarea",
    "html", "head", "body", "meta", "link", "style", "title",
    # Security/web tags that appear in XSS payloads and pentest output
    "script", "svg", "iframe", "object", "embed", "frame",
    "frameset", "applet", "base", "noscript", "template",
    "math", "marquee", "details", "summary", "video", "audio",
    # DeepSeek reasoning wrappers
    # NOTE: "content" intentionally excluded — DeepSeek uses <content> inside
    # tool call blocks as a parameter wrapper; dropping it breaks tool parsing.
    "response", "result", "output", "answer",
}

# Native-tag tool calls for these have real, hard-to-undo side effects
# (write/overwrite a file, run a shell command). If DeepSeek ever produces a
# well-formed one of these while narrating/explaining rather than actually
# intending to act — e.g. demonstrating its own tool syntax mid-explanation —
# there's no reliable way to tell that apart from a genuine call by shape
# alone. See _native_call_has_narrative_preamble below: a high-risk native
# tag preceded by non-trivial prose in the same reply is treated as
# narration, not a real call, and suppressed instead of executed.
_HIGH_RISK_NATIVE_TOOLS = {"Write", "Edit", "MultiEdit", "Bash"}
_LEADING_PROSE_WARN_CHARS = 15

def _strip_json_fences(raw: str) -> str:
    """Remove markdown code fences DeepSeek sometimes wraps JSON in."""
    raw = raw.strip()
    # ```json ... ``` or ``` ... ```
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw


# Valid single-character JSON escape letters (after the leading backslash).
# JSON spec §7: only these are legal after a backslash inside a string.
_VALID_JSON_ESC_CHARS = frozenset({'"', '\\', '/', 'b', 'f', 'n', 'r', 't'})


def _fix_invalid_json_escapes(s: str) -> str:
    """
    Replace invalid JSON escape sequences in a raw JSON string with
    valid equivalents so json.loads can parse it.

    JSON only allows: \\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX
    DeepSeek embeds Python code in JSON strings that may contain Python-style
    escapes such as \\x00, \\a, \\v, \\0, or \\' — all illegal in JSON.

    Conversion rules (applied at the raw-text level, not per parsed value):
      \\xNN  →  \\u00NN   hex escape  → JSON unicode escape (value preserved)
      \\uXXXX             already valid, left untouched
      any other \\<c>    →  \\\\<c>  escape the backslash, emit <c> literally

    This function is a no-op when the input contains no invalid escapes, so
    it is safe to call eagerly on all inputs.
    """
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch != '\\' or i + 1 >= n:
            out.append(ch)
            i += 1
            continue
        nxt = s[i + 1]
        if nxt in _VALID_JSON_ESC_CHARS:
            # Valid single-char escape — keep as-is
            out.append(ch)
            out.append(nxt)
            i += 2
        elif nxt == 'u' and i + 5 <= n and all(
                c in '0123456789abcdefABCDEF' for c in s[i + 2: i + 6]):
            # Valid \uXXXX — keep as-is
            out.append(s[i: i + 6])
            i += 6
        elif nxt == 'x' and i + 3 < n and all(
                c in '0123456789abcdefABCDEF' for c in s[i + 2: i + 4]):
            # \xNN → \u00NN  (Python hex-escape → JSON unicode escape)
            out.append(f'\\u00{s[i + 2: i + 4]}')
            i += 4
        else:
            # Invalid escape (\a \v \0 \' etc.) — escape the backslash.
            # The offending character is emitted normally on the next iteration.
            out.append('\\\\')
            i += 1
    return ''.join(out)


def _escape_raw_control_chars(s: str) -> str:
    """
    Escape literal newline / carriage-return / tab characters that appear
    INSIDE a JSON string literal, so json.loads doesn't choke on them.

    JSON strings may only contain a newline as the two-character escape
    \\n — a literal 0x0A byte inside a quoted string is illegal per the
    spec, and Python's json.loads (strict mode) raises "Invalid control
    character" for it. DeepSeek sometimes wraps a long "command" value
    (e.g. a multi-line `python3 -c "..."` string) across lines in its
    plain-text output, leaving a raw newline sitting inside the JSON
    string. _fix_invalid_json_escapes doesn't catch this because there's
    no leading backslash to key off — this is a bare control byte, not an
    invalid escape sequence.

    String/escape-aware by construction (mirrors _close_unbalanced_json):
    walks the text tracking whether we're inside a string and whether the
    previous char was an unconsumed backslash, and only rewrites control
    chars found while in_string is True. Left untouched outside strings
    (e.g. the newline between "role" and the next key) since those don't
    break parsing and reformatting them would be needlessly invasive.

    No-op when the input has no raw control chars inside strings, so it's
    safe to call eagerly on all inputs.
    """
    out: list[str] = []
    in_string = False
    escape = False
    for ch in s:
        if in_string and not escape and ch in ('\n', '\r', '\t'):
            out.append({'\n': '\\n', '\r': '\\r', '\t': '\\t'}[ch])
            continue
        out.append(ch)
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
    return ''.join(out)


def _close_unbalanced_json(s: str) -> str | None:
    """
    String/escape-aware brace-and-bracket closer. Walks the text tracking
    whether we're inside a JSON string (honoring backslash escapes) and
    keeps a stack of {/[ seen OUTSIDE of strings. If anything is left open
    at the end, returns `s` with the missing closers appended (innermost
    first). Returns None if nothing is open (nothing to fix) or if the
    imbalance is implausibly large (>10 — at that point this probably isn't
    truncated JSON, and guessing a matching close for it is more likely to
    produce a garbage tool call than a correct one).

    Unlike Attempt 3's plain fixed.count("{") - fixed.count("}"), this
    doesn't get confused by braces that happen to appear inside string
    values (e.g. a "content" field containing actual code), since those are
    skipped while in_string is True.
    """
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in s:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
    if not stack or len(stack) > 10:
        return None
    return s + "".join(reversed(stack))


def _parse_json_tool(raw: str) -> dict | None:
    """
    Try to parse a JSON tool-use block. Handles:
      - plain JSON
      - markdown-fenced JSON (```json ... ```)
      - truncated/partial JSON (best-effort repair)
    Returns a tool_use block dict or None on failure.
    """
    cleaned = _strip_json_fences(raw)
    _original_cleaned = cleaned  # kept for the failure log at the bottom

    # Pre-pass: fix invalid JSON escape sequences before any parse attempt.
    # DeepSeek embeds Python code in JSON strings, and Python uses \x00, \a,
    # \v, \' etc. which are not legal JSON escapes — json.loads raises
    # "Invalid \escape" and every repair attempt below also fails.
    # _fix_invalid_json_escapes is a no-op on already-valid JSON so applying
    # it eagerly here is safe; all subsequent attempts use the cleaned form.
    _escape_fixed = _fix_invalid_json_escapes(cleaned)
    if _escape_fixed != cleaned:
        print("[tool_parse] pre-fixed invalid JSON escape sequences", flush=True)
        cleaned = _escape_fixed

    # Pre-pass: escape literal newline/CR/tab bytes found INSIDE JSON string
    # literals (e.g. a "command" value DeepSeek wrapped across two lines).
    # A raw control character in a string is illegal JSON and makes every
    # attempt below fail identically at the same spot, so this must run
    # before Attempt 1 — same reasoning as the escape-sequence pre-fix above.
    _control_fixed = _escape_raw_control_chars(cleaned)
    if _control_fixed != cleaned:
        print("[tool_parse] pre-fixed raw control characters inside JSON string", flush=True)
        cleaned = _control_fixed

    # Attempt 1: straight parse
    try:
        obj = json.loads(cleaned)
        return obj
    except json.JSONDecodeError:
        pass

    # Attempt 1b: unwrap a duplicated <tool_use> open tag. DeepSeek sometimes
    # emits a SECOND literal "<tool_use>" as the value of "name" instead of
    # just writing the inner call directly, e.g.:
    #   {"name": "<tool_use>
    #   {"name": "Bash", "input": {...}}
    # Only one closing </tool_use> ever follows, so parse_response's
    # non-greedy match captures both layers as one blob — the outer "name"
    # string is unterminated, so a straight parse fails outright and none of
    # the repairs below (trailing commas / unclosed braces / missing outer
    # braces) apply, since this isn't truncated or malformed JSON, it's a
    # duplicated wrapper around otherwise-valid JSON. Strip the corrupt
    # prefix and parse what's actually the real inner call.
    m_dup = re.match(r'^\{\s*"name"\s*:\s*"<tool_use>\s*', cleaned)
    if m_dup:
        obj = _parse_json_tool(cleaned[m_dup.end():])
        if obj is not None:
            print(f"[tool_parse] unwrapped duplicated <tool_use> open tag: {repr(cleaned[:80])}", flush=True)
            return obj

    # Attempt 1c: same duplication, but written as a literal second opening
    # tag rather than embedded inside the JSON string value, e.g.:
    #   <tool_use><tool_use>{"name": "Bash", "input": {...}}</tool_use></tool_use>
    # The outer scan's non-greedy match stops at the FIRST </tool_use>, so
    # what reaches here is a dangling leading "<tool_use>" in front of
    # otherwise-valid JSON — strip it and parse what's left.
    m_lit = re.match(r'^<tool_use>\s*', cleaned)
    if m_lit:
        obj = _parse_json_tool(cleaned[m_lit.end():])
        if obj is not None:
            print(f"[tool_parse] stripped literal duplicated <tool_use> open tag: {repr(cleaned[:80])}", flush=True)
            return obj

    # Attempt 2: Remove trailing commas
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
        obj = json.loads(fixed)
        print(f"[tool_parse] fixed trailing commas: {repr(cleaned[:80])}", flush=True)
        return obj
    except json.JSONDecodeError:
        pass
    
    # Attempt 3: Fix unclosed braces/brackets (careful with strings containing braces)
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
        last_char = fixed.rstrip()[-1:] if fixed.rstrip() else ""
        
        # Count brace imbalances
        open_braces = fixed.count("{") - fixed.count("}")
        open_brackets = fixed.count("[") - fixed.count("]")
        
        # Only repair if:
        # 1. There's a small imbalance (≤3)
        # 2. Last char isn't already a closer (suggests truncation)
        # 3. The string doesn't look like it contains code with braces
        if (0 < open_braces <= 3 or 0 < open_brackets <= 3) and last_char not in "}]":
            # Check if it looks like truncated JSON (not code with braces)
            if '"' in fixed or "'" in fixed:  # Has string delimiters
                fixed += "]" * max(0, open_brackets) + "}" * max(0, open_braces)
                obj = json.loads(fixed)
                print(f"[tool_parse] repaired unclosed braces/brackets: {repr(cleaned[:80])}", flush=True)
                return obj
    except (json.JSONDecodeError, IndexError):
        pass

    # Attempt 3b: string/escape-aware close for the shape Attempt 3's
    # last-char heuristic rejects. Attempt 3 refuses to repair whenever the
    # payload already ends in "}" or "]" (its signal for "not truncated").
    # But that's exactly what a real truncated tool call looks like when
    # generation is cut off right after an inner value closes but before
    # the outer object does, e.g.:
    #     {"name": "Bash", "input": {"command": "ls -la"}
    # (the inner "input" object is balanced; only the outer object is
    # still open — the string ends in "}" and Attempt 3 gives up.) This is
    # the main real-world case _fix_dangling_unclosed_tool_use above exists
    # to catch, so without this attempt that repair only fixes the tag
    # wrapper and the inner JSON still fails to parse.
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
        closed = _close_unbalanced_json(fixed)
        if closed is not None:
            obj = json.loads(closed)
            print(f"[tool_parse] repaired unclosed braces/brackets (string-aware): {repr(cleaned[:80])}", flush=True)
            return obj
    except json.JSONDecodeError:
        pass

    # Attempt 4: Handle case where JSON is missing outer braces but has "name" and "input"
    try:
        # Pattern: "name": "ToolName", "input": {...}
        if '"name"' in cleaned and '"input"' in cleaned and not cleaned.strip().startswith("{"):
            fixed = "{" + cleaned + "}"
            fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
            obj = json.loads(fixed)
            print(f"[tool_parse] added missing outer braces: {repr(cleaned[:80])}", flush=True)
            return obj
    except json.JSONDecodeError:
        pass
    
    print(f"[tool_parse] failed to parse JSON: {repr(_original_cleaned[:120])}", flush=True)
    return None


def _default_param_key(tool_name: str) -> str:
    """Return the single-param key name for a genuinely single-param tool."""
    mapping = {
        "Bash":         "command",
        "Read":         "file_path",
        "NotebookRead": "file_path",
        "WebFetch":     "url",
        "WebSearch":    "query",
    }
    return mapping[tool_name]


# Tools where an unstructured tag body can safely become one parameter.
# Multi-param tools (Write, Edit, MultiEdit, ...) can't be guessed this way —
# stuffing raw content into a single field (e.g. a whole file body into
# "file_path") produces a tool call that's wrong in a confusing, hard-to-
# diagnose way rather than cleanly missing. For those, an unstructured tag
# is treated as unparseable instead of guessed.
_SINGLE_PARAM_TOOLS = {"Bash", "Read", "NotebookRead", "WebFetch", "WebSearch"}


def parse_native_tag(tag_name: str, inner: str, valid_tools: set[str] | None = None) -> dict | None:
    """
    Convert a DeepSeek native XML tool call like:
      <Bash><command>ls -la</command></Bash>
    into an Anthropic tool_use block, or None if the tag should be skipped.

    Resolution order:
      1. Exact match in CLAUDE_CODE_TOOL_NAMES (preserves casing e.g. "WebFetch")
      2. Lowercase lookup in DEEPSEEK_TAG_TO_TOOL
      3. If valid_tools provided and resolved name not in it → skip (return None)
      4. Fall back to tag name verbatim

    Sub-tag extraction handles:
      - Simple:  <command>ls -la</command>
      - Nested:  <content><![CDATA[...]]></content>  (CDATA unwrapped)
      - Mixed content with attributes ignored gracefully
    """
    if tag_name in CLAUDE_CODE_TOOL_NAMES:
        tool_name = tag_name
    else:
        tool_name = DEEPSEEK_TAG_TO_TOOL.get(tag_name.lower())
        if tool_name is None:
            # Unknown tag — only accept if it's in the valid tool set
            if valid_tools and tag_name not in valid_tools:
                return None
            tool_name = tag_name

    # Reject if not in the caller's tool list (prevents hallucinated tool names)
    if valid_tools and tool_name not in valid_tools:
        print(f"[tool_parse] dropping unknown tool '{tool_name}' (not in valid_tools)", flush=True)
        return None

    # ── Extract sub-tag parameters ───────────────────────────────────────────
    params: dict[str, str] = {}

    # Unwrap CDATA sections first
    inner_clean = re.sub(r"<!\[CDATA\[(.*?)]]>", r"\1", inner, flags=re.DOTALL)

    # Match immediate child tags. Uses non-greedy for the value but this is
    # sufficient for tool params — pathological cases (file content containing
    # </paramname>) are handled by the CDATA unwrap above.
    # Attributes on child tags (e.g. <file_path encoding="utf-8">) are ignored
    # via (?:\s[^>]*)? — the attribute content is stripped.
    for m in re.finditer(r"<(\w+)(?:\s[^>]*)?>(.*?)</\1>", inner_clean, re.DOTALL):
        key = m.group(1)
        val = m.group(2).strip()
        # Unwrap nested CDATA inside param values
        val = re.sub(r"<!\[CDATA\[(.*?)]]>", r"\1", val, flags=re.DOTALL)
        params[key] = val

    # If inner looks like raw JSON (DeepSeek sometimes puts JSON inside the tag)
    if not params and inner_clean.strip().startswith("{"):
        try:
            obj = json.loads(inner_clean.strip())
            if isinstance(obj, dict):
                params = {k: (json.dumps(v) if not isinstance(v, str) else v)
                          for k, v in obj.items()}
        except json.JSONDecodeError:
            pass

    # Fallback: treat whole inner content as the primary parameter — but only
    # for tools where that's actually safe (a real single required param).
    # For multi-param tools, guessing which one field to stuff unstructured
    # content into produces a tool call that's wrong in a confusing way
    # (e.g. a whole file body ending up as "file_path"), so treat it as
    # unparseable instead — the raw tag falls through as plain text rather
    # than triggering a tool call with garbage params.
    if not params:
        if tool_name in _SINGLE_PARAM_TOOLS:
            params[_default_param_key(tool_name)] = inner_clean.strip()
        else:
            print(
                f"[tool_parse] '{tool_name}' has no parsable sub-tags and isn't a "
                f"single-param tool — treating tag as unparseable",
                flush=True,
            )
            return None

    return {
        "type":  "tool_use",
        "id":    f"toolu_{uuid.uuid4().hex[:16]}",
        "name":  tool_name,
        "input": params,
    }


def _native_call_has_narrative_preamble(tool_name: str, tag_name: str, full_text: str, match_start: int) -> bool:
    """
    A native-tag call (<Bash>, <Write>, etc.) for a tool with real side
    effects, preceded by non-trivial prose in the same reply, is
    indistinguishable here from DeepSeek demonstrating its own tool syntax
    mid-explanation rather than actually intending the action (see
    _HIGH_RISK_NATIVE_TOOLS above). Real calls are never supposed to have
    preamble at all — rule 1 in TOOL_CALL_SYSTEM already requires the tool
    call to be the first thing in the reply, with no announcement first — so
    treating "has meaningful leading prose" as "not a real call" doesn't ban
    anything that was ever supposed to happen; it enforces a policy that was
    already stated. Suppressing it (rather than executing) is safe because
    the caller falls back to plain text, and the existing unexecuted-intent
    retry loop will push DeepSeek to reissue a real, preamble-free call if it
    actually meant to act.
    """
    if tool_name not in _HIGH_RISK_NATIVE_TOOLS:
        return False
    leading = full_text[:match_start].strip()
    if len(leading) > _LEADING_PROSE_WARN_CHARS:
        print(
            f"[tool_parse] suppressing native <{tag_name}> call to high-risk "
            f"tool '{tool_name}' — {len(leading)} chars of narrative text "
            f"precede it in the same reply, treating it as text instead of "
            f"executing it. This shape is consistent with DeepSeek "
            f"demonstrating tool syntax rather than intending the action. "
            f"Leading text (last 200 chars): {leading[-200:]!r}",
            flush=True,
        )
        return True
    return False


def strip_premature_exit_preambles(text: str) -> str:
    """
    Remove "I will now...", "Let me check...", etc. phrases that appear
    BEFORE a tool call. These are announcements without execution.
    
    Only strips if they appear at the start of the text and are followed
    by a tool call (not standalone).
    """
    # Patterns that indicate the model is announcing intent without executing
    preamble_patterns = [
        r"^(?:Now\s+)?(?:let me|let's|I'll|I will)\s+(?:check|run|execute|look at|examine|read|write|edit|search|fetch)\s+[^\n]*?\n+(?=<tool_use>|<\w+>)",
        r"^(?:I'm|I am)\s+(?:going to|about to)\s+(?:check|run|execute|look at|examine|read|write|edit|search|fetch)\s+[^\n]*?\n+(?=<tool_use>|<\w+>)",
        r"^(?:First|Next),?\s+(?:let me|I'll|I will)\s+(?:check|run|execute|look at|examine|read|write|edit|search|fetch)\s+[^\n]*?\n+(?=<tool_use>|<\w+>)",
    ]
    
    for pattern in preamble_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    return text








def strip_fabricated_continuation(text: str) -> str:
    """
    DeepSeek sometimes fabricates the tool result and keeps going after the
    tool call block. Cut at the EARLIEST fabrication marker found.
    Applied ONLY after a real tool call has been found.
    """
    markers = [
        "\nAssistant:", "\nHuman:",
        "\n\nThe result", "\n\nOutput:",
        "\n\nResult:", "\n\nResponse:",
        "\n<tool_result",          # DeepSeek hallucinating its own tool_result
        "\n<tool_use_error",       # DeepSeek hallucinating error responses
        "\nObservation:",          # ReAct-style fabrication
        "\nHuman: <tool_result>",
        "\nHuman: <tool_use_error>",
        "\n\nThe output is",       # Common fabrication phrase
        "\n\nThe command returned", # Common fabrication phrase
    ]
    earliest = len(text)
    for marker in markers:
        idx = text.find(marker)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest].strip()


def _append_text(blocks: list, text: str) -> None:
    """Append text to last text block (merge) or create a new one."""
    if not text:
        return
    if blocks and blocks[-1].get("type") == "text":
        blocks[-1]["text"] += text
    else:
        blocks.append({"type": "text", "text": text})


def _find_first_tool_call_end(text: str) -> int:
    """
    Position just after the first genuine tool-call closing tag, in
    whichever format DeepSeek used — the plain <tool_use> wrapper or a
    native XML tag matching a known tool name/alias (e.g. </Bash>).
    Returns -1 if no real tool call is found.

    Callers must run this AFTER normalizing the <tool_call name="X"> and
    DeepSeek-V3 special-token formats into <tool_use> — otherwise a real
    tool call still in one of those other formats won't be recognized,
    and the fabrication-stripping scoped to "after the first real tool
    call" would wrongly treat none as having happened yet.
    """
    candidates = []
    idx = text.find("</tool_use>")
    if idx != -1:
        candidates.append(idx + len("</tool_use>"))
    for m in re.finditer(r"</(\w+)>", text):
        tag = m.group(1)
        if tag in CLAUDE_CODE_TOOL_NAMES or tag.lower() in DEEPSEEK_TAG_TO_TOOL:
            candidates.append(m.end())
    return min(candidates) if candidates else -1


def _fix_missing_opening_tool_use_tag(text: str) -> str:
    """
    Repair a reply that contains one or more literal </tool_use> closing
    tags but NO opening <tool_use> tag anywhere — e.g.:

        </tool_use>
        {"name": "Bash", "input": {...}}
        </tool_use>

    or simply:

        {"name": "Bash", "input": {...}}
        </tool_use>

    The main scan in parse_response requires a literal opening tag to match
    anything at all, so without this repair a reply shaped like this never
    matches and the whole thing — including a genuine, parseable tool call —
    falls through as inert text.

    Conservative by construction: only rewrites when (a) no real opening tag
    exists anywhere in the text, and (b) the JSON sitting immediately before
    the last closing tag actually parses as a tool-call object with a
    "name" key. Anything else is left untouched rather than guessed at.
    """
    if "<tool_use>" in text:
        return text  # a real opening tag exists — not this case

    closes = list(re.finditer(r"</tool_use>", text))
    if not closes:
        return text

    # The real payload sits before the LAST closing tag — that's DeepSeek's
    # actual intended close. Anything before an earlier closing tag (a
    # stray/hallucinated one) is discarded, mirroring what a genuine
    # <tool_use>...</tool_use> block would keep (just its own inner text).
    last_close = closes[-1]
    head = text[:last_close.start()]
    tail = text[last_close.end():]

    search_from = closes[-2].end() if len(closes) > 1 else 0
    brace_idx = head.find("{", search_from)
    if brace_idx == -1:
        return text  # no JSON-looking payload here — leave untouched

    prose = head[:brace_idx]
    json_candidate = head[brace_idx:].strip()
    obj = _parse_json_tool(json_candidate)
    if not isinstance(obj, dict) or "name" not in obj:
        return text  # doesn't parse as a tool call — leave untouched

    print(
        f"[tool_parse] repaired reply missing opening <tool_use> tag "
        f"({len(closes)} closing tag(s) found, 0 opening): "
        f"{repr(json_candidate[:80])}",
        flush=True,
    )
    return f"{prose}<tool_use>\n{json_candidate}\n</tool_use>{tail}"


def _fix_dangling_unclosed_tool_use(text: str) -> str:
    """
    Repair a reply that has an opening <tool_use> with no matching close —
    typically because generation was cut off mid-call (token limit, stream
    truncation cutting off before the model emitted "</tool_use>"). Without
    this, the main scan below requires a literal closing tag to match
    anything at all, so a truncated-but-otherwise-real call is silently
    dropped and NO tool_use block is ever produced for it.

    Conservative by construction: only looks at the LAST <tool_use> open in
    the text (any earlier ones are assumed already closed and handled by the
    normal scan), only fires if there is no closing tag anywhere after it,
    and only if what follows actually looks like a JSON object (starts with
    "{"). We don't try to validate/repair the JSON itself here — that's what
    _parse_json_tool's existing brace-imbalance repair (Attempt 3) is for;
    this function's only job is making sure the tag wrapper is well-formed
    enough for the main scan to find it in the first place.
    """
    opens = [m.end() for m in re.finditer(r"<tool_use>", text)]
    if not opens:
        return text
    last_open = opens[-1]
    if "</tool_use>" in text[last_open:]:
        return text  # already closes properly somewhere — not this case

    payload = text[last_open:].strip()
    if not payload.startswith("{"):
        return text  # doesn't look like a JSON payload — leave untouched

    print(
        f"[tool_parse] repaired dangling unclosed <tool_use> "
        f"(likely truncated generation): {repr(payload[:80])}",
        flush=True,
    )
    return text[:last_open] + payload + "</tool_use>"


def _find_balanced_tag(text: str, tag: str, start: int) -> tuple[int, int] | None:
    """
    Find a balanced <tag>...</tag> span with the opening tag starting at
    `start`, correctly handling same-name NESTING by counting depth instead
    of stopping at the first closing tag of that name (which is what a
    non-greedy regex with a backreference does, and which mis-closes on the
    inner tag when the model nests a tag inside itself).

    Returns (inner_start, inner_end) — the span strictly between the
    matched opening and closing tags — or None if `start` isn't an opening
    tag for `tag`, or no balancing close exists (unbalanced/truncated).
    """
    open_re = re.compile(rf"<{re.escape(tag)}>")
    close_re = re.compile(rf"</{re.escape(tag)}>")
    m = open_re.match(text, start)
    if not m:
        return None
    depth = 1
    pos = inner_start = m.end()
    while depth > 0:
        nxt_open = open_re.search(text, pos)
        nxt_close = close_re.search(text, pos)
        if not nxt_close:
            return None  # unbalanced — no matching close at this depth
        if nxt_open and nxt_open.start() < nxt_close.start():
            depth += 1
            pos = nxt_open.end()
        else:
            depth -= 1
            pos = nxt_close.end()
            if depth == 0:
                return inner_start, nxt_close.start()
    return None  # pragma: no cover — loop always returns or hits the unbalanced case above


# Recognizes either the literal "tool_use" wrapper or a native "<TagName>"
# tag. Order matters: alternation tries "tool_use" first, so a literal
# <tool_use> tag is always classified as such rather than falling through
# to the generic native-tag branch (which would also technically match it).
_OPEN_TAG_RE = re.compile(r"<(tool_use|[A-Za-z]\w*)>")


def _iter_tool_matches(text: str):
    """
    Depth-aware replacement for the old single non-greedy regex scan.
    Yields dicts {start, end, tag, inner} for each top-level, properly
    balanced <tool_use>...</tool_use> or <NativeTag>...</NativeTag> span,
    left to right, non-overlapping — same contract as re.finditer, but
    using _find_balanced_tag so same-name nesting closes on the correct
    (outermost) matching tag instead of the first one encountered.

    An opening tag with no balancing close (still-unbalanced after the
    dangling-tag repairs above) is simply skipped — its "<...>" text is
    left for the caller to treat as ordinary text, same as if it had never
    matched under the old regex-based scan.
    """
    pos = 0
    n = len(text)
    while pos < n:
        m = _OPEN_TAG_RE.search(text, pos)
        if not m:
            return
        tag = m.group(1)
        span = _find_balanced_tag(text, tag, m.start())
        if span is None:
            pos = m.end()  # unbalanced open — skip it, keep scanning after
            continue
        inner_start, inner_end = span
        close_end = inner_end + len(f"</{tag}>")
        yield {"start": m.start(), "end": close_end, "tag": tag, "inner": text[inner_start:inner_end]}
        pos = close_end


def parse_response(text: str, valid_tools: set[str] | None = None) -> list:
    """
    Parse DeepSeek reply into Anthropic content blocks.

    Handles three tool-call formats emitted by DeepSeek:
      1. Injected JSON format:   <tool_use>{"name":…,"input":…}</tool_use>
      2. DeepSeek native XML:    <Bash><command>…</command></Bash>
      3. Fenced JSON in tool_use:<tool_use>```json{…}```</tool_use>

    Key behaviours:
      - Text BETWEEN tool calls is preserved (not dropped after first tool).
      - Text AFTER a tool call is stripped (fabricated results/continuations).
      - Multiple tool calls in one response all survive.
      - Unknown / skip tags are passed through as literal text.
      - JSON is repaired when possible (trailing commas, unclosed braces).
      - valid_tools: set of tool names from the current request; unknown names
        are dropped rather than forwarded to Claude Code which would error.
    """
    blocks: list[dict] = []
    last = 0
    last_tool_end: int | None = None  # position just after last confirmed tool match

    # Debug: log full raw text to diagnose tool parsing issues
    print(f"[parse_response] full raw text ({len(text)} chars):\n{text}\n---END---", flush=True)

    # ── Pre-pass 0: strip premature exit preambles ───────────────────────────
    # "Now let's check the files" followed by nothing → strip it
    text = strip_premature_exit_preambles(text)

    # ── Pre-pass 1: <tool_call name="ToolName">{json}</tool_call> ────────────
    # DeepSeek sometimes uses this format instead of <tool_use> or <ToolName>.
    # Runs BEFORE fabrication-stripping below, so that format is already
    # normalized to <tool_use> by the time we need to detect "was there a
    # real tool call already" — otherwise a genuine tool call still in this
    # format would be invisible to that check.
    tool_call_attr_pat = re.compile(
        r'<tool_call\s+name=["\']?(\w+)["\']?>\s*(.*?)\s*</tool_call>',
        re.DOTALL,
    )
    def _replace_tool_call_attr(m):
        tool_name = m.group(1).strip()
        raw_inner = m.group(2).strip()
        obj = _parse_json_tool(raw_inner)
        if obj is None:
            obj = {}
        # If the JSON itself has an "input" key (full tool-call envelope),
        # unwrap it rather than nesting {name, input: {name, input: ...}}.
        if isinstance(obj, dict) and "input" in obj and isinstance(obj["input"], dict):
            params = obj["input"]
        elif isinstance(obj, dict) and "name" in obj and set(obj.keys()) <= {"name", "id", "input", "type"}:
            params = obj.get("input", {}) or {}
        else:
            params = obj if isinstance(obj, dict) else {}
        merged = {"name": tool_name, "id": f"toolu_{uuid.uuid4().hex[:16]}", "input": params}
        return f"<tool_use>{json.dumps(merged)}</tool_use>"
    text = tool_call_attr_pat.sub(_replace_tool_call_attr, text)

    # ── Pre-pass 2: DeepSeek-V3 special token format ─────────────────────────
    # DeepSeek-V3 sometimes emits tool calls as:
    #   <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
    #   type\nfunction
    #   <｜tool▁sep｜>
    #   TOOL_NAME
    #   ```json
    #   {"arg": "value"}
    #   ```
    #   <｜tool▁call▁end｜><｜tool▁calls▁end｜>
    # Convert these to <tool_use>{json}</tool_use> before normal parsing.
    special_token_pat = re.compile(
        r"<｜tool▁call▁begin｜>.*?<｜tool▁sep｜>\s*(\w+)\s*```(?:json)?\s*(.*?)```\s*<｜tool▁call▁end｜>",
        re.DOTALL,
    )
    def _replace_special_tokens(m):
        tool_name = m.group(1).strip()
        raw_json  = m.group(2).strip()
        try:
            params = json.loads(raw_json)
        except json.JSONDecodeError:
            params = {"input": raw_json}
        obj = {"name": tool_name, "id": f"toolu_{uuid.uuid4().hex[:16]}", "input": params}
        return f"<tool_use>{json.dumps(obj)}</tool_use>"
    text = special_token_pat.sub(_replace_special_tokens, text)
    # Strip the outer begin/end wrappers if any remain
    text = re.sub(r"<｜tool▁calls▁begin｜>|<｜tool▁calls▁end｜>", "", text)

    # ── Pre-pass 3: repair a missing opening <tool_use> tag ──────────────────
    # Mirror image of the duplicated-OPENING-tag repairs in _parse_json_tool
    # (m_dup/m_lit) — those fire once the main scan below has already matched
    # an opening tag. This handles the opposite shape: DeepSeek sometimes
    # emits a well-formed tool call with NO opening tag at all, e.g.:
    #   </tool_use>
    #   {"name": "Bash", "input": {...}}
    #   </tool_use>
    # (a stray leading close, then the real call, then its real close) or
    # just:
    #   {"name": "Bash", "input": {...}}
    #   </tool_use>
    # Without this repair, the main scan's regex requires a literal opening
    # tag to match anything at all, so text like this never matches ANYWHERE
    # and the entire reply — including a perfectly valid tool call — falls
    # through as inert text. Only fires when there's no real opening tag
    # anywhere in the reply; if one exists, this is not that case and the
    # normal scan (plus the duplicated-open-tag repairs) handles it.
    text = _fix_missing_opening_tool_use_tag(text)

    # ── Pre-pass 3b: repair a dangling unclosed <tool_use> (truncated call) ──
    # Mirror image of pre-pass 3 above: instead of a missing OPEN, this is a
    # missing CLOSE — the reply ends mid-call because generation was cut off
    # before "</tool_use>" was emitted. Must run after pre-pass 3 (which only
    # fires when there's no opening tag anywhere) so the two repairs don't
    # step on each other, and before the fabrication-stripping pass below so
    # a real-but-truncated call is normalized before anything else inspects
    # tool-call boundaries.
    text = _fix_dangling_unclosed_tool_use(text)

    # ── Pre-pass 4: strip hallucinated conversation continuations ────────────
    # DeepSeek sometimes emits a real tool call, then keeps going by
    # fabricating "Human: <tool_result>..." / "Assistant: ..." turns of its
    # own — worst case, a SECOND tool call based on a result it made up.
    # Runs AFTER Pre-passes 1-3 so every real tool-call format (<tool_use>,
    # the <tool_call name="X"> alias, the special-token format, and a
    # missing-opening-tag reply) has already been normalized to <tool_use>
    # by this point.
    #
    # The two protocol-specific markers below are safe to strip unconditionally
    # — nobody writes literal "<tool_result>"/"<tool_use_error>" tags in normal
    # prose. The bare "Human:"/"Assistant:" markers are NOT safe to strip
    # unconditionally: build_prompt() itself uses those exact role prefixes
    # ("Human: {content}" / "Assistant: {content}"), so a response that
    # explains or edits code involving them — e.g. someone working on this
    # very proxy, or on any chat-formatting code — can legitimately contain
    # those substrings before any tool call ever appears. Blindly cutting at
    # the first occurrence would silently destroy a real tool call that
    # follows. So the bare markers are only searched for AFTER the first
    # genuine tool-call close (<tool_use>, or a native tag matching a known
    # tool name/alias, e.g. </Bash>) — i.e. only to catch a fabricated
    # continuation that follows an already-real tool call, never to truncate
    # text that precedes (and may contain) the first one.
    fabrication_markers_always = [
        "\nHuman: <tool_result>",
        "\nHuman: <tool_use_error>",
    ]
    fabrication_markers_after_tool = [
        "\n\nHuman:",
        "\nHuman:",          # single-newline variant
        "\n\nAssistant:",    # fabricated self-reply
    ]
    earliest = len(text)
    for marker in fabrication_markers_always:
        idx = text.find(marker)
        if idx != -1 and idx < earliest:
            earliest = idx
    first_tool_close = _find_first_tool_call_end(text)
    if first_tool_close != -1:
        for marker in fabrication_markers_after_tool:
            idx = text.find(marker, first_tool_close)
            if idx != -1 and idx < earliest:
                earliest = idx
    if earliest < len(text):
        print(f"[pre-pass] stripping hallucinated continuation at pos {earliest}", flush=True)
        text = text[:earliest]

    # ── Pattern 1: <tool_use>…</tool_use>  (case-sensitive tag)
    # ── Pattern 2: <NativeName>…</NativeName>  (tag starts with letter, any case)
    #
    # We process the text in a single left-to-right scan. For each match:
    #   • text before the match → appended to blocks (unless we already cut after a tool)
    #   • tool_use block → parsed as JSON
    #   • native tag → resolved via DEEPSEEK_TAG_TO_TOOL / CLAUDE_CODE_TOOL_NAMES
    #   • skip tag  → passed through as text

    # NOTE: deliberately NOT using the depth-aware _iter_tool_matches walk
    # here, even though it exists above. Tried it — it breaks the existing
    # duplicate-<tool_use>-wrapper repair (_parse_json_tool's m_dup/m_lit
    # attempts), which depends on the OLD shallow non-greedy-regex behavior:
    # for DeepSeek's real "<tool_use><tool_use>{...}</tool_use></tool_use>"
    # artifact, the shallow scan closes on the FIRST </tool_use> (the inner
    # one), capturing json_inner = '<tool_use>{...}' with no trailing close
    # — exactly the shape m_lit expects and strips. A depth-aware matcher
    # instead correctly finds the true OUTER boundary, which includes the
    # inner tag's own close inside "inner" — a shape m_lit does NOT handle,
    # so that (the actually-observed DeepSeek artifact) would regress.
    # True independent nesting of two intentional same-name calls doesn't
    # appear in practice, so the old regex scan (which happens to do the
    # right thing for the artifact that DOES occur) is kept here.
    combined = re.compile(
        r"<tool_use>(?P<json_inner>.*?)</tool_use>"
        r"|<(?P<ntag>[A-Za-z]\w*)>(?P<ninner>.*?)</(?P=ntag)>",
        re.DOTALL,
    )

    for m in combined.finditer(text):
        segment_start = m.start()
        segment_end   = m.end()
        tag   = "tool_use" if m.group("json_inner") is not None else m.group("ntag")
        inner = m.group("json_inner") if m.group("json_inner") is not None else m.group("ninner")

        # ── Text before this match ──────────────────────────────────────────
        # Only emit text that comes BEFORE any tool call in the segment
        # that's still "open" (i.e. we haven't yet found a tool here).
        if last_tool_end is None:
            # Haven't hit a tool yet → all text before this match is kept
            before = text[last:segment_start]
            _append_text(blocks, before)
        else:
            # Already past a tool call → do NOT emit text between tools
            # (it's usually fabricated results or "Now let me do X..." filler).
            pass

        # Raw matched text (equivalent to the old m.group(0)), computed from
        # the span since _iter_tool_matches yields dicts, not match objects.
        matched_text = text[segment_start:segment_end]

        # ── Decide what kind of match this is ───────────────────────────────
        if tag == "tool_use":
            # ── Format 1: <tool_use>{json}</tool_use> ───────────────────────
            raw_inner = inner.strip()
            obj = _parse_json_tool(raw_inner)
            if obj and isinstance(obj, dict) and "name" in obj:
                tool_name = obj.get("name", "")
                tool_input = obj.get("input", {})
                # Validate against caller's tool list
                if valid_tools and tool_name not in valid_tools:
                    print(f"[tool_parse] dropping unknown tool '{tool_name}' in <tool_use>", flush=True)
                    _append_text(blocks, matched_text)
                else:
                    blocks.append({
                        "type":  "tool_use",
                        # Always mint our own id — never trust DeepSeek's. Weaker
                        # models commonly copy the literal "call_abc123" id from
                        # the system-prompt example instead of generating a fresh
                        # one, and duplicate ids across tool_use blocks break
                        # Claude Code's tool_result -> tool_use_id matching. The
                        # proxy is the one handing this id to Claude Code in the
                        # first place, so nothing here ever needs the model's own.
                        "id":    f"toolu_{uuid.uuid4().hex[:16]}",
                        "name":  tool_name,
                        "input": tool_input if isinstance(tool_input, dict) else {},
                    })
                    last_tool_end = segment_end
            else:
                # Couldn't parse — treat as text
                _append_text(blocks, matched_text)

        else:
            # ── Format 2: <NativeTag>…</NativeTag> ──────────────────────────
            tag_name = tag

            if tag_name.lower() in _SKIP_TAGS:
                # Structural / HTML tag — pass through as text
                if last_tool_end is None:
                    _append_text(blocks, matched_text)
                # (after a tool call, skip-tag text is dropped as fabrication)
            else:
                block = parse_native_tag(tag_name, inner, valid_tools=valid_tools)
                if block is not None and _native_call_has_narrative_preamble(
                        block["name"], tag_name, text, segment_start):
                    # Looks narrated/demonstrated, not intended — treat as text.
                    if last_tool_end is None:
                        _append_text(blocks, matched_text)
                elif block is not None:
                    blocks.append(block)
                    last_tool_end = segment_end
                else:
                    # Resolved to None → unknown/invalid, treat as text
                    if last_tool_end is None:
                        _append_text(blocks, matched_text)

        last = segment_end

    # ── Tail text ────────────────────────────────────────────────────────────
    tail = text[last:]
    if tail:
        if last_tool_end is None:
            # No tool calls found at all → keep everything
            _append_text(blocks, tail)
        else:
            # Text after the last real tool call → strip fabricated continuation
            cleaned = strip_fabricated_continuation(tail)
            if cleaned:
                # Only keep if it's substantial and doesn't look like a fabricated result
                _append_text(blocks, cleaned)

    if not blocks:
        blocks.append({"type": "text", "text": text})

    # ── Fallback: if no tool_use blocks found, try ```json fenced JSON ────────
    # DeepSeek sometimes emits: ```json\n{"name":"Bash","input":{...}}\n```
    # Only match explicit ```json fences (not ```javascript, bare ```, etc.)
    # to avoid misfiring on code examples in explanatory text responses.
    if not any(b["type"] == "tool_use" for b in blocks):
        fence_pat = re.compile(r"```json\s*(\{[^`]*?\})\s*```", re.DOTALL)
        rebuilt: list[dict] = []
        replaced = False
        full_text_so_far = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        last_pos = 0
        for fm in fence_pat.finditer(full_text_so_far):
            obj = _parse_json_tool(fm.group(1))
            if obj and isinstance(obj, dict) and "name" in obj and "input" in obj:
                tool_name = obj.get("name", "")
                if not valid_tools or tool_name in valid_tools:
                    before = full_text_so_far[last_pos:fm.start()]
                    if before.strip():
                        rebuilt.append({"type": "text", "text": before})
                    rebuilt.append({
                        "type":  "tool_use",
                        "id":    f"toolu_{uuid.uuid4().hex[:16]}",  # never trust DeepSeek's id — see rationale above
                        "name":  tool_name,
                        "input": obj.get("input", {}),
                    })
                    last_pos = fm.end()
                    replaced = True
        if replaced:
            tail = full_text_so_far[last_pos:]
            if tail.strip():
                rebuilt.append({"type": "text", "text": tail})
            blocks = rebuilt
            print(f"[parse_response] fallback fence parser rescued {sum(1 for b in blocks if b['type']=='tool_use')} tool(s)", flush=True)

    # ── Intentionally NO fallback for plain fenced shell code ─────────────────
    # A ```bash/sh/shell/zsh fence is never treated as a real tool call, even
    # when "Bash" was offered this turn. It used to be auto-"rescued" into a
    # real Bash tool_use — but DeepSeek uses that exact fence shape both for
    # commands it actually wants to run AND for purely illustrative examples
    # ("Recommended Next Steps: `hydra -l admin -P rockyou.txt ...`") that are
    # meant for the human to review and run manually, and the two are not
    # reliably distinguishable from the fence alone. Auto-executing the
    # illustrative case caused unapproved commands (bruteforce attempts,
    # credential templates, etc.) to actually run. The system prompt now
    # forbids DeepSeek from using bash/sh/shell/zsh fences at all — real
    # execution must go through <tool_use>, and illustrative commands must use
    # inline backticks or a non-shell fence tag instead — so a ```bash fence
    # reaching this point is always left as inert display text.

    print(
        f"[parsed blocks] {[b['type'] + ('/' + b.get('name','')) if b['type'] == 'tool_use' else b['type'] for b in blocks]}",
        flush=True,
    )
    return blocks


# ── ToolFilter helpers ───────────────────────────────────────────────────────

def _find_unknown_tools(raw_text: str, valid_tools: set[str]) -> set[str]:
    """
    Scan raw DeepSeek output for any tool-call block referencing a tool name
    not in valid_tools. Covers all tag variants DeepSeek may emit:
      - <tool_use>{json}</tool_use>
      - <tool_call>{json}</tool_call>
      - <tool_call name="ToolName">{json}</tool_call>
    Returns the set of unknown names found (empty = all OK).
    """
    unknown: set[str] = set()
    pattern = re.compile(
        r"<(?:tool_use|tool_call)(?:\s[^>]*)?\s*>\s*(.*?)\s*</(?:tool_use|tool_call)>",
        re.DOTALL,
    )
    for m in pattern.finditer(raw_text):
        inner = m.group(1).strip()
        obj = _parse_json_tool(inner)
        if obj and isinstance(obj, dict):
            name = obj.get("name", "")
            if name and name not in valid_tools:
                unknown.add(name)
    return unknown


# Prefill used only by the unknown-tool retry (Case 1 in call_with_filter).
_TOOL_USE_PREFILL = '<tool_use>\n{"name": "'


# ── ToolFilter ────────────────────────────────────────────────────────────────

class ToolFilter:
    """
    Wraps call_deepseek_managed + parse_response.
    Stores the valid tool names from the current request so parse_response
    can reject hallucinated tool names before they reach Claude Code.
    """

    def __init__(self, tools: list, messages: list | None = None):
        # Valid tools = ONLY what this specific request offered. Do NOT pad
        # with the full CLAUDE_CODE_TOOL_NAMES baseline — that used to make
        # every Claude Code built-in "valid" regardless of what this turn
        # actually declared, so a tool-restricted subagent (e.g. a read-only
        # Explore agent given only Read/Grep) could still have a hallucinated
        # <Edit> or <Bash> native-tag call slip through, defeating the
        # "prevents hallucinated tool names" guarantee for anything with a
        # recognized built-in name. CLAUDE_CODE_TOOL_NAMES / DEEPSEEK_TAG_TO_TOOL
        # are still used elsewhere purely to normalize tag spelling/casing —
        # membership is what's restricted here.
        self._valid_tools: set[str] = set()
        for t in (tools or []):
            name = t.get("name")
            if name:
                self._valid_tools.add(name)

    def call_with_filter(self, session_key: str, prompt: str) -> tuple[str, list]:
        """
        Returns (raw_text, blocks).

        Retries only when the model calls an unknown tool name — injects the
        valid tool list and asks it to retry with a prefill. Budget:
        MAX_UNKNOWN_TOOL_RETRIES. The model is otherwise free to respond with
        or without a tool call as it sees fit.
        """
        MAX_UNKNOWN_TOOL_RETRIES = 2
        unknown_tool_attempts = 0
        current_prompt = prompt
        pending_prefill = ""
        raw_text, blocks = "", []

        while True:
            raw_text, _ = call_deepseek_managed(session_key, current_prompt)
            if pending_prefill:
                raw_text = pending_prefill + raw_text
                pending_prefill = ""

            blocks = parse_response(raw_text, valid_tools=self._valid_tools)

            # ── Unknown tool name: retry with valid tool list ─────────────
            unknown = _find_unknown_tools(raw_text, self._valid_tools)
            if unknown:
                if unknown_tool_attempts >= MAX_UNKNOWN_TOOL_RETRIES:
                    print(
                        f"[tool_parse] unknown tool(s) {unknown} persisted after "
                        f"{MAX_UNKNOWN_TOOL_RETRIES} retries — returning as-is",
                        flush=True,
                    )
                    return raw_text, blocks

                unknown_tool_attempts += 1
                valid_list = ", ".join(sorted(self._valid_tools))
                correction = (
                    f"\n\nHuman: <tool_use_error>\n"
                    f"ERROR: You tried to call unknown tool(s): {', '.join(sorted(unknown))}.\n"
                    f"Those tools DO NOT EXIST. You MUST only use tools from this list:\n"
                    f"{valid_list}\n\n"
                    f"The tag below is already open — continue with a valid tool name from\n"
                    f"the list above, then finish the JSON. Do not write anything else.\n"
                    f"</tool_use_error>\n\nAssistant: {_TOOL_USE_PREFILL}"
                )
                print(
                    f"[tool_parse] unknown tool(s) {unknown} — injecting correction and retrying "
                    f"(attempt {unknown_tool_attempts}/{MAX_UNKNOWN_TOOL_RETRIES})",
                    flush=True,
                )
                current_prompt = current_prompt + correction
                pending_prefill = _TOOL_USE_PREFILL
                continue

            return raw_text, blocks


# ── SSE helpers ───────────────────────────────────────────────────────────────

def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

# ── DeepSeek call (collects full response) ────────────────────────────────────

def call_deepseek(session: dict, prompt: str, parent_message_id: int | None) -> tuple[str, int | None, bool, bool]:
    """
    Send prompt to DeepSeek, return (full_text, new_message_id, rate_limited, server_busy).

    parent_message_id: the DS assistant message_id from the previous reply
                        in this chat session (None for the very first
                        message of the session).
    rate_limited:      True if DeepSeek responded with finish_reason
                        "rate_limit_reached" ("Messages too frequent").
    server_busy:       True if DeepSeek responded with generation_timeout
                        ("Server busy, please try again later").
    """
    http       = session["http"]
    ds_session = session["ds_session_id"]

    challenge_data = get_pow_challenge(http)
    answer = hasher.solve(
        challenge_data["challenge"],
        challenge_data["salt"],
        challenge_data["difficulty"],
        challenge_data["expire_at"],
    )
    pow_response = build_pow_response(challenge_data, answer)

    headers = {
        "X-Ds-Pow-Response": pow_response,
        "Content-Type":      "application/json",
        "Accept":            "text/event-stream",
    }
    # Resolve model_type from CLI flag.
    # "fast"   → model_type = "default"
    # "expert" → model_type = null (None)
    model_type = "default" if _CLI_ARGS.model == "fast" else None

    body = {
        "chat_session_id":   ds_session,
        "parent_message_id": parent_message_id,
        "model_type":        model_type,
        "prompt":            prompt,
        "ref_file_ids":      [],
        "thinking_enabled":  _CLI_ARGS.think,
        "search_enabled":    _CLI_ARGS.search,
        "action":            None,
        "preempt":           False,
    }

    full_text    = ""
    new_msg_id   = None
    line_count   = 0
    last_lines   = []
    rate_limited = False
    server_busy  = False

    with http.post(
        f"{BASE_URL}/api/v0/chat/completion",
        headers=headers,
        json=body,
        stream=True,
        timeout=120,
    ) as resp:
        print(f"[deepseek] HTTP {resp.status_code} content-type={resp.headers.get('content-type')}", flush=True)
        resp.raise_for_status()
        if "application/json" in resp.headers.get("content-type", ""):
            err_body = resp.json()
            print(f"[deepseek] JSON error body: {err_body}", flush=True)
            raise RuntimeError(f"DeepSeek API error: {err_body}")

        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line_count += 1
            last_lines.append(raw_line)
            if len(last_lines) > 5:
                last_lines.pop(0)
            if not raw_line.startswith("data:"):
                continue
            try:
                pl = json.loads(raw_line[5:].strip())
            except json.JSONDecodeError:
                continue

            # Surface explicit error payloads from DeepSeek
            if pl.get("code") not in (None, 0):
                print(f"[deepseek] error payload in stream: {pl}", flush=True)

            if pl.get("finish_reason") == "rate_limit_reached" or pl.get("type") == "error" and "frequent" in str(pl.get("content", "")).lower():
                rate_limited = True

            # Detect "Server busy" / generation_timeout — different from rate_limit;
            # means the backend is overloaded and we should switch to another token.
            if (pl.get("finish_reason") == "generation_timeout"
                    or (pl.get("type") == "error"
                        and "server busy" in str(pl.get("content", "")).lower())):
                server_busy = True
                print(f"[deepseek] server_busy detected: {pl}", flush=True)

            v = pl.get("v")
            p = pl.get("p", "")
            o = pl.get("o", "")
            chunk = None

            if isinstance(v, dict) and "response" in v:
                resp_obj = v["response"]
                new_msg_id = resp_obj.get("message_id")
                # Join ALL RESPONSE-type fragments in order — this snapshot
                # can arrive with more than one already-buffered fragment
                # (e.g. on reconnect), and keeping only the last one (the
                # previous behaviour) silently dropped the earlier content,
                # which could chop the front off an otherwise well-formed
                # tool call (e.g. losing a leading "<tool_use>" open tag).
                parts = [
                    f.get("content", "") for f in resp_obj.get("fragments", [])
                    if f.get("type") == "RESPONSE"
                ]
                if parts:
                    chunk = "".join(parts)
            elif isinstance(v, str) and o == "APPEND" and p == "response/fragments/-1/content":
                chunk = v
            elif isinstance(v, str) and "p" not in pl:
                chunk = v

            if chunk:
                full_text += chunk

    if not full_text.strip():
        print(f"[deepseek] EMPTY result — {line_count} raw lines received, last lines: {last_lines}", flush=True)
        if rate_limited:
            print("[deepseek] detected rate_limit_reached", flush=True)
        if server_busy:
            print("[deepseek] detected server_busy (generation_timeout)", flush=True)

    print(f"[deepseek raw]\n{repr(full_text[:500])}\n", flush=True)
    return full_text, new_msg_id, rate_limited, server_busy


def call_deepseek_managed(session_key: str, prompt: str, _depth: int = 0) -> tuple[str, int | None]:
    """
    Sends `prompt` to DeepSeek. On empty response (session full / rate limited):
      1. Retry the same anchor up to MAX_SAME_ANCHOR_RETRIES times (transient errors).
      2. If server_busy is detected (generation_timeout), rotate to the next token
         in the pool and immediately retry with a fresh session on that token.
      3. If still empty, create a brand-new DeepSeek chat session and resend
         with parent_message_id=None — identical to how the very first message
         is sent. This recovers from "session full" without losing the prompt.
      4. Cap total session resets at MAX_SESSIONS to avoid infinite loops.
    """
    MAX_SAME_ANCHOR_RETRIES = 3
    MAX_SESSIONS            = 3

    session = store.get_or_create(session_key)
    anchor  = session["anchor_message_id"]

    # ── Try against the current anchor (with retries for transient empty) ──
    delay = 2
    for attempt in range(1, MAX_SAME_ANCHOR_RETRIES + 1):
        full_text, new_id, rate_limited, server_busy = call_deepseek(session, prompt, anchor)
        if full_text.strip():
            if new_id is not None:
                session["last_good_message_id"] = new_id
            return full_text, new_id

        # ── Server busy: rotate token and start fresh session immediately ──
        if server_busy and len(_token_pool) > 1:
            new_token = _token_pool.rotate()
            print(f"[deepseek] server_busy — rotating to next token and resetting session", flush=True)
            session = store.reset(session_key, token=new_token)
            anchor  = None  # new session has no history — stale parent_message_id would target the abandoned session
            full_text, new_id, _, _ = call_deepseek(session, prompt, parent_message_id=None)
            if full_text.strip():
                if new_id is not None:
                    session["last_good_message_id"] = new_id
                print(f"[deepseek] token-rotation succeeded (new_id={new_id})", flush=True)
                return full_text, new_id
            print(f"[deepseek] token-rotation attempt also empty — continuing retry loop", flush=True)

        if attempt < MAX_SAME_ANCHOR_RETRIES:
            print(f"[deepseek] empty on anchor={anchor} (attempt {attempt}/{MAX_SAME_ANCHOR_RETRIES}) — retrying in {delay}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 30)

    # ── All retries exhausted — session is full, start a fresh one ─────────
    if _depth + 1 >= MAX_SESSIONS:
        print("[deepseek] session-reset budget exhausted — giving up", flush=True)
        return "", None

    print(f"[deepseek] session appears full (anchor={anchor}) — creating new session and resending as first message", flush=True)
    session = store.reset(session_key)          # fresh ds_session_id, anchor=None
    full_text, new_id, _, _ = call_deepseek(session, prompt, parent_message_id=None)
    if full_text.strip():
        if new_id is not None:
            session["last_good_message_id"] = new_id
        print(f"[deepseek] new session succeeded (new_id={new_id})", flush=True)
        return full_text, new_id

    # New session also returned empty — recurse once more (covers edge cases)
    print("[deepseek] new session also empty — recursing", flush=True)
    return call_deepseek_managed(session_key, prompt, _depth=_depth + 1)


def stream_response_as_anthropic(session_key: str, prompt: str, model: str, input_tokens: int, tool_filter: "ToolFilter"):
    """
    Calls DeepSeek, collects full response, parses content blocks,
    then streams them as proper Anthropic SSE events.
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    yield sse("message_start", {
        "type": "message_start",
        "message": {
            "id":            msg_id,
            "type":          "message",
            "role":          "assistant",
            "content":       [],
            "model":         model,
            "stop_reason":   None,
            "stop_sequence": None,
            "usage":         {"input_tokens": input_tokens, "output_tokens": 0},
        },
    })
    yield sse("ping", {"type": "ping"})

    # Collect full response from DeepSeek. This can take a while (PoW solve +
    # generation, possibly multiplied by ToolFilter's retries), so run it off
    # the generator thread and keep sending SSE pings while we wait — without
    # this, a slow generation or a forced retry could leave the stream idle
    # long enough for the client to time out the connection, which looks
    # exactly like "no tool call happened" from Claude Code's side.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(tool_filter.call_with_filter, session_key, prompt)
        while True:
            try:
                full_text, blocks = future.result(timeout=10)
                break
            except concurrent.futures.TimeoutError:
                yield sse("ping", {"type": "ping"})

    output_tokens = max(1, len(full_text.split()))
    stop_reason   = "end_turn"

    # Merge consecutive text blocks so they stream as one block — avoids
    # spurious newlines that Claude Code inserts between separate content blocks.
    merged = []
    for block in blocks:
        if block["type"] == "text" and merged and merged[-1]["type"] == "text":
            merged[-1]["text"] += block["text"]
        else:
            merged.append(dict(block))
    blocks = merged

    for idx, block in enumerate(blocks):
        if block["type"] == "text":
            yield sse("content_block_start", {
                "type": "content_block_start", "index": idx,
                "content_block": {"type": "text", "text": ""},
            })
            # Stream text in chunks for responsiveness
            text = block["text"]
            chunk_size = 20
            for i in range(0, len(text), chunk_size):
                yield sse("content_block_delta", {
                    "type": "content_block_delta", "index": idx,
                    "delta": {"type": "text_delta", "text": text[i:i+chunk_size]},
                })
            yield sse("content_block_stop", {"type": "content_block_stop", "index": idx})

        elif block["type"] == "tool_use":
            stop_reason = "tool_use"
            yield sse("content_block_start", {
                "type": "content_block_start", "index": idx,
                "content_block": {
                    "type":  "tool_use",
                    "id":    block["id"],
                    "name":  block["name"],
                    "input": {},
                },
            })
            # Stream input as a single JSON delta
            yield sse("content_block_delta", {
                "type": "content_block_delta", "index": idx,
                "delta": {
                    "type":         "input_json_delta",
                    "partial_json": json.dumps(block["input"]),
                },
            })
            yield sse("content_block_stop", {"type": "content_block_stop", "index": idx})

    yield sse("message_delta", {
        "type":  "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield sse("message_stop", {"type": "message_stop"})

# ── Flask app ─────────────────────────────────────────────────────────────────

app   = Flask(__name__)
hasher: DeepSeekHash = None
store = SessionStore()

# ── Per-conversation request pacing ──────────────────────────────────────────
# Requests are serialized PER conversation (session_key), not globally — one
# shared lock would force independent conversations (e.g. a main agent and a
# subagent running in parallel via the Task tool) to queue behind each other
# even though they hit different DeepSeek chat sessions and don't need to.
# Within a single conversation, calls still must be serialized: DeepSeek
# doesn't support concurrent in-flight generations against the same chat
# session, and parent_message_id chaining would race otherwise.

REQUEST_DELAY_SECONDS = 3.0
_session_locks: dict[str, threading.Lock] = {}
_session_locks_meta_lock = threading.Lock()


def _get_session_lock(session_key: str) -> threading.Lock:
    with _session_locks_meta_lock:
        lock = _session_locks.get(session_key)
        if lock is None:
            lock = threading.Lock()
            _session_locks[session_key] = lock
        return lock


def _evict_session_lock(session_key: str) -> None:
    """Drop a session's pacing lock when SessionStore evicts it — otherwise
    _session_locks grows forever, one entry per distinct conversation ever
    seen, for the life of the process."""
    with _session_locks_meta_lock:
        _session_locks.pop(session_key, None)


def enforce_request_pacing(session_key: str) -> threading.Lock:
    """Acquire this conversation's lock, then sleep for the fixed pacing
    delay with a countdown while holding it. Returns the lock so the caller
    can release it when done."""
    lock = _get_session_lock(session_key)
    lock.acquire()
    remaining = int(REQUEST_DELAY_SECONDS)
    frac = REQUEST_DELAY_SECONDS - remaining
    for i in range(remaining, 0, -1):
        print(f"[delay] {i}...", flush=True)
        time.sleep(1)
    if frac > 0:
        time.sleep(frac)
    print("[delay] 0", flush=True)
    return lock


def mark_request_finished(lock: threading.Lock) -> None:
    """Release the given conversation's lock so its next request can proceed."""
    if lock.locked():
        lock.release()


def is_permission_request(system: str) -> bool:
    """
    Detect Claude Code permission-evaluation requests.
    These have a very specific system prompt asking to output <decision>allow</decision>
    or <decision>block</decision>. We match on highly specific phrases only.
    """
    s = system.lower()
    # These phrases appear ONLY in the permission evaluator prompt, not the main prompt
    specific = [
        "<decision>allow</decision>",
        "<decision>block</decision>",
        "output <decision>",
        "respond with <decision>",
        "respond only with",
        "your response must be",
        "either allow or block",
        "allow or block",
    ]
    return any(k in s for k in specific)


def _allow_response(stream: bool, model: str):
    """Return an instant allow decision in Anthropic format."""
    text = "<decision>allow</decision>"
    if stream:
        def _gen():
            mid = f"msg_{uuid.uuid4().hex[:24]}"
            yield sse("message_start", {"type": "message_start", "message": {
                "id": mid, "type": "message", "role": "assistant", "content": [],
                "model": model, "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }})
            yield sse("content_block_start", {"type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""}})
            yield sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": text}})
            yield sse("content_block_stop", {"type": "content_block_stop", "index": 0})
            yield sse("message_delta", {"type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1}})
            yield sse("message_stop", {"type": "message_stop"})
        return Response(_gen(), mimetype="text/event-stream",
                        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})
    return jsonify({
        "id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": text}], "model": model,
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })


@app.route("/v1/messages", methods=["POST"])
def messages():
    body     = request.get_json(force=True)
    msgs     = body.get("messages", [])
    model    = body.get("model", "claude-sonnet-4-20250514")
    max_tok  = body.get("max_tokens", 8096)
    stream   = body.get("stream", False)
    system   = body.get("system", "")
    tools    = body.get("tools", [])

    # Flatten system to string if it's a list of blocks
    if isinstance(system, list):
        system = "\n".join(
            b.get("text", "") for b in system
            if isinstance(b, dict) and b.get("type") == "text"
        )

    # Log first 300 chars of system so we can tune the permission detector
    print(f"[system prefix] {repr(system[:300])}", flush=True)

    # Auto-approve permission evaluation requests from Claude Code
    if is_permission_request(system):
        print("[permission] auto-approving", flush=True)
        return _allow_response(stream, model)

    prompt       = build_prompt(system, msgs, tools)
    input_tokens = max(1, len(prompt.split()))
    session_key  = derive_session_key(system, msgs)   # ← per-conversation, not "global"
    store.get_or_create(session_key)  # ensure session exists
    tf           = ToolFilter(tools, msgs)          # ← filter created per-request

    if stream:
        def generate():
            lock = enforce_request_pacing(session_key)
            try:
                yield from stream_response_as_anthropic(session_key, prompt, model, input_tokens, tf)
            except Exception as e:
                print(f"[error] {e}", file=sys.stderr)
                import traceback; traceback.print_exc()
                yield sse("error", {"type": "error", "error": {"type": "api_error", "message": str(e)}})
            finally:
                mark_request_finished(lock)

        return Response(generate(), mimetype="text/event-stream",
                        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

    # Non-streaming
    lock = enforce_request_pacing(session_key)
    try:
        full_text, blocks = tf.call_with_filter(session_key, prompt)   # ← managed call
        output_toks  = max(1, len(full_text.split()))
        stop_reason  = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
    except Exception as e:
        return jsonify({"type": "error", "error": {"type": "api_error", "message": str(e)}}), 500
    finally:
        mark_request_finished(lock)

    return jsonify({
        "id":            f"msg_{uuid.uuid4().hex[:24]}",
        "type":          "message",
        "role":          "assistant",
        "content":       blocks,
        "model":         model,
        "stop_reason":   stop_reason,
        "stop_sequence": None,
        "usage":         {"input_tokens": input_tokens, "output_tokens": output_toks},
    })


@app.route("/v1/messages/count_tokens", methods=["POST"])
def count_tokens():
    body     = request.get_json(force=True)
    messages = body.get("messages", [])
    system   = body.get("system", "")
    tools    = body.get("tools", [])

    if isinstance(system, list):
        system = "\n".join(
            b.get("text", "") for b in system
            if isinstance(b, dict) and b.get("type") == "text"
        )

    prompt    = build_prompt(system, messages, tools)
    estimated = max(1, int(len(prompt.split()) * 1.3))
    return jsonify({"input_tokens": estimated})


@app.route("/v1/models", methods=["GET"])
def models():
    return jsonify({
        "data": [
            {"id": "claude-opus-4-5",          "object": "model"},
            {"id": "claude-sonnet-4-20250514",  "object": "model"},
            {"id": "claude-haiku-4-5-20251001", "object": "model"},
        ],
        "object": "list",
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "wasm":   os.path.exists(WASM_PATH),
        "tokens": len(_token_pool),
        "current_token_index": _token_pool._index,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _token_pool:
        print(
            "Error: DEEPSEEK_TOKEN not set.\n"
            "  export DEEPSEEK_TOKEN='token1,token2,token3'\n\n"
            "  Get it: DevTools → Network → any DeepSeek request → Authorization header\n"
            "  Multiple tokens separated by commas for automatic rotation on server busy.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(WASM_PATH):
        print(
            f"Error: WASM file not found at {WASM_PATH}\n"
            "  curl -L 'https://github.com/Fundiman/dskpp/raw/refs/heads/main/wasm/sha3_wasm_bg.7b9ca65ddd.wasm'"
            " -o sha3_wasm_bg.wasm",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Loading WASM solver...", end=" ", flush=True)
    hasher = DeepSeekHash(WASM_PATH)
    print("OK")

    model_label  = "fast (model_type=default)" if _CLI_ARGS.model == "fast" else "expert (model_type=null)"
    search_label = "enabled" if _CLI_ARGS.search else "disabled"
    think_label  = "enabled" if _CLI_ARGS.think  else "disabled"
    token_label  = f"{len(_token_pool)} token(s) loaded (rotation on server-busy {'enabled' if len(_token_pool) > 1 else 'disabled — add more tokens to enable'})"

    print(f"\nDeepSeek proxy listening on http://0.0.0.0:{PORT}")
    print(f"  model   : {model_label}")
    print(f"  search  : {search_label}")
    print(f"  thinking: {think_label}")
    print(f"  tokens  : {token_label}")
    print(f"\nIn your shell:")
    print(f'  export ANTHROPIC_BASE_URL="http://localhost:{PORT}"')
    print(f'  export ANTHROPIC_API_KEY="local-proxy-key"')
    print()

    app.run(host="0.0.0.0", port=PORT, threaded=True)
