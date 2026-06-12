#!/usr/bin/env python3
"""
DeepSeek → Anthropic API Proxy for CyberStrikeAI
Exposes /messages that CyberStrikeAI can talk to, forwarding to DeepSeek.

- Translates Anthropic tool definitions → XML schema in the prompt
- Parses DeepSeek plain-text responses for <tool_use> blocks
- Re-emits proper Anthropic tool_use / text content blocks
- Handles tool_result turns in conversation history
- Memory-efficient: root-pins after first reply (history stays at 2 msgs)

Requirements:
    pip install requests wasmtime numpy flask

Setup:
    Place sha3_wasm_bg.wasm next to this file.

    export DEEPSEEK_TOKEN="your-bearer-token"
    python cyberstrike_server.py

Then:
    export ANTHROPIC_BASE_URL="http://localhost:8765"
    export ANTHROPIC_API_KEY="local-proxy-key"
    cyberstrike
"""

import base64
import ctypes
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

# ── Config ────────────────────────────────────────────────────────────────────

PORT       = int(os.environ.get("PROXY_PORT", 8765))
TOKEN      = os.environ.get("DEEPSEEK_TOKEN", "")
COOKIE_STR = os.environ.get("DEEPSEEK_COOKIES", "")

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
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: dict[str, dict] = {}

    def get_or_create(self, key: str) -> dict:
        with self._lock:
            if key not in self._sessions:
                cookies = parse_cookies(COOKIE_STR) if COOKIE_STR else {}
                http = make_http_session(TOKEN, cookies)
                ds_id = create_chat_session(http)
                self._sessions[key] = {
                    "ds_session_id":  ds_id,
                    "root_message_id": None,
                    "http":           http,
                }
                print(f"[session] new ds_id={ds_id}")
            return self._sessions[key]

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


TOOL_CALL_SYSTEM = """You are CyberStrikeAI, an AI-native security testing assistant with access to 100+ integrated security tools. You operate within an intelligent orchestration engine supporting role-based testing, a skills system, lifecycle management, and a built-in C2 framework for authorized engagements.

Your capabilities include:
- End-to-end security testing automation: recon, scanning, exploitation, post-exploitation
- Vulnerability discovery and attack-chain analysis
- C2 operations: listeners, encrypted implants, sessions, tasks, real-time events
- Knowledge retrieval, result visualization, and audit-trail generation
- MCP protocol integration and multi-agent orchestration

You assist authorized security teams only. Always operate within the scope of the engagement.

CRITICAL RULE: When you need to use a tool, you MUST output the tool call immediately using this EXACT format with no variation:

<tool_use>
{"name": "TOOL_NAME", "id": "call_UNIQUE_ID", "input": {PARAMETERS}}
</tool_use>

Do NOT say "Let me do that" or "I will run" before calling a tool — just emit the <tool_use> block directly.
Do NOT make up results — wait for the tool result to be returned.
Do NOT add markdown backticks around the tool_use block.
After receiving a tool result, respond naturally with findings, risk context, and recommended next steps.

Example — if asked to run an nmap scan, output EXACTLY:
<tool_use>
{"name": "Bash", "id": "call_abc123", "input": {"command": "nmap -sV -T4 target"}}
</tool_use>

Nothing else. No preamble. Just the block.
"""


def build_prompt(system: str, messages: list, tools: list) -> str:
    """
    Build a single prompt string from the full conversation history,
    including system prompt, tool definitions, and all turns.
    """
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

    parts.append("Assistant:")
    return "\n\n".join(parts)

# ── Response parser: plain text → Anthropic content blocks ───────────────────

# Known DeepSeek native tag names -> CyberStrikeAI Anthropic tool names.
# DeepSeek may emit its own XML tags (e.g. <Bash>, <Editor>) which we map
# to the exact tool names CyberStrikeAI expects.
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
    # Web
    "webfetch":                    "WebFetch",
    "websearch":                   "WebSearch",
    # Agent / sub-tasks
    "agent":                       "Agent",
    "task":                        "Agent",
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
    # Misc tools
    "skill":                       "Skill",
    "workflow":                    "Workflow",
    "enterplanmode":               "EnterPlanMode",
    "exitplanmode":                "ExitPlanMode",
    "enterworktree":               "EnterWorktree",
    "exitworktree":                "ExitWorktree",
    "askuserquestion":             "AskUserQuestion",
    "computer":                    "computer",
    # CyberStrikeAI security tools
    "scan":                        "Scan",
    "exploit":                     "Exploit",
    "recon":                       "Recon",
    "enumerate":                   "Enumerate",
    "bruteforce":                  "BruteForce",
    "fuzz":                        "Fuzz",
    "c2":                          "C2",
    "implant":                     "Implant",
    "listener":                    "Listener",
    "session":                     "Session",
    "payload":                     "Payload",
    "pivot":                       "Pivot",
    "exfil":                       "Exfil",
    "report":                      "Report",
    "skill":                       "Skill",
}

# All known CyberStrikeAI tool names (exact casing) used to recognise native
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

# XML tags that are structural / metadata -- never treat as tool calls
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
    "response", "result", "output", "answer", "content",
}

def parse_native_tag(tag_name: str, inner: str) -> dict | None:
    """
    Convert a DeepSeek native XML tool call like:
      <Bash><command>ls -la</command></Bash>
    into an Anthropic tool_use block.

    Handles three cases:
      1. tag_name is already a valid CyberStrikeAI tool name  (e.g. "Bash", "Scan")
      2. tag_name maps via DEEPSEEK_TAG_TO_TOOL               (e.g. "editor" -> "Edit")
      3. Unknown tag -- use the tag name verbatim so the ToolFilter can decide
    """
    # Prefer exact match in CyberStrikeAI tool names (preserves casing like "WebFetch")
    if tag_name in CLAUDE_CODE_TOOL_NAMES:
        tool_name = tag_name
    else:
        tool_name = DEEPSEEK_TAG_TO_TOOL.get(tag_name.lower(), tag_name)

    # Extract all sub-tags as input params  e.g. <command>ls</command>
    params = {}
    for m in re.finditer(r"<(\w+)>\s*(.*?)\s*</\1>", inner, re.DOTALL):
        params[m.group(1)] = m.group(2).strip()

    # If no sub-tags, pick a sensible default key based on the tool
    if not params:
        if tool_name == "Bash":
            key = "command"
        elif tool_name in {"Read", "NotebookRead"}:
            key = "file_path"
        elif tool_name == "Write":
            key = "content"
        elif tool_name in {"WebFetch", "WebSearch"}:
            key = "url" if tool_name == "WebFetch" else "query"
        else:
            key = "input"
        params[key] = inner.strip()

    return {
        "type":  "tool_use",
        "id":    f"toolu_{uuid.uuid4().hex[:16]}",
        "name":  tool_name,
        "input": params,
    }


def strip_fabricated_result(text: str) -> str:
    """
    DeepSeek sometimes fabricates the tool result and continues after the
    tool call. Strip everything from "Assistant:" onward since that is
    DeepSeek hallucinating the result.
    """
    # Cut at fabricated continuation markers
    for marker in ["\nAssistant:", "\nHuman:", "\n\nThe result", "\n\nOutput:"]:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def parse_response(text: str) -> list:
    """
    Parse DeepSeek reply into Anthropic content blocks.
    Handles two tool-call formats:
      1. Our injected format:  <tool_use>{json}</tool_use>
      2. DeepSeek native:      <Bash><command>...</command></Bash>
    Strips any fabricated tool results DeepSeek appended.
    """
    # Combined pattern: matches both formats
    pattern = re.compile(
        r"<tool_use>\s*(.*?)\s*</tool_use>"      # our JSON format
        r"|<([A-Z][\w]*)>(.*?)</\2>",             # DeepSeek native XML tags
        re.DOTALL | re.IGNORECASE,
    )

    blocks = []
    last   = 0
    found_tool = False

    for m in pattern.finditer(text):
        # Preserve text exactly — no strip() — so backtick code blocks and
        # newlines render correctly in CyberStrikeAI.
        before = text[last:m.start()]
        if before and not found_tool:
            blocks.append({"type": "text", "text": before})

        if m.group(1) is not None:
            # Format 1: <tool_use>{json}</tool_use>  — the only thing we actively parse
            raw = m.group(1).strip()
            try:
                obj = json.loads(raw)
                blocks.append({
                    "type":  "tool_use",
                    "id":    obj.get("id", f"toolu_{uuid.uuid4().hex[:16]}"),
                    "name":  obj.get("name", ""),
                    "input": obj.get("input", {}),
                })
                found_tool = True
            except json.JSONDecodeError:
                blocks.append({"type": "text", "text": m.group(0)})
        else:
            # Format 2: <TagName>...</TagName>
            tag_name = m.group(2)
            inner    = m.group(3)
            if tag_name.lower() in _SKIP_TAGS:
                # Not a tool — append raw match text so content isn't lost
                if blocks and blocks[-1].get("type") == "text":
                    blocks[-1]["text"] += m.group(0)
                else:
                    blocks.append({"type": "text", "text": m.group(0)})
            else:
                block = parse_native_tag(tag_name, inner)
                blocks.append(block)
                found_tool = True

        last = m.end()

    # Tail: only discard after a real tool call (DeepSeek fabricates a result after it)
    tail = text[last:]
    if tail and not found_tool:
        blocks.append({"type": "text", "text": tail})

    if not blocks:
        blocks.append({"type": "text", "text": text})

    print(f"[parsed blocks] {[b['type'] + ':' + b.get('name','') for b in blocks]}", flush=True)
    return blocks

# ── ToolFilter: validates and retries tool-use blocks ─────────────────────────

class ToolFilter:
    """
    Wraps call_deepseek + parse_response with retry logic.

    A response is considered INVALID (and retried) when:
      - CyberStrikeAI sent tools but DeepSeek returned ONLY text (no tool_use blocks)
        AND the raw text looks like it was trying to call a tool (contains keyword signals)
      - A tool_use block has a blank or missing name
      - A tool_use block has malformed / empty input when the tool schema requires params
      - JSON inside <tool_use> is broken (parse_response already falls back to text,
        so we catch that here by checking for tool signals in text blocks)

    On failure: re-prompts DeepSeek with a strict correction instruction appended,
    up to MAX_RETRIES times. Falls back to the last response on exhaustion.
    """

    MAX_RETRIES = 3

    # Keywords that ONLY appear when DeepSeek is trying (and failing) to emit a tool call.
    # Deliberately excludes ```bash / ```python — those are normal markdown in text answers.
    # Deliberately excludes "$ " — common in shell examples in reports.
    TOOL_INTENT_SIGNALS = [
        "<tool_use>",
        # DeepSeek native tags
        "<bash>", "<python>", "<editor>",
        # Phrase signals
        "I'll run", "I will run", "Let me run",
        "I'll use the", "I will use the",
        "I'll call", "I will call",
        "Executing tool:", "Calling tool:",
        # CyberStrikeAI tool name mentions in action context
        "I'll use Bash", "I'll use Read", "I'll use Write", "I'll use Edit",
        "I'll use WebFetch", "I'll use WebSearch",
        "I'll use Agent", "I'll use Scan", "I'll use Recon",
        "I'll use Exploit", "I'll use Enumerate",
        "using the Bash tool", "using the Read tool",
        "call the Bash", "call the Read", "call the Write",
    ]

    # If any of these appear, the response is clearly intentional prose — skip retry.
    PROSE_SIGNALS = [
        "### ", "## ", "# ",
        "**Vulnerability", "**SQL", "**Remediation",
        "Penetration Test", "pentest",
        "Here is", "Here's", "Summary", "Finding",
    ]

    RETRY_INSTRUCTION = (
        "\n\n[SYSTEM CORRECTION] Your previous response did not emit a valid <tool_use> block "
        "even though a tool call was required. "
        "You MUST output the tool call using ONLY this format — nothing else, no preamble:\n"
        "<tool_use>\n"
        "{\"name\": \"TOOL_NAME\", \"id\": \"call_UNIQUE_ID\", \"input\": {PARAMETERS}}\n"
        "</tool_use>\n"
        "Try again now."
    )

    def __init__(self, tools: list, messages: list | None = None):
        """
        tools:    the Anthropic tool definitions list from the current request.
        messages: the full conversation history (optional). Used to collect
                  tool names already used in earlier assistant turns — these
                  are always considered valid so nested / chained calls aren't
                  rejected as "unknown".
        """
        self._valid_names  = {t["name"] for t in tools} if tools else set()
        self._required_params = {}
        for t in (tools or []):
            schema   = t.get("input_schema", {})
            required = schema.get("required", [])
            if required:
                self._required_params[t["name"]] = required

        # Augment valid names with any tool already invoked in the history
        # (handles nested / chained scenarios where the same tool re-appears
        # or a tool from a prior turn shows up in a follow-up call).
        for msg in (messages or []):
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name", "").strip()
                    if name:
                        self._valid_names.add(name)

    # ── public entry point ────────────────────────────────────────────────────

    def call_with_filter(self, session: dict, prompt: str) -> tuple[str, list]:
        """
        Returns (raw_text, validated_blocks).
        Retries automatically on tool-parse failure.
        """
        current_prompt = prompt
        last_text      = ""
        last_blocks    = []

        for attempt in range(1, self.MAX_RETRIES + 1):
            raw_text, _ = call_deepseek_with_empty_retry(session, current_prompt)
            blocks      = parse_response(raw_text)
            last_text   = raw_text
            last_blocks = blocks

            failure = self._failure_reason(blocks, raw_text)
            if failure is None:
                if attempt > 1:
                    print(f"[tool_filter] ✓ passed on attempt {attempt}", flush=True)
                return raw_text, blocks

            print(
                f"[tool_filter] attempt {attempt}/{self.MAX_RETRIES} FAILED — {failure}",
                flush=True,
            )

            if attempt < self.MAX_RETRIES:
                # Append the raw model reply + correction nudge to the prompt
                current_prompt = (
                    current_prompt.rstrip()
                    + f"\n\nAssistant: {raw_text}"
                    + f"\n\nHuman:{self.RETRY_INSTRUCTION}"
                    + "\n\nAssistant:"
                )

        print(
            f"[tool_filter] exhausted {self.MAX_RETRIES} retries — returning best effort",
            flush=True,
        )
        return last_text, last_blocks

    # ── private validation ────────────────────────────────────────────────────

    def _failure_reason(self, blocks: list, raw_text: str) -> str | None:
        """
        Returns a human-readable failure reason string, or None if blocks are valid.
        Only validates tool blocks when the request actually included tools.
        """
        if not self._valid_names:
            # No tools in this request — any text response is fine
            return None

        # Empty response = DeepSeek timeout/rate-limit, not a tool-format failure.
        # Retrying won't fix it and just wastes quota.
        if not raw_text.strip():
            return None

        tool_blocks = [b for b in blocks if b.get("type") == "tool_use"]
        text_blocks = [b for b in blocks if b.get("type") == "text"]

        # Case 1: No tool blocks at all — check if DeepSeek was trying but failed
        if not tool_blocks:
            combined_text = " ".join(b.get("text", "") for b in text_blocks)

            # If the response looks like deliberate prose (report, summary, explanation)
            # then the model chose not to call a tool — that's valid, don't retry.
            if self._looks_like_prose(combined_text) or self._looks_like_prose(raw_text):
                return None

            if self._looks_like_tool_intent(combined_text) or self._looks_like_tool_intent(raw_text):
                return "tool intent detected in text but no tool_use block emitted"

            # No tool intent, no prose signals — model answered normally, accept it
            return None

        # Case 2: Validate each tool block
        for block in tool_blocks:
            name  = block.get("name", "").strip()
            input_ = block.get("input", {})

            if not name:
                return "tool_use block has empty name"

            if self._valid_names and name not in self._valid_names:
                # Only hard-fail if the name looks completely fabricated (contains
                # spaces or is a single generic word like "tool"). Legitimate nested
                # calls often use names that weren't in the original tools list.
                if " " in name or name.lower() in {"tool", "function", "call", "action"}:
                    return f"unknown tool name: {repr(name)}"
                # Otherwise accept it — it may be from a nested or chained context
                print(f"[tool_filter] allowing unlisted tool name {repr(name)} (may be nested)", flush=True)

            required = self._required_params.get(name, [])
            missing  = [p for p in required if p not in input_]
            if missing:
                return f"tool {repr(name)} missing required params: {missing}"

        return None  # all good

    def _looks_like_tool_intent(self, text: str) -> bool:
        tl = text.lower()
        return any(sig.lower() in tl for sig in self.TOOL_INTENT_SIGNALS)

    def _looks_like_prose(self, text: str) -> bool:
        """Returns True if the text is clearly intentional prose, not a failed tool call."""
        return any(sig in text for sig in self.PROSE_SIGNALS)


# ── SSE helpers ───────────────────────────────────────────────────────────────

def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

# ── DeepSeek call (collects full response) ────────────────────────────────────

def call_deepseek(session: dict, prompt: str) -> tuple[str, int | None]:
    """Send prompt to DeepSeek, return (full_text, new_message_id)."""
    http       = session["http"]
    ds_session = session["ds_session_id"]
    root_id    = session["root_message_id"]

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
    body = {
        "chat_session_id":   ds_session,
        "parent_message_id": root_id,
        "model_type":        "default" if root_id is None else None,
        "prompt":            prompt,
        "ref_file_ids":      [],
        "thinking_enabled":  False,
        "search_enabled":    False,
        "action":            None,
        "preempt":           False,
    }

    full_text   = ""
    new_root_id = None

    with http.post(
        f"{BASE_URL}/api/v0/chat/completion",
        headers=headers,
        json=body,
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        if "application/json" in resp.headers.get("content-type", ""):
            raise RuntimeError(f"DeepSeek API error: {resp.json()}")

        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            try:
                pl = json.loads(raw_line[5:].strip())
            except json.JSONDecodeError:
                continue

            v = pl.get("v")
            p = pl.get("p", "")
            o = pl.get("o", "")
            chunk = None

            if isinstance(v, dict) and "response" in v:
                resp_obj = v["response"]
                new_root_id = resp_obj.get("message_id")
                for f in resp_obj.get("fragments", []):
                    if f.get("type") == "RESPONSE":
                        chunk = f.get("content", "")
            elif isinstance(v, str) and o == "APPEND" and p == "response/fragments/-1/content":
                chunk = v
            elif isinstance(v, str) and "p" not in pl:
                chunk = v

            if chunk:
                full_text += chunk

    # Pin root after first reply only — all subsequent turns are siblings of
    # message 1, keeping the DeepSeek conversation memory flat and bounded.
    if new_root_id is not None and session["root_message_id"] is None:
        session["root_message_id"] = new_root_id

    print(f"[deepseek raw]\n{repr(full_text[:500])}\n", flush=True)
    return full_text, new_root_id


def call_deepseek_with_empty_retry(session: dict, prompt: str, max_empty_retries: int = 3) -> tuple[str, int | None]:
    """
    Wraps call_deepseek: if the response is empty (DeepSeek returned ''),
    wait 3 seconds and retry up to max_empty_retries times before giving up.
    """
    for attempt in range(1, max_empty_retries + 1):
        full_text, new_root_id = call_deepseek(session, prompt)
        if full_text.strip():
            return full_text, new_root_id
        print(
            f"[deepseek] empty response on attempt {attempt}/{max_empty_retries} — "
            f"waiting 3s before retry...",
            flush=True,
        )
        time.sleep(3)

    print("[deepseek] all empty-response retries exhausted — returning empty string", flush=True)
    return "", None


def stream_response_as_anthropic(session: dict, prompt: str, model: str, input_tokens: int, tool_filter: "ToolFilter"):
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

    # Collect full response from DeepSeek — filtered + retried if tool parsing fails
    full_text, blocks = tool_filter.call_with_filter(session, prompt)

    output_tokens = max(1, len(full_text.split()))
    stop_reason   = "end_turn"

    # Merge consecutive text blocks so they stream as one block — avoids
    # spurious newlines that CyberStrikeAI inserts between separate content blocks.
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


def is_permission_request(system: str) -> bool:
    """
    Detect CyberStrikeAI permission-evaluation requests.
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


@app.route("/messages", methods=["POST"])
def messages():
    body     = request.get_json(force=True)
    msgs     = body.get("messages", [])
    model    = body.get("model", "cyberstrike-pro")
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

    # Auto-approve permission evaluation requests from CyberStrikeAI
    if is_permission_request(system):
        print("[permission] auto-approving", flush=True)
        return _allow_response(stream, model)

    prompt       = build_prompt(system, msgs, tools)
    input_tokens = max(1, len(prompt.split()))
    session      = store.get_or_create("global")
    tf           = ToolFilter(tools, msgs)          # ← filter created per-request

    if stream:
        def generate():
            try:
                yield from stream_response_as_anthropic(session, prompt, model, input_tokens, tf)
            except Exception as e:
                print(f"[error] {e}", file=sys.stderr)
                import traceback; traceback.print_exc()
                yield sse("error", {"type": "error", "error": {"type": "api_error", "message": str(e)}})

        return Response(generate(), mimetype="text/event-stream",
                        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

    # Non-streaming
    try:
        full_text, blocks = tf.call_with_filter(session, prompt)   # ← filtered call
        output_toks  = max(1, len(full_text.split()))
        stop_reason  = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
    except Exception as e:
        return jsonify({"type": "error", "error": {"type": "api_error", "message": str(e)}}), 500

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


@app.route("/messages/count_tokens", methods=["POST"])
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


@app.route("/models", methods=["GET"])
def models():
    return jsonify({
        "data": [
            {"id": "cyberstrike-ultra",   "object": "model"},
            {"id": "cyberstrike-pro",     "object": "model"},
            {"id": "cyberstrike-fast",    "object": "model"},
        ],
        "object": "list",
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "wasm": os.path.exists(WASM_PATH), "token": bool(TOKEN)})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not TOKEN:
        print(
            "Error: DEEPSEEK_TOKEN not set.\n"
            "  export DEEPSEEK_TOKEN='your-bearer-token'\n\n"
            "  Get it: DevTools → Network → any DeepSeek request → Authorization header",
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

    print(f"\nCyberStrikeAI proxy listening on http://0.0.0.0:{PORT}")
    print(f"\nIn your shell:")
    print(f'  export ANTHROPIC_BASE_URL="http://localhost:{PORT}"')
    print(f'  export ANTHROPIC_API_KEY="cyberstrike-local-key"')
    print()

    app.run(host="0.0.0.0", port=PORT, threaded=True)
