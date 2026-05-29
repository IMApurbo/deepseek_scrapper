"""
chat.deepseek.com → Anthropic API Proxy
========================================
Exposes a local HTTP server that speaks the Anthropic Messages API,
but routes every request through chat.deepseek.com via Playwright.

Usage:
    pip install playwright flask rich
    playwright install chromium
    python deepseek_server_fixed.py

Then in another shell:
    export ANTHROPIC_BASE_URL="http://localhost:8765"
    export ANTHROPIC_API_KEY="local-proxy-key"
    claude   # or any tool that uses the Anthropic SDK

Supported endpoints:
    POST /v1/messages   (streaming + non-streaming)
    GET  /v1/models
    GET  /health

Claude Code compatibility fixes applied:
  - Reasoning blocks returned as a separate top-level `thinking` content block
    (type="thinking") so Claude Code ignores them cleanly, not injected into text
  - Streaming: proper SSE framing with correct anthropic-beta header support
  - Token counts use character-based estimation (closer to real BPE)
  - system prompt deduplication guard
  - anthropic-version header no longer required (relaxed auth)
"""

import json
import threading
import time
import uuid
from datetime import datetime

import re
import sys
import textwrap

from flask import Flask, request, Response, jsonify

# ── reuse the scraper (must be in same folder) ────────────────
from deepseek_scraper import DeepSeekScraper


# ─────────────────────────────────────────────────────────────
# Debug Logger
# ─────────────────────────────────────────────────────────────

DEBUG_OUTPUT = True   # Set to False to silence all debug output

def _dbg_separator(label: str = "", char: str = "─", width: int = 72):
    if not DEBUG_OUTPUT:
        return
    if label:
        side = (width - len(label) - 2) // 2
        print(f"\n{char * side} {label} {char * (width - side - len(label) - 2)}", file=sys.stderr, flush=True)
    else:
        print(char * width, file=sys.stderr, flush=True)

def _dbg(label: str, value, indent: int = 2):
    """Pretty-print a debug value with a labelled header."""
    if not DEBUG_OUTPUT:
        return
    _dbg_separator(label)
    if isinstance(value, (dict, list)):
        formatted = json.dumps(value, indent=indent, ensure_ascii=False)
        print(formatted, file=sys.stderr, flush=True)
    elif isinstance(value, str):
        # Show whitespace-visible representation for exact format inspection
        print(repr(value) if len(value) < 200 else value, file=sys.stderr, flush=True)
        if "\n" in value:
            _dbg_separator("rendered")
            print(value, file=sys.stderr, flush=True)
    else:
        print(repr(value), file=sys.stderr, flush=True)
    _dbg_separator(char="─")
    print("", file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────
# Response Cleaning  (ported from hackers-ai FreeLLM / ResponseGenerator)
# ─────────────────────────────────────────────────────────────

def _clean_response(text: str, mode: str = "text") -> str:
    """
    Scrub common LLM output artefacts so callers always receive clean prose.

    mode="text"   – full prose cleaning (headers, bold, fences, prompts …)
    mode="json"   – extract a JSON payload, strip surrounding fences/text
    mode="code"   – extract code from a fenced block, normalise indentation
    mode="light"  – only strip markdown links and shell prompt artefacts

    All modes applied in order of increasing aggression; heavier modes apply
    everything the lighter modes do plus their own steps.
    """

    if not text:
        return text

    # ── 1. Normalise line endings ─────────────────────────────
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # ── 2. Strip markdown hyperlinks  [label](url) → label ───
    #    (applied by FreeLLM.ask() on every raw response)
    text = re.sub(r'\[([^\]]+)\]\([^\)]*\)', r'\1', text)

    if mode == "light":
        return text.strip()

    # ── 3. Remove shell prompt artefacts the LLM hallucinates ─
    #    e.g.  (kali㉿kali)-[~]$   or   root@host:/path#
    text = re.sub(
        r'^\s*\([^)]*㉿[^)]*\)-\[.*?\]\$\s*',
        '', text, flags=re.MULTILINE
    )
    text = re.sub(
        r'^\s*root@\S+:[^\$#]*[#\$]\s*',
        '', text, flags=re.MULTILINE
    )

    # ── 4. Strip role-prefix bleed  (USER: / AI: lines the model copies) ─
    text = re.sub(r'^\s*USER:.*?\n', '', text, flags=re.IGNORECASE)

    # ── 5. JSON mode: extract content of ```json … ``` block ──
    if mode == "json":
        # Prefer a fenced ```json block
        fenced = re.search(r'```(?:json)?\s*\n?([\s\S]*?)```', text)
        if fenced:
            text = fenced.group(1).strip()
        else:
            # Strip any leftover fence markers
            text = re.sub(r'^```[a-z]*\n?', '', text).strip()
            text = re.sub(r'\n?```$', '', text).strip()
        # Locate the first { or [ in case there's prose preamble
        m = re.search(r'[{\[]', text)
        if m:
            text = text[m.start():]

        # Fast path: already valid JSON
        try:
            json.loads(text)
            return text.strip()
        except (json.JSONDecodeError, ValueError):
            pass

        # ── Agent-definition repair ───────────────────────────
        # Claude Code agent creation returns {"identifier":…,"whenToUse":…,"systemPrompt":…}
        # DeepSeek often puts unescaped " inside string values (e.g. user: "hello")
        # which breaks json.loads.  Extract each value by known key boundaries,
        # then re-serialize cleanly with json.dumps so all quotes are properly escaped.
        _AGENT_KEYS = ['identifier', 'whenToUse', 'systemPrompt']
        if all(f'"{k}"' in text for k in _AGENT_KEYS):
            result = {}
            for i, key in enumerate(_AGENT_KEYS):
                key_pat = re.compile(r'"' + key + r'"\s*:\s*"')
                km = key_pat.search(text)
                if not km:
                    break
                val_start = km.end()
                if i + 1 < len(_AGENT_KEYS):
                    next_key = _AGENT_KEYS[i + 1]
                    end_pat = re.compile(r'",\s*\n?\s*"' + next_key + r'"')
                    em = end_pat.search(text, val_start)
                    raw_val = text[val_start: em.start()] if em else text[val_start:]
                else:
                    em = re.search(r'"\s*\n?\s*\}', text[val_start:])
                    raw_val = text[val_start: val_start + em.start()] if em else text[val_start:]
                # raw_val is the JSON-encoded string content; decode it
                try:
                    decoded = json.loads('"' + raw_val + '"')
                except (json.JSONDecodeError, ValueError):
                    decoded = raw_val  # keep as-is if decode fails
                result[key] = decoded
            if len(result) == len(_AGENT_KEYS):
                return json.dumps(result)

        # ── Generic fallback: trim trailing garbage ───────────
        for end_char in ('}', ']'):
            idx = text.rfind(end_char)
            while idx > 0:
                candidate = text[:idx + 1]
                try:
                    json.loads(candidate)
                    return candidate.strip()
                except (json.JSONDecodeError, ValueError):
                    idx = text.rfind(end_char, 0, idx)

        return text.strip()

    # ── 6. Code mode: extract + normalise indentation ─────────
    if mode == "code":
        fenced = re.search(r'```[^\n]*\n(.*?)(?:```|$)', text, re.DOTALL)
        if fenced:
            # Check whether the language tag itself leaked into the group
            fence_line = re.search(r'```([^\n]+)\n', text)
            if fence_line:
                tag = fence_line.group(1).strip()
                if tag and not re.match(r'^[a-zA-Z0-9]+$', tag):
                    text = tag + "\n" + fenced.group(1).strip()
                else:
                    text = fenced.group(1).strip()
            else:
                text = fenced.group(1).strip()
        else:
            text = re.sub(r'^```[a-z]*\n?', '', text).strip()
            text = re.sub(r'\n?```$', '', text).strip()

        # Remove lone language-keyword lines ("python", "python3", "py")
        cleaned_lines = [
            line for line in text.splitlines()
            if line.strip() not in ("python", "python3", "py")
        ]
        text = "\n".join(cleaned_lines)

        # Normalise tabs → 4 spaces (consistent with PythonExecutor._clean_code)
        fixed = []
        for line in text.splitlines():
            stripped  = line.lstrip("\t")
            n_tabs    = len(line) - len(stripped)
            stripped2 = stripped.lstrip(" ")
            n_spaces  = len(stripped) - len(stripped2)
            fixed.append("    " * n_tabs + " " * n_spaces + stripped2)
        text = "\n".join(fixed).rstrip("\n") + "\n"
        return text

    # ── 7. Text / full mode: strip Markdown decoration ────────
    # Remove all fenced code blocks (```…```) – keep just the inner text
    text = re.sub(r'```[a-zA-Z]*\n?', '', text)
    text = re.sub(r'```', '', text)

    # **bold** → plain
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)

    # *italic* / _italic_ → plain
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)

    # ## Headers → plain text (strip leading #+ and space)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Horizontal rules ---  ═══  ───  → empty line
    text = re.sub(r'^[-=─═]{3,}\s*$', '', text, flags=re.MULTILINE)

    # Remove lone language-keyword lines that slip through after fence removal
    cleaned = []
    for line in text.splitlines():
        if line.strip() in ("python", "python3", "py", "bash", "sh"):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)

    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

# ─────────────────────────────────────────────────────────────
# Global scraper instance (one browser, one session)
# ─────────────────────────────────────────────────────────────

_scraper: DeepSeekScraper | None = None
_scraper_lock   = threading.Lock()
_headless_flag  = False
_search_flag    = False
_deepthink_flag = False
_expert_flag    = False


def get_scraper() -> DeepSeekScraper:
    global _scraper
    with _scraper_lock:
        if _scraper is None:
            _scraper = DeepSeekScraper(
                headless=_headless_flag,
                enable_search=_search_flag,
                enable_deepthink=_deepthink_flag,
                enable_expert=_expert_flag,
            )
            _scraper.start()
        return _scraper


# ─────────────────────────────────────────────────────────────
# Token estimation
# ─────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough BPE estimate: ~4 chars per token, minimum 1."""
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────────────────────
# Message Formatting Helpers
# ─────────────────────────────────────────────────────────────

def _content_to_text(content) -> str:
    """
    Normalise a message content field into plain text for DeepSeek.
    Uses DeepSeek's native <tool_call>/<tool_result> tags so the
    model recognises its own output format in multi-turn history.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                t = block.get("text", "").strip()
                if t:
                    parts.append(t)
            elif btype == "tool_use":
                name = block.get("name", "tool")
                inp  = block.get("input", {})
                param_lines = "\n".join(
                    f'<parameter name="{k}">{v}</parameter>' for k, v in inp.items()
                )
                parts.append(f'<tool_call name="{name}">\n{param_lines}\n</tool_call>')
            elif btype == "tool_result":
                tool_id = block.get("tool_use_id", "")
                inner = block.get("content", "")
                if isinstance(inner, list):
                    texts = [rb.get("text", "") for rb in inner
                             if isinstance(rb, dict) and rb.get("type") == "text"]
                    result_text = "\n".join(texts)
                else:
                    result_text = str(inner)
                parts.append(f'<tool_result for="{tool_id}">\n{result_text}\n</tool_result>')
        return "\n".join(parts)

    return str(content)


def _messages_to_prompt(messages: list) -> str:
    """
    Flatten an Anthropic `messages` array into a plain-text prompt.
    system entries are already prepended by _extract_prompt — do NOT add them again.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        text = _content_to_text(msg.get("content", ""))
        if role == "system":
            # guard: should not reach here, but handle gracefully
            parts.append(f"[System Instructions]\n{text}")
        elif role == "user":
            parts.append(f"Human: {text}")
        elif role == "assistant":
            parts.append(f"Assistant: {text}")
        else:
            parts.append(text)
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _tools_to_xml(tools: list) -> str:
    """
    Describe available tools to DeepSeek and instruct it to respond
    using JSON code blocks — a format DeepSeek knows reliably from
    training data, unlike custom XML schemas.
    """
    if not tools:
        return ""

    import json as _json

    # Build a compact tool listing
    tool_lines = ["## Available Tools\n"]
    for t in tools:
        name     = t.get("name", "")
        desc     = t.get("description", "").strip().split("\n")[0][:300]
        schema   = t.get("input_schema", {})
        props    = schema.get("properties", {})
        required = schema.get("required", [])
        params   = {}
        for pname, pinfo in props.items():
            req_mark = " (required)" if pname in required else ""
            params[pname] = pinfo.get("description", pinfo.get("type", "string"))[:120] + req_mark
        tool_lines.append(f"- **{name}**: {desc}")
        if params:
            tool_lines.append(f"  Parameters: {_json.dumps(params)}")

    tool_lines += [
        "",
        "## How to call a tool",
        "",
        "Use this exact format — one block per tool call:",
        "",
        '<tool_call name="TOOL_NAME">',
        '<parameter name="param1">value1</parameter>',
        '<parameter name="param2">value2</parameter>',
        "</tool_call>",
        "",
        "Example:",
        "",
        '<tool_call name="Bash">',
        '<parameter name="command">ls -la</parameter>',
        '<parameter name="description">list files</parameter>',
        "</tool_call>",
        "",
        "RULES:",
        "- Use <tool_call name=\"TOOL_NAME\"> with the exact tool name.",
        "- Add one <parameter name=\"key\">value</parameter> per parameter.",
        "- You may write text before or after the tool_call block.",
        "- For multiple calls, emit multiple tool_call blocks in sequence.",
        "- If no tool is needed, respond normally with no tool_call block.",
    ]
    return "\n".join(tool_lines)


def _extract_prompt(body: dict) -> str:
    """
    Build the final prompt string to send to DeepSeek.

    Rules:
      • system is the TOP-LEVEL "system" field in the Anthropic request body,
        NOT a message with role="system" in the messages array. Treat them the
        same but deduplicate so the system text only appears once.
      • Tool schemas are converted to XML instructions so DeepSeek knows the
        format to use when calling tools.
      • Single user turn, no system, no tools → send raw user text.
      • Otherwise → flatten into Human:/Assistant: dialogue with system header.
    """
    messages = body.get("messages", [])
    tools    = body.get("tools", [])

    # Collect system text from top-level field (Anthropic convention)
    system_text = ""
    raw_system = body.get("system", "")
    if isinstance(raw_system, str):
        system_text = raw_system.strip()
    elif isinstance(raw_system, list):
        # system can be a list of content blocks in newer API versions
        # Skip billing/internal header blocks (x-anthropic-billing-header lines)
        filtered_sys_parts = []
        for blk in raw_system:
            if not isinstance(blk, dict):
                continue
            if blk.get("type") != "text":
                continue
            t = blk.get("text", "").strip()
            # Drop the internal billing/session header injected by Claude Code
            if t.startswith("x-anthropic-billing-header:"):
                continue
            if t:
                filtered_sys_parts.append(t)
        system_text = "\n\n".join(filtered_sys_parts).strip()

    # Append tool XML instructions to system prompt so DeepSeek knows the format
    tools_xml = _tools_to_xml(tools)
    if tools_xml:
        system_text = (system_text + "\n\n" + tools_xml).strip() if system_text else tools_xml

    # Filter out any role="system" messages from the messages array
    # (some SDKs inject them there; deduplicate against top-level system)
    filtered_messages = []
    for m in messages:
        if m.get("role") == "system":
            extra = _content_to_text(m.get("content", "")).strip()
            # Only append if it's different from the top-level system text
            if extra and extra not in system_text:
                system_text = (system_text + "\n\n" + extra).strip() if system_text else extra
        else:
            filtered_messages.append(m)

    user_msgs = [m for m in filtered_messages if m.get("role") == "user"]

    # Simple case: single user turn, no system, no tools → raw text
    if len(user_msgs) == 1 and len(filtered_messages) == 1 and not system_text:
        return _content_to_text(user_msgs[0].get("content", ""))

    # Multi-turn or system/tools present → structured dialogue
    all_messages = []
    if system_text:
        all_messages.append({"role": "system", "content": system_text})
    all_messages.extend(filtered_messages)
    return _messages_to_prompt(all_messages)


# ─────────────────────────────────────────────────────────────
# Response Builders
# ─────────────────────────────────────────────────────────────

# ── Tool-call parser ──────────────────────────────────────────
# PRIMARY format: ```tool JSON ``` — reliable because DeepSeek is
# trained on code blocks and handles JSON naturally.
# All XML variants kept as fallbacks for old-prompt compatibility.

# PRIMARY: ```tool {"name":…,"input":{…}} ```
_TOOL_BLOCK_RE = re.compile(
    r'```tool\s*\n(?P<body>[\s\S]*?)\n?```',
    re.DOTALL,
)
# Backward-compat: ```json {"type":"tool_use",...} ```
_FENCED_TOOL_RE = re.compile(
    r'''```(?:json)?\s*\n\s*\{[\s\S]*?"type"\s*:\s*"tool_use"[\s\S]*?\}\s*\n```''',
    re.DOTALL,
)
# <bash><command>…</command></bash>
_BASH_CMD_RE = re.compile(
    r'''<bash>\s*<command>(?P<cmd>[\s\S]*?)</command>\s*</bash>''',
    re.DOTALL,
)
# {"tool": "name", ...} blob
_TOOL_USE_JSON_RE = re.compile(
    r'''\{\s*"tool"\s*:\s*"(?P<name>[^"]+)"[\s\S]*?\}''',
    re.DOTALL,
)

# <tool_call name="X">{JSON}</tool_call> — format DeepSeek actually emits
_TOOL_CALL_TAG_RE = re.compile(
    r'<tool_call\s+name="(?P<name>[^"]+)"[^>]*>\s*(?P<body>[\s\S]*?)\s*</tool_call>',
    re.IGNORECASE | re.DOTALL,
)

# XML fallback sub-patterns
_BLOCK_RE       = re.compile(r'<(tool_?use)(?:\s+[^>]*)?>'
                              r'[\s\S]*?</\1>',
                              re.IGNORECASE | re.DOTALL)
_NAME_ATTR_RE   = re.compile(r'<tool_?use[^>]*\sname="([^"]+)"', re.IGNORECASE)
_NAME_TAG_RE    = re.compile(r'<name>([\s\S]*?)</name>', re.IGNORECASE | re.DOTALL)
_INPUT_TAG_RE   = re.compile(r'<input>([\s\S]*?)</input>', re.IGNORECASE | re.DOTALL)
_PARAM_NAMED_RE = re.compile(r'<parameter\s+name="([^"]+)"[^>]*>([\s\S]*?)</parameter>',
                              re.IGNORECASE | re.DOTALL)
_PARAM_SINGLE_RE= re.compile(r'<parameter>([\s\S]*?)</parameter>', re.IGNORECASE | re.DOTALL)
_CHILD_TAG_RE   = re.compile(r'<([a-zA-Z_][a-zA-Z0-9_]*)>([^<]*)</\1>',
                              re.IGNORECASE | re.DOTALL)


def _parse_tool_input(body: str) -> dict:
    """
    Parse tool-call body into a plain dict.

    Handles every format observed from DeepSeek:
      1. JSON object  {"command": "ls", ...}
      2. Named-attr params  <parameter name="command" index="0">ls</parameter>
      3. Simple child tags  <command>ls</command><description>list</description>
      4. Mixed/stray closing tags like </invoke> are silently dropped

    Never returns {"content": raw_xml} — that causes Claude Code to fail with
    "unexpected parameter `content`".
    """
    body = body.strip()
    if not body:
        return {}

    # Drop known stray closing tags DeepSeek appends (</invoke>, </function_calls>, etc.)
    body = re.sub(r'</(?:invoke|function_calls|parameters|functions)>\s*', '', body,
                  flags=re.IGNORECASE)
    body = body.strip()
    if not body:
        return {}

    # 1. JSON
    try:
        result = json.loads(body)
        if isinstance(result, dict):
            return result
        return {"value": result}
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. <parameter name="key" ...>value</parameter>
    named = re.findall(
        r'<parameter\s+name="([^"]+)"[^>]*>([\s\S]*?)</parameter>',
        body, re.DOTALL | re.IGNORECASE,
    )
    if named:
        return {k.strip(): v.strip() for k, v in named}

    # 3. <key>value</key>
    simple = re.findall(r'<([a-zA-Z_][a-zA-Z0-9_]*)>([\s\S]*?)</\1>', body, re.DOTALL)
    if simple:
        return {k.strip(): v.strip() for k, v in simple}

    # Last resort — shouldn't reach here if DeepSeek follows any known format
    return {}


def _extract_tool_calls(text: str):
    """
    Unified single-pass tool-call extractor.

    Handles every format DeepSeek has been observed to emit:
      A. <tool_use><name>X</name><input>{JSON}</input></tool_use>
      B. <tool_use name="X"><input>{JSON}</input></tool_use>
      C. <tool_use name="X"><cmd>val</cmd>...</tool_use>   (direct child tags)
      D. <tool_use name="X">{raw JSON}</tool_use>
      E. <tool_use><name>X</name><parameter name="p">v</parameter>...</tool_use>
      F. <tool_use><name>X</name><parameter>{JSON}</parameter></tool_use>
      G. outer <tool_use> wrapper containing inner calls of any format above
      H. <bash><command>…</command></bash>
      I. {"tool": "name", …} JSON blob
      J. ```json {"type":"tool_use",...} ``` fenced block
      — also handles <tooluse> (no underscore) throughout
    Returns (tool_calls: list[dict], remaining_text: str).
    """
    tool_calls = []
    # Track (start, end) spans of blocks we already consumed so nested
    # inner blocks are not double-counted
    consumed_spans: list[tuple[int, int]] = []

    # ── 0. ```tool {"name":…,"input":{…}} ``` blocks (PRIMARY) ───
    for m in _TOOL_BLOCK_RE.finditer(text):
        body = m.group("body").strip()
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            obj = _parse_tool_input(body)
        name = obj.get("name", "")
        inp  = obj.get("input", {})
        if not name:
            continue
        if not isinstance(inp, dict):
            inp = {"value": str(inp)}
        tool_calls.append({
            "type":  "tool_use",
            "id":    f"toolu_{uuid.uuid4().hex[:24]}",
            "name":  name,
            "input": inp,
        })
        consumed_spans.append((m.start(), m.end()))

    if not tool_calls:
        # ── 0b. <tool_call name="X">…</tool_call> ────────────────
        # Body may be JSON *or* XML child tags like <command>…</command>
        for m in _TOOL_CALL_TAG_RE.finditer(text):
            name = m.group("name").strip()
            body = m.group("body").strip()
            inp = _parse_tool_input(body)
            if not isinstance(inp, dict):
                inp = {"value": str(inp)}
            tool_calls.append({
                "type":  "tool_use",
                "id":    f"toolu_{uuid.uuid4().hex[:24]}",
                "name":  name,
                "input": inp,
            })
            consumed_spans.append((m.start(), m.end()))

    # Return early if primary formats matched
    if tool_calls:
        remaining = text
        for start, end in sorted(consumed_spans, reverse=True):
            remaining = remaining[:start] + remaining[end:]
        return tool_calls, remaining.strip()

    for m in _BLOCK_RE.finditer(text):
        raw_block = m.group(0)
        span = (m.start(), m.end())

        # Skip if this span is fully inside an already-consumed outer block
        if any(s <= m.start() and m.end() <= e for s, e in consumed_spans):
            continue

        # ── Resolve tool name ──────────────────────────────────
        attr_m = _NAME_ATTR_RE.search(raw_block)
        tag_m  = _NAME_TAG_RE.search(raw_block)

        if attr_m:
            name = attr_m.group(1).strip()
        elif tag_m:
            name = tag_m.group(1).strip()
        else:
            # Bare <tool_use> with no name — it is a container; mark consumed
            # so inner calls (matched separately) are not skipped
            consumed_spans.append(span)
            continue

        # ── Strip outer tags + <name>…</name> to get the inner body ───
        inner = raw_block
        if tag_m:
            inner = inner[:tag_m.start()] + inner[tag_m.end():]
        inner = re.sub(r'^<tool_?use[^>]*>', '', inner.strip(), flags=re.IGNORECASE)
        inner = re.sub(r'</tool_?use>$',     '', inner.strip(), flags=re.IGNORECASE)
        inner = inner.strip()

        # ── Resolve input with priority chain ─────────────────
        inp = None

        # P1: <input>JSON</input>
        im = _INPUT_TAG_RE.search(inner)
        if im:
            inp = _parse_tool_input(im.group(1))

        # P2: one or more  <parameter name="k" ...>v</parameter>
        if inp is None:
            named = _PARAM_NAMED_RE.findall(inner)
            if named:
                inp = {k.strip(): v.strip() for k, v in named}

        # P3: single bare <parameter>content</parameter>  (may be JSON)
        if inp is None:
            pm = _PARAM_SINGLE_RE.search(inner)
            if pm:
                parsed = _parse_tool_input(pm.group(1).strip())
                inp = parsed  # even {"content":...} is acceptable here

        # P4: body IS a raw JSON object/array
        if inp is None and (inner.startswith('{') or inner.startswith('[')):
            parsed = _parse_tool_input(inner)
            if list(parsed.keys()) != ["content"]:
                inp = parsed
            else:
                # fall through to child-tag extraction in case JSON failed
                child_map = {cm.group(1).lower(): cm.group(2).strip()
                             for cm in _CHILD_TAG_RE.finditer(inner)
                             if cm.group(1).lower() not in
                                ('tool_use', 'tooluse', 'name', 'input', 'parameter')}
                inp = child_map if child_map else parsed

        # P5: direct XML child tags  <command>…</command>
        if inp is None:
            child_map = {cm.group(1).lower(): cm.group(2).strip()
                         for cm in _CHILD_TAG_RE.finditer(inner)
                         if cm.group(1).lower() not in
                            ('tool_use', 'tooluse', 'name', 'input', 'parameter')}
            inp = child_map if child_map else {"content": inner}

        tool_calls.append({
            "type":  "tool_use",
            "id":    f"toolu_{uuid.uuid4().hex[:24]}",
            "name":  name,
            "input": inp,
        })
        consumed_spans.append(span)

    # ── Fallback: <bash><command>…</command></bash> ────────────
    if not tool_calls:
        for m in _BASH_CMD_RE.finditer(text):
            tool_calls.append({
                "type":  "tool_use",
                "id":    f"toolu_{uuid.uuid4().hex[:24]}",
                "name":  "Bash",
                "input": {"command": m.group("cmd").strip()},
            })

    # ── Fallback: {"tool": "name", …} JSON blob ───────────────
    if not tool_calls:
        for m in _TOOL_USE_JSON_RE.finditer(text):
            try:
                obj = json.loads(m.group(0))
                if "tool" in obj:
                    name = obj.pop("tool")
                    tool_calls.append({
                        "type":  "tool_use",
                        "id":    f"toolu_{uuid.uuid4().hex[:24]}",
                        "name":  name,
                        "input": obj,
                    })
            except json.JSONDecodeError:
                continue

    # ── Fallback: markdown-fenced tool_use block ───────────────
    if not tool_calls:
        for m in _FENCED_TOOL_RE.finditer(text):
            raw = m.group(0)
            inner = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
            try:
                obj = json.loads(inner)
                if obj.get("type") == "tool_use":
                    tool_calls.append({
                        "type":  "tool_use",
                        "id":    obj.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                        "name":  obj.get("name", ""),
                        "input": obj.get("input", {}),
                    })
            except json.JSONDecodeError:
                continue

    # ── Build remaining text (strip all matched tool_use blocks) ─
    remaining = text
    for m in reversed(list(_BLOCK_RE.finditer(text))):
        remaining = remaining[:m.start()] + remaining[m.end():]
    # Strip ```tool ``` blocks and <tool_call> blocks
    remaining = _TOOL_BLOCK_RE.sub('', remaining)
    remaining = _TOOL_CALL_TAG_RE.sub('', remaining)
    # Also strip <bash>…</bash> fallback blocks
    remaining = _BASH_CMD_RE.sub('', remaining).strip()

    return tool_calls, remaining.strip()


def _build_content_blocks(response_md: str, reasoning_blocks: list) -> tuple:
    """
    Build the `content` array for an Anthropic response.

    Claude Code (and all Anthropic SDK consumers) expect:
      - Optional thinking blocks FIRST  (type="thinking", thinking=<text>)
      - Optional tool_use blocks        (type="tool_use", id, name, input)
      - Then a text block (may be empty if only tool calls)

    Returns (content_blocks, stop_reason).
    stop_reason is "tool_use" if tool calls were detected, else "end_turn".
    """
    blocks = []

    # Reasoning blocks as proper "thinking" type — Claude Code skips these
    for rb in reasoning_blocks:
        if rb.strip():
            blocks.append({
                "type":      "thinking",
                "thinking":  rb,
            })

    # Try to extract tool_use calls from the response text
    tool_calls, remaining_text = _extract_tool_calls(response_md)

    blocks.extend(tool_calls)

    # Always include a text block (even empty) so the SDK doesn't choke
    blocks.append({
        "type": "text",
        "text": remaining_text if remaining_text else response_md,
    })

    stop_reason = "tool_use" if tool_calls else "end_turn"
    return blocks, stop_reason


def _detect_clean_mode(prompt: str, has_tools: bool = False) -> str:
    """
    Heuristic: decide which cleaning mode to use based on the prompt content.

    - has_tools=True  → "light" (preserve any tool XML / JSON DeepSeek emits)
    - Prompts that expect ```json output  → "json"
    - Prompts that ask for Python/bash    → "code"
    - Everything else                    → "text"
    """
    # When tools are declared, DeepSeek may return structured tool-call markup.
    # Any aggressive cleaning would destroy it, so use light mode only.
    if has_tools:
        return "light"

    lower = prompt.lower()
    # Explicit JSON-output instructions
    json_signals = [
        "respond with only a ```json",
        "output only a ```json",
        "return only a json",
        "return only this json",
        "reply with only one word",
        "respond with only a json",
        '"fixed_command"',
        '"alternative_command"',
        '"intent"',
        '"steps"',
        # Claude Code agent creation — expects {"identifier":...,"whenToUse":...,"systemPrompt":...}
        '"identifier"',
        '"whentouse"',
        '"systemprompt"',
        'create a new agent',
        'generate an agent definition',
        'agent definition json',
        # Other structured-output patterns Claude Code uses
        '"bash_command"',
        '"description"',
        'output only json',
        'respond only with json',
        'only output json',
        'only respond with json',
    ]
    if any(s in lower for s in json_signals):
        return "json"

    # Code-generation prompts
    code_signals = [
        "output only a ```python",
        "start your response with: ```python",
        "python3 code generator",
        "write the python 3 script",
        "fenced block",
    ]
    if any(s in lower for s in code_signals):
        return "code"

    return "text"


def _build_response_body(
    response_md: str,
    reasoning_blocks: list,
    model: str,
    input_tokens: int = 0,
) -> dict:
    """Build a valid non-streaming Anthropic /v1/messages response."""
    content, stop_reason = _build_content_blocks(response_md, reasoning_blocks)
    output_tokens = _estimate_tokens(response_md)
    for rb in reasoning_blocks:
        output_tokens += _estimate_tokens(rb)

    return {
        "id":            f"msg_{uuid.uuid4().hex[:24]}",
        "type":          "message",
        "role":          "assistant",
        "content":       content,
        "model":         model,
        "stop_reason":   stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens":  max(input_tokens, 1),
            "output_tokens": max(output_tokens, 1),
        },
    }


# ─────────────────────────────────────────────────────────────
# SSE Streaming
# ─────────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _stream_response(response_md: str, reasoning_blocks: list, model: str):
    """
    Yield SSE events matching the Anthropic streaming wire format.

    Block order:
      [thinking block(s)] → [tool_use block(s)] → [text block]

    This is critical for Claude Code: it uses the content_block_start `type`
    field to decide how to handle each block. Reasoning in a "thinking" block
    is silently consumed; tool_use blocks trigger tool execution.
    """
    msg_id        = f"msg_{uuid.uuid4().hex[:24]}"
    chunk_size    = 64          # larger chunks = fewer events = better perf
    output_tokens = _estimate_tokens(response_md)
    for rb in reasoning_blocks:
        output_tokens += _estimate_tokens(rb)

    # Parse tool calls and remaining text up-front so we know stop_reason
    tool_calls, remaining_text = _extract_tool_calls(response_md)
    stop_reason = "tool_use" if tool_calls else "end_turn"
    text_to_stream = remaining_text if remaining_text else (response_md if not tool_calls else "")

    # ── message_start ─────────────────────────────────────────
    yield _sse({
        "type": "message_start",
        "message": {
            "id":            msg_id,
            "type":          "message",
            "role":          "assistant",
            "content":       [],
            "model":         model,
            "stop_reason":   None,
            "stop_sequence": None,
            "usage":         {"input_tokens": 1, "output_tokens": 1},
        },
    })

    block_index = 0

    # ── thinking blocks (one per reasoning step) ──────────────
    for rb in reasoning_blocks:
        if not rb.strip():
            continue

        yield _sse({
            "type":          "content_block_start",
            "index":         block_index,
            "content_block": {"type": "thinking", "thinking": ""},
        })

        for i in range(0, len(rb), chunk_size):
            yield _sse({
                "type":  "content_block_delta",
                "index": block_index,
                "delta": {"type": "thinking_delta", "thinking": rb[i: i + chunk_size]},
            })

        yield _sse({"type": "content_block_stop", "index": block_index})
        block_index += 1

    # ── tool_use blocks ───────────────────────────────────────
    for tc in tool_calls:
        input_json = json.dumps(tc["input"])
        yield _sse({
            "type":          "content_block_start",
            "index":         block_index,
            "content_block": {
                "type":  "tool_use",
                "id":    tc["id"],
                "name":  tc["name"],
                "input": {},
            },
        })
        # Stream the input JSON as input_json_delta
        for i in range(0, len(input_json), chunk_size):
            yield _sse({
                "type":  "content_block_delta",
                "index": block_index,
                "delta": {"type": "input_json_delta", "partial_json": input_json[i: i + chunk_size]},
            })
        yield _sse({"type": "content_block_stop", "index": block_index})
        block_index += 1

    # ── main text block (always present; may be empty for tool-only responses) ──
    yield _sse({
        "type":          "content_block_start",
        "index":         block_index,
        "content_block": {"type": "text", "text": ""},
    })

    if text_to_stream:
        for i in range(0, len(text_to_stream), chunk_size):
            yield _sse({
                "type":  "content_block_delta",
                "index": block_index,
                "delta": {"type": "text_delta", "text": text_to_stream[i: i + chunk_size]},
            })

    yield _sse({"type": "content_block_stop", "index": block_index})

    # ── message_delta + message_stop ──────────────────────────
    yield _sse({
        "type":  "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": max(output_tokens, 1)},
    })

    yield _sse({"type": "message_stop"})
    # The Anthropic SDK expects [DONE] as the final SSE event
    yield "data: [DONE]\n\n"


def _stream_response_debug(response_md: str, reasoning_blocks: list, model: str):
    """
    Wrapper around _stream_response that prints every SSE event to stderr
    so you can see the exact wire-format bytes being sent to the client.
    """
    _dbg_separator("SSE STREAM START", char="═")
    event_index = 0
    for event in _stream_response(response_md, reasoning_blocks, model):
        if DEBUG_OUTPUT:
            label = f"SSE EVENT #{event_index}"
            # Pretty-print parseable events; pass through [DONE] as-is
            if event.strip() == "data: [DONE]":
                print(f"\n[{label}] data: [DONE]", file=sys.stderr, flush=True)
            else:
                # Strip "data: " prefix and parse JSON for readable output
                raw_json = event.removeprefix("data: ").strip()
                try:
                    parsed = json.loads(raw_json)
                    print(f"\n[{label}]", file=sys.stderr, flush=True)
                    print(json.dumps(parsed, indent=2, ensure_ascii=False),
                          file=sys.stderr, flush=True)
                except json.JSONDecodeError:
                    print(f"\n[{label}] (raw) {repr(event)}", file=sys.stderr, flush=True)
        event_index += 1
        yield event
    _dbg_separator("SSE STREAM END", char="═")


# ─────────────────────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)

_CORS_HEADERS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Headers": (
        "Content-Type, Authorization, x-api-key, "
        "anthropic-version, anthropic-beta"
    ),
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
}


@app.after_request
def _add_cors(response):
    for k, v in _CORS_HEADERS.items():
        response.headers[k] = v
    return response


def _cors_preflight():
    resp = Response("", status=204)
    for k, v in _CORS_HEADERS.items():
        resp.headers[k] = v
    return resp


# ── POST /v1/messages ─────────────────────────────────────────

@app.route("/v1/messages", methods=["POST", "OPTIONS"])
def messages():
    if request.method == "OPTIONS":
        return _cors_preflight()

    try:
        body = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": {
            "type":    "invalid_request_error",
            "message": "Invalid JSON body",
        }}), 400

    model  = body.get("model", "deepseek-chat")
    stream = body.get("stream", False)

    if not body.get("messages"):
        return jsonify({"error": {
            "type":    "invalid_request_error",
            "message": "messages field is required",
        }}), 400

    prompt = _extract_prompt(body).strip()
    if not prompt:
        return jsonify({"error": {
            "type":    "invalid_request_error",
            "message": "Prompt is empty after extraction",
        }}), 400

    # ── Send to DeepSeek ──────────────────────────────────────
    try:
        scraper = get_scraper()
        response_md, reasoning_blocks, _elapsed = scraper.send_message(prompt)
    except Exception as e:
        return jsonify({"error": {
            "type":    "api_error",
            "message": f"DeepSeek scraper error: {e}",
        }}), 500

    if response_md.startswith("[Error]"):
        return jsonify({"error": {
            "type":    "api_error",
            "message": response_md,
        }}), 500

    # ── Debug: raw response from DeepSeek ────────────────────
    _dbg("RAW RESPONSE FROM DEEPSEEK (exact bytes)", response_md)
    _dbg(f"RAW REASONING BLOCKS ({len(reasoning_blocks)} block(s))", reasoning_blocks)

    # ── Clean the response (same pipeline as hackers-ai) ─────
    has_tools   = bool(body.get("tools"))
    clean_mode  = _detect_clean_mode(prompt, has_tools=has_tools)

    response_md = _clean_response(response_md, mode=clean_mode)
    # Reasoning blocks: always light-clean (strip links + prompt artefacts)
    reasoning_blocks = [_clean_response(rb, mode="light") for rb in reasoning_blocks]

    # ── Debug: cleaned response ───────────────────────────────
    _dbg("CLEANED RESPONSE (after _clean_response)", response_md)
    _dbg(f"CLEANED REASONING BLOCKS ({len(reasoning_blocks)} block(s))", reasoning_blocks)

    input_tokens = _estimate_tokens(prompt)

    # ── Stream or return ──────────────────────────────────────
    if stream:
        _dbg_separator("STREAMING SSE RESPONSE  —  events will follow below")
        return Response(
            _stream_response_debug(response_md, reasoning_blocks, model),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    final_body = _build_response_body(response_md, reasoning_blocks, model, input_tokens)
    _dbg("FINAL NON-STREAMING RESPONSE BODY (exact JSON)", final_body)
    return jsonify(final_body)


# ── GET /v1/models ────────────────────────────────────────────

@app.route("/v1/models", methods=["GET", "OPTIONS"])
def list_models():
    if request.method == "OPTIONS":
        return _cors_preflight()

    now = 1720000000
    return jsonify({
        "data": [
            {"id": "deepseek-chat",      "object": "model", "created": now, "owned_by": "deepseek"},
            {"id": "deepseek-reasoner",  "object": "model", "created": now, "owned_by": "deepseek"},
            # Anthropic aliases so tools that hard-code Claude model names still work
            {"id": "claude-opus-4-5",    "object": "model", "created": now, "owned_by": "anthropic"},
            {"id": "claude-sonnet-4-5",  "object": "model", "created": now, "owned_by": "anthropic"},
            {"id": "claude-haiku-3-5",   "object": "model", "created": now, "owned_by": "anthropic"},
            {"id": "claude-opus-4-6",    "object": "model", "created": now, "owned_by": "anthropic"},
            {"id": "claude-sonnet-4-6",  "object": "model", "created": now, "owned_by": "anthropic"},
            {"id": "claude-haiku-4-5",   "object": "model", "created": now, "owned_by": "anthropic"},
        ]
    })


# ── GET /health ───────────────────────────────────────────────

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "proxy":   "chat.deepseek.com → Anthropic API",
        "browser": "ready" if _scraper is not None else "not started",
        "time":    datetime.now().isoformat(),
    })


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="chat.deepseek.com → Anthropic API Proxy")
    ap.add_argument("--host",      default="0.0.0.0",      help="Bind host (default: 0.0.0.0)")
    ap.add_argument("--port",      default=8765, type=int,  help="Port (default: 8765)")
    ap.add_argument("--headless",  action="store_true",     help="Run Chromium headless")
    ap.add_argument("--search",    action="store_true",     help="Enable Search toggle")
    ap.add_argument("--deepthink", action="store_true",     help="Enable DeepThink toggle")
    ap.add_argument("--expert",    action="store_true",     help="Enable Expert model")
    ap.add_argument("--no-warmup", action="store_true",     help="Lazy-init browser")
    args = ap.parse_args()

    _headless_flag  = args.headless
    _search_flag    = args.search
    _deepthink_flag = args.deepthink
    _expert_flag    = args.expert

    banner = f"""
{'=' * 60}
   chat.deepseek.com  →  Anthropic API Proxy
   Claude Code compatible edition
{'=' * 60}
  Listening on : http://{args.host}:{args.port}
  Headless     : {args.headless}
  Search       : {args.search}
  DeepThink    : {args.deepthink}
  Expert model : {args.expert}

  Configure your tool:
    export ANTHROPIC_BASE_URL="http://localhost:{args.port}"
    export ANTHROPIC_API_KEY="local-proxy-key"

  Endpoints:
    POST /v1/messages   (streaming + non-streaming)
    GET  /v1/models
    GET  /health
{'=' * 60}
"""
    print(banner)

    if not args.no_warmup:
        print("[*] Pre-launching browser (pass --no-warmup to skip) ...")
        get_scraper()
        print("[+] Browser ready. Proxy is live!\n")

    app.run(host=args.host, port=args.port, threaded=False, debug=False)
