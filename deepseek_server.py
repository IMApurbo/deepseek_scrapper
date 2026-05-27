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
    Normalise the `content` field of a single message into a plain string
    suitable for sending to DeepSeek as part of a prompt.
    Handles: str, list of content blocks (text / tool_result / tool_use / image).
    cache_control fields are silently dropped — DeepSeek doesn't support them.
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
                parts.append(block.get("text", ""))
            elif btype == "tool_result":
                tool_id = block.get("tool_use_id", "")
                inner = block.get("content", "")
                if isinstance(inner, list):
                    texts = [rb.get("text", "") for rb in inner
                             if isinstance(rb, dict) and rb.get("type") == "text"]
                    result_text = "\n".join(texts)
                else:
                    result_text = str(inner)
                parts.append(f"[Tool Result id={tool_id}]\n{result_text}")
            elif btype == "tool_use":
                name = block.get("name", "tool")
                tool_id = block.get("id", "")
                inp  = json.dumps(block.get("input", {}), indent=2)
                parts.append(f"[Tool Call: {name} id={tool_id}]\n{inp}")
            # skip image blocks and cache_control-only blocks — DeepSeek can't use them
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
    Convert Anthropic tool definitions to an XML block that DeepSeek can
    understand and use to emit correctly-formatted tool calls.
    """
    if not tools:
        return ""
    lines = ["<available_tools>"]
    for t in tools:
        name = t.get("name", "")
        desc = t.get("description", "").strip()
        schema = t.get("input_schema", {})
        props = schema.get("properties", {})
        required = schema.get("required", [])
        lines.append(f"  <tool>")
        lines.append(f"    <name>{name}</name>")
        if desc:
            # Only first sentence of description to keep prompt lean
            short_desc = desc.split("\n")[0][:300]
            lines.append(f"    <description>{short_desc}</description>")
        if props:
            lines.append(f"    <parameters>")
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else ""
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")[:200]
                lines.append(f"      <parameter name=\"{pname}\" type=\"{ptype}\"{req}>{pdesc}</parameter>")
            lines.append(f"    </parameters>")
        lines.append(f"  </tool>")
    lines.append("</available_tools>")
    lines.append("")
    lines.append("IMPORTANT: When you need to use a tool, you MUST respond with ONLY this exact XML format and nothing else:")
    lines.append("<tool_use>")
    lines.append("<name>tool_name_here</name>")
    lines.append("<input>{\"param1\": \"value1\", \"param2\": \"value2\"}</input>")
    lines.append("</tool_use>")
    lines.append("")
    lines.append("Rules:")
    lines.append("- The <input> tag must contain valid JSON.")
    lines.append("- Do NOT include any text before or after the <tool_use> block when calling a tool.")
    lines.append("- After the tool result is returned, you may respond normally or call another tool.")
    lines.append("- If you do not need a tool, respond normally without any <tool_use> block.")
    return "\n".join(lines)


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

# Pattern 1: Our injected format  <tool_use><name>X</name><input>{…}</input></tool_use>
_TOOL_USE_RE = re.compile(
    r'''<tool_use>\s*<name>(?P<name>[^<]+)</name>\s*<input>(?P<input>[\s\S]*?)</input>\s*</tool_use>''',
    re.DOTALL,
)
# Pattern 2: JSON blob with a "tool" key
_TOOL_USE_JSON_RE = re.compile(
    r'''\{\s*"tool"\s*:\s*"(?P<name>[^"]+)"[\s\S]*?\}''',
    re.DOTALL,
)
# Pattern 3: <bash><command>…</command></bash>  (DeepSeek sometimes mimics Claude's internal format)
_BASH_CMD_RE = re.compile(
    r'''<bash>\s*<command>(?P<cmd>[\s\S]*?)</command>\s*</bash>''',
    re.DOTALL,
)
# Pattern 4: ```json\n{"tool_use": ...}\n```  (markdown-fenced tool use)
_FENCED_TOOL_RE = re.compile(
    r'''```(?:json)?\s*\n\s*\{[\s\S]*?"type"\s*:\s*"tool_use"[\s\S]*?\}\s*\n```''',
    re.DOTALL,
)


def _parse_tool_input(raw_input: str) -> dict:
    """
    Robustly parse a tool <input> JSON string from DeepSeek.

    DeepSeek frequently emits malformed JSON inside <input> tags, e.g.:
      - Unescaped double quotes inside string values:
          {"command": "echo "hello"", "description": "test"}
      - Missing closing </input> tag (handled upstream by the regex)
      - Wrapping markdown fences around the JSON

    Strategy:
      1. Try json.loads() as-is (fast path for well-formed JSON).
      2. Strip markdown fences and retry.
      3. Use a field-by-field regex to extract key:value pairs from
         common Claude Code tool schemas (command, description, file_path,
         content, etc.) — avoids the "raw" fallback that breaks tool calls.
      4. Last resort: return {"command": raw_input} for Bash-like tools,
         or {"content": raw_input} for Write-like tools, so the call at
         least has *some* usable input rather than {"raw": ...}.
    """
    # Fast path
    try:
        return json.loads(raw_input)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    stripped = re.sub(r'^```[a-z]*\n?|\n?```$', '', raw_input.strip())
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Try fixing unescaped inner quotes with a simple heuristic:
    # Replace " inside string values with \" by scanning character by character.
    def fix_json_quotes(s):
        result = []
        in_string = False
        escape_next = False
        i = 0
        while i < len(s):
            c = s[i]
            if escape_next:
                result.append(c)
                escape_next = False
                i += 1
                continue
            if c == '\\':
                escape_next = True
                result.append(c)
                i += 1
                continue
            if c == '"':
                if not in_string:
                    in_string = True
                    result.append(c)
                else:
                    # Peek ahead: if next non-space char is :, , or }, this " closes the string
                    j = i + 1
                    while j < len(s) and s[j] in ' \t\n\r':
                        j += 1
                    if j >= len(s) or s[j] in ':,}]':
                        in_string = False
                        result.append(c)
                    else:
                        # Inner quote — escape it
                        result.append('\\"')
            else:
                result.append(c)
            i += 1
        return ''.join(result)

    try:
        fixed = fix_json_quotes(stripped)
        return json.loads(fixed)
    except (json.JSONDecodeError, Exception):
        pass

    # Regex-based field extraction for known Claude Code tool schemas
    result = {}
    # Extract all "key": "value" pairs, allowing for unescaped inner quotes
    # by being greedy-careful: match up to the next  ", "key" or end of object
    field_re = re.compile(
        r'"(?P<key>[^"]+)"\s*:\s*'
        r'(?:"(?P<str_val>(?:[^"\\]|\\.)*)"|(?P<other_val>[^,}\]]+))',
        re.DOTALL
    )
    for fm in field_re.finditer(stripped):
        key = fm.group("key")
        val = fm.group("str_val") if fm.group("str_val") is not None else fm.group("other_val")
        if val is not None:
            val = val.strip()
            # Try to parse booleans/numbers
            if val.lower() == 'true':
                val = True
            elif val.lower() == 'false':
                val = False
            else:
                try:
                    val = int(val)
                except (ValueError, TypeError):
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        pass
            result[key] = val

    if result:
        return result

    # Absolute last resort: preserve the raw string under a meaningful key
    # so the model at least gets *something* executable rather than crashing
    if "command" in raw_input.lower() or "bash" in raw_input.lower():
        # Try to salvage just the command value
        cmd_m = re.search(r'"command"\s*:\s*"(.*)"', raw_input, re.DOTALL)
        if cmd_m:
            return {"command": cmd_m.group(1)}
    return {"content": raw_input}


def _extract_tool_calls(text: str):
    """
    Try to extract tool_use blocks from DeepSeek's response.
    DeepSeek may emit them as:
      1. <tool_use><name>…</name><input>{…}</input></tool_use>  (our injected XML format)
      2. A bare JSON object with a "tool" key
      3. <bash><command>…</command></bash>  (DeepSeek mimicking Claude's internal notation)
      4. Markdown-fenced JSON with type=tool_use
    Returns (tool_calls: list[dict], remaining_text: str).
    """
    tool_calls = []
    remaining = text

    # Strategy 1: Our XML format <tool_use>…</tool_use>
    for m in _TOOL_USE_RE.finditer(text):
        name = m.group("name").strip()
        raw_input = m.group("input").strip()
        inp = _parse_tool_input(raw_input)
        tool_calls.append({
            "type":  "tool_use",
            "id":    f"toolu_{uuid.uuid4().hex[:24]}",
            "name":  name,
            "input": inp,
        })
        remaining = remaining.replace(m.group(0), "")

    if tool_calls:
        return tool_calls, remaining.strip()

    # Strategy 2: <bash><command>…</command></bash>
    for m in _BASH_CMD_RE.finditer(text):
        cmd = m.group("cmd").strip()
        tool_calls.append({
            "type":  "tool_use",
            "id":    f"toolu_{uuid.uuid4().hex[:24]}",
            "name":  "Bash",
            "input": {"command": cmd},
        })
        remaining = remaining.replace(m.group(0), "")

    if tool_calls:
        return tool_calls, remaining.strip()

    # Strategy 3: JSON blob with "tool" key
    for m in _TOOL_USE_JSON_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "tool" in obj:
            name = obj.pop("tool")
            tool_calls.append({
                "type":  "tool_use",
                "id":    f"toolu_{uuid.uuid4().hex[:24]}",
                "name":  name,
                "input": obj,
            })
            remaining = remaining.replace(m.group(0), "")

    # Strategy 4: Markdown-fenced {"type":"tool_use",...}
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
                    remaining = remaining.replace(raw, "")
            except json.JSONDecodeError:
                continue

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
