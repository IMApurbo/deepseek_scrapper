"""
╔══════════════════════════════════════════════════════════════╗
║           chat.deepseek.com  —  Playwright Scraper           ║
║              Modern Edition  ·  Powered by Rich              ║
╚══════════════════════════════════════════════════════════════╝
"""

import time
import json
import re
import sys
from pathlib import Path
from html.parser import HTMLParser
from playwright.sync_api import sync_playwright

# ── Rich imports ──────────────────────────────────────────────
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich import box
from rich.theme import Theme
from rich.live import Live
from rich.spinner import Spinner
from rich.align import Align
from rich.padding import Padding

# ── Optional html2text (preferred path) ───────────────────────
try:
    import html2text as _html2text_mod
    _H2T = _html2text_mod.HTML2Text()
    _H2T.ignore_links = False
    _H2T.ignore_images = True
    _H2T.body_width = 0
    _H2T.protect_links = True
    _H2T.wrap_links = False
    HAS_HTML2TEXT = True
except ImportError:
    HAS_HTML2TEXT = False

# ── Persistent session directory ──────────────────────────────
SESSION_DIR  = Path.home() / ".deepseek_scraper"
COOKIES_FILE = SESSION_DIR / "cookies.json"
STORAGE_FILE = SESSION_DIR / "storage.json"
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# Far-future expiry (year 2099) — forces session cookies to persist
FAR_FUTURE = 4070908800

# ── Theme ─────────────────────────────────────────────────────
THEME = Theme({
    "primary":     "bold #2563EB",
    "secondary":   "#60A5FA",
    "accent":      "bold #0EA5E9",
    "success":     "bold #10B981",
    "warning":     "bold #F59E0B",
    "error":       "bold #EF4444",
    "muted":       "#6B7280",
    "user_label":  "bold #38BDF8",
    "ai_label":    "bold #818CF8",
    "think_label": "bold #FCD34D",
    "cmd":         "#94A3B8",
    "info":        "#60A5FA",
})

console = Console(theme=THEME, highlight=False)


# ─────────────────────────────────────────────────────────────
# Fallback HTML → Markdown Parser
# ─────────────────────────────────────────────────────────────

class _MDParser(HTMLParser):
    """
    Minimal but correct HTML-to-Markdown converter used as a fallback
    when html2text is not installed.

    Key fixes vs the original:
      • Code-fence language hints are captured from <code class="language-*">
        inside <pre> blocks → produces ```python, ```bash, etc.
      • <p> inside <li> no longer inserts double-newlines that break bullet layout.
      • <br> inside <li> emits a space instead of a hard line-break.
      • Language tag is injected on the opening ``` fence, not when </code> fires.
    """

    IGNORE_TAGS = frozenset({
        "svg", "script", "style", "noscript", "button", "input",
        "select", "form", "head", "meta", "link",
        "sup", "sub", "cite", "time", "footer", "nav", "aside",
    })
    BOLD_TAGS   = frozenset({"strong", "b"})
    ITALIC_TAGS = frozenset({"em", "i"})
    CODE_TAGS   = frozenset({"code"})
    DEL_TAGS    = frozenset({"s", "del", "strike"})
    UNDER_TAGS  = frozenset({"u"})

    def __init__(self):
        super().__init__()
        self.out          = []       # output token list
        self.stack        = []       # tag stack (str or ("a", href))
        self.ignore_depth = 0        # depth inside ignored subtrees
        self.pre_depth    = 0        # depth inside <pre> blocks
        self.list_stack   = []       # [["ul"|"ol", counter], ...]
        self.li_depth     = 0        # how many <li> deep we are
        self.in_link      = False
        self.link_buf     = []       # text accumulator for <a>
        self.link_href    = ""
        self.in_cell      = False
        self.cell_buf     = []       # text accumulator for <td>/<th>
        self.td_buf       = []       # cell texts for current <tr>
        self.header_row   = False
        self.in_table     = False
        self._pending_fence = None   # fence token index awaiting language patch

    # ── helpers ───────────────────────────────────────────────

    def _in_li(self) -> bool:
        return self.li_depth > 0

    def _emit(self, text: str):
        """Route output to the correct buffer."""
        if self.in_link:
            self.link_buf.append(text)
        elif self.in_cell:
            self.cell_buf.append(text)
        else:
            self.out.append(text)

    # ── tag handlers ──────────────────────────────────────────

    def handle_starttag(self, tag: str, attrs):
        tag   = tag.lower()
        adict = dict(attrs)

        # ── ignored subtrees ──────────────────────────────────
        if self.ignore_depth or tag in self.IGNORE_TAGS:
            self.ignore_depth += 1
            self.stack.append(("__ignore__", tag))
            return

        self.stack.append(tag)

        # ── headings ──────────────────────────────────────────
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.out.append(f"\n\n{'#' * int(tag[1])} ")

        # ── paragraphs ────────────────────────────────────────
        elif tag == "p":
            # Inside <li>: <p> is inline spacing only — no block break.
            # The double-newline here was the root cause of the blank-line
            # bullet bug in the original code.
            if not self._in_li():
                self.out.append("\n\n")

        # ── lists ─────────────────────────────────────────────
        elif tag in ("ul", "ol"):
            self.list_stack.append([tag, 0])
            self.out.append("\n")

        elif tag == "li":
            self.li_depth += 1
            if self.list_stack:
                kind, count = self.list_stack[-1]
                if kind == "ol":
                    self.list_stack[-1][1] += 1
                    marker = f"{self.list_stack[-1][1]}. "
                else:
                    marker = "- "
                indent = "  " * (len(self.list_stack) - 1)
                self.out.append(f"\n{indent}{marker}")

        # ── blockquote ────────────────────────────────────────
        elif tag == "blockquote":
            self.out.append("\n\n> ")

        # ── pre / code ────────────────────────────────────────
        elif tag == "pre":
            self.pre_depth += 1
            # Emit opening fence; record its index so we can patch in
            # the language hint when the inner <code class="language-*"> fires.
            fence = "```"
            self.out.append(f"\n\n{fence}")
            self._pending_fence = len(self.out) - 1   # index of the fence token

        elif tag == "code":
            if self.pre_depth > 0:
                # ── fenced code block ─────────────────────────
                # Extract language from class="language-python" etc.
                cls  = adict.get("class", "")
                m    = re.search(r"\blanguage-(\w+)\b", cls)
                lang = m.group(1) if m else ""

                # Patch the pending opening fence with the language hint.
                # e.g. "```" → "```python"
                if lang and self._pending_fence is not None:
                    self.out[self._pending_fence] = (
                        self.out[self._pending_fence].rstrip("`") + f"```{lang}"
                    )
                self._pending_fence = None
                # A newline after the opening fence keeps the first line of
                # code on its own line (required by CommonMark).
                self.out.append("\n")
            else:
                # ── inline code ───────────────────────────────
                self._emit("`")

        # ── inline formatting ─────────────────────────────────
        elif tag in self.BOLD_TAGS:
            self._emit("**")
        elif tag in self.ITALIC_TAGS:
            self._emit("*")
        elif tag in self.DEL_TAGS:
            self._emit("~~")
        elif tag in self.UNDER_TAGS:
            self._emit("__")

        # ── links ─────────────────────────────────────────────
        elif tag == "a":
            self.stack[-1] = ("a", adict.get("href", ""))
            self.in_link   = True
            self.link_href = adict.get("href", "")
            self.link_buf  = []

        # ── line-level ────────────────────────────────────────
        elif tag == "br":
            # Inside <li>: <br> is a soft wrap, not a hard break.
            if self._in_li():
                self._emit(" ")
            else:
                self._emit("  \n")

        elif tag == "hr":
            self.out.append("\n\n---\n\n")

        # ── tables ────────────────────────────────────────────
        elif tag == "table":
            self.in_table = True
            self.out.append("\n\n")

        elif tag == "tr":
            self.td_buf = []

        elif tag == "thead":
            self.header_row = True

        elif tag in ("th", "td"):
            self.in_cell  = True
            self.cell_buf = []

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if not self.stack:
            return

        # Pop the matching open-tag entry (search from the top).
        entry = None
        for i in range(len(self.stack) - 1, -1, -1):
            item = self.stack[i]
            item_tag = item[0] if isinstance(item, tuple) else item
            if item_tag == tag or (item_tag == "__ignore__" and item[1] == tag):
                entry = self.stack.pop(i)
                break
        else:
            return  # unmatched close tag — ignore

        # ── ignored subtrees ──────────────────────────────────
        if isinstance(entry, tuple) and entry[0] == "__ignore__":
            self.ignore_depth = max(0, self.ignore_depth - 1)
            return

        # ── headings ──────────────────────────────────────────
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.out.append("\n")

        # ── paragraphs ────────────────────────────────────────
        elif tag == "p":
            if not self._in_li():
                self.out.append("\n")

        # ── lists ─────────────────────────────────────────────
        elif tag == "li":
            self.li_depth = max(0, self.li_depth - 1)

        elif tag in ("ul", "ol"):
            if self.list_stack:
                self.list_stack.pop()
            self.out.append("\n")

        # ── blockquote ────────────────────────────────────────
        elif tag == "blockquote":
            self.out.append("\n\n")

        # ── pre / code ────────────────────────────────────────
        elif tag == "pre":
            self.pre_depth = max(0, self.pre_depth - 1)
            self._pending_fence = None
            self.out.append("\n```\n\n")

        elif tag == "code":
            if self.pre_depth == 0:
                self._emit("`")
            # For fenced blocks the closing ``` is emitted by </pre>

        # ── inline formatting ─────────────────────────────────
        elif tag in self.BOLD_TAGS:
            self._emit("**")
        elif tag in self.ITALIC_TAGS:
            self._emit("*")
        elif tag in self.DEL_TAGS:
            self._emit("~~")
        elif tag in self.UNDER_TAGS:
            self._emit("__")

        # ── links ─────────────────────────────────────────────
        elif tag == "a":
            self.in_link = False
            lt   = "".join(self.link_buf).strip()
            href = self.link_href
            if lt and href:
                self.out.append(f"[{lt}]({href})")
            elif lt:
                self.out.append(lt)
            elif href:
                self.out.append(href)
            self.link_buf  = []
            self.link_href = ""

        # ── tables ────────────────────────────────────────────
        elif tag in ("td", "th"):
            self.in_cell = False
            self.td_buf.append("".join(self.cell_buf).strip())
            self.cell_buf = []

        elif tag == "tr":
            row = "| " + " | ".join(self.td_buf) + " |"
            self.out.append(row + "\n")
            if self.header_row:
                sep = "| " + " | ".join(["---"] * len(self.td_buf)) + " |"
                self.out.append(sep + "\n")
                self.header_row = False
            self.td_buf = []

        elif tag == "table":
            self.in_table = False
            self.out.append("\n")

    def handle_data(self, data: str):
        if self.ignore_depth:
            return
        if self.pre_depth:
            # Inside a <pre> block: preserve whitespace exactly.
            self.out.append(data)
        else:
            # Collapse internal whitespace to a single space.
            self._emit(re.sub(r"\s+", " ", data))

    def handle_entityref(self, name: str):
        _ENTITIES = {
            "amp": "&", "lt": "<", "gt": ">", "quot": '"',
            "nbsp": " ", "mdash": "—", "ndash": "–", "hellip": "…",
            "ldquo": "\u201c", "rdquo": "\u201d",
            "lsquo": "\u2018", "rsquo": "\u2019",
        }
        self._emit(_ENTITIES.get(name, f"&{name};"))

    def handle_charref(self, name: str):
        try:
            ch = chr(int(name[1:], 16) if name.lower().startswith("x") else int(name))
            self._emit(ch)
        except (ValueError, OverflowError):
            pass

    def get_md(self) -> str:
        raw = "".join(str(x) for x in self.out)
        # Collapse 3+ blank lines to 2
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        # Remove spurious blank lines between a list marker and its text body
        # e.g. "- \n\nSome text" → "- Some text"
        raw = re.sub(r"((?:^|\n)[ \t]*(?:[-*+]|\d+[.)]) *)\n\n+", r"\1", raw)
        # Trim trailing whitespace per line
        raw = "\n".join(line.rstrip() for line in raw.split("\n"))
        return raw.strip()


def _html_to_md(html: str) -> str:
    """
    Convert an HTML string to Markdown.

    Prefers html2text when available (handles edge cases better).
    Falls back to the custom _MDParser above, which correctly emits
    language-annotated fenced code blocks (```python, ```bash, etc.).
    """
    if not html or not html.strip():
        return ""

    if HAS_HTML2TEXT:
        try:
            return _H2T.handle(html).strip()
        except Exception:
            pass

    try:
        p = _MDParser()
        p.feed(html)
        return p.get_md()
    except Exception:
        # Last-resort: strip all tags
        return re.sub(r"<[^>]+>", "", html).strip()


# ─────────────────────────────────────────────────────────────
# UI Helpers
# ─────────────────────────────────────────────────────────────

def print_banner():
    lines = [
        ("  ╔════════════════════════════════════════╗\n", "primary"),
        ("  ║  ", "primary"),
        ("🐋 chat.deepseek.com  ", "bold #60A5FA"),
        ("Playwright Scraper  ", "#60A5FA"),
        ("║\n", "primary"),
        ("  ║  ", "primary"),
        ("      Modern Edition · Rich UI            ", "muted"),
        ("║\n", "primary"),
        ("  ╚════════════════════════════════════════╝", "primary"),
    ]
    t = Text()
    for s, style in lines:
        t.append(s, style=style)
    console.print()
    console.print(t)
    console.print()


def print_help():
    table = Table(
        box=box.ROUNDED,
        border_style="#2563EB",
        show_header=True,
        header_style="bold #0EA5E9",
        padding=(0, 2),
    )
    table.add_column("Command",     style="bold #0EA5E9", no_wrap=True)
    table.add_column("Description", style="#94A3B8")
    for cmd, desc in [
        ("/new",     "Start a new conversation"),
        ("/history", "Display full conversation history"),
        ("/debug",   "Inspect DOM state"),
        ("/refresh", "Reload the page"),
        ("/quit",    "Exit"),
    ]:
        table.add_row(cmd, desc)
    console.print(Panel(table, title="[primary]Commands[/primary]",
                        border_style="#2563EB", padding=(1, 2)))
    console.print()


def _spinner_ctx(message: str, style: str = "#60A5FA"):
    renderable = Align.left(Spinner("dots2", text=Text(f"  {message}", style=style)))
    return Live(renderable, console=console, refresh_per_second=12, transient=True)


def render_response(md_text: str, reasoning_blocks: list, elapsed: float):
    tags = ["[ai_label]🐋 DeepSeek[/ai_label]"]
    if reasoning_blocks:
        tags.append("[think_label]💭 Thought[/think_label]")
    title = "  ".join(tags) + f"  [muted]({elapsed:.1f}s)[/muted]"
    console.print(Panel(
        Padding(Markdown(md_text, code_theme="monokai", hyperlinks=True), (1, 2)),
        title=title, title_align="left",
        border_style="secondary",
        box=box.ROUNDED, padding=(0, 1),
    ))
    console.print()


def render_thinking(reasoning_blocks: list):
    total = len(reasoning_blocks)
    for i, block in enumerate(reasoning_blocks, 1):
        label = f"💭 Reasoning{f' [{i}/{total}]' if total > 1 else ''}"
        console.print(Panel(
            Padding(Markdown(block, code_theme="monokai"), (1, 2)),
            title=f"[think_label]{label}[/think_label]", title_align="left",
            border_style="think_label", box=box.ROUNDED,
        ))
        console.print()


# ─────────────────────────────────────────────────────────────
# Scraper
# ─────────────────────────────────────────────────────────────

class DeepSeekScraper:
    INPUT_SEL = "textarea._27c9245"
    MSG_SEL   = "div.ds-message._63c77b1"

    # Generating state:   ._52c986b present  AND  .bd74640a ABSENT  AND  aria-disabled=false
    # Idle/done state:    ._52c986b present  AND  .bd74640a PRESENT AND  aria-disabled=true
    STOP_BTN_ACTIVE_SEL = "div._52c986b:not(.bd74640a)"

    def __init__(
        self,
        headless:        bool = False,
        enable_search:   bool = False,
        enable_deepthink: bool = False,
        enable_expert:   bool = False,
    ):
        self.headless         = headless
        self.enable_search    = enable_search
        self.enable_deepthink = enable_deepthink
        self.enable_expert    = enable_expert
        self.browser          = None
        self.context          = None
        self.page             = None
        self.playwright       = None

    # ─────────────────────────────────────────────────────────
    # Browser Setup
    # ─────────────────────────────────────────────────────────

    def start(self):
        with _spinner_ctx("Launching Chromium…"):
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(
                headless=self.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--start-maximized",
                ],
            )
            self.context = self.browser.new_context(
                viewport=None,
                no_viewport=True,
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                locale="en-US",
            )
            self.page = self.context.new_page()
        console.print("[success]✔[/success]  Browser launched")

        self._load_cookies_silent()

        with _spinner_ctx("Loading chat.deepseek.com…"):
            self.page.goto("https://chat.deepseek.com/",
                           wait_until="domcontentloaded", timeout=30000)
        console.print("[success]✔[/success]  Page loaded")

        restored = self._restore_storage()
        if restored:
            with _spinner_ctx("Applying saved session…"):
                self.page.reload(wait_until="domcontentloaded", timeout=20000)
                time.sleep(1.2)

        self._handle_auth()
        self._handle_captcha()

        with _spinner_ctx("Waiting for input box…"):
            self._wait_for_input()
        console.print("[success]✔[/success]  Chat input ready\n")

        self._configure_toggles()

    # ─────────────────────────────────────────────────────────
    # Session Persistence
    # ─────────────────────────────────────────────────────────

    def _load_cookies_silent(self):
        if not COOKIES_FILE.exists():
            return
        try:
            raw     = json.loads(COOKIES_FILE.read_text())
            cleaned = []
            for c in raw:
                ss = c.get("sameSite", "Lax")
                if ss not in ("Strict", "Lax", "None"):
                    c["sameSite"] = "Lax"
                if c.get("expires", -1) in (-1, 0, None):
                    c["expires"] = FAR_FUTURE
                cleaned.append(c)
            self.context.add_cookies(cleaned)
            console.print(
                f"[success]✔[/success]  Cookies loaded "
                f"([muted]{len(cleaned)} cookies[/muted])"
            )
        except Exception as e:
            console.print(f"[warning]⚠  Could not load cookies: {e}[/warning]")

    def _restore_storage(self) -> bool:
        if not STORAGE_FILE.exists():
            return False
        try:
            data    = json.loads(STORAGE_FILE.read_text())
            local   = data.get("localStorage",  {})
            session = data.get("sessionStorage", {})
            if not local and not session:
                return False
            self.page.evaluate("""(state) => {
                try {
                    for (const [k, v] of Object.entries(state.localStorage || {}))
                        localStorage.setItem(k, v);
                    for (const [k, v] of Object.entries(state.sessionStorage || {}))
                        sessionStorage.setItem(k, v);
                } catch(e) { console.warn('storage restore error', e); }
            }""", {"localStorage": local, "sessionStorage": session})
            console.print(
                f"[success]✔[/success]  Storage restored "
                f"([muted]{len(local)} local, {len(session)} session keys[/muted])"
            )
            return True
        except Exception as e:
            console.print(f"[warning]⚠  Could not restore storage: {e}[/warning]")
            return False

    def _page_is_deepseek(self) -> bool:
        try:
            url = self.page.url
            return "deepseek.com" in url and not url.startswith("about:")
        except Exception:
            return False

    def _save_session(self):
        errors = []

        try:
            cookies = self.context.cookies()
            for c in cookies:
                if c.get("expires", -1) in (-1, 0, None):
                    c["expires"] = FAR_FUTURE
                ss = c.get("sameSite", "Lax")
                if ss not in ("Strict", "Lax", "None"):
                    c["sameSite"] = "Lax"
            COOKIES_FILE.write_text(json.dumps(cookies, indent=2))
        except Exception as e:
            errors.append(f"cookies: {e}")

        if self._page_is_deepseek():
            try:
                storage = self.page.evaluate("""() => {
                    const safe = (fn) => { try { return fn(); } catch(e) { return {}; } };
                    return {
                        localStorage: safe(() => Object.fromEntries(
                            Object.keys(localStorage).map(k => [k, localStorage.getItem(k)])
                        )),
                        sessionStorage: safe(() => Object.fromEntries(
                            Object.keys(sessionStorage).map(k => [k, sessionStorage.getItem(k)])
                        )),
                    };
                }""")
                total = (len(storage.get("localStorage",  {})) +
                         len(storage.get("sessionStorage", {})))
                if total > 0:
                    STORAGE_FILE.write_text(json.dumps(storage, indent=2))
            except Exception as e:
                errors.append(f"storage: {e}")

        if errors:
            console.print(f"[warning]⚠  Session saved with warnings: {', '.join(errors)}[/warning]")
        else:
            console.print(f"[success]✔[/success]  Session saved → [muted]{SESSION_DIR}[/muted]")

    # ─────────────────────────────────────────────────────────
    # Auth / Captcha
    # ─────────────────────────────────────────────────────────

    def _handle_auth(self):
        time.sleep(1.5)
        url = self.page.url
        if any(x in url for x in ("/sign_in", "/login", "/auth")):
            console.print(Panel(
                f"[warning]Login required.[/warning]\n[muted]{url}[/muted]\n\n"
                "Log in inside the browser window.\n"
                "When done, press [accent]Enter[/accent] here.",
                title="[warning]🔐 Authentication[/warning]",
                border_style="warning", box=box.ROUNDED,
            ))
            input()
            try:
                self.page.wait_for_function(
                    "() => !window.location.href.includes('/sign_in')",
                    timeout=60000,
                )
            except Exception:
                pass
            time.sleep(1.5)
            self._save_session()
        else:
            self._save_session()

    def _handle_captcha(self):
        for sig in [
            "iframe[src*='recaptcha']",
            "iframe[src*='captcha']",
            ".g-recaptcha",
            "text=Verify you are human",
            "text=Security Check",
        ]:
            try:
                if self.page.locator(sig).count() > 0:
                    console.print(Panel(
                        "[warning]CAPTCHA detected.[/warning]\n"
                        "Solve it in the browser, then press [accent]Enter[/accent].",
                        title="[warning]⚠  CAPTCHA[/warning]",
                        border_style="warning", box=box.ROUNDED,
                    ))
                    input()
                    self._save_session()
                    return
            except Exception:
                pass

    def _wait_for_input(self):
        try:
            self.page.wait_for_selector(self.INPUT_SEL, state="visible", timeout=20000)
        except Exception:
            console.print("[error]✘  Input not found — page may need login[/error]")
            console.print(f"[muted]   {self.page.url}[/muted]")

    # ─────────────────────────────────────────────────────────
    # Toggle Configuration (Search / DeepThink / Expert)
    # ─────────────────────────────────────────────────────────

    def _configure_toggles(self):
        """
        Enable or disable the Search, DeepThink, and Expert toggles
        according to the flags set at construction time.

        Toggle state is read from aria-pressed (Search / DeepThink) and
        aria-checked (Expert).  The class ``ds-toggle-button--selected``
        is also present when a toggle is active, but aria-pressed is the
        canonical source of truth.

        Search is ON by default in the DeepSeek UI, so we always run this
        method — even when no flags are set — so we can turn Search off.

        Button identification
        ────────────────────
        Search and DeepThink share the same CSS classes:
          off: ds-atom-button f79352dc ds-toggle-button ds-toggle-button--md
          on:  … ds-toggle-button--selected …

        We locate them by inner text (case-insensitive) so the logic stays
        correct even if DeepSeek reorders the toolbar.

        Expert uses [data-model-type="expert"] with aria-checked.
        """
        console.print("[info]  Configuring feature toggles…[/info]")

        # ── helpers ───────────────────────────────────────────

        def _find_toggle_by_text(label_text: str):
            """
            Return the first ds-toggle-button whose visible text contains
            label_text (case-insensitive).  Returns None when not found.
            """
            try:
                # aria-pressed is present on both enabled and disabled states
                candidates = self.page.locator(
                    "button.ds-toggle-button[aria-pressed], "
                    "div.ds-toggle-button[aria-pressed]"
                )
                count = candidates.count()
                for i in range(count):
                    el = candidates.nth(i)
                    try:
                        text = (el.inner_text() or "").strip().lower()
                    except Exception:
                        text = ""
                    if label_text.lower() in text:
                        return el
                # Fallback: return None — caller will warn
                return None
            except Exception:
                return None

        def _toggle_button(label: str, want_enabled: bool):
            """
            Click a Search/DeepThink toggle only when its current state
            differs from want_enabled.
            """
            btn = _find_toggle_by_text(label)
            if btn is None:
                console.print(
                    f"[warning]⚠  {label} toggle not found — skipping[/warning]"
                )
                return
            try:
                current = (btn.get_attribute("aria-pressed") or "false").lower()
                is_on   = current == "true"
                if want_enabled and not is_on:
                    btn.click()
                    time.sleep(0.4)
                    console.print(f"[success]✔[/success]  {label} enabled")
                elif not want_enabled and is_on:
                    btn.click()
                    time.sleep(0.4)
                    console.print(f"[muted]  {label} disabled[/muted]")
                else:
                    state = "on" if is_on else "off"
                    console.print(
                        f"[muted]  {label} already {state} — no change[/muted]"
                    )
            except Exception as e:
                console.print(f"[warning]⚠  {label} toggle error: {e}[/warning]")

        def _toggle_expert(want_enabled: bool):
            """
            Select or deselect the Expert model option.
            Uses aria-checked (not aria-pressed) on [data-model-type="expert"].
            """
            try:
                btn = self.page.locator("[data-model-type='expert']").first
                if btn.count() == 0:
                    console.print(
                        "[warning]⚠  Expert model button not found — skipping[/warning]"
                    )
                    return
                current = (btn.get_attribute("aria-checked") or "false").lower()
                is_on   = current == "true"
                if want_enabled and not is_on:
                    btn.click()
                    time.sleep(0.4)
                    console.print("[success]✔[/success]  Expert model enabled")
                elif not want_enabled and is_on:
                    btn.click()
                    time.sleep(0.4)
                    console.print("[muted]  Expert model disabled[/muted]")
                else:
                    state = "on" if is_on else "off"
                    console.print(
                        f"[muted]  Expert model already {state} — no change[/muted]"
                    )
            except Exception as e:
                console.print(f"[warning]⚠  Expert toggle error: {e}[/warning]")

        # ── Search ────────────────────────────────────────────
        # Default in the DeepSeek UI is ON — turn it off unless --search was given.
        _toggle_button("Search", want_enabled=self.enable_search)

        # ── DeepThink ─────────────────────────────────────────
        # Default is OFF — only enable when --deepthink was given.
        _toggle_button("DeepThink", want_enabled=self.enable_deepthink)

        # ── Expert model ──────────────────────────────────────
        # Default is OFF — only enable when --expert was given.
        if self.enable_expert:
            _toggle_expert(want_enabled=True)

        console.print()

    # ─────────────────────────────────────────────────────────
    # Core: Send Message
    # ─────────────────────────────────────────────────────────

    def send_message(self, message: str) -> tuple:
        """Returns (md_response: str, reasoning_blocks: list[str], elapsed: float)."""
        if not self.page:
            return ("[Error] Browser not started.", [], 0.0)
        try:
            inp = self.page.locator(self.INPUT_SEL)
            inp.click()
            inp.fill(message)
            time.sleep(0.15)
            inp.press("Enter")

            t0 = time.time()
            self._wait_for_response_complete(timeout=300)
            elapsed = time.time() - t0

            md, reasoning_blocks = self._scrape_last_response()
            return (md, reasoning_blocks, elapsed)
        except Exception as e:
            return (f"[Error] {e}", [], 0.0)

    # ─────────────────────────────────────────────────────────
    # Wait Logic
    # ─────────────────────────────────────────────────────────

    def _stop_btn_active(self) -> bool:
        """True while DeepSeek is generating a response."""
        try:
            return self.page.locator(self.STOP_BTN_ACTIVE_SEL).count() > 0
        except Exception:
            return False

    def _wait_for_response_complete(self, timeout: int = 300):
        deadline = time.time() + timeout

        def _spin(label: str, style: str = "#60A5FA"):
            return Align.left(Spinner("dots2", text=Text(f"  {label}", style=style)))

        with Live(_spin("Waiting for DeepSeek…"), console=console,
                  refresh_per_second=12, transient=True) as live:

            # Wait up to 10 s for generation to start
            start_deadline = time.time() + 10
            while time.time() < start_deadline:
                if self._stop_btn_active():
                    break
                time.sleep(0.2)

            live.update(_spin("Generating response…"))

            # Wait for generation to finish
            while time.time() < deadline:
                if not self._stop_btn_active():
                    live.update(Text(""))
                    return
                try:
                    thinking = self.page.locator(
                        "div.e1675d8b.ds-think-content._767406f"
                    ).count() > 0
                    live.update(
                        _spin("Reasoning…", "#FCD34D") if thinking
                        else _spin("Generating response…")
                    )
                except Exception:
                    pass
                time.sleep(0.25)

        console.print("[warning]⚠  Response timed out.[/warning]")

    # ─────────────────────────────────────────────────────────
    # Scraping
    # ─────────────────────────────────────────────────────────

    def _scrape_last_response(self) -> tuple:
        """
        Returns (response_md: str, reasoning_blocks: list[str]).

        DOM layout inside the last .ds-message:
          1 ds-markdown block  → pure response, no reasoning
          2+ ds-markdown blocks → blocks[:-1] are reasoning steps; blocks[-1] is the answer
        """
        try:
            info = self.page.evaluate("""() => {
                const msgs = document.querySelectorAll("div.ds-message._63c77b1");
                if (!msgs.length) return { count: 0, texts: [] };
                const last   = msgs[msgs.length - 1];
                const blocks = Array.from(last.querySelectorAll("div.ds-markdown"));
                const texts  = blocks.map(b => {
                    const c = b.cloneNode(true);
                    c.querySelectorAll(
                        "svg,script,style,noscript,button,span.ds-markdown-cite"
                    ).forEach(el => el.remove());
                    return c.outerHTML;
                });
                if (!texts.length) {
                    const c2 = last.cloneNode(true);
                    c2.querySelectorAll(
                        "svg,script,style,noscript,button,span.ds-markdown-cite"
                    ).forEach(el => el.remove());
                    return { count: 0, texts: [c2.outerHTML] };
                }
                return { count: blocks.length, texts };
            }""")

            texts = info.get("texts", [])
            count = info.get("count", 0)

            if not texts:
                return ("[Error] Empty response", [])

            if count <= 1:
                return (_html_to_md(texts[0]) or "[Error] Empty response", [])

            response_md      = _html_to_md(texts[-1])
            reasoning_blocks = [_html_to_md(h) for h in texts[:-1] if h.strip()]
            return (response_md or "[Error] Empty response", reasoning_blocks)

        except Exception as e:
            return (f"[Error] {e}", [])

    # ─────────────────────────────────────────────────────────
    # New Chat
    # ─────────────────────────────────────────────────────────

    def new_chat(self):
        for sel in [
            "button[aria-label*='new' i]",
            "button[aria-label*='New' i]",
            "button:has-text('New Chat')",
            "button:has-text('New chat')",
            "[class*='new-chat']",
            "a[href='/']",
        ]:
            try:
                btn = self.page.locator(sel).first
                if btn.count() > 0 and btn.is_visible():
                    btn.click()
                    time.sleep(1)
                    self._wait_for_input()
                    console.print("[success]✔[/success]  New conversation started.")
                    return
            except Exception:
                continue

        with _spinner_ctx("Starting new chat…"):
            self.page.goto("https://chat.deepseek.com/",
                           wait_until="domcontentloaded", timeout=15000)
            self._wait_for_input()
        console.print("[success]✔[/success]  New conversation started.")

    # ─────────────────────────────────────────────────────────
    # Conversation History
    # ─────────────────────────────────────────────────────────

    def get_full_conversation(self) -> list:
        try:
            return self.page.evaluate("""() => {
                const history = [];
                const msgs = document.querySelectorAll("div.ds-message._63c77b1");
                msgs.forEach(msg => {
                    const mdBlocks = Array.from(msg.querySelectorAll("div.ds-markdown"));
                    const el       = mdBlocks.length
                        ? mdBlocks[mdBlocks.length - 1] : msg;
                    const clone    = el.cloneNode(true);
                    clone.querySelectorAll(
                        "svg,script,style,noscript,button"
                    ).forEach(e => e.remove());
                    history.push({ role: "assistant", content: clone.innerHTML.trim() });
                });
                for (const sel of [
                    "[class*='user-message']",
                    "[class*='human-message']",
                    "[data-role='user']",
                ]) {
                    const els = document.querySelectorAll(sel);
                    if (els.length) {
                        els.forEach(e => history.push({
                            role: "user",
                            content: e.innerText?.trim() || "",
                        }));
                        break;
                    }
                }
                return history;
            }""") or []
        except Exception as e:
            console.print(f"[error]History error: {e}[/error]")
            return []

    # ─────────────────────────────────────────────────────────
    # DOM Debug
    # ─────────────────────────────────────────────────────────

    def debug_dom(self):
        try:
            info = self.page.evaluate("""() => {
                const stopBtn = document.querySelector("button._52c986b.bd74640a");
                const msgs    = document.querySelectorAll("div.ds-message._63c77b1");
                const last    = msgs.length ? msgs[msgs.length - 1] : null;
                const lsKeys  = Object.keys(localStorage);
                return {
                    url:                window.location.href,
                    inputExists:        !!document.querySelector("textarea._27c9245"),
                    stopBtnExists:      !!stopBtn,
                    stopBtnAriaDisabled: stopBtn
                        ? stopBtn.getAttribute("aria-disabled") : null,
                    dsMessageCount:     msgs.length,
                    lastMsgMdBlocks:    last
                        ? last.querySelectorAll("div.ds-markdown").length : 0,
                    lastMsgThinkBlocks: last
                        ? last.querySelectorAll(
                            "div.e1675d8b.ds-think-content._767406f"
                          ).length : 0,
                    localStorageTotal:  lsKeys.length,
                    authRelatedKeys:    lsKeys.filter(k =>
                        /token|auth|user|session/i.test(k)),
                    lastMsgPreview:     last
                        ? (last.innerText?.substring(0, 200) || "") : "none",
                    viewportWidth:      window.innerWidth,
                    viewportHeight:     window.innerHeight,
                };
            }""")

            t = Table(
                title="[primary]DOM State[/primary]",
                box=box.ROUNDED, border_style="#2563EB",
                show_header=False, padding=(0, 2),
            )
            t.add_column("Key",   style="bold #0EA5E9", no_wrap=True)
            t.add_column("Value", style="#60A5FA")

            def tick(v):
                return "[success]✔  yes[/success]" if v else "[error]✘  no[/error]"

            t.add_row("URL",                   f"[muted]{info.get('url')}[/muted]")
            t.add_row("Textarea input",        tick(info.get("inputExists")))
            t.add_row("Stop button exists",    tick(info.get("stopBtnExists")))
            t.add_row("Stop aria-disabled",    str(info.get("stopBtnAriaDisabled")))
            t.add_row(".ds-message count",     str(info.get("dsMessageCount")))
            t.add_row("Last msg ds-markdown",  str(info.get("lastMsgMdBlocks")) + " blocks")
            t.add_row("Last msg think blocks", str(info.get("lastMsgThinkBlocks")))
            t.add_row("localStorage total",    str(info.get("localStorageTotal")))
            t.add_row("Auth-related keys",     str(info.get("authRelatedKeys", [])))
            t.add_row("Viewport size",
                      f"{info.get('viewportWidth')}×{info.get('viewportHeight')} px")
            t.add_row("Last 200 chars",
                      f"[muted]{str(info.get('lastMsgPreview', ''))[:200]}…[/muted]")

            console.print()
            console.print(t)
            console.print()
        except Exception as e:
            console.print(f"[error]Debug error: {e}[/error]")

    # ─────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────

    def close(self):
        try:
            self._save_session()
        except Exception:
            pass
        try:
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# Main CLI
# ─────────────────────────────────────────────────────────────

def main():
    scraper = None
    print_banner()
    console.print(f"[muted]  Session folder: {SESSION_DIR}[/muted]\n")

    try:
        scraper = DeepSeekScraper(headless=False)
        scraper.start()
        print_help()

        while True:
            try:
                console.print(Rule(style="muted"))
                user_input = console.input(
                    "[user_label]  You  [/user_label][muted] › [/muted]"
                ).strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    cmd = user_input.lower().strip()

                    if cmd == "/quit":
                        break
                    elif cmd == "/new":
                        scraper.new_chat()
                    elif cmd == "/history":
                        history = scraper.get_full_conversation()
                        if not history:
                            console.print("[muted]  No messages found.[/muted]\n")
                        for msg in history:
                            if msg["role"] == "assistant":
                                md = _html_to_md(msg["content"])
                                console.print(Panel(
                                    Padding(Markdown(md, code_theme="monokai"), (1, 2)),
                                    title="[ai_label]🐋 DeepSeek[/ai_label]",
                                    border_style="secondary", box=box.ROUNDED,
                                ))
                            else:
                                console.print(Panel(
                                    Padding(Text(msg["content"]), (0, 2)),
                                    title="[user_label]  You  [/user_label]",
                                    border_style="user_label", box=box.ROUNDED,
                                ))
                            console.print()
                    elif cmd == "/debug":
                        scraper.debug_dom()
                    elif cmd == "/refresh":
                        with _spinner_ctx("Refreshing…"):
                            scraper.page.goto(
                                "https://chat.deepseek.com/",
                                wait_until="domcontentloaded", timeout=15000,
                            )
                            scraper._wait_for_input()
                        console.print("[success]✔[/success]  Refreshed.")
                    else:
                        console.print(
                            f"[error]Unknown command:[/error] [cmd]{cmd}[/cmd]  "
                            "[muted]— type /help to see commands[/muted]"
                        )
                    continue

                # ── Send & render ──────────────────────────────
                console.print()
                md, reasoning_blocks, elapsed = scraper.send_message(user_input)
                if reasoning_blocks:
                    render_thinking(reasoning_blocks)
                render_response(md, reasoning_blocks, elapsed)

            except KeyboardInterrupt:
                console.print("\n[muted]  Interrupted.[/muted]")
                break
            except Exception as e:
                console.print(f"[error]Error: {e}[/error]")

    finally:
        if scraper:
            scraper.close()
        console.print()
        console.print(Panel(
            Align.center(Text("Session ended  ·  Goodbye!", style="muted")),
            border_style="#2563EB", box=box.ROUNDED,
        ))


if __name__ == "__main__":
    main()
