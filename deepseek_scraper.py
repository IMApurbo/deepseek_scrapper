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
import os
from pathlib import Path
from datetime import datetime
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

# ── Optional html2text ────────────────────────────────────────
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
STORAGE_FILE = SESSION_DIR / "storage.json"   # localStorage + sessionStorage
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# Far-future expiry (year 2099) — forces session cookies to persist across launches
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


def spinner_ctx(message: str, style: str = "#60A5FA"):
    renderable = Align.left(Spinner("dots2", text=Text(f"  {message}", style=style)))
    return Live(renderable, console=console, refresh_per_second=12, transient=True)


def render_response(md_text: str, was_thinking: bool, elapsed: float):
    tags = ["[ai_label]🐋 DeepSeek[/ai_label]"]
    if was_thinking:
        tags.append("[think_label]💭 Thought[/think_label]")
    title = "  ".join(tags) + f"  [muted]({elapsed:.1f}s)[/muted]"
    console.print(Panel(
        Padding(Markdown(md_text, code_theme="monokai", hyperlinks=True), (1, 2)),
        title=title, title_align="left",
        border_style="secondary",
        box=box.ROUNDED, padding=(0, 1),
    ))
    console.print()


def render_thinking(thinking_text: str):
    console.print(Panel(
        Padding(Text(thinking_text, style="muted italic"), (1, 2)),
        title="[think_label]💭 Reasoning[/think_label]", title_align="left",
        border_style="think_label", box=box.ROUNDED,
    ))
    console.print()


# ─────────────────────────────────────────────────────────────
# Scraper
# ─────────────────────────────────────────────────────────────

class DeepSeekScraper:
    INPUT_SEL    = "textarea._27c9245"
    STOP_BTN_SEL = "button._52c986b.bd74640a"
    MSG_SEL      = "div.ds-message._63c77b1"

    def __init__(self, headless: bool = False):
        self.headless   = headless
        self.browser    = None
        self.context    = None
        self.page       = None
        self.playwright = None

    # ─────────────────────────────────────────────────────────
    # Browser Setup
    # ─────────────────────────────────────────────────────────

    def start(self):
        with spinner_ctx("Launching Chromium…"):
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(
                headless=self.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--start-maximized",          # open browser maximized
                ],
            )
            # ── FIX: viewport=None disables the fixed viewport so the browser
            #         window controls page size — resizing works naturally. ──
            self.context = self.browser.new_context(
                viewport=None,                    # no fixed viewport → resize works
                no_viewport=True,                 # explicit opt-out of viewport emulation
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                locale="en-US",
            )
            self.page = self.context.new_page()
        console.print("[success]✔[/success]  Browser launched")

        # Step 1: inject saved cookies into context BEFORE navigation
        self._load_cookies_silent()

        # Step 2: navigate to site
        with spinner_ctx("Loading chat.deepseek.com…"):
            self.page.goto("https://chat.deepseek.com/",
                           wait_until="domcontentloaded", timeout=30000)
        console.print("[success]✔[/success]  Page loaded")

        # Step 3: restore localStorage/sessionStorage (must happen AFTER navigation,
        #         as storage is origin-scoped and needs a live page context)
        restored = self._restore_storage()
        if restored:
            with spinner_ctx("Applying saved session…"):
                # Reload so React/auth state picks up restored tokens
                self.page.reload(wait_until="domcontentloaded", timeout=20000)
                time.sleep(1.2)

        # Step 4: check auth
        self._handle_auth()
        self._handle_captcha()

        with spinner_ctx("Waiting for input box…"):
            self._wait_for_input()
        console.print("[success]✔[/success]  Chat input ready\n")

    # ─────────────────────────────────────────────────────────
    # Session Persistence
    # ─────────────────────────────────────────────────────────

    def _load_cookies_silent(self):
        """Inject saved cookies into Playwright context before first navigation."""
        if not COOKIES_FILE.exists():
            return
        try:
            raw = json.loads(COOKIES_FILE.read_text())
            cleaned = []
            for c in raw:
                # Playwright requires sameSite to be Strict / Lax / None
                ss = c.get("sameSite", "Lax")
                if ss not in ("Strict", "Lax", "None"):
                    c["sameSite"] = "Lax"
                # Promote session-only cookies (expires -1) to persistent
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
        """Inject saved localStorage/sessionStorage into current page.
        Returns True if any data was restored."""
        if not STORAGE_FILE.exists():
            return False
        try:
            data = json.loads(STORAGE_FILE.read_text())
            local   = data.get("localStorage",   {})
            session = data.get("sessionStorage",  {})
            if not local and not session:
                return False

            self.page.evaluate("""(state) => {
                try {
                    for (const [k, v] of Object.entries(state.localStorage || {})) {
                        localStorage.setItem(k, v);
                    }
                    for (const [k, v] of Object.entries(state.sessionStorage || {})) {
                        sessionStorage.setItem(k, v);
                    }
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
        """True only when current page is on deepseek.com (localStorage accessible)."""
        try:
            url = self.page.url
            return "deepseek.com" in url and not url.startswith("about:")
        except:
            return False

    def _save_session(self):
        """Persist cookies (bumped to far-future) + localStorage + sessionStorage."""
        errors = []

        # Cookies (always safe — read from context, not from page)
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

        # localStorage + sessionStorage — only accessible when on deepseek.com origin.
        # Calling this on about:blank or a redirect raises SecurityError.
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
                total = len(storage.get("localStorage", {})) + len(storage.get("sessionStorage", {}))
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
        if any(x in url for x in ["/sign_in", "/login", "/auth"]):
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
            except:
                pass
            time.sleep(1.5)
            # Save immediately after successful login
            self._save_session()
        else:
            # Already authenticated — refresh saved session
            self._save_session()

    def _handle_captcha(self):
        for sig in ["iframe[src*='recaptcha']", "iframe[src*='captcha']",
                    ".g-recaptcha", "text=Verify you are human", "text=Security Check"]:
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
            except:
                pass

    def _wait_for_input(self):
        try:
            self.page.wait_for_selector(self.INPUT_SEL, state="visible", timeout=20000)
        except:
            console.print("[error]✘  Input not found — page may need login[/error]")
            console.print(f"[muted]   {self.page.url}[/muted]")

    # ─────────────────────────────────────────────────────────
    # Core: Send Message
    # ─────────────────────────────────────────────────────────

    def send_message(self, message: str) -> tuple:
        """Returns (md_response, thinking_md, elapsed)."""
        if not self.page:
            return ("[Error] Browser not started.", "", 0.0)
        try:
            inp = self.page.locator(self.INPUT_SEL)
            inp.click()
            inp.fill(message)
            time.sleep(0.15)
            inp.press("Enter")

            t0 = time.time()
            self._wait_for_response_complete(timeout=300)
            elapsed = time.time() - t0

            md, thinking = self._scrape_last_response()
            return (md, thinking, elapsed)
        except Exception as e:
            return (f"[Error] {e}", "", 0.0)

    # ─────────────────────────────────────────────────────────
    # Wait Logic
    # ─────────────────────────────────────────────────────────

    def _stop_btn_active(self) -> bool:
        """True while DeepSeek is generating (stop button aria-disabled=false)."""
        try:
            btn = self.page.locator(self.STOP_BTN_SEL).first
            if btn.count() == 0:
                return False
            return btn.get_attribute("aria-disabled") == "false"
        except:
            return False

    def _wait_for_response_complete(self, timeout: int = 300):
        deadline = time.time() + timeout

        def _spin(label, style="#60A5FA"):
            return Align.left(Spinner("dots2", text=Text(f"  {label}", style=style)))

        with Live(_spin("Waiting for DeepSeek…"), console=console,
                  refresh_per_second=12, transient=True) as live:

            # Wait up to 10s for generation to start
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
                    think_count = self.page.locator(
                        "div.e1675d8b.ds-think-content._767406f"
                    ).count()
                    if think_count > 0:
                        live.update(_spin("Reasoning…", "#FCD34D"))
                    else:
                        live.update(_spin("Generating response…"))
                except:
                    pass
                time.sleep(0.25)

        console.print("[warning]⚠  Response timed out.[/warning]")

    # ─────────────────────────────────────────────────────────
    # Scraping
    # ─────────────────────────────────────────────────────────

    def _scrape_last_response(self) -> tuple:
        """
        Returns (response_md, thinking_md).

        Rules (per spec):
          - Get last div.ds-message._63c77b1
          - Count div.ds-markdown blocks inside it:
            1 block  -> plain response, no thinking
            2 blocks -> [0]=thinking, [1]=response
            3+ blocks-> [0..n-2]=reasoning joined, [-1]=response

        NOTE: We return innerText (plain text) from JS, not innerHTML, to avoid
        SyntaxError from backticks/special chars in HTML breaking the JS string.
        The html2text conversion then happens on the raw outer HTML fetched separately.
        """
        try:
            # Step 1: get block count and plain text safely via JS
            info = self.page.evaluate("""() => {
                var msgs = document.querySelectorAll("div.ds-message._63c77b1");
                if (!msgs.length) return { count: 0, texts: [] };
                var last = msgs[msgs.length - 1];
                var blocks = last.querySelectorAll("div.ds-markdown");
                var texts = [];
                for (var i = 0; i < blocks.length; i++) {
                    var c = blocks[i].cloneNode(true);
                    var junk = c.querySelectorAll("svg,script,style,noscript,button");
                    for (var j = 0; j < junk.length; j++) junk[j].remove();
                    texts.push(c.outerHTML);
                }
                if (texts.length === 0) {
                    var c2 = last.cloneNode(true);
                    var junk2 = c2.querySelectorAll("svg,script,style,noscript,button");
                    for (var j = 0; j < junk2.length; j++) junk2[j].remove();
                    texts.push(c2.outerHTML);
                }
                return { count: blocks.length, texts: texts };
            }""")

            texts = info.get("texts", [])
            count = info.get("count", 0)

            if not texts:
                return ("[Error] Empty response", "")

            if count <= 1:
                # Single block or fallback: just the response
                response_md = self._html_to_md(texts[0])
                return (response_md or "[Error] Empty response", "")
            else:
                # Last = response, everything before = reasoning
                response_md = self._html_to_md(texts[-1])
                thinking_md = self._html_to_md(" ".join(texts[:-1]))
                return (response_md or "[Error] Empty response", thinking_md)

        except Exception as e:
            return (f"[Error] {e}", "")

    def _html_to_md(self, html: str) -> str:
        if not html or not html.strip():
            return ""
        if HAS_HTML2TEXT:
            try:
                return _H2T.handle(html).strip()
            except:
                pass

        # Fallback minimal HTML-to-Markdown parser
        try:
            from html.parser import HTMLParser

            class _MDParser(HTMLParser):
                IGNORE = {'svg','script','style','noscript','button','input',
                          'select','form','head','meta','link',
                          'sup','sub','cite','time','footer','nav','aside'}
                BOLD={'strong','b'}; ITALIC={'em','i'}; CODE={'code'}
                DEL={'s','del','strike'}; UNDER={'u'}

                def __init__(self):
                    super().__init__()
                    self.out=[];self.stack=[];self.ignore_depth=0;self.pre_depth=0
                    self.list_stack=[];self.in_table=False;self.td_buf=[]
                    self.header_row=False;self.link_buf=[];self.in_link=False
                    self.cell_buf=[];self.in_cell=False

                def handle_starttag(self, tag, attrs):
                    tag=tag.lower(); adict=dict(attrs)
                    if self.ignore_depth or tag in self.IGNORE:
                        self.ignore_depth+=1; self.stack.append(tag); return
                    self.stack.append(tag)
                    if tag in ('h1','h2','h3','h4','h5','h6'):
                        self.out.append('\n\n'+'#'*int(tag[1])+' ')
                    elif tag=='p': self.out.append('\n\n')
                    elif tag in ('ul','ol'): self.list_stack.append([tag,0]); self.out.append('\n')
                    elif tag=='li':
                        if self.list_stack:
                            k,_=self.list_stack[-1]
                            if k=='ol': self.list_stack[-1][1]+=1; p=f"{self.list_stack[-1][1]}. "
                            else: p='• '
                            self.out.append(f'\n{"  "*(len(self.list_stack)-1)}{p}')
                    elif tag=='blockquote': self.out.append('\n\n> ')
                    elif tag=='pre': self.pre_depth+=1; self.out.append('\n\n```')
                    elif tag=='code':
                        if self.pre_depth==0: self.out.append('`')
                    elif tag in self.BOLD: self.out.append('**')
                    elif tag in self.ITALIC: self.out.append('*')
                    elif tag in self.DEL: self.out.append('~~')
                    elif tag in self.UNDER: self.out.append('__')
                    elif tag=='a':
                        self.stack[-1]=('a',adict.get('href','')); self.in_link=True; self.link_buf=[]
                    elif tag=='br': self.out.append('  \n')
                    elif tag=='hr': self.out.append('\n\n---\n\n')
                    elif tag=='table': self.in_table=True; self.out.append('\n\n')
                    elif tag=='tr': self.td_buf=[]
                    elif tag in ('th','thead'):
                        if tag=='thead': self.header_row=True
                        else: self.in_cell=True; self.cell_buf=[]
                    elif tag=='td': self.in_cell=True; self.cell_buf=[]

                def handle_endtag(self, tag):
                    tag=tag.lower()
                    if not self.stack: return
                    for i in range(len(self.stack)-1,-1,-1):
                        if self.stack[i]==tag or (isinstance(self.stack[i],tuple) and self.stack[i][0]==tag):
                            entry=self.stack.pop(i); break
                    else: return
                    if self.ignore_depth: self.ignore_depth-=1; return
                    if tag in ('h1','h2','h3','h4','h5','h6'): self.out.append('\n')
                    elif tag=='p': self.out.append('\n')
                    elif tag in ('ul','ol'):
                        if self.list_stack: self.list_stack.pop()
                        self.out.append('\n')
                    elif tag=='blockquote': self.out.append('\n\n')
                    elif tag=='pre': self.pre_depth-=1; self.out.append('\n```\n\n')
                    elif tag=='code':
                        if self.pre_depth==0: self.out.append('`')
                    elif tag in self.BOLD: self.out.append('**')
                    elif tag in self.ITALIC: self.out.append('*')
                    elif tag in self.DEL: self.out.append('~~')
                    elif tag in self.UNDER: self.out.append('__')
                    elif tag=='a':
                        href=entry[1] if isinstance(entry,tuple) else ''
                        self.in_link=False
                        lt=''.join(self.link_buf).strip()
                        if lt and href: self.out.append(f'[{lt}]({href})')
                        elif lt: self.out.append(lt)
                        elif href: self.out.append(href)
                        self.link_buf=[]
                    elif tag in ('td','th'):
                        self.in_cell=False
                        self.td_buf.append(''.join(self.cell_buf).strip())
                        self.cell_buf=[]
                    elif tag=='tr':
                        self.out.append(f'| {" | ".join(self.td_buf)} |\n')
                        if self.header_row:
                            self.out.append(f'| {" | ".join(["---"]*len(self.td_buf))} |\n')
                            self.header_row=False
                        self.td_buf=[]
                    elif tag=='table': self.in_table=False; self.out.append('\n')

                def handle_data(self, data):
                    if self.ignore_depth: return
                    if self.pre_depth: self.out.append(data)
                    elif self.in_link: self.link_buf.append(re.sub(r'\s+',' ',data))
                    elif self.in_cell: self.cell_buf.append(re.sub(r'\s+',' ',data))
                    else: self.out.append(re.sub(r'\s+',' ',data))

                def handle_entityref(self, name):
                    e={'amp':'&','lt':'<','gt':'>','quot':'"','nbsp':' ',
                       'mdash':'—','ndash':'–','hellip':'…',
                       'ldquo':'\u201c','rdquo':'\u201d','lsquo':'\u2018','rsquo':'\u2019'}
                    self.out.append(e.get(name, f'&{name};'))

                def handle_charref(self, name):
                    try:
                        self.out.append(
                            chr(int(name[1:],16) if name.startswith('x') else int(name))
                        )
                    except: pass

                def get_md(self):
                    return re.sub(r'\n{3,}','\n\n',
                                  ''.join(str(x) for x in self.out)).strip()

            p = _MDParser(); p.feed(html); return p.get_md()
        except:
            return re.sub(r'<[^>]+>', '', html).strip()

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
            except:
                continue

        with spinner_ctx("Starting new chat…"):
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
                const msgs = document.querySelectorAll('div.ds-message._63c77b1');
                msgs.forEach(msg => {
                    const mdBlocks = Array.from(msg.querySelectorAll('div.ds-markdown'));
                    const el = mdBlocks.length ? mdBlocks[mdBlocks.length - 1] : msg;
                    const clone = el.cloneNode(true);
                    clone.querySelectorAll('svg,script,style,noscript,button').forEach(e => e.remove());
                    history.push({ role: 'assistant', content: clone.innerHTML.trim() });
                });
                for (const sel of ['[class*="user-message"]','[class*="human-message"]',
                                    '[data-role="user"]']) {
                    const els = document.querySelectorAll(sel);
                    if (els.length) {
                        els.forEach(e => history.push({
                            role: 'user', content: e.innerText?.trim() || ''
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
                const stopBtn = document.querySelector('button._52c986b.bd74640a');
                const msgs = document.querySelectorAll('div.ds-message._63c77b1');
                const last = msgs.length ? msgs[msgs.length - 1] : null;
                const lsKeys = Object.keys(localStorage);
                const authKeys = lsKeys.filter(k =>
                    /token|auth|user|session/i.test(k)
                );
                return {
                    url: window.location.href,
                    inputExists: !!document.querySelector('textarea._27c9245'),
                    stopBtnExists: !!stopBtn,
                    stopBtnAriaDisabled: stopBtn ? stopBtn.getAttribute('aria-disabled') : null,
                    dsMessageCount: msgs.length,
                    lastMsgMdBlocks: last ? last.querySelectorAll('div.ds-markdown').length : 0,
                    lastMsgThinkBlocks: last
                        ? last.querySelectorAll('div.e1675d8b.ds-think-content._767406f').length
                        : 0,
                    localStorageTotal: lsKeys.length,
                    authRelatedKeys: authKeys,
                    lastMsgPreview: last ? (last.innerText?.substring(0, 200) || '') : 'none',
                    viewportWidth: window.innerWidth,
                    viewportHeight: window.innerHeight,
                };
            }""")

            t = Table(
                title="[primary]DOM State[/primary]",
                box=box.ROUNDED, border_style="#2563EB",
                show_header=False, padding=(0, 2),
            )
            t.add_column("Key",   style="bold #0EA5E9", no_wrap=True)
            t.add_column("Value", style="#60A5FA")

            def tick(v): return "[success]✔  yes[/success]" if v else "[error]✘  no[/error]"
            t.add_row("URL",                   f"[muted]{info.get('url')}[/muted]")
            t.add_row("Textarea input",        tick(info.get('inputExists')))
            t.add_row("Stop button exists",    tick(info.get('stopBtnExists')))
            t.add_row("Stop aria-disabled",    str(info.get('stopBtnAriaDisabled')))
            t.add_row(".ds-message count",     str(info.get('dsMessageCount')))
            t.add_row("Last msg ds-markdown",  str(info.get('lastMsgMdBlocks')) + " blocks")
            t.add_row("Last msg think blocks", str(info.get('lastMsgThinkBlocks')))
            t.add_row("localStorage total",    str(info.get('localStorageTotal')))
            t.add_row("Auth-related keys",     str(info.get('authRelatedKeys', [])))
            t.add_row("Viewport size",
                      f"{info.get('viewportWidth')}×{info.get('viewportHeight')} px")
            t.add_row("Last 200 chars",
                      f"[muted]{str(info.get('lastMsgPreview',''))[:200]}…[/muted]")

            console.print(); console.print(t); console.print()
        except Exception as e:
            console.print(f"[error]Debug error: {e}[/error]")

    # ─────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────

    def close(self):
        try:
            self._save_session()
        except:
            pass
        try:
            if self.browser:    self.browser.close()
            if self.playwright: self.playwright.stop()
        except:
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
                                md = scraper._html_to_md(msg["content"])
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
                        with spinner_ctx("Refreshing…"):
                            scraper.page.goto("https://chat.deepseek.com/",
                                              wait_until="domcontentloaded", timeout=15000)
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
                md, thinking, elapsed = scraper.send_message(user_input)
                if thinking:
                    render_thinking(thinking)
                render_response(md, bool(thinking), elapsed)

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
