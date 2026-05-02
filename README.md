# 🐋 DeepSeek Playwright Scraper

A Playwright-based terminal client and Anthropic API proxy for **chat.deepseek.com** — use DeepSeek's web interface from your terminal, or wire it up as a drop-in backend for any tool that speaks the Anthropic Messages API (Claude Code, LLM CLI, custom apps, etc.).

---

## Files

| File | Purpose |
|---|---|
| `deepseek_scraper.py` | Terminal chat client + core scraper class |
| `server.py` | Anthropic API proxy — exposes `POST /v1/messages` backed by DeepSeek |

---

## Requirements

- Python 3.10+
- Google Chrome / Chromium (installed by Playwright)

---

## Installation

```bash
# 1. Install Python dependencies
pip install playwright flask rich html2text

# 2. Install the Chromium browser
playwright install chromium
```

> `html2text` is optional but recommended — it produces cleaner Markdown from DeepSeek's HTML than the built-in fallback parser.

---

## Usage

### Terminal chat client

```bash
python deepseek_scraper.py
```

A Chromium window opens and navigates to `chat.deepseek.com`. If you are not logged in, the browser will pause and ask you to log in manually — after that the session is saved automatically and future launches skip the login step.

#### Chat commands

| Command | Description |
|---|---|
| `/new` | Start a new conversation |
| `/history` | Print the full conversation so far |
| `/debug` | Inspect live DOM state (selectors, viewport size, localStorage) |
| `/refresh` | Reload the page |
| `/quit` | Exit |

---

### Anthropic API proxy

```bash
python server.py
```

Then point any Anthropic-compatible tool at the local server:

```bash
export ANTHROPIC_BASE_URL="http://localhost:8765"
export ANTHROPIC_API_KEY="local-proxy-key"

claude        # Claude Code CLI
```

#### Server options

| Flag | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8765` | Port |
| `--headless` | off | Run browser without a visible window |
| `--no-warmup` | off | Skip pre-launching browser (lazy init on first request) |

```bash
python server.py --headless --port 9000
```

#### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/messages` | Anthropic Messages API — streaming and non-streaming |
| `GET` | `/v1/models` | Lists available model IDs |
| `GET` | `/health` | Health check + browser status |

Both `deepseek-*` model IDs and `claude-*` aliases are accepted — tools that hard-code Claude model names will work without changes.

---

## Session persistence

On first run you log in once inside the browser window. After that, cookies and `localStorage` tokens are saved to:

```
~/.deepseek_scraper/
    cookies.json    ← session cookies (bumped to year 2099 expiry)
    storage.json    ← localStorage + sessionStorage tokens
```

On every subsequent launch the saved session is restored before the page loads, so the login screen is skipped automatically. The session is re-saved after each clean exit.

---
