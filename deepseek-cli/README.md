# DeepSeek → Claude Code Proxy — Setup Guide

## Requirements

```bash
pip install requests wasmtime numpy flask
```

---

## Step 1 — Get your DeepSeek token

Open **chat.deepseek.com** in your browser, log in, then open DevTools.

### Firefox
`F12` → **Storage** tab → **Local Storage** → `https://chat.deepseek.com`
→ find the key **`userToken`** → copy the `value` field (the string inside the outer quotes, not the whole JSON)

![Firefox DevTools Storage tab showing userToken]

### Chrome / Brave / Edge
`F12` → **Application** tab → **Local Storage** → `https://chat.deepseek.com`
→ find the key **`userToken`** → copy the value on the right (the string starting with `"value":...`, copy only the token string inside)

> The token looks like: `twZij1gbCCmHabkTKDSrUtxRmOOmq1g2KoHfz8Rc38PnHG5kQtEd94kBj83/dyHk`
>
<img width="1828" height="986" alt="image" src="https://github.com/user-attachments/assets/103ce2c8-45a3-431e-a8de-46488d03a1ba" />
<img width="1219" height="651" alt="image" src="https://github.com/user-attachments/assets/740471b6-cecd-41d1-a0bc-ecd8ea946fe1" />


---

## Step 2 — Start the proxy

```bash
export DEEPSEEK_TOKEN="paste-your-token-here"
python deepseek_server.py
```

You should see:

```
Loading WASM solver... OK

DeepSeek proxy listening on http://0.0.0.0:8765
```

---

## Step 3 — Point Claude Code at the proxy

In a new terminal (or add to your `.bashrc` / `.zshrc`):

```bash
export ANTHROPIC_BASE_URL="http://localhost:8765"
export ANTHROPIC_API_KEY="local-proxy-key"
```

Then run:

```bash
claude
```

---

## Optional — pass cookies too

If requests fail with auth errors, also grab the **`smidV2`** value from the same Storage panel and pass it:

```bash
export DEEPSEEK_COOKIES="smidV2=your-smidV2-value"
```

---

## Files

| File | Purpose |
|---|---|
| `deepseek_server.py` | The proxy server |
| `deepseek_cli.py` | Standalone CLI (no Claude Code needed) |
| `sha3_wasm_bg.wasm` | PoW solver — must be in the same directory |
