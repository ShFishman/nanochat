"""
Local chat UI that can switch between multiple trained nanochat variants on the fly.

Each variant is a subfolder of a HuggingFace repo (see chat_versions.json). On
first selection of a variant the relevant files are downloaded to a local cache
and the model is loaded into memory. Switching swaps the loaded model.

Run:
    python -m scripts_he.chat_local_ui              # autodetect device
    python -m scripts_he.chat_local_ui --device-type cpu
    python -m scripts_he.chat_local_ui --port 8765 --version v2-sft-aya

To add a future model, edit scripts_he/chat_versions.json -- no code changes needed.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from nanochat.checkpoint_manager import (
    _patch_missing_config_keys,
    _patch_missing_keys,
    load_checkpoint,
)
from nanochat.common import autodetect_device_type
from nanochat.engine import Engine
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer

logger = logging.getLogger("chat_local_ui")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# -----------------------------------------------------------------------------
# Config

DEFAULT_CONFIG = Path(__file__).parent / "chat_versions.json"
DEFAULT_CACHE = Path.home() / ".cache" / "nanochat_local_ui"


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "versions" not in cfg or not cfg["versions"]:
        sys.exit(f"No versions defined in {path}")
    return cfg


# -----------------------------------------------------------------------------
# Model loading

def _find_last_step(checkpoint_dir: Path) -> int:
    candidates = list(checkpoint_dir.glob("model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt found in {checkpoint_dir}")
    return max(int(p.stem.split("_")[1]) for p in candidates)


def _build_model_from_dir(version_dir: Path, tokenizer_dir: Path, device: torch.device):
    """Mirror of nanochat.checkpoint_manager.build_model, but takes explicit dirs."""
    step = _find_last_step(version_dir)
    model_data, _, meta_data = load_checkpoint(str(version_dir), step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()
        }
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    cfg_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(cfg_kwargs)
    cfg = GPTConfig(**cfg_kwargs)
    _patch_missing_keys(model_data, cfg)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()
    tokenizer = RustBPETokenizer.from_directory(str(tokenizer_dir))
    assert tokenizer.get_vocab_size() == cfg_kwargs["vocab_size"], (
        f"Tokenizer vocab size {tokenizer.get_vocab_size()} != model vocab {cfg_kwargs['vocab_size']}"
    )
    return model, tokenizer, meta_data, step


def _download_version(version: dict, cache_root: Path) -> tuple[Path, Path]:
    """Download a version snapshot from HF (cached) and return (version_dir, tokenizer_dir)."""
    from huggingface_hub import snapshot_download

    pir = version["path_in_repo"].rstrip("/")
    patterns = [
        f"{pir}/model_*.pt",
        f"{pir}/meta_*.json",
        f"{pir}/tokenizer/*",
    ] if pir else ["model_*.pt", "meta_*.json", "tokenizer/*"]

    per_version_cache = cache_root / version["id"]
    per_version_cache.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {version['repo_id']}/{pir or '<root>'} (cache: {per_version_cache})")
    snap = snapshot_download(
        repo_id=version["repo_id"],
        allow_patterns=patterns,
        cache_dir=str(per_version_cache / "hf"),
    )
    snap = Path(snap)
    version_dir = (snap / pir) if pir else snap
    tokenizer_dir = version_dir / "tokenizer"
    if not version_dir.exists():
        raise FileNotFoundError(f"Expected version dir not found after download: {version_dir}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer dir missing after download: {tokenizer_dir}")
    return version_dir, tokenizer_dir


# -----------------------------------------------------------------------------
# Per-process state

@dataclass
class LoadedModel:
    version_id: str
    label: str
    model: torch.nn.Module
    tokenizer: RustBPETokenizer
    engine: Engine
    meta: dict
    step: int


class AppState:
    def __init__(self, config: dict, cache_root: Path, device: torch.device):
        self.config = config
        self.cache_root = cache_root
        self.device = device
        self.versions_by_id = {v["id"]: v for v in config["versions"]}
        self.loaded: Optional[LoadedModel] = None
        self.load_lock = asyncio.Lock()
        self.generate_lock = asyncio.Lock()

    def is_cached(self, version_id: str) -> bool:
        v = self.versions_by_id.get(version_id)
        if not v:
            return False
        pir = v["path_in_repo"].rstrip("/")
        per_version_cache = self.cache_root / v["id"] / "hf"
        if not per_version_cache.exists():
            return False
        for snap in per_version_cache.rglob("snapshots/*"):
            candidate = (snap / pir) if pir else snap
            if (candidate / "tokenizer").exists() and any(candidate.glob("model_*.pt")):
                return True
        return False

    async def ensure_loaded(self, version_id: str) -> LoadedModel:
        async with self.load_lock:
            if self.loaded and self.loaded.version_id == version_id:
                return self.loaded
            if version_id not in self.versions_by_id:
                raise HTTPException(status_code=404, detail=f"Unknown version: {version_id}")
            version = self.versions_by_id[version_id]

            # Free previous model first so download + load can use the memory back.
            if self.loaded is not None:
                logger.info(f"Unloading previous model: {self.loaded.version_id}")
                self.loaded = None
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            version_dir, tokenizer_dir = await asyncio.to_thread(
                _download_version, version, self.cache_root
            )
            logger.info(f"Loading model from {version_dir} onto {self.device}")
            model, tokenizer, meta, step = await asyncio.to_thread(
                _build_model_from_dir, version_dir, tokenizer_dir, self.device
            )
            engine = Engine(model, tokenizer)
            self.loaded = LoadedModel(
                version_id=version_id,
                label=version["label"],
                model=model,
                tokenizer=tokenizer,
                engine=engine,
                meta=meta,
                step=step,
            )
            logger.info(f"Loaded {version_id} (step {step})")
            return self.loaded


# -----------------------------------------------------------------------------
# FastAPI app

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    version_id: str
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 50
    max_tokens: Optional[int] = 512


def build_app(state: AppState) -> FastAPI:
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return HTMLResponse(INDEX_HTML)

    @app.get("/api/versions")
    async def list_versions():
        return {
            "default_id": state.config.get("default_id") or state.config["versions"][0]["id"],
            "current_id": state.loaded.version_id if state.loaded else None,
            "versions": [
                {
                    "id": v["id"],
                    "label": v["label"],
                    "repo_id": v["repo_id"],
                    "path_in_repo": v["path_in_repo"],
                    "cached": state.is_cached(v["id"]),
                }
                for v in state.config["versions"]
            ],
        }

    class LoadRequest(BaseModel):
        id: str

    @app.post("/api/load")
    async def load(req: LoadRequest):
        loaded = await state.ensure_loaded(req.id)
        return {
            "id": loaded.version_id,
            "label": loaded.label,
            "step": loaded.step,
            "vocab_size": loaded.tokenizer.get_vocab_size(),
        }

    @app.post("/chat/completions")
    async def chat(req: ChatRequest):
        if not req.messages:
            raise HTTPException(400, "messages required")
        loaded = await state.ensure_loaded(req.version_id)
        tokenizer = loaded.tokenizer
        engine = loaded.engine

        bos = tokenizer.get_bos_token_id()
        user_start = tokenizer.encode_special("<|user_start|>")
        user_end = tokenizer.encode_special("<|user_end|>")
        assistant_start = tokenizer.encode_special("<|assistant_start|>")
        assistant_end = tokenizer.encode_special("<|assistant_end|>")

        tokens = [bos]
        for m in req.messages:
            if m.role == "user":
                tokens.append(user_start)
                tokens.extend(tokenizer.encode(m.content))
                tokens.append(user_end)
            elif m.role == "assistant":
                tokens.append(assistant_start)
                tokens.extend(tokenizer.encode(m.content))
                tokens.append(assistant_end)
            else:
                raise HTTPException(400, f"unknown role: {m.role}")
        tokens.append(assistant_start)

        async def stream():
            # Serialize generation so concurrent requests don't fight over the same model.
            await state.generate_lock.acquire()
            try:
                accumulated: list[int] = []
                last_clean = ""
                seed = random.randint(0, 2**31 - 1)
                kwargs = dict(
                    num_samples=1,
                    max_tokens=req.max_tokens or 512,
                    temperature=req.temperature if req.temperature is not None else 0.7,
                    top_k=req.top_k if req.top_k is not None else 50,
                    seed=seed,
                )
                for token_column, _ in engine.generate(tokens, **kwargs):
                    tok = token_column[0]
                    if tok == assistant_end or tok == bos:
                        break
                    accumulated.append(tok)
                    text = tokenizer.decode(accumulated)
                    if text.endswith("�"):
                        continue
                    new_text = text[len(last_clean):]
                    if new_text:
                        yield f"data: {json.dumps({'token': new_text}, ensure_ascii=False)}\n\n"
                        last_clean = text
                    # Cooperatively yield control so the client can render.
                    await asyncio.sleep(0)
                yield f"data: {json.dumps({'done': True})}\n\n"
            finally:
                state.generate_lock.release()

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


# -----------------------------------------------------------------------------
# Inline UI

INDEX_HTML = r"""<!doctype html>
<html lang="he">
<head>
<meta charset="utf-8">
<title>nanochat-hebrew local UI</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {
    --bg: #f7f7f8;
    --card: #ffffff;
    --border: #e5e7eb;
    --text: #111827;
    --muted: #6b7280;
    --user: #2563eb;
    --user-bg: #eff6ff;
    --assistant-bg: #f3f4f6;
    --accent: #4f46e5;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue",
                 "Arial Hebrew", "Noto Sans Hebrew", Arial, sans-serif;
    background: var(--bg); color: var(--text);
    display: flex; flex-direction: column;
  }
  header {
    background: var(--card);
    border-bottom: 1px solid var(--border);
    padding: 12px 20px;
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header .muted { color: var(--muted); font-size: 13px; }
  .controls { margin-inline-start: auto; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  select, button {
    font: inherit; padding: 6px 10px; border-radius: 6px;
    border: 1px solid var(--border); background: white; cursor: pointer;
  }
  button:hover { background: #f3f4f6; }
  button.primary { background: var(--accent); color: white; border-color: var(--accent); }
  button.primary:hover { background: #3730a3; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  main {
    flex: 1; overflow: hidden; display: flex; flex-direction: column;
    max-width: 900px; width: 100%; margin: 0 auto; padding: 0 16px;
  }
  #status {
    font-size: 13px; color: var(--muted); padding: 8px 4px;
  }
  #status.error { color: #b91c1c; }
  #chat {
    flex: 1; overflow-y: auto; padding: 8px 0 16px;
    display: flex; flex-direction: column; gap: 12px;
  }
  .msg {
    max-width: 80%; padding: 10px 14px; border-radius: 14px;
    white-space: pre-wrap; word-wrap: break-word; line-height: 1.5;
  }
  .msg.user {
    align-self: flex-end; background: var(--user-bg); color: var(--user);
    border: 1px solid #dbeafe;
  }
  .msg.assistant {
    align-self: flex-start; background: var(--assistant-bg); color: var(--text);
    border: 1px solid var(--border);
  }
  .msg.assistant.streaming::after {
    content: "▍"; opacity: 0.5; animation: blink 1s steps(2, start) infinite;
    margin-inline-start: 2px;
  }
  @keyframes blink { to { visibility: hidden; } }
  .meta {
    font-size: 11px; color: var(--muted); margin-bottom: 4px;
  }
  form {
    display: flex; gap: 8px; padding: 12px 0 20px; border-top: 1px solid var(--border);
  }
  textarea {
    flex: 1; resize: none; padding: 10px 12px; font: inherit;
    border: 1px solid var(--border); border-radius: 8px; max-height: 200px;
    min-height: 44px;
  }
  textarea:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  .sliders { display: flex; gap: 14px; align-items: center; font-size: 13px; color: var(--muted); }
  .sliders label { display: flex; align-items: center; gap: 6px; }
  .sliders input[type=number] { width: 64px; padding: 3px 4px; }
</style>
</head>
<body>
<header>
  <h1>nanochat-hebrew</h1>
  <span class="muted" id="device-info"></span>
  <div class="controls">
    <select id="version-select" disabled></select>
    <button id="new-conv">שיחה חדשה</button>
    <details>
      <summary class="muted" style="cursor:pointer;font-size:13px;">הגדרות</summary>
      <div class="sliders" style="margin-top:6px;">
        <label>טמפ׳ <input type="number" id="temperature" value="0.7" min="0" max="2" step="0.05"></label>
        <label>top-k <input type="number" id="top-k" value="50" min="0" max="200" step="1"></label>
        <label>max <input type="number" id="max-tokens" value="512" min="16" max="2048" step="16"></label>
      </div>
    </details>
  </div>
</header>
<main>
  <div id="status">טוען רשימת מודלים…</div>
  <div id="chat"></div>
  <form id="input-form">
    <textarea id="input" dir="auto" placeholder="כתבו הודעה (Enter כדי לשלוח, Shift+Enter שורה חדשה)…" rows="1"></textarea>
    <button class="primary" id="send-btn" type="submit" disabled>שלח</button>
  </form>
</main>
<script>
const $ = (s) => document.querySelector(s);
const chatEl = $("#chat");
const statusEl = $("#status");
const versionSel = $("#version-select");
const sendBtn = $("#send-btn");
const inputEl = $("#input");
const form = $("#input-form");
const newConvBtn = $("#new-conv");

let history = [];
let currentVersion = null;
let busy = false;

function setStatus(msg, isError) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", !!isError);
}

function setBusy(b, label) {
  busy = b;
  sendBtn.disabled = b || !currentVersion;
  versionSel.disabled = b;
  if (label) setStatus(label);
}

function renderMessage(role, text, streaming) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role + (streaming ? " streaming" : "");
  wrap.setAttribute("dir", "auto");
  wrap.textContent = text;
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
  return wrap;
}

async function fetchVersions() {
  const r = await fetch("/api/versions");
  if (!r.ok) throw new Error("failed to list versions");
  const data = await r.json();
  versionSel.innerHTML = "";
  for (const v of data.versions) {
    const opt = document.createElement("option");
    opt.value = v.id;
    opt.textContent = (v.cached ? "✓ " : "") + v.label;
    versionSel.appendChild(opt);
  }
  versionSel.value = data.current_id || data.default_id;
  versionSel.disabled = false;
  return versionSel.value;
}

async function loadVersion(id) {
  setBusy(true, `טוען מודל ${id}… (בפעם הראשונה זה כולל הורדה של ~2GB)`);
  try {
    const r = await fetch("/api/load", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({id})
    });
    if (!r.ok) {
      const err = await r.text();
      throw new Error(err);
    }
    const info = await r.json();
    currentVersion = info.id;
    setStatus(`מודל פעיל: ${info.label} (שלב ${info.step})`);
    // refresh "cached" markers
    await fetchVersions();
    versionSel.value = currentVersion;
  } catch (e) {
    setStatus("שגיאה בטעינת המודל: " + e.message, true);
    currentVersion = null;
  } finally {
    setBusy(false);
    sendBtn.disabled = !currentVersion;
  }
}

async function sendMessage(text) {
  if (!currentVersion || busy) return;
  history.push({role: "user", content: text});
  renderMessage("user", text, false);
  inputEl.value = "";
  inputEl.style.height = "auto";

  setBusy(true, "המודל חושב…");
  const assistantEl = renderMessage("assistant", "", true);
  let accumulated = "";

  try {
    const resp = await fetch("/chat/completions", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        version_id: currentVersion,
        messages: history,
        temperature: parseFloat($("#temperature").value),
        top_k: parseInt($("#top-k").value),
        max_tokens: parseInt($("#max-tokens").value),
      })
    });
    if (!resp.ok) throw new Error(await resp.text());
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {stream: true});
      const events = buf.split("\n\n");
      buf = events.pop();
      for (const ev of events) {
        const line = ev.split("\n").find(l => l.startsWith("data: "));
        if (!line) continue;
        const payload = JSON.parse(line.slice(6));
        if (payload.token) {
          accumulated += payload.token;
          assistantEl.textContent = accumulated;
          chatEl.scrollTop = chatEl.scrollHeight;
        }
        if (payload.done) break;
      }
    }
    assistantEl.classList.remove("streaming");
    if (accumulated.trim().length > 0) {
      history.push({role: "assistant", content: accumulated});
    } else {
      assistantEl.textContent = "(תגובה ריקה)";
    }
    setStatus(`מודל פעיל: ${versionSel.options[versionSel.selectedIndex].textContent}`);
  } catch (e) {
    assistantEl.classList.remove("streaming");
    assistantEl.textContent = "שגיאה: " + e.message;
    setStatus("שגיאה: " + e.message, true);
  } finally {
    setBusy(false);
  }
}

versionSel.addEventListener("change", () => loadVersion(versionSel.value));

newConvBtn.addEventListener("click", () => {
  history = [];
  chatEl.innerHTML = "";
});

inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(200, inputEl.scrollHeight) + "px";
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (text) sendMessage(text);
});

(async () => {
  try {
    const initial = await fetchVersions();
    await loadVersion(initial);
  } catch (e) {
    setStatus("שגיאה: " + e.message, true);
  }
})();
</script>
</body>
</html>
"""


# -----------------------------------------------------------------------------
# Entry point

def main():
    parser = argparse.ArgumentParser(description="Local chat UI with model switcher")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="Path to chat_versions.json")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE),
                        help="Where to cache downloaded model snapshots")
    parser.add_argument("--device-type", default="", choices=["", "cuda", "cpu", "mps"],
                        help="Device for inference (default: autodetect)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--version", default=None,
                        help="Pre-load this version on startup (default: config default_id)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    cache_root = Path(args.cache_dir).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)

    dt = args.device_type or autodetect_device_type()
    if dt == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA requested but not available")
    device = torch.device(dt)

    state = AppState(config=cfg, cache_root=cache_root, device=device)
    app = build_app(state)

    # Optional eager preload (the UI also triggers a load via /api/load on first visit).
    preload_id = args.version or cfg.get("default_id") or cfg["versions"][0]["id"]
    if preload_id:
        async def _preload():
            try:
                await state.ensure_loaded(preload_id)
            except Exception as e:
                logger.warning(f"Preload of {preload_id} failed (UI will retry): {e}")
        @app.on_event("startup")
        async def _startup():
            asyncio.create_task(_preload())

    print(f"\n  nanochat-hebrew local UI")
    print(f"  device: {device}")
    print(f"  cache:  {cache_root}")
    print(f"  open:   http://{args.host}:{args.port}\n")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
