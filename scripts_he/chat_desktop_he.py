"""
Standalone desktop chat app for nanochat-hebrew (Tkinter, CPU-only).

Downloads both 'aya-heavy' and 'balanced' SFT variants from HuggingFace on
first launch into a local cache, lets you switch between them via a
dropdown, and runs inference on CPU.

Run from anywhere:
    python scripts_he/chat_desktop_he.py

Or use the bundled chat_he_desktop.bat launcher which sets up a venv first.
"""

from __future__ import annotations

import gc
import os
import queue
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import ttk
from typing import Optional

# Make 'nanochat' importable even when this script is launched by absolute path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


VERSIONS = [
    {
        "id": "v2-sft-balanced",
        "label": "v2 - Balanced SFT (recommended)",
        "repo_id": "ShFishman/nanochat-hebrew-d20",
        "path_in_repo": "v2-fw2-hebrew-only/sft",
    },
    {
        "id": "v2-sft-aya",
        "label": "v2 - Aya-heavy SFT",
        "repo_id": "ShFishman/nanochat-hebrew-d20",
        "path_in_repo": "v2-fw2-hebrew-only/sft-aya-heavy",
    },
]

CACHE_ROOT = Path.home() / ".cache" / "nanochat_desktop"


# -----------------------------------------------------------------------------
# Model handling (lazy imports — keep startup snappy)

def _find_last_step(version_dir: Path) -> int:
    candidates = list(version_dir.glob("model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt found in {version_dir}")
    return max(int(p.stem.split("_")[1]) for p in candidates)


def download_version(version: dict, on_progress=None) -> tuple[Path, Path]:
    """Snapshot-download just one variant's files; return (version_dir, tokenizer_dir)."""
    from huggingface_hub import snapshot_download

    pir = version["path_in_repo"].rstrip("/")
    patterns = [
        f"{pir}/model_*.pt",
        f"{pir}/meta_*.json",
        f"{pir}/tokenizer/*",
    ]
    per_version_cache = CACHE_ROOT / version["id"]
    per_version_cache.mkdir(parents=True, exist_ok=True)
    if on_progress:
        on_progress(f"מוריד את {version['label']} מ-HuggingFace…")
    snap = snapshot_download(
        repo_id=version["repo_id"],
        allow_patterns=patterns,
        cache_dir=str(per_version_cache / "hf"),
    )
    snap = Path(snap)
    version_dir = snap / pir
    tokenizer_dir = version_dir / "tokenizer"
    if not version_dir.exists():
        raise FileNotFoundError(f"Expected version dir not found: {version_dir}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer dir missing: {tokenizer_dir}")
    return version_dir, tokenizer_dir


def build_model(version_dir: Path, tokenizer_dir: Path, device):
    import torch
    from nanochat.checkpoint_manager import (
        _patch_missing_config_keys, _patch_missing_keys, load_checkpoint,
    )
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import RustBPETokenizer

    step = _find_last_step(version_dir)
    model_data, _, meta = load_checkpoint(str(version_dir), step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()
        }
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    cfg_kwargs = meta["model_config"]
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
    return model, tokenizer, meta, step


# -----------------------------------------------------------------------------
# UI

@dataclass
class LoadedModel:
    version_id: str
    label: str
    model: object
    tokenizer: object
    engine: object
    step: int


@dataclass
class GenerationJob:
    history: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event)


class ChatApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("nanochat-hebrew (CPU)")
        self.root.geometry("820x720")
        self.root.minsize(500, 500)

        self.device = None  # set after torch import
        self.loaded: Optional[LoadedModel] = None
        self.cached: dict[str, tuple[Path, Path]] = {}
        self.history: list[dict] = []
        self.event_q: queue.Queue = queue.Queue()
        self.current_job: Optional[GenerationJob] = None
        self._assistant_start_idx: Optional[str] = None

        self._build_ui()
        self.root.after(50, self._poll_events)

        # Kick off background bootstrap (imports + downloads + first model load)
        threading.Thread(target=self._bootstrap, daemon=True).start()

    # ---------- widgets ----------

    def _build_ui(self):
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=11)

        # Header
        header = ttk.Frame(self.root, padding=(10, 8))
        header.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(header, text="מודל:").pack(side=tk.LEFT, padx=(0, 6))
        self.version_var = tk.StringVar()
        self.version_combo = ttk.Combobox(
            header, textvariable=self.version_var, state="disabled",
            values=[v["label"] for v in VERSIONS], width=42,
        )
        self.version_combo.pack(side=tk.LEFT)
        self.version_combo.bind("<<ComboboxSelected>>", self._on_version_change)

        ttk.Button(header, text="שיחה חדשה", command=self._new_conversation).pack(side=tk.LEFT, padx=8)

        # Status
        self.status_var = tk.StringVar(value="טוען…")
        status = ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W,
                           padding=(10, 4), foreground="#444")
        status.pack(side=tk.TOP, fill=tk.X)

        # Chat area (Text widget with tags for user/assistant)
        chat_frame = ttk.Frame(self.root, padding=(10, 4))
        chat_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.chat = tk.Text(
            chat_frame, wrap=tk.WORD, state=tk.DISABLED, bg="#fafafa",
            relief=tk.FLAT, padx=8, pady=8, font=("Segoe UI", 11), spacing3=6,
        )
        sb = ttk.Scrollbar(chat_frame, orient=tk.VERTICAL, command=self.chat.yview)
        self.chat.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Tags
        self.chat.tag_configure("user_label", foreground="#1f4ed8", font=("Segoe UI", 10, "bold"))
        self.chat.tag_configure("user_text", lmargin1=12, lmargin2=12, spacing1=2, spacing3=10)
        self.chat.tag_configure("assistant_label", foreground="#16a34a", font=("Segoe UI", 10, "bold"))
        self.chat.tag_configure("assistant_text", lmargin1=12, lmargin2=12, spacing1=2, spacing3=10)
        self.chat.tag_configure("error", foreground="#b91c1c", lmargin1=12, lmargin2=12)
        self.chat.tag_configure("system", foreground="#888", lmargin1=12, lmargin2=12)

        # Input row
        input_frame = ttk.Frame(self.root, padding=(10, 8))
        input_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.input = tk.Text(input_frame, height=3, wrap=tk.WORD, font=("Segoe UI", 11))
        self.input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.input.bind("<Return>", self._on_return_key)
        self.input.bind("<Shift-Return>", lambda e: None)  # allow newline

        btn_col = ttk.Frame(input_frame)
        btn_col.pack(side=tk.RIGHT, padx=(8, 0), fill=tk.Y)
        self.send_btn = ttk.Button(btn_col, text="שלח", command=self._on_send, state="disabled")
        self.send_btn.pack(side=tk.TOP, fill=tk.X)
        self.stop_btn = ttk.Button(btn_col, text="עצור", command=self._on_stop, state="disabled")
        self.stop_btn.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

    # ---------- helpers for chat area ----------

    def _set_status(self, text: str, error: bool = False):
        self.status_var.set(text)

    def _append(self, text: str, tag: str = None):
        self.chat.configure(state=tk.NORMAL)
        if tag:
            self.chat.insert(tk.END, text, tag)
        else:
            self.chat.insert(tk.END, text)
        self.chat.see(tk.END)
        self.chat.configure(state=tk.DISABLED)

    def _begin_message(self, who: str):
        label = "המשתמש" if who == "user" else "המודל"
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(tk.END, f"{label}\n", f"{who}_label")
        if who == "assistant":
            self._assistant_start_idx = self.chat.index(tk.END + " -1c")
        self.chat.configure(state=tk.DISABLED)

    def _stream_token(self, text: str):
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(tk.END, text, "assistant_text")
        self.chat.see(tk.END)
        self.chat.configure(state=tk.DISABLED)

    def _end_message(self):
        self._append("\n")

    # ---------- bootstrap (background) ----------

    def _bootstrap(self):
        try:
            self.event_q.put(("status", "מטעין PyTorch…"))
            import torch
            self.device = torch.device("cpu")

            self.event_q.put(("status", "מוודא ש-huggingface_hub זמין…"))
            import huggingface_hub  # noqa: F401

            # Download both variants
            for v in VERSIONS:
                self.event_q.put(("status", f"מוריד את {v['label']} (פעם ראשונה, ~2GB)…"))
                version_dir, tokenizer_dir = download_version(v)
                self.cached[v["id"]] = (version_dir, tokenizer_dir)
                self.event_q.put(("status", f"הורד: {v['label']}"))

            # Enable dropdown
            self.event_q.put(("enable_dropdown", VERSIONS[0]["label"]))

            # Load default
            self._load_version_blocking(VERSIONS[0])

            self.event_q.put(("ready", None))
        except Exception as e:
            import traceback
            self.event_q.put(("fatal", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))

    def _load_version_blocking(self, version: dict):
        import torch
        # Free old model first
        if self.loaded is not None:
            self.event_q.put(("status", f"משחרר את {self.loaded.label}…"))
            self.loaded = None
            gc.collect()
        self.event_q.put(("status", f"טוען את {version['label']} לזיכרון…"))
        version_dir, tokenizer_dir = self.cached[version["id"]]
        model, tokenizer, meta, step = build_model(version_dir, tokenizer_dir, self.device)
        from nanochat.engine import Engine
        engine = Engine(model, tokenizer)
        self.loaded = LoadedModel(
            version_id=version["id"], label=version["label"],
            model=model, tokenizer=tokenizer, engine=engine, step=step,
        )
        self.event_q.put(("status", f"מוכן: {version['label']} (שלב {step})"))

    # ---------- generation ----------

    def _generate_worker(self, job: GenerationJob):
        try:
            tokenizer = self.loaded.tokenizer
            engine = self.loaded.engine
            bos = tokenizer.get_bos_token_id()
            user_start = tokenizer.encode_special("<|user_start|>")
            user_end = tokenizer.encode_special("<|user_end|>")
            assistant_start = tokenizer.encode_special("<|assistant_start|>")
            assistant_end = tokenizer.encode_special("<|assistant_end|>")

            tokens = [bos]
            for m in job.history:
                if m["role"] == "user":
                    tokens.append(user_start)
                    tokens.extend(tokenizer.encode(m["content"]))
                    tokens.append(user_end)
                elif m["role"] == "assistant":
                    tokens.append(assistant_start)
                    tokens.extend(tokenizer.encode(m["content"]))
                    tokens.append(assistant_end)
            tokens.append(assistant_start)

            accumulated: list[int] = []
            last_clean = ""
            import random
            seed = random.randint(0, 2**31 - 1)
            for token_column, _ in engine.generate(tokens, num_samples=1, seed=seed, **job.settings):
                if job.cancel.is_set():
                    break
                tok = token_column[0]
                if tok == assistant_end or tok == bos:
                    break
                accumulated.append(tok)
                text = tokenizer.decode(accumulated)
                if text.endswith("�"):
                    continue
                new_text = text[len(last_clean):]
                if new_text:
                    self.event_q.put(("token", new_text))
                    last_clean = text
            self.event_q.put(("gen_done", last_clean))
        except Exception as e:
            import traceback
            self.event_q.put(("gen_error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))

    # ---------- event handlers ----------

    def _on_send(self):
        if not self.loaded:
            return
        text = self.input.get("1.0", tk.END).strip()
        if not text:
            return
        self.input.delete("1.0", tk.END)

        self.history.append({"role": "user", "content": text})
        self._begin_message("user")
        self._append(text + "\n", "user_text")
        self._begin_message("assistant")

        self.send_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.version_combo.configure(state="disabled")
        self._set_status("המודל יוצר תגובה… (CPU, סבלנות)")

        job = GenerationJob(
            history=list(self.history),
            settings={"max_tokens": 384, "temperature": 0.7, "top_k": 50},
        )
        self.current_job = job
        threading.Thread(target=self._generate_worker, args=(job,), daemon=True).start()

    def _on_stop(self):
        if self.current_job:
            self.current_job.cancel.set()
            self._set_status("עוצר…")

    def _on_return_key(self, event):
        if event.state & 0x0001:  # Shift held -> allow newline
            return None
        self._on_send()
        return "break"

    def _new_conversation(self):
        if self.current_job:
            self.current_job.cancel.set()
        self.history = []
        self.chat.configure(state=tk.NORMAL)
        self.chat.delete("1.0", tk.END)
        self.chat.configure(state=tk.DISABLED)
        if self.loaded:
            self._set_status(f"מוכן: {self.loaded.label} (שלב {self.loaded.step})")

    def _on_version_change(self, event=None):
        label = self.version_var.get()
        target = next((v for v in VERSIONS if v["label"] == label), None)
        if not target or (self.loaded and self.loaded.version_id == target["id"]):
            return
        self.version_combo.configure(state="disabled")
        self.send_btn.configure(state="disabled")
        threading.Thread(
            target=lambda: (self._load_version_blocking(target), self.event_q.put(("ready", None))),
            daemon=True,
        ).start()

    # ---------- event pump ----------

    def _poll_events(self):
        try:
            while True:
                kind, payload = self.event_q.get_nowait()
                if kind == "status":
                    self._set_status(payload)
                elif kind == "enable_dropdown":
                    self.version_combo.configure(state="readonly")
                    self.version_var.set(payload)
                elif kind == "ready":
                    self.send_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.version_combo.configure(state="readonly")
                    if self.loaded:
                        self._set_status(f"מוכן: {self.loaded.label} (שלב {self.loaded.step})")
                elif kind == "token":
                    self._stream_token(payload)
                elif kind == "gen_done":
                    assistant_text = payload or ""
                    if assistant_text.strip():
                        self.history.append({"role": "assistant", "content": assistant_text})
                    else:
                        self._append("(תגובה ריקה)", "system")
                    self._end_message()
                    self.send_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.version_combo.configure(state="readonly")
                    if self.loaded:
                        self._set_status(f"מוכן: {self.loaded.label} (שלב {self.loaded.step})")
                    self.current_job = None
                elif kind == "gen_error":
                    self._end_message()
                    self._append(f"שגיאה: {payload}\n", "error")
                    self.send_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.version_combo.configure(state="readonly")
                    self.current_job = None
                elif kind == "fatal":
                    self._set_status("שגיאה קריטית — ראו את החלון")
                    self._append(f"\n{payload}\n", "error")
        except queue.Empty:
            pass
        self.root.after(80, self._poll_events)


def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.25)
    except Exception:
        pass
    app = ChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
