"""
Upload a trained nanochat Hebrew model to the Hugging Face Hub.

Uploads, by default, the final SFT checkpoint at
    $NANOCHAT_BASE_DIR/chatsft_checkpoints/<model_tag>/
plus the tokenizer directory at
    $NANOCHAT_BASE_DIR/tokenizer/
and the latest report markdown if present.

Optimizer shards (optim_*.pt) are excluded -- they are only useful for
resuming training, not inference.

A README.md is generated locally and uploaded alongside the weights.

By default each upload lives in its own subfolder of the repo so
multiple model variants can coexist:
    <path-in-repo>/             (e.g. v2-fw2-hebrew-only/base or .../sft)
        model_*.pt
        meta_*.json
        README.md
        tokenizer/
        report.md

Authentication:
    Set HF_TOKEN env var, or run `huggingface-cli login` once.

Usage:
    python -m scripts_he.upload_to_hf \\
        --repo-id ShFishman/nanochat-hebrew-d20 \\
        --model-tag he_fw2_d20 \\
        --source sft \\
        --path-in-repo v2-fw2-hebrew-only/sft
"""

import os
import sys
import json
import argparse
from pathlib import Path

from nanochat.common import get_base_dir


def find_last_step(checkpoint_dir: Path) -> int:
    candidates = list(checkpoint_dir.glob("model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt in {checkpoint_dir}")
    return max(int(p.stem.split("_")[1]) for p in candidates)


def build_readme(repo_id: str, model_tag: str, source: str, path_in_repo: str, meta: dict | None) -> str:
    cfg = (meta or {}).get("model_config", {})
    user_cfg = (meta or {}).get("user_config", {})
    n_layer = cfg.get("n_layer", "?")
    n_embd = cfg.get("n_embd", "?")
    vocab_size = cfg.get("vocab_size", "?")
    seq_len = cfg.get("sequence_len", "?")
    step = (meta or {}).get("step", "?")
    val_bpb = (meta or {}).get("val_bpb", "?")

    stage_label = "post-trained (SFT)" if source == "sft" else "pretrained (base)"
    snapshot_pattern = f"{path_in_repo}/*" if path_in_repo else "*"

    lines = [
        "---",
        "license: apache-2.0",
        "language:",
        "- he",
        "library_name: nanochat",
        "tags:",
        "- nanochat",
        "- hebrew",
        "- causal-lm",
        f"- {source}",
        "---",
        "",
        f"# {repo_id} / {path_in_repo or '(root)'}",
        "",
        f"Hebrew-only nanochat-d{n_layer if n_layer != '?' else '?'} model, **{stage_label}**.",
        "Pretrained on `HuggingFaceFW/fineweb-2` (`heb_Hebr` subset) with nikkud stripped"
        + (", then supervised-fine-tuned on a Hebrew-only chat mixture "
           "(HeQ, ParaShoot, Dolly-HE, OpenHermes-2.5-Hebrew, plus hand-templated identity conversations)." if source == "sft"
           else "."),
        "",
        "## Model details",
        "",
        f"- **Model tag:** `{model_tag}`",
        f"- **Stage:** `{source}` ({stage_label})",
        f"- **Step:** {step}",
        f"- **Layers (depth):** {n_layer}",
        f"- **Hidden dim:** {n_embd}",
        f"- **Vocab size:** {vocab_size}",
        f"- **Sequence length:** {seq_len}",
        f"- **Final validation BPB:** {val_bpb}",
        "",
        "## Files",
        "",
        "- `model_*.pt` - model weights (PyTorch `state_dict`).",
        "- `meta_*.json` - model config and training metadata.",
        "- `tokenizer/` - BPE tokenizer (HuggingFace `tokenizers` format) and "
        "`token_bytes.pt` for BPB scoring. **Use this tokenizer, not v1's.**",
        "- `report.md` - training report excerpt (if available).",
        "",
        "## Loading",
        "",
        "This model is in nanochat's native format, not HF `transformers`. To load:",
        "",
        "```python",
        "from huggingface_hub import snapshot_download",
        "from nanochat.checkpoint_manager import load_model_from_dir",
        "",
        f'local = snapshot_download("{repo_id}", allow_patterns=["{snapshot_pattern}"])',
        f'# this version lives at: {{local}}/{path_in_repo}' if path_in_repo else "",
        f'model, tokenizer, meta = load_model_from_dir(local, device="cuda", phase="eval", model_tag="{model_tag}")',
        "```",
        "",
        "## Training",
        "",
        f"Trained with [nanochat](https://github.com/karpathy/nanochat) on 8x A100 80GB (bf16, no FP8, "
        f"`--window-pattern=L`). See `report.md` for full details.",
        "",
        f"User config: ```{json.dumps(user_cfg, ensure_ascii=False)[:1500]}```",
    ]
    return "\n".join(l for l in lines if l != "")


def upload(repo_id: str, model_tag: str, source: str, path_in_repo: str, private: bool, token: str | None):
    from huggingface_hub import HfApi, create_repo

    base_dir = Path(get_base_dir())
    checkpoint_root = {
        "base": base_dir / "base_checkpoints",
        "sft": base_dir / "chatsft_checkpoints",
    }[source]
    checkpoint_dir = checkpoint_root / model_tag
    if not checkpoint_dir.exists():
        sys.exit(f"ERROR: checkpoint dir not found: {checkpoint_dir}")

    last_step = find_last_step(checkpoint_dir)
    meta_path = checkpoint_dir / f"meta_{last_step:06d}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None

    api = HfApi(token=token)
    print(f"Creating repo {repo_id} (private={private}) if needed ...")
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True, token=token)

    # Write README.md inside the checkpoint dir so it's uploaded with the rest.
    readme_path = checkpoint_dir / "README.md"
    readme_path.write_text(build_readme(repo_id, model_tag, source, path_in_repo, meta), encoding="utf-8")
    print(f"Wrote {readme_path}")

    tokenizer_dir = base_dir / "tokenizer"
    report_md = base_dir / "report" / "report.md"

    pir = path_in_repo.rstrip("/")  # may be ""

    print(f"Uploading checkpoint dir: {checkpoint_dir} -> {pir or '<root>'}")
    api.upload_folder(
        folder_path=str(checkpoint_dir),
        path_in_repo=pir,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["optim_*.pt", "*.tmp"],
        commit_message=f"Upload {source} checkpoint step {last_step} -> {pir or '<root>'}",
        token=token,
    )

    if tokenizer_dir.exists():
        tok_path = f"{pir}/tokenizer" if pir else "tokenizer"
        print(f"Uploading tokenizer dir: {tokenizer_dir} -> {tok_path}")
        api.upload_folder(
            folder_path=str(tokenizer_dir),
            path_in_repo=tok_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload tokenizer -> {tok_path}",
            token=token,
        )

    if report_md.exists():
        rpt_path = f"{pir}/report.md" if pir else "report.md"
        print(f"Uploading report.md -> {rpt_path}")
        api.upload_file(
            path_or_fileobj=str(report_md),
            path_in_repo=rpt_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload training report -> {rpt_path}",
            token=token,
        )

    print()
    print(f"Done. View it at https://huggingface.co/{repo_id}/tree/main/{pir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload trained nanochat-he model to HuggingFace")
    parser.add_argument("--repo-id", required=True,
                        help="Target HF repo id, e.g. 'ShFishman/nanochat-hebrew-d20'")
    parser.add_argument("--model-tag", required=True,
                        help="nanochat model tag (e.g. 'he_fw2_d20')")
    parser.add_argument("--source", choices=["base", "sft"], default="sft",
                        help="Which stage's checkpoint to upload (default: sft)")
    parser.add_argument("--path-in-repo", default="",
                        help="Subfolder inside the repo to upload to (default: repo root). "
                             "Example: 'v2-fw2-hebrew-only/base'")
    parser.add_argument("--private", action="store_true",
                        help="Create the repo as private (no effect if it already exists)")
    parser.add_argument("--token", default=None,
                        help="HF token (defaults to HF_TOKEN env var / cached login)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    upload(args.repo_id, args.model_tag, args.source, args.path_in_repo, args.private, token)
