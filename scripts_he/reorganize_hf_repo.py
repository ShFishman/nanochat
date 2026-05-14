"""
One-shot migration: move all existing files in a HF model repo into a
single subfolder (default 'v1-mixed-75he-25en/') so a new version can be
uploaded alongside under its own subfolder.

Server-side copy is used for LFS files (no download/re-upload). Small
non-LFS files are downloaded once and re-uploaded at the new path. The
entire move is committed as one atomic commit on the Hub.

After running, also writes a top-level README.md describing the layout.

Usage:
    huggingface-cli login                       # one-time, write-scope token
    python -m scripts_he.reorganize_hf_repo \\
        --repo-id ShFishman/nanochat-hebrew-d20 \\
        --target-folder v1-mixed-75he-25en
"""

import os
import sys
import argparse
import tempfile

from huggingface_hub import (
    HfApi,
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
    hf_hub_download,
)


REPO_README = """\
# nanochat-hebrew-d20

This repo hosts multiple versions of a Hebrew nanochat model. Each version is
self-contained: its `base/` (and `sft/` where present) weights are paired with
the matching `tokenizer/` in the same subfolder. **Tokenizers are not
interchangeable across versions.**

## Versions

### `v1-mixed-75he-25en/`
First Hebrew nanochat. Pretrained on a 75% Hebrew / 25% English mixture of
OSCAR-23.01 (he), Hebrew Wikipedia, CC-100 (he), Knesset corpus, and
FineWeb-Edu English. Tokenizer includes English. No SFT, no nikkud stripping.

### `v2-fw2-hebrew-only/`
Pure Hebrew nanochat. Pretrained on `HuggingFaceFW/fineweb-2` config
`heb_Hebr` only, with nikkud (U+0591-U+05C7) stripped before tokenization.
Tokenizer is Hebrew-only. Followed by SFT on HeQ, ParaShoot, Dolly-HE,
OpenHermes-2.5-Hebrew, plus hand-templated identity conversations.

## Loading

Each version uses nanochat's native format. Point `nanochat.checkpoint_manager.
load_model_from_dir` at the version subfolder after downloading:

```python
from huggingface_hub import snapshot_download
from nanochat.checkpoint_manager import load_model_from_dir

local = snapshot_download("ShFishman/nanochat-hebrew-d20",
                         allow_patterns=["v2-fw2-hebrew-only/*"])
# point tokenizer loader at v2's tokenizer dir before calling load_model_from_dir
```
"""

V1_README = """\
# v1 - mixed 75% Hebrew / 25% English

Legacy Hebrew nanochat-d20 base model, trained before the FineWeb-2 pure-Hebrew
recipe. Kept here for reproducibility.

- **Pretrain data:** 75% Hebrew (OSCAR-23.01 he, Hebrew Wikipedia, CC-100 he,
  Knesset corpus) + 25% English (FineWeb-Edu sample-10BT)
- **Tokenizer:** mixed Hebrew+English BPE, nikkud retained
- **SFT:** none in this version
- **Architecture:** nanochat d20 (~560M params)

## Layout

- `base/` - pretrained model weights and metadata
- `tokenizer/` - mixed Hebrew/English BPE tokenizer (use ONLY with this base)
- `report/` - training report
- `training_log.md` - full stdout log of the training run
"""


def main():
    parser = argparse.ArgumentParser(description="Reorganize an HF repo into a versioned subfolder layout")
    parser.add_argument("--repo-id", required=True, help="HF repo id, e.g. ShFishman/nanochat-hebrew-d20")
    parser.add_argument("--target-folder", default="v1-mixed-75he-25en",
                        help="Subfolder to move existing files into (default: v1-mixed-75he-25en)")
    parser.add_argument("--keep-at-root", nargs="*", default=[".gitattributes", "README.md"],
                        help="Files to leave at the repo root (default: .gitattributes, README.md)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned moves without committing")
    parser.add_argument("--token", default=None, help="HF token (defaults to HF_TOKEN env / cached login)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    print(f"Inspecting repo {args.repo_id} ...")
    all_files = api.list_repo_files(repo_id=args.repo_id, repo_type="model")
    print(f"Found {len(all_files)} files.")

    files_to_move = []
    for path in all_files:
        if path in args.keep_at_root:
            continue
        if path.startswith(f"{args.target_folder}/"):
            print(f"  already migrated: {path}")
            continue
        files_to_move.append(path)

    if not files_to_move:
        print("Nothing to move. Repo already organized?")
    else:
        print(f"\nWill move {len(files_to_move)} files into {args.target_folder}/")
        for p in files_to_move[:20]:
            print(f"  {p}  ->  {args.target_folder}/{p}")
        if len(files_to_move) > 20:
            print(f"  ... and {len(files_to_move) - 20} more")

    if args.dry_run:
        print("\n--dry-run: not committing.")
        return

    if not files_to_move:
        # Still ensure top-level README and v1 README are present
        _write_readmes_only(api, args.repo_id, args.target_folder)
        return

    # Get LFS info for each path
    print("\nFetching LFS info for each file ...")
    infos = api.get_paths_info(repo_id=args.repo_id, paths=files_to_move, repo_type="model")
    lfs_paths = set()
    for entry in infos:
        if getattr(entry, "lfs", None):
            lfs_paths.add(entry.path)
    print(f"  LFS files: {len(lfs_paths)} / {len(files_to_move)}")

    operations = []

    # 1. Server-side copy for LFS files
    for src in files_to_move:
        if src in lfs_paths:
            dst = f"{args.target_folder}/{src}"
            operations.append(CommitOperationCopy(src_path_in_repo=src, path_in_repo=dst))

    # 2. Download + re-upload for non-LFS files
    non_lfs = [p for p in files_to_move if p not in lfs_paths]
    if non_lfs:
        print(f"\nDownloading {len(non_lfs)} non-LFS files for re-upload ...")
        with tempfile.TemporaryDirectory() as tmpdir:
            payloads: list[tuple[str, bytes]] = []
            for src in non_lfs:
                print(f"  downloading {src} ...")
                local = hf_hub_download(repo_id=args.repo_id, filename=src,
                                        local_dir=tmpdir, repo_type="model", token=token)
                with open(local, "rb") as fh:
                    payloads.append((src, fh.read()))
            for src, data in payloads:
                dst = f"{args.target_folder}/{src}"
                operations.append(CommitOperationAdd(path_in_repo=dst, path_or_fileobj=data))

    # 3. Delete originals (after copy/upload)
    for src in files_to_move:
        operations.append(CommitOperationDelete(path_in_repo=src))

    # 4. Write the v1 README inside the new folder
    operations.append(CommitOperationAdd(
        path_in_repo=f"{args.target_folder}/README.md",
        path_or_fileobj=V1_README.encode("utf-8"),
    ))

    # 5. Top-level repo README
    operations.append(CommitOperationAdd(
        path_in_repo="README.md",
        path_or_fileobj=REPO_README.encode("utf-8"),
    ))

    print(f"\nCommitting {len(operations)} operations ...")
    api.create_commit(
        repo_id=args.repo_id,
        repo_type="model",
        operations=operations,
        commit_message=f"Reorganize: move existing model into {args.target_folder}/",
    )
    print(f"\nDone. View at https://huggingface.co/{args.repo_id}")


def _write_readmes_only(api, repo_id, target_folder):
    """If migration is already done, still keep the top-level + version READMEs current."""
    ops = [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=REPO_README.encode("utf-8")),
        CommitOperationAdd(path_in_repo=f"{target_folder}/README.md", path_or_fileobj=V1_README.encode("utf-8")),
    ]
    api.create_commit(repo_id=repo_id, repo_type="model", operations=ops,
                      commit_message="Refresh top-level and version READMEs")
    print("Refreshed READMEs.")


if __name__ == "__main__":
    main()
