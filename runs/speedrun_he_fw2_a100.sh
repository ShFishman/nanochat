#!/usr/bin/env bash
# runs/speedrun_he_fw2_a100.sh
# ============================================================
# Hebrew-only nanochat speedrun on 8 x A100 SXM4.
#
# Pretrain on HuggingFaceFW/fineweb-2 config "heb_Hebr" ONLY (no English).
# SFT on Hebrew-only chat data (no SmolTalk).
#
# A100 constraints:
#   - No FP8  (Hopper-only) -> bf16 is auto-selected
#   - No FA3  (Hopper-only) -> SDPA fallback
#   - SDPA + sliding window is very slow, so --window-pattern=L is mandatory
#
# Run as:
#   bash runs/speedrun_he_fw2_a100.sh
#   # with wandb:
#   WANDB_RUN=he_fw2_d20 bash runs/speedrun_he_fw2_a100.sh
#   # with HuggingFace upload at the end (set HF_TOKEN once or `huggingface-cli login`):
#   #   HF_REPO_ID         = single repo hosting both base + SFT (default: ShFishman/nanochat-hebrew-d20)
#   #   HF_VERSION_FOLDER  = subfolder inside the repo  (default: v2-fw2-hebrew-only)
#   #   Layout after run:  <repo>/<version>/base/...   <repo>/<version>/sft/...
#   #   Run scripts_he/reorganize_hf_repo.py FIRST to move the old model into v1-mixed-75he-25en/.
#   HF_PRIVATE=1 bash runs/speedrun_he_fw2_a100.sh
# ============================================================

set -euo pipefail

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat_he_fw2}"
export OMP_NUM_THREADS=1
export WANDB_RUN="${WANDB_RUN:-dummy}"

DEPTH="${DEPTH:-20}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"   # 16 on A100 80GB, drop to 8 (or 4) on 40GB
NUM_SHARDS="${NUM_SHARDS:-24}"                  # ~6 GB parquet, ~12B Hebrew tokens after BPE
INITIAL_SHARDS="${INITIAL_SHARDS:-8}"           # downloaded synchronously for tokenizer training

mkdir -p "$NANOCHAT_BASE_DIR/base_data_climbmix"

echo "============================================================"
echo "  Hebrew-only nanochat speedrun on A100"
echo "    base dir   : $NANOCHAT_BASE_DIR"
echo "    depth      : $DEPTH"
echo "    GPUs       : $NPROC_PER_NODE"
echo "    bs/device  : $DEVICE_BATCH_SIZE"
echo "    shards     : $NUM_SHARDS (initial $INITIAL_SHARDS for tokenizer)"
echo "    wandb run  : $WANDB_RUN"
echo "============================================================"

# -----------------------------------------------------------------------------
# venv (uv)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# 1) Materialize Hebrew FineWeb-2 -> parquet shards (writes to base_data_climbmix/)
echo "[1/6] Downloading Hebrew FineWeb-2 -> parquet ..."
python -m nanochat_he.fineweb2_to_parquet -n $INITIAL_SHARDS
python -m nanochat_he.fineweb2_to_parquet -n $NUM_SHARDS --start-idx=$INITIAL_SHARDS &
DL_PID=$!

# -----------------------------------------------------------------------------
# 2) Tokenizer (vocab 65536 for Hebrew morphology). Original scripts/tok_train.py
#    reads parquet via nanochat.dataset.parquets_iter_batched, so it Just Works.
echo "[2/6] Training Hebrew BPE tokenizer (vocab=65536) ..."
python -m scripts.tok_train --vocab-size=65536 --max-chars=2000000000
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Wait for remaining shards before pretraining
echo "[3/6] Waiting for shard download to finish ..."
wait $DL_PID || true

# -----------------------------------------------------------------------------
# 3) Pretrain on Hebrew. NO --fp8 (A100 can't), --window-pattern=L (SDPA can't
#    do sliding window efficiently). target-param-data-ratio=12 = compute-optimal.
echo "[4/6] Pretraining base model d=$DEPTH ..."
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
    --depth=$DEPTH \
    --target-param-data-ratio=12 \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --window-pattern=L \
    --run="$WANDB_RUN" \
    --model-tag="he_fw2_d${DEPTH}"

# -----------------------------------------------------------------------------
# 4) Base eval. We SKIP CORE entirely (English HELM tasks are meaningless on a
#    Hebrew-only model) and run only BPB on the Hebrew val shard + a few samples.
echo "[5/6] Evaluating base model (BPB only; CORE and English sample prompts skipped) ..."
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- \
    --eval=bpb \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --model-tag="he_fw2_d${DEPTH}"

# -----------------------------------------------------------------------------
# 5) Prepare Hebrew-only SFT data (no SmolTalk, no English) + Hebrew identity rows
echo "[6/6] Preparing Hebrew SFT data and running SFT ..."
python -m scripts_he.prepare_sft_data_he

# 6) SFT through the Hebrew-only wrapper (bypasses the hard-coded English mixture
#    in scripts/chat_sft.py). ChatCORE disabled because it benchmarks English tasks.
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts_he.chat_sft_he -- \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --model-tag="he_fw2_d${DEPTH}" \
    --chatcore-every=-1 \
    --run="${WANDB_RUN}_sft"

# -----------------------------------------------------------------------------
# Report
python -m nanochat.report generate

# -----------------------------------------------------------------------------
# 7) Upload BOTH the pretrained (base) and post-trained (SFT) models to HF,
#    into versioned subfolders of a single repo. Defaults target
#    ShFishman/nanochat-hebrew-d20 with the v2 layout described in its README.
#    Override HF_REPO_ID and HF_VERSION_FOLDER to point elsewhere.
#    Set HF_TOKEN env var or run `huggingface-cli login` once. Skipped if HF_REPO_ID="" .
HF_REPO_ID="${HF_REPO_ID-ShFishman/nanochat-hebrew-d20}"
HF_VERSION_FOLDER="${HF_VERSION_FOLDER:-v2-fw2-hebrew-only}"

if [ -n "${HF_REPO_ID}" ]; then
    echo "[7/7a] Uploading BASE checkpoint to https://huggingface.co/$HF_REPO_ID/tree/main/$HF_VERSION_FOLDER/base ..."
    python -m scripts_he.upload_to_hf \
        --repo-id "$HF_REPO_ID" \
        --model-tag "he_fw2_d${DEPTH}" \
        --source base \
        --path-in-repo "$HF_VERSION_FOLDER/base" \
        ${HF_PRIVATE:+--private}

    echo "[7/7b] Uploading SFT checkpoint to https://huggingface.co/$HF_REPO_ID/tree/main/$HF_VERSION_FOLDER/sft ..."
    python -m scripts_he.upload_to_hf \
        --repo-id "$HF_REPO_ID" \
        --model-tag "he_fw2_d${DEPTH}" \
        --source sft \
        --path-in-repo "$HF_VERSION_FOLDER/sft" \
        ${HF_PRIVATE:+--private}
else
    echo "[7/7] HF_REPO_ID is empty, skipping HuggingFace upload."
    echo "      To upload later:"
    echo "        python -m scripts_he.upload_to_hf --repo-id ShFishman/nanochat-hebrew-d20 --model-tag he_fw2_d${DEPTH} --source base --path-in-repo v2-fw2-hebrew-only/base"
    echo "        python -m scripts_he.upload_to_hf --repo-id ShFishman/nanochat-hebrew-d20 --model-tag he_fw2_d${DEPTH} --source sft  --path-in-repo v2-fw2-hebrew-only/sft"
fi

echo
echo "Done. Try a Hebrew prompt:"
echo "  python -m scripts.chat_cli -p 'מה בירת צרפת?'"
echo "  python -m scripts.chat_web   # ChatGPT-style UI"
