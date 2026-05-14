"""
Stream HuggingFaceFW/fineweb-2 (config 'heb_Hebr') and materialize parquet
shards into $NANOCHAT_BASE_DIR/base_data_climbmix/ so the existing
nanochat dataloader and tokenizer trainer work UNCHANGED.

The dataloader at nanochat/dataloader.py iterates parquet row_groups and
reads the column literally named 'text', which is exactly the FineWeb-2
schema. Each shard is written with row_group_size=1000 for efficient DDP
sharding (each rank picks every num_ranks-th row group).

Usage:
    python -m nanochat_he.fineweb2_to_parquet -n 24

This will write shards shard_00000.parquet ... shard_00023.parquet.
The last shard is reserved as the validation shard by the dataloader
(nanochat/dataloader.py:38).
"""

import os
import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.common import get_base_dir
from nanochat_he.text_utils import strip_nikkud

HF_DATASET = "HuggingFaceFW/fineweb-2"
HF_CONFIG = "heb_Hebr"
SHARD_TARGET_BYTES = 250 * 1024 * 1024  # ~250 MB compressed parquet per shard
ROW_GROUP_SIZE = 1000


def get_shard_dir() -> Path:
    base_dir = get_base_dir()
    shard_dir = Path(base_dir) / "base_data_climbmix"
    shard_dir.mkdir(parents=True, exist_ok=True)
    return shard_dir


def _stream_fineweb2_hebrew():
    from datasets import load_dataset
    ds = load_dataset(
        HF_DATASET,
        name=HF_CONFIG,
        split="train",
        streaming=True,
    )
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        text = strip_nikkud(text)
        if text:
            yield text


def write_shards(num_shards: int, shard_dir: Path, start_idx: int = 0):
    stream = _stream_fineweb2_hebrew()
    print(f"Writing {num_shards} shards (target ~250 MB each) to {shard_dir}")

    for shard_idx in range(start_idx, start_idx + num_shards):
        shard_path = shard_dir / f"shard_{shard_idx:05d}.parquet"
        if shard_path.exists():
            print(f"  shard_{shard_idx:05d}.parquet already exists, skipping")
            continue

        tmp_path = shard_path.with_suffix(".parquet.tmp")
        writer = None
        rows_in_group: list[str] = []
        bytes_in_shard = 0
        rows_in_shard = 0

        try:
            for text in stream:
                rows_in_group.append(text)
                bytes_in_shard += len(text.encode("utf-8"))
                rows_in_shard += 1

                if len(rows_in_group) >= ROW_GROUP_SIZE:
                    table = pa.table({"text": rows_in_group})
                    if writer is None:
                        writer = pq.ParquetWriter(
                            tmp_path,
                            table.schema,
                            compression="zstd",
                        )
                    writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
                    rows_in_group = []

                if bytes_in_shard >= SHARD_TARGET_BYTES:
                    break

            if rows_in_group:
                table = pa.table({"text": rows_in_group})
                if writer is None:
                    writer = pq.ParquetWriter(
                        tmp_path,
                        table.schema,
                        compression="zstd",
                    )
                writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
                rows_in_group = []
        finally:
            if writer is not None:
                writer.close()

        if not tmp_path.exists() or rows_in_shard == 0:
            print(f"  WARNING: no rows written for shard {shard_idx:05d} (stream exhausted?)")
            if tmp_path.exists():
                tmp_path.unlink()
            break

        os.replace(tmp_path, shard_path)
        size_mb = shard_path.stat().st_size / 1e6
        print(f"  wrote shard_{shard_idx:05d}.parquet  ({rows_in_shard:,} docs, {size_mb:.1f} MB on disk)")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Materialize Hebrew FineWeb-2 as parquet shards")
    parser.add_argument("-n", "--num-shards", type=int, default=24,
                        help="Number of parquet shards to write (default: 24)")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Starting shard index (default: 0)")
    args = parser.parse_args()

    shard_dir = get_shard_dir()
    write_shards(args.num_shards, shard_dir, start_idx=args.start_idx)
    # HF datasets streaming workers can throw at interpreter shutdown
    # (`terminate called without an active exception`). Force clean exit
    # after successful writes so the speedrun shell sees a 0 exit code.
    os._exit(0)
