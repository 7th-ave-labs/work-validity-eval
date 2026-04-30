#!/usr/bin/env python3
"""Embed O*NET task statements with OpenAI text-embedding-3-small.

Variants (for sensitivity analysis; both are written under ``data/derived/embeddings/``):

* ``task_only`` — embed ``task_text_clean`` only.
* ``title_plus_task`` — embed ``"{occupation_title}: {task_text_clean}"`` (legacy comparison).
* ``operation_only`` — embed the domain-stripped ``operation_phrase``
  produced by ``normalize_task_operations.py`` (operation-first
  clustering variant).

Reads ``data/derived/onet_tasks_prepared.parquet``. Writes:

  data/derived/embeddings/onet_task_embeddings__<variant>.npz
  data/derived/embeddings/onet_task_embeddings__<variant>_meta.parquet

Environment: OPENAI_API_KEY or OPENAI_API.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
EMB_DIR = DERIVED / "embeddings"

MODEL_ID = "text-embedding-3-small"
BATCH = 256


def build_input_title_plus_task(title: str, task: str) -> str:
    title = (title or "").strip()
    task = (task or "").strip()
    if title and task:
        return f"{title}: {task}"
    return task or title


def default_paths(variant: str) -> tuple[Path, Path]:
    stem = f"onet_task_embeddings__{variant}"
    return EMB_DIR / f"{stem}.npz", EMB_DIR / f"{stem}_meta.parquet"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(DERIVED / "onet_tasks_prepared.parquet"))
    p.add_argument(
        "--variant",
        choices=("task_only", "title_plus_task", "operation_only"),
        default="task_only",
        help=(
            "Naming + input template. ``operation_only`` embeds the domain-"
            "stripped ``operation_phrase`` column from "
            "``data/derived/task_operations_normalized.parquet``."
        ),
    )
    p.add_argument(
        "--operations",
        default=str(DERIVED / "task_operations_normalized.parquet"),
        help="Path to normalized operations parquet (for --variant operation_only).",
    )
    p.add_argument(
        "--out-npz",
        default=None,
        help="Override output .npz path (default: derived/embeddings/onet_task_embeddings__<variant>.npz).",
    )
    p.add_argument(
        "--out-meta",
        default=None,
        help="Override output .parquet path (default: ..._meta.parquet next to npz).",
    )
    p.add_argument("--core-only", action="store_true", help="Keep only Core tasks.")
    p.add_argument(
        "--min-relevance",
        type=float,
        default=None,
        help="Optional: keep tasks with relevance_pct >= this.",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    default_npz, default_meta = default_paths(args.variant)
    out_npz = Path(args.out_npz) if args.out_npz else default_npz
    out_meta = Path(args.out_meta) if args.out_meta else default_meta

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY / OPENAI_API env var not set.")

    df = pd.read_parquet(args.input)
    if args.core_only:
        df = df[df["task_type"] == "Core"].copy()
    if args.min_relevance is not None:
        df = df[df["relevance_pct"].fillna(0) >= args.min_relevance].copy()
    df = df.drop_duplicates(subset=["onetsoc_code", "task_id"]).reset_index(drop=True)

    if args.variant == "task_only":
        df["embed_input"] = df["task_text_clean"].fillna("").astype(str).str.strip()
    elif args.variant == "title_plus_task":
        df["embed_input"] = [
            build_input_title_plus_task(t, x)
            for t, x in zip(df["occupation_title"], df["task_text_clean"])
        ]
    else:  # operation_only
        ops = pd.read_parquet(args.operations)[
            ["onetsoc_code", "task_id", "operation_phrase", "operation_object_type"]
        ]
        ops["task_id"] = ops["task_id"].astype(str)
        df["task_id"] = df["task_id"].astype(str)
        before = len(df)
        df = df.merge(ops, on=["onetsoc_code", "task_id"], how="inner")
        if len(df) != before:
            print(f"[warn] {before - len(df)} tasks dropped (no normalized op)")
        df["embed_input"] = df["operation_phrase"].fillna("").astype(str).str.strip()

    print(f"Variant: {args.variant}  Rows to embed: {len(df)}")
    if args.dry_run:
        return

    client = OpenAI(api_key=api_key)
    vecs: list[np.ndarray] = []
    inputs = df["embed_input"].tolist()

    for start in tqdm(range(0, len(inputs), BATCH), desc="embed"):
        chunk = inputs[start:start + BATCH]
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=MODEL_ID, input=chunk)
                break
            except Exception as exc:  # pragma: no cover
                wait = 2 ** attempt
                print(f"[warn] embed batch failed ({exc!r}); retrying in {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError("embedding batch failed after retries")
        for item in sorted(resp.data, key=lambda x: x.index):
            vecs.append(np.asarray(item.embedding, dtype=np.float32))

    mat = np.vstack(vecs)
    assert mat.shape[0] == len(df), (mat.shape, len(df))

    meta_cols = [
        "onetsoc_code",
        "task_id",
        "occupation_title",
        "task_text_clean",
        "task_type",
        "job_zone",
        "importance_mean",
        "relevance_pct",
        "embed_input",
    ]
    meta = df[meta_cols].copy()
    meta["row_index"] = np.arange(len(meta), dtype=np.int64)
    meta["embedding_variant"] = args.variant

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, embeddings=mat, row_index=meta["row_index"].to_numpy())
    meta.to_parquet(out_meta, index=False)
    print(f"Wrote {out_npz} shape={mat.shape}")
    print(f"Wrote {out_meta} rows={len(meta)}")


if __name__ == "__main__":
    main()
