#!/usr/bin/env python3
"""Embed scoped ESCO items with ``text-embedding-3-small``.

Mirrors the O*NET ``scripts/embed_onet_tasks.py`` interface so the two corpora
are embedded in the same vector space. Writes:

    data/derived/esco/embeddings/esco_item_embeddings__<variant>.npz
    data/derived/esco/embeddings/esco_item_embeddings__<variant>_meta.parquet

Variants
--------
* ``label_only`` — ``preferredLabel`` only. Primary variant; most directly
  comparable to the O*NET ``task_only`` primary because ESCO preferred labels
  are already imperative operation phrases (``manage musical staff``,
  ``compile content for...``).
* ``label_plus_description`` — ``preferredLabel: description`` (description
  is the longer paragraph-form gloss).

Reads ``data/derived/esco/esco_items_scoped.parquet`` produced by
``scripts/esco/prepare_esco_items.py``.

Environment: ``OPENAI_API_KEY`` or ``OPENAI_API``.
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

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived" / "esco"
EMB_DIR = DERIVED / "embeddings"

MODEL_ID = "text-embedding-3-small"
BATCH = 256


def build_input_label_plus_description(label: str, desc: str) -> str:
    label = (label or "").strip()
    desc = (desc or "").strip()
    if label and desc:
        return f"{label}: {desc}"
    return label or desc


def default_paths(variant: str) -> tuple[Path, Path]:
    stem = f"esco_item_embeddings__{variant}"
    return EMB_DIR / f"{stem}.npz", EMB_DIR / f"{stem}_meta.parquet"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(DERIVED / "esco_items_scoped.parquet"))
    p.add_argument(
        "--variant",
        choices=("label_only", "label_plus_description"),
        default="label_only",
    )
    p.add_argument("--out-npz", default=None)
    p.add_argument("--out-meta", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    default_npz, default_meta = default_paths(args.variant)
    out_npz = Path(args.out_npz) if args.out_npz else default_npz
    out_meta = Path(args.out_meta) if args.out_meta else default_meta

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY / OPENAI_API env var not set.")

    df = pd.read_parquet(args.input).drop_duplicates("esco_uri").reset_index(drop=True)

    if args.variant == "label_only":
        df["embed_input"] = df["preferred_label"].fillna("").astype(str).str.strip()
    else:
        df["embed_input"] = [
            build_input_label_plus_description(lbl, desc)
            for lbl, desc in zip(df["preferred_label"], df["description"])
        ]

    df = df[df["embed_input"].str.len() > 0].reset_index(drop=True)

    print(f"Variant: {args.variant}  Rows to embed: {len(df)}")
    if args.dry_run:
        return

    EMB_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key)
    vecs: list[np.ndarray] = []
    inputs = df["embed_input"].tolist()

    for start in tqdm(range(0, len(inputs), BATCH), desc="embed"):
        chunk = inputs[start : start + BATCH]
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=MODEL_ID, input=chunk)
                break
            except Exception as exc:  # pragma: no cover
                wait = 2**attempt
                print(f"[warn] embeddings.create failed ({exc!r}); retry in {wait}s")
                time.sleep(wait)
        else:
            raise SystemExit("embedding API failed after retries")
        vecs.extend(np.asarray(d.embedding, dtype=np.float32) for d in resp.data)

    arr = np.vstack(vecs)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, embeddings=arr, model=MODEL_ID, variant=args.variant)
    df.to_parquet(out_meta, index=False)
    print(f"Wrote {out_npz} ({arr.shape}) and {out_meta}")


if __name__ == "__main__":
    main()
