#!/usr/bin/env python3
"""Export cluster rows ranked by cosine distance to the cluster centroid.

For human review: within each HDBSCAN cluster (excluding noise), L2-normalize
embeddings, compute the mean direction as the centroid, renormalize, then rank
tasks by ascending cosine distance (1 - cosine similarity) to that centroid.

The first column is the suggested theme/operation label taken from the cluster
artifact (`cluster_label` preferred, else `figure_label`).

Inputs:
  data/derived/embeddings/onet_task_embeddings__task_only.npz  (or any variant)
  data/derived/onet_task_clusters.parquet (same row order as embedding rows)

Output:
  CSV (default: data/derived/cluster_review_by_centroid.csv)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
EMB = DERIVED / "embeddings"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--npz",
        default=str(EMB / "onet_task_embeddings__task_only.npz"),
        help="Embedding matrix (must align row-wise with clusters parquet).",
    )
    p.add_argument(
        "--clusters",
        default=str(DERIVED / "onet_task_clusters.parquet"),
        help="Output of scripts/cluster_onet_tasks.py.",
    )
    p.add_argument(
        "--out",
        default=str(DERIVED / "cluster_review_by_centroid.csv"),
        help="Output CSV path.",
    )
    p.add_argument(
        "--include-noise",
        action="store_true",
        help="Also output noise rows (-1) with empty centroid ranking.",
    )
    args = p.parse_args()

    z = np.load(args.npz)
    emb = z["embeddings"].astype(np.float64)
    meta = pd.read_parquet(args.clusters)
    if len(meta) != emb.shape[0]:
        raise SystemExit(
            f"Row mismatch: embeddings {emb.shape[0]} vs clusters {len(meta)}"
        )

    emb_n = l2_normalize(emb)
    labels = meta["hdbscan_cluster"].to_numpy()

    rows: list[dict] = []
    for cid in sorted({int(x) for x in labels if int(x) != -1}):
        mask = labels == cid
        sub_emb = emb_n[mask]
        centroid = sub_emb.mean(axis=0, keepdims=True)
        centroid = l2_normalize(centroid)[0]
        cos_sim = sub_emb @ centroid
        cos_dist = 1.0 - cos_sim
        order = np.argsort(cos_dist)
        sub = meta.loc[mask].reset_index(drop=True)
        theme_col = None
        if "cluster_label" in sub.columns and sub["cluster_label"].notna().any():
            theme_col = sub["cluster_label"].iloc[0]
        elif "figure_label" in sub.columns and sub["figure_label"].notna().any():
            theme_col = sub["figure_label"].iloc[0]
        summary = None
        if "cluster_summary" in sub.columns and sub["cluster_summary"].notna().any():
            summary = sub["cluster_summary"].iloc[0]

        for rank, idx in enumerate(order, start=1):
            r = sub.iloc[int(idx)]
            rows.append(
                {
                    "suggested_cluster_theme": theme_col or "",
                    "cluster_theme_one_liner": summary or "",
                    "hdbscan_cluster": int(cid),
                    "rank_in_cluster_by_centroid_distance": int(rank),
                    "cosine_distance_to_centroid": float(cos_dist[int(idx)]),
                    "cosine_similarity_to_centroid": float(cos_sim[int(idx)]),
                    "onetsoc_code": r.get("onetsoc_code", ""),
                    "task_id": r.get("task_id", ""),
                    "occupation_title": r.get("occupation_title", ""),
                    "task_text_clean": r.get("task_text_clean", ""),
                    "job_zone": r.get("job_zone", ""),
                    "importance_mean": r.get("importance_mean", ""),
                    "relevance_pct": r.get("relevance_pct", ""),
                }
            )

    if args.include_noise:
        noise = meta[labels == -1].copy()
        for _, r in noise.iterrows():
            rows.append(
                {
                    "suggested_cluster_theme": "",
                    "cluster_theme_one_liner": "",
                    "hdbscan_cluster": -1,
                    "rank_in_cluster_by_centroid_distance": "",
                    "cosine_distance_to_centroid": "",
                    "cosine_similarity_to_centroid": "",
                    "onetsoc_code": r.get("onetsoc_code", ""),
                    "task_id": r.get("task_id", ""),
                    "occupation_title": r.get("occupation_title", ""),
                    "task_text_clean": r.get("task_text_clean", ""),
                    "job_zone": r.get("job_zone", ""),
                    "importance_mean": r.get("importance_mean", ""),
                    "relevance_pct": r.get("relevance_pct", ""),
                }
            )

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
