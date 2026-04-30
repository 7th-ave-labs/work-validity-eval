#!/usr/bin/env python3
"""Project the unsupervised cluster labels onto every task as ``auto_label_v2``.

This is a sensitivity artifact only. It takes the operation-first labels the
LLM proposed per HDBSCAN cluster during ``cluster_onet_tasks.py`` and
projects them onto every task using ``hdbscan_cluster_assigned`` (so noise
points are mapped to their nearest cluster for full corpus coverage), then
writes the per-task auto label alongside the frozen taxonomy.

The frozen 28-label taxonomy in ``data/work_pattern_labels.csv`` remains the
primary coding target; ``auto_label_v2`` is only used to see where a
bottom-up unsupervised pass agrees or disagrees with the frozen labels
(see ``scripts/build_label_crosstab.py``).

Inputs:
  data/derived/onet_task_clusters__<variant>.parquet

Outputs:
  data/derived/auto_labels_v2__<variant>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"


def _variant_suffix(variant: str) -> str:
    return f"__{variant}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant",
        default="task_only",
        choices=("task_only", "title_plus_task", "operation_only", "operation_only_r2"),
    )
    p.add_argument("--clusters", default=None, help="Override cluster parquet path.")
    p.add_argument("--out", default=None, help="Override output CSV path.")
    args = p.parse_args()

    suffix = _variant_suffix(args.variant)
    clusters_path = Path(args.clusters or DERIVED / f"onet_task_clusters{suffix}.parquet")
    out_path = Path(args.out or DERIVED / f"auto_labels_v2{suffix}.csv")

    df = pd.read_parquet(clusters_path)
    required = {
        "onetsoc_code",
        "task_id",
        "hdbscan_cluster",
        "hdbscan_cluster_assigned",
        "cluster_label",
        "cluster_summary",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"{clusters_path} is missing required columns: {sorted(missing)}. "
            "Rerun scripts/cluster_onet_tasks.py with the new recipe first."
        )

    # Build cluster_id -> (label, summary) from the rows that were NOT noise at
    # clustering time (those rows already have the LLM-proposed label attached).
    hard = df[df["hdbscan_cluster"] != -1]
    label_map = (
        hard.dropna(subset=["cluster_label"])
        .groupby("hdbscan_cluster")
        .agg(cluster_label=("cluster_label", "first"), cluster_summary=("cluster_summary", "first"))
        .reset_index()
        .rename(columns={"hdbscan_cluster": "cluster_id"})
    )

    result = df[[
        "onetsoc_code",
        "task_id",
        "occupation_title",
        "task_text_clean",
        "hdbscan_cluster",
        "hdbscan_cluster_assigned",
    ]].copy()
    result = result.merge(
        label_map.rename(
            columns={
                "cluster_id": "hdbscan_cluster_assigned",
                "cluster_label": "auto_label_v2",
                "cluster_summary": "auto_label_definition",
            }
        ),
        on="hdbscan_cluster_assigned",
        how="left",
    )
    result["auto_label_source"] = "cluster_onet_tasks.py:hdbscan_cluster_assigned"
    result["variant"] = args.variant
    result["was_hdbscan_noise"] = result["hdbscan_cluster"] == -1

    missing_lab = int(result["auto_label_v2"].isna().sum())
    if missing_lab:
        print(f"[warn] {missing_lab} rows could not be mapped to a cluster label (should be 0).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    n_unique = int(result["auto_label_v2"].nunique())
    print(
        f"Wrote {out_path} rows={len(result)} unique_auto_labels={n_unique} "
        f"noise_rows_soft_assigned={int(result['was_hdbscan_noise'].sum())}"
    )


if __name__ == "__main__":
    main()
