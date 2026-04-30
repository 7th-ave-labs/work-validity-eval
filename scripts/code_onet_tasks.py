#!/usr/bin/env python3
"""Code each O*NET task against the 24-op frozen taxonomy via deterministic join.

Post-pass-5 redesign (2026-04-21, see docs/research_paper.md §3.6): the
taxonomy is no longer the output of an independent LLM coder over task text.
It is the emergent hierarchy derived from clustering, so assigning a task to
an operation is just a deterministic join:

    task
      -> hdbscan_cluster_assigned   (hard + soft-assigned noise;
                                     from scripts/cluster_onet_tasks.py)
      -> cluster_id (one of 68)
      -> label_id   (one of 24;
                     from data/derived/cluster_to_label__operation_only_r2.csv)

This guarantees by construction that every task in cluster X gets the same
operation label, removing LLM drift as a source of disagreement between
geometry-driven clusters and label-assignment.

Per-task *requirements* (explicit_requirements tokens) are not coded per task
anymore: those are properties of the operation, not individual tasks, and are
produced by scripts/code_operation_requirements.py (one call per operation).

Writes:
  data/derived/onet_tasks_coded.parquet
  data/derived/onet_tasks_coded.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"


def salience_tier(importance: float | None, relevance: float | None, task_type: str | None) -> str:
    imp = importance if importance is not None and not pd.isna(importance) else 3.0
    rel = relevance if relevance is not None and not pd.isna(relevance) else 50.0
    core = (task_type == "Core")
    score = imp + (rel / 100.0) * 2.0 + (1.0 if core else 0.0)
    if score >= 6.2:
        return "high"
    if score >= 5.3:
        return "medium"
    return "low"


def run() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--clusters",
        default=str(DERIVED / "onet_task_clusters__operation_only_r2.parquet"),
        help="Per-task cluster assignment parquet (must have task_id, onetsoc_code, "
             "hdbscan_cluster_assigned).",
    )
    p.add_argument(
        "--cluster-to-label",
        default=str(DERIVED / "cluster_to_label__operation_only_r2.csv"),
        help="cluster_id -> label_id map produced by scripts/finalize_work_patterns.py.",
    )
    p.add_argument(
        "--labels-csv",
        default=str(ROOT / "data" / "work_pattern_labels.csv"),
        help="The 24-op frozen taxonomy.",
    )
    p.add_argument("--out-parquet", default=str(DERIVED / "onet_tasks_coded.parquet"))
    p.add_argument("--out-csv", default=str(DERIVED / "onet_tasks_coded.csv"))
    args = p.parse_args()

    tasks = pd.read_parquet(args.clusters)
    c2l = pd.read_csv(args.cluster_to_label)
    labels = pd.read_csv(args.labels_csv)

    assert "hdbscan_cluster_assigned" in tasks.columns, (
        "clusters parquet missing hdbscan_cluster_assigned; "
        "re-run scripts/cluster_onet_tasks.py with prediction_data=True."
    )
    assert set(c2l["cluster_id"]).issubset(set(tasks["hdbscan_cluster_assigned"])), (
        "cluster_to_label has ids not present in the cluster assignment; stale map?"
    )

    merged = tasks.merge(
        c2l,
        how="left",
        left_on="hdbscan_cluster_assigned",
        right_on="cluster_id",
        validate="many_to_one",
    )

    # Every soft-assigned task should land on a label (clusters partition the
    # 68 sub-ops across 24 labels; soft-assignment routes noise to one of them).
    unmapped = merged["label_id"].isna().sum()
    if unmapped:
        print(f"[warn] {unmapped} tasks did not map to any label_id; check cluster_to_label coverage.")

    merged["primary_label_id"] = merged["label_id"].fillna("other")
    merged["coding_method"] = "cluster_to_label_deterministic"
    merged["coding_confidence"] = merged["hdbscan_cluster"].apply(
        lambda hc: "high" if (pd.notna(hc) and int(hc) != -1) else "medium"
    )
    merged["salience_tier"] = merged.apply(
        lambda r: salience_tier(
            r.get("importance_mean"), r.get("relevance_pct"), r.get("task_type")
        ),
        axis=1,
    )

    out_cols = [
        "onetsoc_code",
        "task_id",
        "occupation_title",
        "task_text_clean",
        "task_type",
        "job_zone",
        "importance_mean",
        "relevance_pct",
        "salience_tier",
        "hdbscan_cluster",
        "hdbscan_cluster_assigned",
        "kmeans_cluster",
        "primary_label_id",
        "coding_method",
        "coding_confidence",
    ]
    out = merged[out_cols].copy()

    out.to_parquet(args.out_parquet, index=False)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_parquet} rows={len(out)}")
    print("Primary-label distribution:")
    print(
        out["primary_label_id"]
        .value_counts()
        .rename_axis("label_id")
        .reset_index(name="n_tasks")
        .to_string(index=False)
    )

    total_labels = labels["label_id"].nunique()
    matched = out["primary_label_id"].nunique()
    soft_assigned = int((out["hdbscan_cluster"] == -1).sum())
    hard = len(out) - soft_assigned
    print(
        f"\n{len(out)} tasks coded against {matched}/{total_labels} labels "
        f"({hard} hard-cluster + {soft_assigned} soft-assigned noise)."
    )


if __name__ == "__main__":
    run()
