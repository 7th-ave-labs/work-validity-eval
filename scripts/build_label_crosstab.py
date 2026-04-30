#!/usr/bin/env python3
"""Cross-tab the frozen 28-label taxonomy against the unsupervised ``auto_label_v2``.

Purpose: sensitivity check. Read both the frozen coding (primary_label_id on
``onet_tasks_coded.parquet``) and the bottom-up unsupervised label
(``auto_label_v2`` from ``auto_labels_v2__<variant>.csv``) and produce:

  data/derived/label_crosstab__<variant>.csv
    Long-format table: frozen_label x auto_label with n_tasks and
    salience-weighted share. One row per (frozen_label, auto_label) pair.

  data/derived/label_crosstab__<variant>__diag.csv
    Per frozen-label diagnostic: purity (share of dominant auto_label),
    dominant_auto_label, and number of auto_labels touched. Useful for
    spotting which frozen labels split into multiple auto-buckets (possible
    sub-structure) and which auto-buckets are scattered across many frozen
    labels (possible frozen-side conflation).

Inputs:
  data/derived/onet_tasks_coded.parquet
  data/derived/auto_labels_v2__<variant>.csv

Outputs:
  data/derived/label_crosstab__<variant>.csv
  data/derived/label_crosstab__<variant>__diag.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"


SALIENCE_WEIGHTS = {
    "core_high": 1.0,
    "core": 0.75,
    "supplemental_high": 0.5,
    "supplemental": 0.25,
}


def _variant_suffix(variant: str) -> str:
    return f"__{variant}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant",
        default="task_only",
        choices=("task_only", "title_plus_task", "operation_only", "operation_only_r2"),
    )
    p.add_argument("--coded", default=str(DERIVED / "onet_tasks_coded.parquet"))
    p.add_argument("--auto", default=None, help="Override auto_labels_v2 CSV path.")
    p.add_argument("--out", default=None, help="Override long-format crosstab CSV path.")
    p.add_argument(
        "--diag-out", default=None, help="Override per-frozen-label diagnostic CSV path."
    )
    args = p.parse_args()

    suffix = _variant_suffix(args.variant)
    auto_path = Path(args.auto or DERIVED / f"auto_labels_v2{suffix}.csv")
    out_path = Path(args.out or DERIVED / f"label_crosstab{suffix}.csv")
    diag_path = Path(args.diag_out or DERIVED / f"label_crosstab{suffix}__diag.csv")

    coded = pd.read_parquet(args.coded)[
        ["onetsoc_code", "task_id", "primary_label_id", "salience_tier", "importance_mean"]
    ].copy()
    auto = pd.read_csv(auto_path)[
        ["onetsoc_code", "task_id", "auto_label_v2", "hdbscan_cluster_assigned", "was_hdbscan_noise"]
    ].copy()
    coded["task_id"] = coded["task_id"].astype(str)
    auto["task_id"] = auto["task_id"].astype(str)

    merged = coded.merge(auto, on=["onetsoc_code", "task_id"], how="inner")
    if len(merged) != len(coded):
        print(
            f"[warn] merge kept {len(merged)} / {len(coded)} coded rows; "
            "some tasks may be missing an auto_label_v2."
        )

    merged["salience_weight"] = merged["salience_tier"].map(SALIENCE_WEIGHTS).fillna(0.5)

    # Long-format crosstab.
    ct = (
        merged.groupby(["primary_label_id", "auto_label_v2"], dropna=False)
        .agg(
            n_tasks=("task_id", "size"),
            n_occupations=("onetsoc_code", "nunique"),
            salience_weighted_n=("salience_weight", "sum"),
            mean_importance=("importance_mean", "mean"),
            n_soft_assigned_noise=("was_hdbscan_noise", "sum"),
        )
        .reset_index()
        .sort_values(["primary_label_id", "n_tasks"], ascending=[True, False])
    )
    # Share within each frozen label.
    totals = ct.groupby("primary_label_id")["n_tasks"].transform("sum")
    ct["share_of_frozen_label"] = ct["n_tasks"] / totals.replace(0, np.nan)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ct.to_csv(out_path, index=False)
    print(f"Wrote {out_path} rows={len(ct)}")

    # Per-frozen-label diagnostic: purity, dominant auto_label, n_auto_labels.
    diag_rows = []
    for frozen, sub in ct.groupby("primary_label_id"):
        sub_sorted = sub.sort_values("n_tasks", ascending=False)
        total = int(sub_sorted["n_tasks"].sum())
        dom = sub_sorted.iloc[0] if len(sub_sorted) else None
        diag_rows.append(
            {
                "primary_label_id": frozen,
                "n_tasks": total,
                "n_auto_labels": int(sub_sorted["auto_label_v2"].nunique(dropna=False)),
                "dominant_auto_label": None if dom is None else dom["auto_label_v2"],
                "dominant_share": 0.0 if (dom is None or total == 0) else float(dom["n_tasks"]) / total,
                "top2_auto_labels": "; ".join(
                    str(x)
                    for x in sub_sorted["auto_label_v2"].head(2).tolist()
                    if pd.notna(x)
                ),
            }
        )
    diag = pd.DataFrame(diag_rows).sort_values("dominant_share", ascending=False)
    diag.to_csv(diag_path, index=False)
    print(f"Wrote {diag_path} rows={len(diag)}")

    # Short console summary.
    print("\nPurity by frozen label (share of dominant auto_label):")
    for _, row in diag.iterrows():
        print(
            f"  {row['primary_label_id']:<40s}  "
            f"n={int(row['n_tasks']):>5d}  "
            f"purity={row['dominant_share']:.2f}  "
            f"k_auto={int(row['n_auto_labels']):>3d}  "
            f"dominant={row['dominant_auto_label']}"
        )


if __name__ == "__main__":
    main()
