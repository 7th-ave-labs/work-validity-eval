#!/usr/bin/env python3
"""Assemble ESCO coverage tables and the per-op summary for the paper.

Outputs:
  data/derived/esco/esco_coverage_by_op.csv
  data/derived/esco/esco_coverage_summary.json
  data/derived/esco/esco_unmapped_sample.csv

The numbers collected here back the external-robustness section of the
paper (§5 and appendix), mirroring the role of
``data/derived/coverage_by_label.csv`` for the O*NET side.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ESCO_DERIVED = ROOT / "data" / "derived" / "esco"
DERIVED = ROOT / "data" / "derived"


ONET_OP_TASK_COUNTS = {
    "analysis": 1732,
    "administration": 1517,
    "design": 992,
    "inspection": 963,
    "troubleshooting": 902,
    "record_keeping": 882,
    "teaching": 814,
    "protocol_execution": 735,
    "representation": 615,
    "coordination": 603,
    "creative_production": 543,
    "fabrication": 505,
    "appraisal": 453,
    "advising": 403,
    "investigation": 377,
    "emergency_response": 212,
    "self_study": 186,
    "rule_enforcement": 30,
}


def main() -> None:
    ESCO_DERIVED.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(ESCO_DERIVED / "esco_op_assignments.parquet")

    total = len(df)
    none_rate = (df["llm_op_id"] == "none_of_the_above").mean()

    # Per-op coverage against the LLM label (primary) with centroid-agreement
    # stats alongside.
    rows: list[dict] = []
    for op in ONET_OP_TASK_COUNTS.keys():
        llm_mask = df["llm_op_id"] == op
        sub = df[llm_mask]
        n_items = int(llm_mask.sum())
        agree = int(((df["centroid_op_id"] == op) & llm_mask).sum())
        top2 = int(
            (((df["centroid_op_id"] == op) | (df["centroid_top2_op_id"] == op)) & llm_mask).sum()
        )
        conf_counts = sub["llm_confidence"].value_counts().to_dict()
        rows.append(
            {
                "op_id": op,
                "onet_n_tasks": ONET_OP_TASK_COUNTS[op],
                "esco_n_items_llm": n_items,
                "esco_pct_of_corpus": round(100 * n_items / total, 2),
                "centroid_agree_n": agree,
                "centroid_agree_pct": round(100 * agree / max(n_items, 1), 1),
                "centroid_top2_agree_pct": round(100 * top2 / max(n_items, 1), 1),
                "llm_conf_high": int(conf_counts.get("high", 0)),
                "llm_conf_medium": int(conf_counts.get("medium", 0)),
                "llm_conf_low": int(conf_counts.get("low", 0)),
                "centroid_cos_top1_median": (
                    float(sub["centroid_cos_top1"].median()) if n_items else None
                ),
            }
        )
    per_op = pd.DataFrame(rows).sort_values("esco_n_items_llm", ascending=False)
    per_op.to_csv(ESCO_DERIVED / "esco_coverage_by_op.csv", index=False)

    # Overall summary.
    summary = {
        "n_esco_items": total,
        "n_none_of_the_above": int((df["llm_op_id"] == "none_of_the_above").sum()),
        "pct_none_of_the_above": round(100 * none_rate, 2),
        "n_covered_by_any_op_llm": int(total - (df["llm_op_id"] == "none_of_the_above").sum()),
        "pct_covered_by_any_op_llm": round(
            100 * (1 - none_rate), 2
        ),
        "n_ops_with_any_esco_items_llm": int(
            df[df["llm_op_id"] != "none_of_the_above"]["llm_op_id"].nunique()
        ),
        "n_ops_with_ge_40_esco_items_llm": int(
            (df[df["llm_op_id"] != "none_of_the_above"]["llm_op_id"]
             .value_counts() >= 40).sum()
        ),
        "centroid_llm_strict_agreement_pct": round(
            100 * float((df["centroid_op_id"] == df["llm_op_id"]).mean()), 2
        ),
        "centroid_top2_llm_agreement_pct": round(
            100
            * float(
                ((df["centroid_op_id"] == df["llm_op_id"])
                 | (df["centroid_top2_op_id"] == df["llm_op_id"])).mean()
            ),
            2,
        ),
        "llm_confidence_distribution": df["llm_confidence"].value_counts().to_dict(),
    }

    # Include the structural recoverability metrics if present.
    rec_path = ESCO_DERIVED / "esco_structural_recoverability.json"
    if rec_path.exists():
        rec = json.loads(rec_path.read_text())
        summary["hdbscan_ari_vs_llm"] = rec["vs_llm_op_id"]["ari"]
        summary["hdbscan_nmi_vs_llm"] = rec["vs_llm_op_id"]["nmi"]
        summary["kmeans18_ari_vs_llm"] = rec["vs_llm_op_id_kmeans18"]["ari"]
        summary["kmeans18_nmi_vs_llm"] = rec["vs_llm_op_id_kmeans18"]["nmi"]
        summary["knn_same_op_rate"] = rec["knn_label_consistency_vs_llm"]
        summary["knn_same_op_rate_null"] = rec["knn_label_consistency_null"]
        summary["silhouette_llm_op_id_umap15"] = rec["silhouette_llm_op_id"]

    (ESCO_DERIVED / "esco_coverage_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    # Unmapped sample: ESCO items the LLM declared none_of_the_above.
    unmapped = df[df["llm_op_id"] == "none_of_the_above"].copy()
    unmapped = unmapped[
        [
            "esco_uri",
            "preferred_label",
            "description",
            "centroid_op_id",
            "centroid_cos_top1",
            "centroid_top2_op_id",
            "llm_rationale",
        ]
    ].sort_values("centroid_cos_top1", ascending=True)
    unmapped.to_csv(ESCO_DERIVED / "esco_unmapped_sample.csv", index=False)

    print("per-op coverage:")
    print(per_op.to_string(index=False))
    print("\nsummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
