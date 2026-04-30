#!/usr/bin/env python3
"""Build the ESCO atlas figure.

Produces a 2-D UMAP of the ESCO items coloured by the consensus op
assignment (or LLM op when consensus is missing). Visual palette matches
the O*NET operation atlas (``scripts/build_operation_atlas.py``) so the
two atlases can be compared side by side.

Outputs:
  docs/figures/esco_atlas_2d.png
  docs/figures/esco_atlas_op_bars.png  (per-op n_items bar chart)

Run:
  python3 scripts/esco/build_esco_atlas.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ESCO_DERIVED = ROOT / "data" / "derived" / "esco"
FIG_DIR = ROOT / "docs" / "figures"


OP_COLOR_ORDER = [
    "analysis",
    "administration",
    "design",
    "inspection",
    "troubleshooting",
    "record_keeping",
    "teaching",
    "protocol_execution",
    "representation",
    "coordination",
    "creative_production",
    "fabrication",
    "appraisal",
    "advising",
    "investigation",
    "emergency_response",
    "self_study",
    "rule_enforcement",
]

OP_COLOR: dict[str, str] = {
    "analysis":            "#4C79A7",
    "administration":      "#F28E2B",
    "design":              "#59A14F",
    "inspection":          "#E15759",
    "troubleshooting":     "#B07AA1",
    "record_keeping":      "#9C755F",
    "teaching":            "#76B7B2",
    "protocol_execution":  "#EDC948",
    "representation":      "#FF9DA7",
    "coordination":        "#BAB0AC",
    "creative_production": "#86BCB6",
    "fabrication":         "#D37295",
    "appraisal":           "#A0CBE8",
    "advising":            "#FABFD2",
    "investigation":       "#E6A3A3",
    "emergency_response":  "#F4B98A",
    "self_study":          "#B6BCC9",
    "rule_enforcement":    "#D5BBA1",
}


def build_atlas() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(ESCO_DERIVED / "esco_hdbscan_clusters.parquet")

    # Primary label is the LLM op (semantic), falling back to centroid
    # where the LLM returned ``none_of_the_above``. This matches the
    # paper-text narrative: LLM defines membership, centroid geometry is
    # the independent agreement test.
    label = df["llm_op_id"].where(df["llm_op_id"] != "none_of_the_above")
    label = label.fillna("none_of_the_above")
    df["plot_op"] = label

    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=220)

    other_mask = df["plot_op"] == "none_of_the_above"
    ax.scatter(
        df.loc[other_mask, "umap_x"],
        df.loc[other_mask, "umap_y"],
        s=3,
        c="#D0D0D0",
        alpha=0.25,
        linewidths=0,
        label="none_of_the_above",
    )

    for op in OP_COLOR_ORDER:
        m = df["plot_op"] == op
        if not m.any():
            continue
        ax.scatter(
            df.loc[m, "umap_x"],
            df.loc[m, "umap_y"],
            s=5,
            c=OP_COLOR[op],
            alpha=0.75,
            linewidths=0,
            label=op,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(
        "ESCO skill/competence atlas (n=5,826)  \u2014  coloured by LLM op assignment",
        fontsize=9,
    )

    legend_patches = [
        mpatches.Patch(color=OP_COLOR[op], label=op) for op in OP_COLOR_ORDER
    ] + [mpatches.Patch(color="#D0D0D0", label="none_of_the_above")]
    ax.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=6,
        frameon=False,
        handlelength=0.8,
        handleheight=0.8,
    )
    fig.tight_layout()
    out_png = FIG_DIR / "esco_atlas_2d.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def build_bar_chart() -> None:
    df = pd.read_parquet(ESCO_DERIVED / "esco_op_assignments.parquet")
    counts = (
        df[df["llm_op_id"] != "none_of_the_above"]["llm_op_id"]
        .value_counts()
        .reindex(OP_COLOR_ORDER)
        .fillna(0)
        .astype(int)
    )
    onet_counts = pd.Series(
        {
            "analysis": 1732, "administration": 1517, "design": 992,
            "inspection": 963, "troubleshooting": 902, "record_keeping": 882,
            "teaching": 814, "protocol_execution": 735, "representation": 615,
            "coordination": 603, "creative_production": 543, "fabrication": 505,
            "appraisal": 453, "advising": 403, "investigation": 377,
            "emergency_response": 212, "self_study": 186, "rule_enforcement": 30,
        }
    ).reindex(OP_COLOR_ORDER)

    # Normalise each side to % of its own corpus so the two distributions
    # can sit on the same axis.
    onet_pct = 100 * onet_counts / onet_counts.sum()
    esco_pct = 100 * counts / counts.sum()

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=220)
    x = np.arange(len(OP_COLOR_ORDER))
    width = 0.4
    ax.bar(x - width / 2, onet_pct.values, width, color="#4C79A7", label="O*NET (% of 12,464 tasks)")
    ax.bar(x + width / 2, esco_pct.values, width, color="#F28E2B", label="ESCO (% of 5,730 mapped items)")
    ax.set_xticks(x)
    ax.set_xticklabels(OP_COLOR_ORDER, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("% of own corpus", fontsize=8)
    ax.set_title(
        "Per-op share: O*NET task corpus vs ESCO skill/competence corpus", fontsize=9
    )
    ax.legend(fontsize=7, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "esco_onet_op_share.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    build_atlas()
    build_bar_chart()


if __name__ == "__main__":
    main()
