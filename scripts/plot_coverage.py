#!/usr/bin/env python3
"""Render the coverage/validity figure for the paper-style synthesis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
FIGS = DERIVED / "figs"


def main() -> None:
    cov = pd.read_csv(DERIVED / "coverage_by_label.csv")
    cov = cov[cov["primary_label_id"] != "other"].copy()
    cov = cov.sort_values("salience_share", ascending=True)

    FIGS.mkdir(parents=True, exist_ok=True)

    colors = {
        "coverage_missing": "#d62728",
        "coverage_weak": "#ff7f0e",
        "coverage_moderate": "#1f77b4",
        "coverage_broad": "#2ca02c",
    }
    fig, ax = plt.subplots(figsize=(10, 9), dpi=120)
    y = np.arange(len(cov))
    ax.barh(y, cov["salience_share"].to_numpy(), color=[colors.get(t, "#888") for t in cov["coverage_tier"]])
    for i, r in enumerate(cov.itertuples()):
        ax.text(
            r.salience_share + 0.001,
            i,
            f"{int(r.n_scored_units)}u",
            va="center",
            fontsize=8,
            color="#444",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(cov["primary_label_id"], fontsize=8)
    ax.set_xlabel("Salience share of O*NET tasks (Job Zones 3–5)")
    ax.set_title(
        "O*NET work-pattern salience vs. benchmark sample coverage\n"
        "bar = share of importance×relevance-weighted task coding; number = scored units covering pattern"
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[k])
        for k in ["coverage_missing", "coverage_weak", "coverage_moderate", "coverage_broad"]
    ]
    ax.legend(
        handles,
        ["0 units (missing)", "1 unit (weak)", "2–3 units (moderate)", ">=4 units (broad)"],
        loc="lower right",
        fontsize=9,
    )
    fig.tight_layout()
    out = FIGS / "coverage_vs_salience.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
