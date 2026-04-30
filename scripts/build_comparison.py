#!/usr/bin/env python3
"""Build coverage-only artifacts: per-op benchmark coverage tier summary.

**Scope (post-demotion, see docs/research_paper.md §3.6).** This script produces
only the *coverage* signal — does any benchmark in `data/benchmarks.csv` map to
this operation, and via how many scored units? It does NOT produce any
validity / "what the benchmark preserves vs. drops" analysis: a previous
revision emitted a hand-curated `PRESERVED` dict and `data/onet_benchmark_comparison.csv`
that were written from benchmark-level *literature-review summaries*, not from
reading actual benchmark items or rubric code. Those artifacts were removed
because the "preserves X / drops Y" claim needs item-level grounding we have
not yet produced. Item-level benchmark validity is a follow-up PR.

Inputs:
  * data/work_pattern_labels.csv           -- 24-op promoted codebook
  * data/benchmarks.csv                    -- scored unit registry
  * data/labels.csv                        -- bench × op hand-link (lit-review,
                                              bench-level, source="manual-literature-review")
  * data/derived/onet_tasks_coded.parquet  -- 8,338 tasks coded via
                                              cluster→label join (deterministic)

Outputs:
  * data/derived/coverage_by_label.csv     -- per-op tasks / occupations /
                                              salience-weight / n_scored_units /
                                              scored_units list / coverage tier.
  * docs/coverage_synthesis.md             -- auto-generated readoff of coverage
                                              (no validity observations).

Coverage tiers:
  * coverage_broad     ≥ 4 scored units
  * coverage_moderate  2-3 scored units
  * coverage_weak      1 scored unit
  * coverage_missing   0 scored units

Caveat: the coverage tier uses bench-level hand-links. It answers "does any
scored unit in this sample claim to target this op", not "does this scored
unit's scoring actually exercise this op's contract on its items". Sharpening
that is the item-level follow-up.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"


def main() -> None:
    labels = pd.read_csv(ROOT / "data" / "work_pattern_labels.csv")
    label_by_id = labels.set_index("label_id").to_dict(orient="index")
    label_links = pd.read_csv(ROOT / "data" / "labels.csv")
    coded = pd.read_parquet(DERIVED / "onet_tasks_coded.parquet")

    # --- ONET per-op salience roll-up -------------------------------------------
    coded["core_weight"] = coded["task_type"].map({"Core": 1.0}).fillna(0.5)
    coded["salience_weight"] = (
        coded["importance_mean"].fillna(3.0)
        * (coded["relevance_pct"].fillna(50.0) / 100.0)
        * coded["core_weight"]
    )

    onet_primary = (
        coded.groupby("primary_label_id")
        .agg(
            n_tasks=("task_id", "size"),
            n_core_tasks=("task_type", lambda s: (s == "Core").sum()),
            n_occupations=("onetsoc_code", "nunique"),
            salience_weight=("salience_weight", "sum"),
            mean_importance=("importance_mean", "mean"),
            mean_relevance=("relevance_pct", "mean"),
        )
        .reset_index()
    )
    total_weight = onet_primary["salience_weight"].sum()
    onet_primary["salience_share"] = onet_primary["salience_weight"] / total_weight

    # --- Benchmark coverage (bench-level hand-link) -----------------------------
    cov = (
        label_links.groupby("label_value")
        .agg(
            n_scored_units=("benchmark_id", "nunique"),
            scored_units=("benchmark_id", lambda s: ";".join(sorted(set(s)))),
        )
        .reset_index()
        .rename(columns={"label_value": "primary_label_id"})
    )

    coverage = onet_primary.merge(cov, on="primary_label_id", how="left")
    coverage["n_scored_units"] = coverage["n_scored_units"].fillna(0).astype(int)
    coverage["scored_units"] = coverage["scored_units"].fillna("")

    def tier(row: pd.Series) -> str:
        if row["primary_label_id"] == "other":
            return "excluded"
        if row["n_scored_units"] == 0:
            return "coverage_missing"
        if row["n_scored_units"] <= 1:
            return "coverage_weak"
        if row["n_scored_units"] <= 3:
            return "coverage_moderate"
        return "coverage_broad"

    coverage["coverage_tier"] = coverage.apply(tier, axis=1)
    coverage = coverage.sort_values("salience_share", ascending=False).reset_index(drop=True)
    coverage.to_csv(DERIVED / "coverage_by_label.csv", index=False)
    print(f"Wrote {DERIVED / 'coverage_by_label.csv'} ({len(coverage)} labels)")

    # --- Coverage-only synthesis text -------------------------------------------
    missing = coverage[coverage["coverage_tier"] == "coverage_missing"]
    weak = coverage[coverage["coverage_tier"] == "coverage_weak"]
    strong = coverage[coverage["coverage_tier"].isin(["coverage_moderate", "coverage_broad"])]

    out_md: list[str] = [
        "# Benchmark coverage summary (auto-generated)",
        "",
        "**Scope.** This document reports *coverage only*: for each of the 24",
        "promoted operations, how many scored units in the benchmark sample",
        "(`data/benchmarks.csv`) have been hand-linked to this operation in",
        "`data/labels.csv` (source: `manual-literature-review`, bench-level).",
        "",
        "**Not in scope.** Item-level validity — i.e., whether a scored unit's",
        "*scoring* actually tests the operation's `minimum_task_contract` on",
        "its items — is **not** produced here. That requires pulling benchmark",
        "items + rubric/scoring code and coding each item against the contract,",
        "which is the next PR. Treat coverage tiers below as an upper bound on",
        "what the sample claims to measure, not on what it meaningfully tests.",
        "",
        "**Reference set.** O*NET 30.2 task statements, Job Zones 3-5, filtered",
        "through the pass-0 knowledge-work prescreen and the pass-5 round-2",
        "task-level knowledge screen (see docs/research_paper.md §3). 8,338",
        "tasks, deterministically coded into 24 operations via the cluster→label",
        "join (`data/derived/cluster_to_label__operation_only_r2.csv`).",
        "",
        "## Per-operation task counts and salience share",
        "",
        "| op | tasks | occupations | salience share | n scored units | tier |",
        "| -- | ----: | ----------: | -------------: | -------------: | ---- |",
    ]
    for _, r in coverage.iterrows():
        lab = label_by_id.get(r["primary_label_id"], {})
        out_md.append(
            f"| {lab.get('label_name', r['primary_label_id'])} | "
            f"{int(r['n_tasks'])} | {int(r['n_occupations'])} | "
            f"{r['salience_share']:.1%} | {int(r['n_scored_units'])} | "
            f"{r['coverage_tier']} |"
        )

    out_md += ["", "## Operations with no scored unit in the sample", ""]
    for _, r in missing.iterrows():
        lab = label_by_id.get(r["primary_label_id"], {})
        out_md.append(
            f"- **{lab.get('label_name', r['primary_label_id'])}** "
            f"(`{r['primary_label_id']}`) — "
            f"{int(r['n_tasks'])} O*NET tasks across {int(r['n_occupations'])} occupations, "
            f"salience share ≈ {r['salience_share']:.1%}."
        )
    if not len(missing):
        out_md.append("- (none: every op has at least one benchmark unit)")

    out_md += ["", "## Operations covered by a single scored unit", ""]
    for _, r in weak.iterrows():
        lab = label_by_id.get(r["primary_label_id"], {})
        out_md.append(
            f"- **{lab.get('label_name', r['primary_label_id'])}** "
            f"(`{r['primary_label_id']}`) — salience share {r['salience_share']:.1%}; "
            f"covered by `{r['scored_units']}`."
        )

    out_md += ["", "## Operations covered by 2+ scored units", ""]
    for _, r in strong.iterrows():
        lab = label_by_id.get(r["primary_label_id"], {})
        out_md.append(
            f"- **{lab.get('label_name', r['primary_label_id'])}** "
            f"(`{r['primary_label_id']}`) — salience share {r['salience_share']:.1%}; "
            f"covered by {int(r['n_scored_units'])} units: `{r['scored_units']}`."
        )

    summary_path = ROOT / "docs" / "coverage_synthesis.md"
    summary_path.write_text("\n".join(out_md) + "\n")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
