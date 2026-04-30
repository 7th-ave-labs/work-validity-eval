"""Generate a compact operations table for the paper caption.

For each of the 19 top-level operations, emits one markdown table
with:

  - op name
  - n member clusters (included-only)
  - n included tasks
  - 2 example task statements (deterministic pick per op)

Source of truth:
  - data/derived/onet_task_clusters__task_only.parquet  (clusters + tasks)
  - data/derived/onet_tasks_coded.parquet               (round-2 included set)
  - data/derived/manual_labeling/cluster_cross_job_labels.json (cluster -> op)

Inclusion matches build_operation_atlas.py exactly (round-2 screen
plus the force-included cluster audit list). Output:

  docs/figures/operation_examples_table.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_operation_atlas import (
    CLUSTER_PARQUET,
    CODED_PARQUET,
    FORCE_INCLUDE_CLUSTER_IDS,
    LABELS_JSON,
    MERGE_TOP_LEVEL,
    OP_BASE_COLOR,
    OP_COLOR_ORDER,
    REPO,
)

OUT_MD = REPO / "docs" / "figures" / "operation_examples_table.md"

# Tasks picked as representative examples for each op. Selected
# deterministically from the included, non-outlier population so the
# table is reproducible but readable (not random fragments).
RANDOM_SEED = 42
N_EXAMPLES_PER_OP = 2
MAX_EXAMPLE_LEN = 140


def _truncate(s: str, n: int = MAX_EXAMPLE_LEN) -> str:
    s = " ".join(s.split())
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "\u2026"


def _pick_examples(sub: pd.DataFrame, n: int) -> list[tuple[str, str]]:
    """Return n (occupation, task) pairs preferring distinct occupations
    and cluster centres. Deterministic via a fixed seed."""
    if sub.empty:
        return []
    # Rank tasks by closeness to cluster centroid in the 2D UMAP plane
    # so we pick prototypical, not edge-of-cluster, task statements.
    rows: list[tuple[str, str]] = []
    seen_occ: set[str] = set()
    for cid, grp in sub.groupby("cluster_id", sort=False):
        cx = grp["umap_x"].median(); cy = grp["umap_y"].median()
        d = np.hypot(grp["umap_x"] - cx, grp["umap_y"] - cy)
        for i in d.argsort().to_numpy():
            r = grp.iloc[int(i)]
            occ = str(r["occupation_title"])
            if occ in seen_occ:
                continue
            rows.append((occ, str(r["task_text_clean"])))
            seen_occ.add(occ)
            break
        if len(rows) >= n:
            break
    # Fallback if we still do not have n examples (tiny ops).
    if len(rows) < n:
        for _, r in sub.head(n).iterrows():
            occ = str(r["occupation_title"])
            if occ in seen_occ:
                continue
            rows.append((occ, str(r["task_text_clean"])))
            seen_occ.add(occ)
            if len(rows) >= n:
                break
    return rows[:n]


def main() -> None:
    df = pd.read_parquet(CLUSTER_PARQUET)
    df["cluster_id"] = df["hdbscan_cluster_assigned"].astype(int)
    coded = pd.read_parquet(CODED_PARQUET)

    labels_doc = json.loads(LABELS_JSON.read_text())
    per_cluster = {
        int(c["cluster_id"]): {
            "sub_op": c["sub_op"],
            "top_level": MERGE_TOP_LEVEL.get(c["top_level"], c["top_level"]),
            "note": c.get("note", ""),
        }
        for c in labels_doc["clusters"]
    }
    df = df[df["cluster_id"].isin(per_cluster)].copy()
    df["top_level"] = df["cluster_id"].map(lambda c: per_cluster[c]["top_level"])
    df["sub_op"] = df["cluster_id"].map(lambda c: per_cluster[c]["sub_op"])

    inc_keys = set(zip(coded["onetsoc_code"], coded["task_id"]))
    screen = np.array([(s, t) in inc_keys for s, t in zip(df["onetsoc_code"], df["task_id"])])
    force = df["cluster_id"].isin(FORCE_INCLUDE_CLUSTER_IDS).to_numpy()
    df["included"] = screen | force

    incl = df[df["included"]].copy()

    # Per-op counts.
    stats = (
        incl.groupby("top_level")
        .agg(
            n_clusters=("cluster_id", "nunique"),
            n_tasks=("task_id", "size"),
        )
        .to_dict("index")
    )

    rng = np.random.default_rng(RANDOM_SEED)
    lines: list[str] = []
    lines.append("# Operation examples (19 top-level operations)\n")
    lines.append("")
    lines.append(
        "Table compiles, for each of the 19 top-level knowledge-work "
        "operations (v5 / pass-7 taxonomy), the number of member "
        "HDBSCAN clusters, the number of included O*NET tasks, and "
        f"{N_EXAMPLES_PER_OP} representative task statements. Included "
        "counts match the 2D / 3D atlas figure (round-2 knowledge-work "
        "screen plus manually re-included clusters)."
    )
    lines.append("")
    lines.append("| color | op | clusters | tasks | example tasks |")
    lines.append("| :---: | --- | ---: | ---: | --- |")

    for op in OP_COLOR_ORDER:
        sub = incl[incl["top_level"] == op].copy()
        # Stable sort so examples are deterministic per run.
        sub = sub.sort_values(["cluster_id", "task_id"]).reset_index(drop=True)
        n_c = stats.get(op, {}).get("n_clusters", 0)
        n_t = stats.get(op, {}).get("n_tasks", 0)
        ex = _pick_examples(sub, N_EXAMPLES_PER_OP)
        formatted = " <br> ".join(
            f"_{occ}_ — {_truncate(task)}" for occ, task in ex
        ) if ex else "(no included tasks)"
        color = OP_BASE_COLOR[op]
        swatch = f"![{op}](https://singlecolorimage.com/get/{color.lstrip('#')}/14x14.png)"
        lines.append(
            f"| {swatch} | `{op}` | {n_c} | {n_t:,} | {formatted} |"
        )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
