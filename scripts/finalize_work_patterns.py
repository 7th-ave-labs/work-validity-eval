#!/usr/bin/env python3
"""Produce the frozen work-pattern taxonomy from the post-r2 emergent hierarchy.

Reads  : data/derived/operation_hierarchy__operation_only_r2__t0p55.json
         (23 emergent top-level operations from the bottom-up agglomerative
         merge at cosine-distance threshold 0.55 on the 68 post-pass-5
         HDBSCAN clusters; see docs/research_paper.md §3.5.2).

Applies: one manual split of the "Produce and Communicate Research Reports"
         bucket (421 tasks) into "Prepare Reports and Presentations" (clusters
         45 + 9 = 307 tasks) and "Conduct Scientific Research" (cluster 53,
         114 tasks) so benchmark-coverage claims can separate research
         execution from research communication. Decision recorded in the
         paper's reversals log.

Writes :
  data/work_pattern_labels.csv              (authoritative 24-label set)
  data/derived/work_pattern_labels_frozen.json
  data/derived/cluster_to_label__operation_only_r2.csv
        (68 sub-op clusters -> 24 top-level label_id mapping, used to build
        the auto_labels_v2 cross-tab.)

The earlier hand-typed 28-label finalize script is retained for audit as
scripts/finalize_work_patterns_pre_r2_archive.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"

HIERARCHY_JSON = DERIVED / "operation_hierarchy__operation_only_r2__t0p55.json"
CLUSTER_THEMES_JSON = DERIVED / "work_pattern_labels_by_cluster__operation_only_r2.json"

CLUSTER_OR_METHOD = (
    "text-embedding-3-small on gpt-5.4-mini-normalized operation phrases "
    "(domain-stripped) -> UMAP(n_components=15, n_neighbors=30, min_dist=0.0, "
    "metric=cosine) -> HDBSCAN(min_cluster_size=30, min_samples=10, eom, "
    "prediction_data=True) + soft-assign noise -> 68 hard clusters -> "
    "gpt-5.4-mini per-cluster theme + agent-doability/eval rubric -> "
    "agglomerative merge on theme embeddings at cosine-distance threshold "
    "0.55 -> 23 emergent top-level operations -> manual split of research "
    "bucket -> 24-label frozen taxonomy."
)

# cluster_ids to split out of the LLM-produced bucket
# 'produce_and_communicate_research_reports' and promote as their own op.
CONDUCT_RESEARCH_CLUSTER_IDS = {53}
PREPARE_REPORTS_CLUSTER_IDS = {45, 9}

CONDUCT_RESEARCH_LABEL = {
    "label_id": "conduct_scientific_research",
    "label_name": "Conduct Scientific Research",
    "definition": (
        "Plan, conduct, direct, or analyze a research study intended to "
        "produce new scientific, medical, environmental, social, technical, "
        "or economic findings. The shared operation is executing the "
        "research itself (design, data collection, analysis) rather than "
        "merely reporting prior findings, routine assistance, or fundraising."
    ),
    "minimum_task_contract": (
        "The item must present a task whose core operation is planning, "
        "conducting, directing, or analyzing a research study that generates "
        "new findings, with enough context to distinguish it from "
        "information lookup or from writing up already-completed research."
    ),
    "agent_doability": "high",
    "agent_doability_rationale": (
        "Study design, protocol drafting, and results analysis are largely "
        "text-and-data operations an agent can perform with standard tools; "
        "the binding constraint is that real experimental execution and "
        "credentialed field collection often require physical work or "
        "regulatory authorization outside the agent's reach."
    ),
    "evaluation_artifact": "study protocol, analysis plan, or analysis report",
    "evaluation_method": "expert_review",
    "sibling_labels": [
        "prepare_reports_and_presentations",
        "analyze_and_interpret_data_for_decisions",
        "inspect_evaluate_and_verify_compliance_and_performance",
    ],
}

PREPARE_REPORTS_LABEL = {
    "label_id": "prepare_reports_and_presentations",
    "label_name": "Prepare Reports and Presentations",
    "definition": (
        "Compile, synthesize, and communicate completed findings, results, "
        "status, recommendations, or compliance information in a written, "
        "oral, tabular, or graphic artifact (report, presentation, "
        "publication, brief, summary) for a specified audience. The shared "
        "operation is producing the communicative artifact, not performing "
        "the underlying research or analysis that feeds it."
    ),
    "minimum_task_contract": (
        "The item must require producing a concrete communicative artifact "
        "summarizing completed findings for a specified audience, with "
        "enough source material or results to distinguish reporting from "
        "doing the underlying research or analysis itself."
    ),
    "agent_doability": "high",
    "agent_doability_rationale": (
        "Given source findings and an audience specification, drafting "
        "reports, presentations, and briefs is a textbook knowledge-work "
        "operation for an LLM agent with standard writing and charting "
        "tools."
    ),
    "evaluation_artifact": "report or presentation document",
    "evaluation_method": "rubric_on_artifact",
    "sibling_labels": [
        "conduct_scientific_research",
        "analyze_and_interpret_data_for_decisions",
        "develop_and_maintain_instructional_and_operational_documents",
    ],
}


def load_cluster_themes() -> dict[int, dict]:
    raw = json.loads(CLUSTER_THEMES_JSON.read_text())
    if isinstance(raw, list):
        themes = raw
    elif isinstance(raw, dict) and "clusters" in raw:
        themes = raw["clusters"]
    else:
        themes = list(raw.values())
    return {int(t["cluster_id"]): t for t in themes}


def inclusion_examples_for_clusters(
    cluster_ids: list[int], themes: dict[int, dict], max_examples: int = 4
) -> list[str]:
    examples: list[str] = []
    for cid in cluster_ids:
        theme = themes.get(int(cid))
        if not theme:
            continue
        text = (theme.get("best_theme_task_text") or "").strip()
        if text:
            examples.append(text)
        if len(examples) >= max_examples:
            break
    return examples


def build_labels() -> tuple[list[dict], list[dict]]:
    hierarchy = json.loads(HIERARCHY_JSON.read_text())
    themes = load_cluster_themes()

    labels: list[dict] = []
    cluster_map: list[dict] = []  # cluster_id -> label_id rows

    for op in hierarchy:
        op_id = op["operation_id"]
        sub_cluster_ids = [int(s["cluster_id"]) for s in op["sub_operations"]]

        if op_id == "produce_and_communicate_research_reports":
            # Apply manual research split: conduct vs. prepare reports.
            conduct_subs = [
                int(s["cluster_id"]) for s in op["sub_operations"]
                if int(s["cluster_id"]) in CONDUCT_RESEARCH_CLUSTER_IDS
            ]
            prepare_subs = [
                int(s["cluster_id"]) for s in op["sub_operations"]
                if int(s["cluster_id"]) in PREPARE_REPORTS_CLUSTER_IDS
            ]
            conduct_tasks = sum(
                int(s["n_tasks"]) for s in op["sub_operations"]
                if int(s["cluster_id"]) in CONDUCT_RESEARCH_CLUSTER_IDS
            )
            prepare_tasks = sum(
                int(s["n_tasks"]) for s in op["sub_operations"]
                if int(s["cluster_id"]) in PREPARE_REPORTS_CLUSTER_IDS
            )

            conduct = dict(CONDUCT_RESEARCH_LABEL)
            conduct["sub_operation_cluster_ids"] = conduct_subs
            conduct["n_sub_ops"] = len(conduct_subs)
            conduct["n_tasks"] = conduct_tasks
            conduct["inclusion_examples"] = inclusion_examples_for_clusters(
                conduct_subs, themes
            )
            conduct["exclusion_examples"] = [
                "Prepare and present a report summarizing research already "
                "completed by others (that is prepare_reports_and_presentations).",
                "Analyze existing operational data to support a decision "
                "(that is analyze_and_interpret_data_for_decisions).",
            ]
            conduct["source_bucket_coherence_note"] = (
                op.get("bucket_coherence", "") + " | Split out of the emergent "
                "bucket 'Produce and Communicate Research Reports' at promotion "
                "time: research execution and research communication are "
                "different top-level operations that must be separated for "
                "benchmark-coverage analysis."
            ).strip()
            labels.append(conduct)
            for cid in conduct_subs:
                cluster_map.append({"cluster_id": cid, "label_id": conduct["label_id"]})

            prepare = dict(PREPARE_REPORTS_LABEL)
            prepare["sub_operation_cluster_ids"] = prepare_subs
            prepare["n_sub_ops"] = len(prepare_subs)
            prepare["n_tasks"] = prepare_tasks
            prepare["inclusion_examples"] = inclusion_examples_for_clusters(
                prepare_subs, themes
            )
            prepare["exclusion_examples"] = [
                "Plan or conduct a research study to produce new findings "
                "(that is conduct_scientific_research).",
                "Draft a policy or procedure manual for internal operations "
                "(that is develop_and_maintain_instructional_and_operational_documents).",
            ]
            prepare["source_bucket_coherence_note"] = (
                op.get("bucket_coherence", "") + " | Split out of the emergent "
                "bucket 'Produce and Communicate Research Reports' at promotion "
                "time."
            ).strip()
            labels.append(prepare)
            for cid in prepare_subs:
                cluster_map.append({"cluster_id": cid, "label_id": prepare["label_id"]})
            continue

        label = {
            "label_id": op_id,
            "label_name": op["operation_name"],
            "definition": op["definition"],
            "minimum_task_contract": op["minimum_task_contract"],
            "agent_doability": op["agent_doability"],
            "agent_doability_rationale": op["agent_doability_rationale"],
            "evaluation_artifact": op["evaluation_artifact"],
            "evaluation_method": op["evaluation_method"],
            "sub_operation_cluster_ids": sub_cluster_ids,
            "n_sub_ops": int(op["n_sub_ops"]),
            "n_tasks": int(op["n_tasks"]),
            "inclusion_examples": inclusion_examples_for_clusters(sub_cluster_ids, themes),
            "exclusion_examples": [],
            "source_bucket_coherence_note": op.get("bucket_coherence", ""),
            "sibling_labels": [],  # populated after labels are fully built
        }
        labels.append(label)
        for cid in sub_cluster_ids:
            cluster_map.append({"cluster_id": cid, "label_id": op_id})

    # Sort top-level labels by n_tasks descending for stable downstream order.
    labels.sort(key=lambda x: (-int(x.get("n_tasks", 0)), x["label_id"]))

    # Sanity checks.
    assert len(labels) == 24, f"expected 24 labels, got {len(labels)}"
    assert len({x["label_id"] for x in labels}) == 24, "duplicate label_id"
    assert len(cluster_map) == 68, f"expected 68 cluster->label rows, got {len(cluster_map)}"
    # n_tasks in the hierarchy JSON counts hard-cluster members only
    # (HDBSCAN noise is soft-assigned separately by cluster_onet_tasks.py
    # via `hdbscan_cluster_assigned`; noise share ~27% on the post-r2 run,
    # i.e. ~6,080 hard + ~2,258 soft-assigned = 8,338 total).
    total_tasks_hard = sum(int(x["n_tasks"]) for x in labels)
    print(
        f"Hard-cluster tasks across {len(labels)} labels: {total_tasks_hard} "
        f"(soft-assigned noise routed via cluster_to_label map)."
    )

    return labels, cluster_map


def main() -> None:
    labels, cluster_map = build_labels()

    (DERIVED / "work_pattern_labels_frozen.json").write_text(
        json.dumps(labels, indent=2)
    )

    rows = []
    for item in labels:
        rows.append(
            {
                "label_id": item["label_id"],
                "label_name": item["label_name"],
                "description": item["definition"],
                "cluster_or_method": CLUSTER_OR_METHOD,
                "frozen": True,
                "example_onetsoc_tasks": " | ".join(item["inclusion_examples"]),
                "minimum_task_contract": item["minimum_task_contract"],
                "sibling_labels": ",".join(item.get("sibling_labels") or []),
                "exclusion_examples": " | ".join(item.get("exclusion_examples") or []),
                "agent_doability": item["agent_doability"],
                "agent_doability_rationale": item["agent_doability_rationale"],
                "evaluation_artifact": item["evaluation_artifact"],
                "evaluation_method": item["evaluation_method"],
                "n_sub_ops": item["n_sub_ops"],
                "n_tasks_in_corpus": item["n_tasks"],
                "sub_operation_cluster_ids": ",".join(
                    str(c) for c in item["sub_operation_cluster_ids"]
                ),
                "source_bucket_coherence_note": item.get("source_bucket_coherence_note", ""),
                "notes": "",
            }
        )
    out_csv = ROOT / "data" / "work_pattern_labels.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(rows)} labels)")

    cluster_map_df = pd.DataFrame(cluster_map).sort_values("cluster_id")
    cluster_map_csv = DERIVED / "cluster_to_label__operation_only_r2.csv"
    cluster_map_df.to_csv(cluster_map_csv, index=False)
    print(f"Wrote {cluster_map_csv} ({len(cluster_map_df)} rows)")


if __name__ == "__main__":
    main()
