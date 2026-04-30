#!/usr/bin/env python3
"""Populate the benchmark inventory + operationalization coding.

We (a) fill `data/benchmarks.csv` with a canonical row per scored unit / task
template, (b) update `data/benchmark_sample.csv` harvest status, and (c) write
labels to `data/labels.csv` mapping each benchmark scored unit to one or more
work-pattern labels.

This script relies on hand-curated knowledge of each benchmark source (paper,
task form, scoring rule). All rows include a `source_ref` (URL or arxiv id)
and `notes` with the task form and scoring rule, so a reviewer can audit
without re-running the scripts.
"""

from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_UNITS: list[dict] = [
    # ---------- LegalBench ----------
    {
        "benchmark_id": "bench_legalbench_issue_spotting",
        "parent_id": "bench_legalbench",
        "source": "LegalBench",
        "source_ref": "https://arxiv.org/abs/2308.11462",
        "scored_unit": "Issue-spotting prompts: 'does this fact pattern implicate rule X?'",
        "task_form": "binary / multi-class classification over legal fact patterns",
        "input_form": "short legal fact pattern (single document)",
        "scoring_rule": "exact-match accuracy vs. expert-labeled key",
        "task_text": "Given a short fact pattern and a candidate legal rule, answer whether the rule is implicated.",
        "labels": ["compare_records_against_standards"],
        "notes": "LegalBench aggregates many issue-spotting tasks from IRAC-style inputs; scoring is exact match.",
    },
    {
        "benchmark_id": "bench_legalbench_rule_application",
        "parent_id": "bench_legalbench",
        "source": "LegalBench",
        "source_ref": "https://arxiv.org/abs/2308.11462",
        "scored_unit": "Rule-application prompts (IRAC application step)",
        "task_form": "short-text classification / generation with expert key",
        "input_form": "fact pattern + rule (single document)",
        "scoring_rule": "exact match or expert-labeled key",
        "task_text": "Apply a supplied rule to a supplied fact pattern and give the conclusion.",
        "labels": ["compare_records_against_standards", "evaluate_compliance"],
        "notes": "Strips away drafting, client interaction, and downstream document production.",
    },
    {
        "benchmark_id": "bench_legalbench_rhetorical_analysis",
        "parent_id": "bench_legalbench",
        "source": "LegalBench",
        "source_ref": "https://arxiv.org/abs/2308.11462",
        "scored_unit": "Rhetorical-analysis / clause classification prompts",
        "task_form": "span or sentence classification",
        "input_form": "single contract/opinion excerpt",
        "scoring_rule": "exact-match accuracy",
        "task_text": "Classify sentences or clauses in a legal text by their rhetorical/functional role.",
        "labels": ["extract_or_verify_information_from_documents"],
        "notes": "",
    },

    # ---------- FinanceBench ----------
    {
        "benchmark_id": "bench_financebench_core",
        "parent_id": "bench_financebench",
        "source": "FinanceBench",
        "source_ref": "https://arxiv.org/abs/2311.11944",
        "scored_unit": "Open-book QA over public filings (10-K, 10-Q, earnings)",
        "task_form": "question answering with grounded citation",
        "input_form": "long-form filing PDF (retrieve + extract)",
        "scoring_rule": "answer correctness vs. human-annotated reference + citation check",
        "task_text": "Answer a financial question by retrieving and quoting evidence from the supplied filing.",
        "labels": ["extract_or_verify_information_from_documents", "analyze_data_for_decisions"],
        "notes": "Does not require the downstream memo, recommendation, or analyst judgement.",
    },

    # ---------- BrowseComp ----------
    {
        "benchmark_id": "bench_browsecomp_core",
        "parent_id": "bench_browsecomp",
        "source": "BrowseComp",
        "source_ref": "https://openai.com/index/browsecomp/",
        "scored_unit": "Hard-to-find fact queries; agent must browse to verify",
        "task_form": "multi-hop web-browsing QA",
        "input_form": "open web via browser tool",
        "scoring_rule": "exact-match / string match vs. reference answer",
        "task_text": "Given a hard factual question, use a browser to find and return the single verified answer.",
        "labels": ["gather_field_evidence", "extract_or_verify_information_from_documents"],
        "notes": "No multi-artifact deliverable; no client interaction.",
    },

    # ---------- GAIA ----------
    {
        "benchmark_id": "bench_gaia_core",
        "parent_id": "bench_gaia",
        "source": "GAIA",
        "source_ref": "https://arxiv.org/abs/2311.12983",
        "scored_unit": "Open-domain assistant tasks with tool use",
        "task_form": "multi-step tool-using QA with single canonical answer",
        "input_form": "web + files + images (heterogeneous)",
        "scoring_rule": "exact match of final answer string",
        "task_text": "Solve a multi-step problem requiring tool use (browsing, files) and return one final answer.",
        "labels": ["gather_field_evidence", "analyze_data_for_decisions"],
        "notes": "Final-answer-only scoring drops intermediate artifacts and reasoning quality.",
    },

    # ---------- WebArena ----------
    {
        "benchmark_id": "bench_webarena_core",
        "parent_id": "bench_webarena",
        "source": "WebArena",
        "source_ref": "https://arxiv.org/abs/2307.13854",
        "scored_unit": "Realistic web-agent tasks in reproducible sandboxes (shopping, git, maps, CMS, reddit)",
        "task_form": "sequential web action → goal state",
        "input_form": "browser over sandboxed web apps",
        "scoring_rule": "programmatic check of end-state (URL, DB state, rendered text)",
        "task_text": "Complete a natural-language web task by driving a browser to change the goal application state.",
        "labels": ["process_or_verify_transactional_artifacts", "serve_clients_or_public"],
        "notes": "End-state only; does not model other actors, policies, or review steps.",
    },

    # ---------- WorkArena ----------
    {
        "benchmark_id": "bench_workarena_core",
        "parent_id": "bench_workarena",
        "source": "WorkArena",
        "source_ref": "https://arxiv.org/abs/2403.07718",
        "scored_unit": "Enterprise knowledge-worker tasks on ServiceNow",
        "task_form": "sequential UI action → goal state in enterprise record system",
        "input_form": "browser over ServiceNow sandbox",
        "scoring_rule": "programmatic goal state (ticket state, record edits)",
        "task_text": "Complete enterprise workflow tasks (dashboard filtering, form filling, list sort) on an enterprise record system.",
        "labels": ["collect_and_maintain_records", "process_or_verify_transactional_artifacts"],
        "notes": "Tasks are short and atomic; no multi-party coordination or approval chain.",
    },

    # ---------- OSWorld ----------
    {
        "benchmark_id": "bench_osworld_core",
        "parent_id": "bench_osworld",
        "source": "OSWorld",
        "source_ref": "https://arxiv.org/abs/2404.07972",
        "scored_unit": "Computer-use tasks across desktop OS apps",
        "task_form": "GUI agent: screen+keyboard+mouse to goal state",
        "input_form": "full desktop environment (files, browsers, office apps)",
        "scoring_rule": "task-specific programmatic checks on final state",
        "task_text": "Complete a desktop task using mouse and keyboard actions across multiple applications.",
        "labels": ["process_or_verify_transactional_artifacts", "collect_and_maintain_records"],
        "notes": "Broader UI envelope than WebArena, but still scored on terminal goal state.",
    },

    # ---------- OfficeBench ----------
    {
        "benchmark_id": "bench_officebench_core",
        "parent_id": "bench_officebench",
        "source": "OfficeBench",
        "source_ref": "https://arxiv.org/abs/2407.19056",
        "scored_unit": "Office-software tasks across Word, Excel, PPT",
        "task_form": "multi-step office-app actions to produce a target document",
        "input_form": "office files",
        "scoring_rule": "document-state comparison to reference artifact",
        "task_text": "Edit or produce an office document (text/spreadsheet/slides) to match a specification.",
        "labels": ["prepare_or_produce_reports_and_documentation".replace('prepare_or_produce_reports_and_documentation','write_research_or_analytic_reports'), "collect_and_maintain_records"],
        "notes": "Reference-artifact scoring may admit many acceptable variants in real work.",
    },

    # ---------- MedAgentBench ----------
    {
        "benchmark_id": "bench_medagentbench_core",
        "parent_id": "bench_medagentbench",
        "source": "MedAgentBench",
        "source_ref": "https://arxiv.org/abs/2501.14654",
        "scored_unit": "Clinical tasks against a simulated EHR with FHIR tools",
        "task_form": "API-using agent against simulated EHR",
        "input_form": "simulated EHR + FHIR tools",
        "scoring_rule": "programmatic checks on resulting resources (orders, notes, chart entries)",
        "task_text": "Use EHR tools to complete a clinical administrative or decision-support task end-to-end.",
        "labels": ["collect_and_maintain_records", "diagnose_or_assess_needs", "process_or_verify_transactional_artifacts"],
        "notes": "Closer to real clinician desk work than most LLM benches; still no live patient interaction.",
    },

    # ---------- APEX-Agents ----------
    {
        "benchmark_id": "bench_apex_swe_core",
        "parent_id": "bench_apex_agents",
        "source": "APEX-SWE",
        "source_ref": "https://arxiv.org/abs/2506.12326",
        "scored_unit": "Software engineering tasks scored by rubric + tests with observability",
        "task_form": "multi-file code change with passing tests + rubric review",
        "input_form": "repository + issue + test environment",
        "scoring_rule": "unit tests + rubric pass/fail with review lane",
        "task_text": "Implement a change in a repository that meets an issue and passes hidden tests, with rubric compliance.",
        "labels": ["design_technical_artefact", "develop_methods_or_techniques", "evaluate_compliance"],
        "notes": "Integrates test-pass with rubric / review lane (related to SWE-PRBench framing).",
    },
]


def normalize_label(lid: str) -> str:
    # handle accidental concatenation in the curated dicts
    if "prepare_or_produce_reports_and_documentation" in lid:
        return "write_research_or_analytic_reports"
    return lid


def main() -> None:
    labels_df = pd.read_csv(ROOT / "data" / "work_pattern_labels.csv")
    allowed = set(labels_df["label_id"])
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    benchmark_rows = []
    label_rows = []
    parents_covered: dict[str, list[str]] = {}

    for entry in BENCHMARK_UNITS:
        parent_id = entry["parent_id"]
        used_labels = []
        for raw in entry["labels"]:
            lid = normalize_label(raw)
            if lid not in allowed:
                raise SystemExit(f"Unknown label: {lid}")
            used_labels.append(lid)
        benchmark_rows.append(
            {
                "benchmark_id": entry["benchmark_id"],
                "source": entry["source"],
                "source_ref": entry["source_ref"],
                "task_text": entry["task_text"],
                "notes": f"[unit={entry['scored_unit']}] [task_form={entry['task_form']}] "
                         f"[input_form={entry['input_form']}] [scoring={entry['scoring_rule']}] {entry['notes']}",
            }
        )
        parents_covered.setdefault(parent_id, []).extend(used_labels)
        for lid in used_labels:
            label_rows.append(
                {
                    "label_id": f"lab_{entry['benchmark_id']}__{lid}",
                    "benchmark_id": entry["benchmark_id"],
                    "label_value": lid,
                    "annotator": "manual-literature-review",
                    "created_at": now,
                    "notes": "",
                }
            )

    pd.DataFrame(benchmark_rows).to_csv(ROOT / "data" / "benchmarks.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(label_rows).to_csv(ROOT / "data" / "labels.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    sample = pd.read_csv(ROOT / "data" / "benchmark_sample.csv")
    sample["scored_unit"] = sample["benchmark_id"].map(
        lambda b: "; ".join(sorted({e["scored_unit"] for e in BENCHMARK_UNITS if e["parent_id"] == b}))
        or ""
    )
    sample["paper_ref"] = sample["benchmark_id"].map(
        lambda b: next((e["source_ref"] for e in BENCHMARK_UNITS if e["parent_id"] == b), "")
    )
    sample["work_patterns_covered"] = sample["benchmark_id"].map(
        lambda b: "|".join(sorted(set(parents_covered.get(b, []))))
    )
    sample["harvest_status"] = sample["benchmark_id"].map(
        lambda b: "coded" if b in parents_covered else "pending"
    )
    sample.to_csv(ROOT / "data" / "benchmark_sample.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote benchmarks.csv ({len(benchmark_rows)} scored units)")
    print(f"Wrote labels.csv ({len(label_rows)} label links)")
    print(f"Updated benchmark_sample.csv")


if __name__ == "__main__":
    main()
