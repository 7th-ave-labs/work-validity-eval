#!/usr/bin/env python3
"""Per-cluster refresh of candidate work-pattern labels (no sampling).

For each HDBSCAN cluster from ``cluster_onet_tasks.py`` (new recipe), send
*every* task statement in the cluster to ``gpt-5.4-mini`` and ask it to
produce:

  - ``theme_label``: tight operation-first label (2-6 words)
  - ``definition``: one sentence, operation-first
  - ``best_theme_task``: the *single* exemplar task_id from the cluster
    that most cleanly represents the theme (verbatim, not synthesised)
  - ``inclusion_examples``: 2-3 additional task_ids from the cluster that
    fit the theme
  - ``exclusion_examples``: 0-3 task_ids from the cluster that are
    outliers (the theme does not have to cover these)
  - ``minimum_task_contract``: one sentence describing what a benchmark
    task has to expose to count as this operation.
  - ``n_tasks_covered_estimate``: rough share of cluster tasks the theme
    cleanly covers (0.0-1.0).

This is Option B of the 2026-04-21 taxonomy refresh conversation. It
replaces the older ``propose_work_patterns.py`` stratified-sample path,
which sampled ~12 tasks per cluster. Per user instruction: no sampling;
the LLM sees the whole cluster and picks the best theme task; it is OK
if the theme does not cover every outlier task in the cluster.

Inputs:
  data/derived/onet_task_clusters__<variant>.parquet
    (must have hdbscan_cluster, task_id, task_text_clean, occupation_title,
     onetsoc_code, job_zone, importance_mean, relevance_pct)

Outputs:
  data/derived/work_pattern_labels_by_cluster__<variant>.json
  data/derived/work_pattern_labels_by_cluster__<variant>.csv
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"

DEFAULT_MODEL = os.environ.get("REFRESH_WORK_PATTERNS_MODEL", "gpt-5.4-mini")


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are refining a taxonomy of *work patterns* observed in U.S. O*NET
    occupation task statements (Job Zones 3-5, mostly knowledge and
    professional/technical work).

    A "work pattern" names the kind of OPERATION a worker performs
    (compare, draft, diagnose, advise, supervise, negotiate, ...) on an
    OBJECT (records, patients, equipment, contracts, ...), with any key
    CONSTRAINT that matters for validity (e.g. "against an explicit
    standard", "under regulatory requirements").

    Rules for the label:
      * Operation-first. Not tools, not occupations, not benchmark/AI
        terminology. Good: "Compare records against standards". Bad:
        "Uses Excel", "Nurse documentation", "Question answering".
      * 2-6 words, sentence case.
      * Tight, not an umbrella. It is OK if the label does not cover every
        outlier task in the cluster; a sharp label that covers the core is
        better than a vague label that covers everything.
      * CROSS-JOB. The label must describe an OPERATION that plausibly
        applies across multiple O*NET occupations - not a single
        profession's idiosyncratic jargon. Even if this cluster is
        dominated by one occupation (check the occupation list), label at
        an abstraction level that would make sense for workers in at
        least 3-5 other occupations performing the same operation on an
        analogous object. NEVER name the profession, product, regulatory
        regime, or industry in the label.
          - BAD (job-specific): "Prescribe medications and dosages"
            (physicians only), "Compile subject bibliographies"
            (librarians only), "Grade student exams" (teachers only),
            "Testify in court proceedings" (lawyers only).
          - GOOD (cross-job): "Recommend and document intervention plans"
            (covers physicians, dietitians, case managers, therapists),
            "Compile topic-specific reference lists" (covers researchers,
            librarians, legal assistants, policy analysts), "Assess and
            score submitted work" (covers teachers, reviewers, auditors,
            inspectors), "Present evidence to a deciding body" (covers
            lawyers, auditors, investigators, hearing officers).
        If the cluster's tasks only make sense for one profession,
        abstract UP so the operation names the WORK, not the worker.

    You will be given ALL tasks from one cluster (not a sample). Read all
    of them, identify the dominant theme, and:

      1. Pick the single ``best_theme_task`` by its ``task_id`` - the task
         in the cluster that most cleanly exemplifies the theme. Do NOT
         synthesise or edit the task text; just return its ``task_id``.
      2. Draft ``theme_label`` and ``definition`` from that theme.
      3. Pick 2-3 additional ``inclusion_examples`` (task_ids) that also
         fit the theme.
      4. Pick 0-3 ``exclusion_examples`` (task_ids) from the cluster that
         are outliers - tasks that are in the cluster but do NOT fit the
         theme. These document where the theme boundary sits.
      5. Write a ``minimum_task_contract``: one sentence describing what a
         benchmark item must expose to legitimately count as this work
         pattern (e.g. "Must present the artifact to check, the explicit
         standard, and a scoring rule that credits specific deviation
         calls.").
      6. Estimate ``n_tasks_covered_estimate`` in [0.0, 1.0]: roughly what
         share of the tasks in this cluster the theme cleanly covers.

    Return JSON only. Shape:
      {
        "theme_label": "...",
        "definition": "...",
        "best_theme_task": "<task_id from the cluster>",
        "inclusion_examples": ["<task_id>", "<task_id>", "<task_id>"],
        "exclusion_examples": ["<task_id>", ...],
        "minimum_task_contract": "...",
        "n_tasks_covered_estimate": 0.0
      }
    """
).strip()


def format_cluster_for_prompt(cluster_id: int, sub: pd.DataFrame, max_chars_per_task: int) -> str:
    header = textwrap.dedent(
        f"""
        Cluster id: {cluster_id}
        Number of tasks: {len(sub)}
        Number of distinct occupations: {sub["onetsoc_code"].nunique()}
        Sample occupations (up to 10, sorted by n tasks):
        """
    ).rstrip()
    occ_counts = (
        sub.groupby(["onetsoc_code", "occupation_title"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .head(10)
    )
    occ_lines = "\n".join(
        f"  - {row.occupation_title} ({row.onetsoc_code}, n={int(row.n)})"
        for row in occ_counts.itertuples()
    )
    task_lines = []
    for _, row in sub.iterrows():
        text = str(row["task_text_clean"])[:max_chars_per_task]
        tid = str(row["task_id"])
        occ = str(row["occupation_title"])[:60]
        zone = int(row["job_zone"]) if pd.notna(row["job_zone"]) else -1
        imp = row["importance_mean"] if pd.notna(row["importance_mean"]) else None
        imp_s = f"imp={imp:.2f}" if imp is not None else ""
        task_lines.append(f"[{tid} | zone {zone} | {occ} {imp_s}] {text}")
    return header + "\n" + occ_lines + "\n\nAll tasks in cluster:\n" + "\n".join(task_lines)


def call_llm_for_cluster(
    client: OpenAI,
    model: str,
    cluster_id: int,
    sub: pd.DataFrame,
    max_chars_per_task: int,
) -> dict:
    user_msg = format_cluster_for_prompt(cluster_id, sub, max_chars_per_task)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)

    valid_ids = set(str(t) for t in sub["task_id"].tolist())

    def _keep(ids: list[str]) -> list[str]:
        return [i for i in ids if i in valid_ids]

    data["best_theme_task"] = (
        str(data.get("best_theme_task", "")) if data.get("best_theme_task") in valid_ids else ""
    )
    data["inclusion_examples"] = _keep(
        [str(x) for x in (data.get("inclusion_examples") or [])]
    )[:3]
    data["exclusion_examples"] = _keep(
        [str(x) for x in (data.get("exclusion_examples") or [])]
    )[:3]
    data.setdefault("theme_label", "")
    data.setdefault("definition", "")
    data.setdefault("minimum_task_contract", "")
    try:
        data["n_tasks_covered_estimate"] = float(data.get("n_tasks_covered_estimate", 0.0))
    except (TypeError, ValueError):
        data["n_tasks_covered_estimate"] = 0.0

    data["cluster_id"] = int(cluster_id)
    data["n_tasks"] = int(len(sub))
    data["n_occupations"] = int(sub["onetsoc_code"].nunique())
    data["importance_mean"] = float(sub["importance_mean"].mean())
    data["relevance_mean"] = float(sub["relevance_pct"].mean())

    # Resolve task_ids to readable text for the CSV / audit.
    tid_to_text = dict(zip(sub["task_id"].astype(str), sub["task_text_clean"]))
    data["best_theme_task_text"] = tid_to_text.get(data["best_theme_task"], "")
    data["inclusion_examples_text"] = [
        tid_to_text.get(tid, "") for tid in data["inclusion_examples"]
    ]
    data["exclusion_examples_text"] = [
        tid_to_text.get(tid, "") for tid in data["exclusion_examples"]
    ]

    return data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant",
        default="task_only",
        choices=("task_only", "title_plus_task", "operation_only"),
    )
    p.add_argument("--clusters", default=None, help="Override cluster parquet path.")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument(
        "--max-chars-per-task",
        type=int,
        default=300,
        help="Truncate each task_text before sending to the LLM.",
    )
    p.add_argument(
        "--include-noise",
        action="store_true",
        help="Also send the soft-assigned noise points through the prompt. Default: only"
        " non-noise HDBSCAN clusters.",
    )
    args = p.parse_args()

    suffix = f"__{args.variant}"
    clusters_path = Path(args.clusters or DERIVED / f"onet_task_clusters{suffix}.parquet")
    out_json = Path(args.out_json or DERIVED / f"work_pattern_labels_by_cluster{suffix}.json")
    out_csv = Path(args.out_csv or DERIVED / f"work_pattern_labels_by_cluster{suffix}.csv")

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY / OPENAI_API missing")

    df = pd.read_parquet(clusters_path)
    df["task_id"] = df["task_id"].astype(str)
    cluster_col = "hdbscan_cluster_assigned" if args.include_noise else "hdbscan_cluster"
    df = df[df[cluster_col] != -1].copy()
    grouped = {int(cid): sub for cid, sub in df.groupby(cluster_col)}
    print(
        f"Refreshing labels for {len(grouped)} clusters "
        f"(variant={args.variant}, col={cluster_col}, model={args.model})"
    )

    client = OpenAI(api_key=api_key)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                call_llm_for_cluster,
                client,
                args.model,
                cid,
                sub,
                args.max_chars_per_task,
            ): cid
            for cid, sub in grouped.items()
        }
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                results.append(fut.result())
                print(
                    f"  [cluster {cid:>3d}] {results[-1]['theme_label']!r}  "
                    f"(n={results[-1]['n_tasks']}, occ={results[-1]['n_occupations']}, "
                    f"cover~{results[-1]['n_tasks_covered_estimate']:.2f})"
                )
            except Exception as exc:  # pragma: no cover
                print(f"  [cluster {cid:>3d}] FAILED: {exc!r}")

    results.sort(key=lambda r: r["cluster_id"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote {out_json}")

    rows = []
    for r in results:
        rows.append(
            {
                "cluster_id": r["cluster_id"],
                "n_tasks": r["n_tasks"],
                "n_occupations": r["n_occupations"],
                "importance_mean": round(r["importance_mean"], 3),
                "relevance_mean": round(r["relevance_mean"], 3),
                "theme_label": r["theme_label"],
                "definition": r["definition"],
                "best_theme_task_id": r["best_theme_task"],
                "best_theme_task_text": r["best_theme_task_text"],
                "n_tasks_covered_estimate": round(r["n_tasks_covered_estimate"], 3),
                "minimum_task_contract": r["minimum_task_contract"],
                "inclusion_examples_text": " || ".join(r["inclusion_examples_text"]),
                "exclusion_examples_text": " || ".join(r["exclusion_examples_text"]),
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
