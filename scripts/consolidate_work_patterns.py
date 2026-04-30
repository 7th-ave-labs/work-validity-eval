#!/usr/bin/env python3
"""Consolidate per-cluster theme labels into a curated frozen taxonomy.

Reads ``data/derived/work_pattern_labels_by_cluster__<variant>.json`` (per-
cluster theme labels produced by ``refresh_work_patterns.py``) and rolls
them up to a curated taxonomy in two deterministic passes:

  1. **Bucket pass (embedding-driven).** Embed each cluster's
     ``theme_label``, ``definition``, and ``best_theme_task_text`` with
     ``text-embedding-3-small``. Run agglomerative clustering (cosine,
     average linkage) on the L2-normalised embeddings with
     ``n_clusters == --target``. This gives us exactly K merged groups
     with no LLM refusal-to-collapse behaviour.

  2. **Describe pass (LLM).** For each bucket, send all constituent
     cluster-level themes to ``gpt-5.4-mini`` and ask it to produce the
     final operation-first label text (definition,
     minimum_task_contract, inclusion / exclusion examples, sibling
     labels).

The earlier single-shot and cascading-merge prompts were unable to
collapse the 108 themes below ~80 buckets even with aggressive prompting
- the LLM consistently preserved fine-grained distinctions. Using an
embedding-distance partition for the structural collapse (and LLM only
for naming / describing) avoids that failure mode and gives us a fixed
K that we can target by argument.

Outputs:
  data/derived/work_pattern_labels_refresh__<variant>.json
  data/derived/work_pattern_labels_refresh__<variant>.csv
  data/derived/work_pattern_labels_refresh__<variant>__map.csv

This is an intermediate artefact; promotion to
``data/work_pattern_labels.csv`` is a separate human-in-the-loop edit
via ``finalize_work_patterns.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.cluster import AgglomerativeClustering

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"

DEFAULT_MODEL = os.environ.get("CONSOLIDATE_WORK_PATTERNS_MODEL", "gpt-5.4-mini")
DEFAULT_EMBED_MODEL = os.environ.get(
    "CONSOLIDATE_WORK_PATTERNS_EMBED_MODEL", "text-embedding-3-small"
)

SNAKE_RE = re.compile(r"[^a-z0-9_]+")


def _snake(s: str) -> str:
    s = s.strip().lower().replace(" ", "_")
    return SNAKE_RE.sub("_", s).strip("_") or "unknown"


DESCRIBE_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are finalising a single FROZEN work-pattern label for a taxonomy
    over U.S. O*NET occupation task statements (Job Zones 3-5,
    knowledge / professional / technical work).

    You will receive one bucket: a list of cluster-level theme labels
    that were automatically grouped together by embedding distance on
    their operation-first descriptions. Each cluster has ``theme_label``,
    ``definition``, ``best_theme_task_text``, ``inclusion_examples_text``,
    ``minimum_task_contract``, ``n_tasks``, and ``n_occupations``.

    HARD REQUIREMENT: the frozen label MUST be CROSS-JOB. The operation
    must plausibly apply to workers in many different O*NET occupations,
    not a single profession or a tight professional family. Even if the
    member clusters are dominated by one occupation, label at an
    abstraction level that spans jobs. NEVER emit a label that only one
    profession can satisfy.

      - BAD (job-specific): ``prescribe_treatments_and_medications``
        (physicians / NPs only), ``present_evidence_in_hearings``
        (lawyers only), ``compile_subject_bibliographies`` (librarians
        only), ``facilitate_group_discussions`` (counselors only),
        ``assess_and_grade_work`` (teachers only),
        ``conduct_scientific_research`` (scientists only).
      - GOOD (cross-job): ``advise_and_counsel_people`` (counselors,
        physicians, financial advisors, social workers, HR),
        ``produce_and_maintain_records_and_reports`` (crosses most
        knowledge workers), ``inspect_evaluate_and_verify_compliance``
        (auditors, inspectors, nurses, QA engineers, code inspectors),
        ``develop_and_deliver_instruction`` (teachers, trainers, clinical
        educators, corporate L&D).

    If the members only make sense for one profession, abstract UP so
    the label names the WORK, not the worker (e.g. "prescribe
    medications" -> "recommend and document treatment or intervention
    plans").

    Your job is to produce the final frozen-label record representing
    the OPERATION common to these clusters:

      * label_id: snake_case, stable, operation-first (verb + object +
        optional key constraint). No tools, no occupations, no
        profession-specific jargon, no benchmark / AI terminology.
        Examples: ``supervise_personnel``,
        ``diagnose_and_repair_equipment``, ``analyze_data_for_decisions``,
        ``train_or_instruct_people``, ``manage_budgets_and_resources``.
      * label_name: concise title-case name.
      * definition: 1-2 sentences, job-task-side, neutral across the
        member clusters. Capture the shared operation, not any single
        cluster's idiosyncrasy.
      * minimum_task_contract: ONE sentence specifying what a benchmark
        item MUST expose to legitimately count as this operation. This
        is the load-bearing validity constraint.
      * inclusion_examples: EXACTLY 3 short O*NET-style task strings
        that together span the breadth of the bucket (choose or
        lightly paraphrase from the member clusters).
      * exclusion_examples: EXACTLY 2 short strings that look similar
        but belong under a DIFFERENT frozen label (briefly say why).
      * sibling_labels: list of other likely-frozen ``label_id`` strings
        (snake_case) this label is easy to confuse with.
      * bucket_coherence: one short paragraph describing how well the
        member clusters actually share a single operation, and whether
        any member is a weak fit.

    If the embedding-grouped bucket is clearly incoherent (spans two or
    more unrelated operations), pick the DOMINANT operation for the
    required fields and call it out in ``bucket_coherence``. Do NOT
    invent two labels.

    Return JSON only.
    """
).strip()


def embed_themes(
    client: OpenAI,
    embed_model: str,
    per_cluster: list[dict],
) -> np.ndarray:
    """Embed each cluster's theme+definition+exemplar into a single vector."""
    texts: list[str] = []
    for r in per_cluster:
        parts = [
            r.get("theme_label", ""),
            r.get("definition", ""),
            r.get("best_theme_task_text", ""),
        ]
        texts.append("\n".join(p for p in parts if p))
    # Single batched embedding call (108 items).
    resp = client.embeddings.create(model=embed_model, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    # L2-normalise so cosine == (1 - euclidean_sq / 2) monotonically.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    vecs = vecs / norms
    return vecs


def agglomerate(vecs: np.ndarray, k: int) -> np.ndarray:
    """Return an (n,) array of cluster labels in [0, k)."""
    # Use cosine + average linkage. sklearn needs metric='cosine'.
    model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    return model.fit_predict(vecs)


def describe_pass(
    client: OpenAI,
    model: str,
    bucket_idx: int,
    members: list[dict],
) -> dict:
    user_msg = json.dumps(
        {
            "bucket_index": bucket_idx,
            "members": [
                {
                    "cluster_id": int(m["cluster_id"]),
                    "theme_label": m.get("theme_label", ""),
                    "definition": m.get("definition", ""),
                    "best_theme_task_text": m.get("best_theme_task_text", ""),
                    "inclusion_examples_text": m.get("inclusion_examples_text", []),
                    "minimum_task_contract": m.get("minimum_task_contract", ""),
                    "n_tasks": int(m.get("n_tasks", 0)),
                    "n_occupations": int(m.get("n_occupations", 0)),
                }
                for m in members
            ],
        },
        ensure_ascii=False,
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DESCRIBE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)
    data["label_id"] = _snake(str(data.get("label_id", f"bucket_{bucket_idx}")))
    data.setdefault("label_name", data["label_id"].replace("_", " ").title())
    data.setdefault("definition", "")
    data.setdefault("minimum_task_contract", "")
    data.setdefault("bucket_coherence", "")
    inc = [str(x) for x in (data.get("inclusion_examples") or [])][:3]
    while len(inc) < 3:
        inc.append("")
    exc = [str(x) for x in (data.get("exclusion_examples") or [])][:2]
    while len(exc) < 2:
        exc.append("")
    data["inclusion_examples"] = inc
    data["exclusion_examples"] = exc
    data["sibling_labels"] = [_snake(str(x)) for x in (data.get("sibling_labels") or [])]
    return data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="task_only", choices=("task_only", "title_plus_task", "operation_only"))
    p.add_argument("--in-json", default=None)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--out-map", default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--target", type=int, default=28, help="exact number of frozen labels")
    args = p.parse_args()

    suffix = f"__{args.variant}"
    in_json = Path(args.in_json or DERIVED / f"work_pattern_labels_by_cluster{suffix}.json")
    out_json = Path(args.out_json or DERIVED / f"work_pattern_labels_refresh{suffix}.json")
    out_csv = Path(args.out_csv or DERIVED / f"work_pattern_labels_refresh{suffix}.csv")
    out_map = Path(args.out_map or DERIVED / f"work_pattern_labels_refresh{suffix}__map.csv")

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY / OPENAI_API missing")

    per_cluster = json.loads(in_json.read_text())
    print(
        f"Input: {len(per_cluster)} cluster-level themes from {in_json}; "
        f"target exactly {args.target} frozen labels."
    )
    if len(per_cluster) < args.target:
        raise SystemExit(
            f"target={args.target} > n_clusters={len(per_cluster)}; nothing to collapse"
        )

    client = OpenAI(api_key=api_key)

    print("Pass 1: embed + agglomerative-cluster themes")
    vecs = embed_themes(client, args.embed_model, per_cluster)
    labels_arr = agglomerate(vecs, args.target)

    by_cluster = {int(r["cluster_id"]): r for r in per_cluster}
    buckets: dict[int, list[dict]] = {}
    cluster_to_bucket_idx: dict[int, int] = {}
    for r, lab in zip(per_cluster, labels_arr):
        cid = int(r["cluster_id"])
        bucket_idx = int(lab)
        buckets.setdefault(bucket_idx, []).append(r)
        cluster_to_bucket_idx[cid] = bucket_idx

    # Sort buckets by rolled-up task count (largest first).
    ordered = sorted(
        buckets.items(),
        key=lambda kv: -sum(int(m.get("n_tasks", 0)) for m in kv[1]),
    )
    print(f"Pass 2: describe {len(ordered)} buckets with {args.model}")

    labels: list[dict] = []
    bucket_idx_to_label_id: dict[int, str] = {}
    for rank, (bucket_idx, members) in enumerate(ordered):
        lab = describe_pass(client, args.model, bucket_idx, members)
        # Guarantee label_id uniqueness.
        base = lab["label_id"]
        cand = base
        i = 2
        while cand in bucket_idx_to_label_id.values():
            cand = f"{base}_{i}"
            i += 1
        lab["label_id"] = cand
        lab["source_cluster_ids"] = sorted(int(m["cluster_id"]) for m in members)
        lab["n_source_clusters"] = len(members)
        lab["n_tasks_rollup"] = int(sum(int(m.get("n_tasks", 0)) for m in members))
        lab["n_occupations_max"] = int(max(int(m.get("n_occupations", 0)) for m in members))
        lab["member_theme_labels"] = [m.get("theme_label", "") for m in members]
        labels.append(lab)
        bucket_idx_to_label_id[bucket_idx] = lab["label_id"]
        print(
            f"  [{rank:>2d}] {lab['label_id']:<40s}  "
            f"n_clusters={lab['n_source_clusters']:>2d}  "
            f"n_tasks={lab['n_tasks_rollup']:>4d}"
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(labels, indent=2, ensure_ascii=False))

    rows = []
    for lab in labels:
        inc = lab.get("inclusion_examples") or []
        exc = lab.get("exclusion_examples") or []
        sib = lab.get("sibling_labels") or []
        rows.append(
            {
                "label_id": lab["label_id"],
                "label_name": lab.get("label_name", ""),
                "definition": lab.get("definition", ""),
                "minimum_task_contract": lab.get("minimum_task_contract", ""),
                "inclusion_examples": " || ".join(inc),
                "exclusion_examples": " || ".join(exc),
                "sibling_labels": ",".join(sib),
                "n_source_clusters": lab["n_source_clusters"],
                "n_tasks_rollup": lab["n_tasks_rollup"],
                "n_occupations_max": lab["n_occupations_max"],
                "source_cluster_ids": ",".join(str(c) for c in lab["source_cluster_ids"]),
                "member_theme_labels": " || ".join(lab["member_theme_labels"]),
                "bucket_coherence": lab.get("bucket_coherence", ""),
            }
        )
    map_rows = []
    for cid, bucket_idx in cluster_to_bucket_idx.items():
        lab_id = bucket_idx_to_label_id[bucket_idx]
        cluster_theme = by_cluster[cid].get("theme_label", "")
        map_rows.append(
            {
                "cluster_id": cid,
                "cluster_theme_label": cluster_theme,
                "refreshed_label_id": lab_id,
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    pd.DataFrame(map_rows).sort_values("cluster_id").to_csv(out_map, index=False)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_map} ({len(map_rows)} cluster -> label rows)")
    print(f"Final: {len(labels)} frozen labels covering {len(map_rows)} clusters.")


if __name__ == "__main__":
    main()
