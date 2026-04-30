#!/usr/bin/env python3
"""Build a 2-level operation hierarchy from per-cluster themes.

Unlike ``consolidate_work_patterns.py``, this script does NOT prespecify
a target label count. Instead it applies a cosine-distance threshold to
the agglomerative merge of cluster-theme embeddings and lets the number
of top-level operations emerge from the data.

Each bucket (top-level operation) is then described by ``gpt-5.4-mini``
as an agent-doable + evaluable operation, and the constituent cluster
themes are preserved as sub-operations under it.

Inputs:
  data/derived/work_pattern_labels_by_cluster__<variant>.json
  data/derived/onet_task_clusters__<variant>.parquet (optional; used for
      task-level coverage rollup under each sub-op)

Outputs:
  data/derived/operation_hierarchy__<variant>__t<threshold>.json
  data/derived/operation_hierarchy__<variant>__t<threshold>.csv     (flat)
  data/derived/operation_hierarchy__<variant>__t<threshold>__tree.txt

The tree.txt file is a human-readable print of the hierarchy for quick
review.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.cluster import AgglomerativeClustering

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"

DEFAULT_MODEL = os.environ.get("OPERATION_HIERARCHY_MODEL", "gpt-5.4-mini")
DEFAULT_EMBED_MODEL = os.environ.get(
    "OPERATION_HIERARCHY_EMBED_MODEL", "text-embedding-3-small"
)

SNAKE_RE = re.compile(r"[^a-z0-9_]+")


def _snake(s: str) -> str:
    s = (s or "").strip().lower().replace(" ", "_")
    return SNAKE_RE.sub("_", s).strip("_") or "unknown"


DESCRIBE_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are naming a TOP-LEVEL OPERATION in a taxonomy of knowledge work
    derived from U.S. O*NET occupational task statements (Job Zones
    3-5). The candidate was formed bottom-up by embedding-distance
    agglomeration over per-cluster operation themes. Each theme is
    already an operation-first, domain-stripped phrase (verb + generic
    object).

    HARD REQUIREMENT: the top-level operation MUST be CROSS-JOB. That
    is, the operation must plausibly be performed by workers in many
    different O*NET occupations, not just one profession or a tight
    family of professions. Even if the member sub-ops are dominated by
    a single occupation, you must label at a level of abstraction that
    spans jobs. NEVER emit an operation that only one profession can
    do. Concretely:

      - BAD (job-specific): "prescribe_treatments_and_medications"
        (physicians / nurse practitioners only),
        "present_evidence_in_hearings" (lawyers / judges only),
        "compile_subject_bibliographies" (librarians only),
        "facilitate_group_discussions" (counselors only),
        "assess_and_grade_work" (teachers only),
        "conduct_scientific_research" (scientists only).
      - GOOD (cross-job): "advise_and_counsel_people" (covers
        counselors, physicians, financial advisors, social workers,
        HR), "produce_and_maintain_records_and_reports" (crosses most
        knowledge workers), "inspect_evaluate_and_verify_compliance"
        (auditors, inspectors, nurses, QA engineers, code inspectors),
        "develop_and_deliver_instruction" (teachers, trainers,
        clinical educators, corporate L&D), "analyze_and_interpret_
        data_for_decisions" (analysts, physicians, researchers,
        managers).

    Abstraction rule of thumb: the operation should apply to workers
    across AT LEAST FIVE distinct O*NET occupations at job-zone 3-5
    level. If the members only make sense for one profession, abstract
    UP (e.g. "prescribe medications" -> "recommend and document
    treatment or intervention plans", which covers physicians,
    dietitians, therapists, social workers, case managers).

    For this bucket you will receive its member cluster themes and
    the list of occupations represented across them. Your job is to
    (a) decide whether the bucket names a SINGLE agent-doable +
    evaluable + CROSS-JOB operation, and (b) produce the label record
    for it.

    "Agent-doable" = an AI agent equipped with standard tools (code
    execution, retrieval, file I/O, a browser, APIs, and text / image
    I/O) could plausibly execute the operation end-to-end, not just
    advise about it. "Evaluable" = success of that execution can be
    checked against a concrete artifact, rubric, ground-truth
    comparison, or expert judgement, not only by self-report.

    Produce JSON with:

      operation_id: snake_case, stable, operation-first (verb + generic
        object + optional key constraint). No industry / occupation /
        product / benchmark terms, and no single-profession framing.
        Examples:
        "produce_and_maintain_records_and_reports",
        "install_maintain_and_repair_equipment",
        "advise_and_counsel_people",
        "supervise_and_coordinate_work".
      operation_name: concise title-case name.
      definition: 1-2 sentences, operation-side (what the WORKER does),
        neutral across the member sub-ops. State the shared operation,
        not any member's idiosyncrasy.
      agent_doability: one of {"high", "medium", "low"}. A rough
        assessment of how much of this operation an LLM-agent with
        standard tooling can credibly execute today.
      agent_doability_rationale: one short sentence justifying the
        rating, citing the operation's main bottleneck (e.g. "requires
        physical manipulation", "purely text artifacts", "needs real
        stakeholder interviews", "requires regulated professional
        licensure").
      evaluation_artifact: one short phrase naming the concrete output
        that would be graded (e.g. "written report", "repair work-order
        log", "lesson plan document", "treatment plan with rationale",
        "calibration record").
      evaluation_method: one short phrase, one of
        {"rubric_on_artifact", "unit_tests_or_execution",
         "ground_truth_comparison", "expert_review",
         "simulation_outcome", "other"}; pick the most natural primary.
      minimum_task_contract: ONE sentence stating what a benchmark item
        for this operation MUST expose to legitimately count. This is
        the load-bearing validity constraint.
      bucket_coherence: one sentence naming the weakest-fit member (if
        any) and whether the bucket really represents one operation or
        a merger that should be split in a later pass.
      split_suggestion: list of {"operation_id", "member_cluster_ids"}
        (possibly empty). Use this ONLY if the bucket is genuinely two
        different operations stapled together; otherwise return [].

    Return JSON only.
    """
).strip()


def _load_cluster_occupations(
    clusters_parquet: str | None, variant: str
) -> dict[int, set[str]]:
    """Return {cluster_id: {onetsoc_code, ...}} from the clusters parquet.

    Uses ``hdbscan_cluster_assigned`` when available (soft-assigned noise is
    included so the pooled occupation count reflects all tasks routed to the
    sub-op cluster), else falls back to ``hdbscan_cluster``. Returns an empty
    mapping if the parquet cannot be loaded; downstream callers treat an
    empty mapping as "skip the cross-job absorption pass".
    """
    path = Path(
        clusters_parquet or DERIVED / f"onet_task_clusters__{variant}.parquet"
    )
    if not path.exists():
        print(
            f"[cross-job] {path} not found; skipping pooled-occupation "
            "computation (absorption pass will no-op)"
        )
        return {}
    df = pd.read_parquet(path)
    col = (
        "hdbscan_cluster_assigned"
        if "hdbscan_cluster_assigned" in df.columns
        else "hdbscan_cluster"
    )
    if "onetsoc_code" not in df.columns:
        print(
            "[cross-job] onetsoc_code column missing from clusters parquet; "
            "skipping pooled-occupation computation"
        )
        return {}
    out: dict[int, set[str]] = {}
    for cid, sub in df[[col, "onetsoc_code"]].groupby(col):
        cid_int = int(cid)
        if cid_int == -1:
            continue
        out[cid_int] = set(sub["onetsoc_code"].astype(str).tolist())
    print(
        f"[cross-job] loaded occupation sets for {len(out)} sub-op clusters "
        f"from {path}"
    )
    return out


def _bucket_occupation_sets(
    buckets: dict[int, list[dict]],
    cluster_occupations: dict[int, set[str]],
) -> dict[int, set[str]]:
    """Union occupation sets across all sub-op clusters in each bucket."""
    out: dict[int, set[str]] = {}
    for bid, members in buckets.items():
        pooled: set[str] = set()
        for m in members:
            cid = int(m["cluster_id"])
            pooled |= cluster_occupations.get(cid, set())
        out[bid] = pooled
    return out


def _log_bucket_occupation_counts(
    buckets: dict[int, list[dict]],
    bucket_occupations: dict[int, set[str]],
    label: str,
) -> None:
    counts = [len(bucket_occupations.get(bid, set())) for bid in buckets]
    sub_counts = [len(members) for members in buckets.values()]
    if not counts:
        return
    print(
        f"[cross-job] {label}: n_buckets={len(buckets)} "
        f"pooled_occs min/median/max={min(counts)}/"
        f"{sorted(counts)[len(counts)//2]}/{max(counts)} "
        f"sub_ops min/median/max={min(sub_counts)}/"
        f"{sorted(sub_counts)[len(sub_counts)//2]}/{max(sub_counts)}"
    )


def _bucket_centroid(vecs: np.ndarray, idx: list[int]) -> np.ndarray:
    v = vecs[idx].mean(axis=0)
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else v / n


def _absorb_job_specific_buckets(
    *,
    buckets: dict[int, list[dict]],
    bucket_vec_idx: dict[int, list[int]],
    bucket_occupations: dict[int, set[str]],
    vecs: np.ndarray,
    min_pooled_occupations: int,
    min_sub_ops: int,
) -> tuple[dict[int, list[dict]], dict[int, list[int]], dict[int, set[str]]]:
    """Iteratively merge each job-specific bucket into its nearest neighbour.

    A bucket is considered job-specific when BOTH
      (a) ``len(pooled_occupations) < min_pooled_occupations``, and
      (b) ``n_sub_ops < min_sub_ops`` OR the pooled occupation count is
          below ``2 * min_pooled_occupations``.

    A lone sub-op bucket with pooled occupations >= ``2 *
    min_pooled_occupations`` is allowed to stand on its own (it is already
    clearly cross-occupational, just fine-grained).

    If occupation sets are unavailable (e.g. clusters parquet missing), the
    function only applies the ``min_sub_ops`` criterion.
    """
    if not buckets:
        return buckets, bucket_vec_idx, bucket_occupations

    have_occs = any(len(s) > 0 for s in bucket_occupations.values())

    def is_job_specific(bid: int) -> bool:
        n_sub_ops = len(buckets[bid])
        n_occs = len(bucket_occupations.get(bid, set()))
        if have_occs:
            too_narrow_occs = n_occs < min_pooled_occupations
            too_few_sub_ops = n_sub_ops < min_sub_ops
            clearly_cross_job = n_occs >= 2 * max(1, min_pooled_occupations)
            if too_narrow_occs:
                return True
            if too_few_sub_ops and not clearly_cross_job:
                return True
            return False
        # No occupation signal available: only enforce min_sub_ops.
        return n_sub_ops < min_sub_ops

    iteration = 0
    while True:
        iteration += 1
        bad = [bid for bid in buckets if is_job_specific(bid)]
        if not bad:
            break
        if len(buckets) <= 1:
            break

        # Merge the "worst" job-specific bucket first: fewest pooled occs,
        # then fewest sub-ops, then smallest cluster_id for determinism.
        def _rank(bid: int) -> tuple[int, int, int]:
            return (
                len(bucket_occupations.get(bid, set())),
                len(buckets[bid]),
                bid,
            )

        bad.sort(key=_rank)
        bid = bad[0]

        # Find the nearest OTHER bucket by centroid cosine distance.
        src_centroid = _bucket_centroid(vecs, bucket_vec_idx[bid])
        best_dst = None
        best_dist = float("inf")
        for other, idx in bucket_vec_idx.items():
            if other == bid:
                continue
            dst_centroid = _bucket_centroid(vecs, idx)
            dist = float(1.0 - np.dot(src_centroid, dst_centroid))
            if dist < best_dist:
                best_dist = dist
                best_dst = other
        if best_dst is None:
            break

        print(
            f"[cross-job] absorb (iter {iteration}): bucket {bid} "
            f"(n_sub_ops={len(buckets[bid])}, "
            f"n_occs={len(bucket_occupations.get(bid, set()))}) -> "
            f"bucket {best_dst} "
            f"(n_sub_ops={len(buckets[best_dst])}, "
            f"n_occs={len(bucket_occupations.get(best_dst, set()))}) "
            f"at cosine distance {best_dist:.3f}"
        )
        buckets[best_dst].extend(buckets[bid])
        bucket_vec_idx[best_dst].extend(bucket_vec_idx[bid])
        bucket_occupations[best_dst] = bucket_occupations.get(
            best_dst, set()
        ) | bucket_occupations.get(bid, set())
        del buckets[bid]
        del bucket_vec_idx[bid]
        if bid in bucket_occupations:
            del bucket_occupations[bid]

        # Safety: bound the number of iterations to the starting bucket count
        # (each iteration strictly reduces len(buckets)).
        if iteration > 10_000:
            print("[cross-job] absorption loop bounded at 10000 iterations")
            break

    return buckets, bucket_vec_idx, bucket_occupations


def embed_themes(
    client: OpenAI, embed_model: str, per_cluster: list[dict]
) -> np.ndarray:
    texts = []
    for r in per_cluster:
        parts = [
            r.get("theme_label", ""),
            r.get("definition", ""),
            r.get("best_theme_task_text", ""),
        ]
        texts.append("\n".join(p for p in parts if p))
    resp = client.embeddings.create(model=embed_model, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms


def describe_bucket(
    client: OpenAI,
    model: str,
    bucket_idx: int,
    members: list[dict],
    occupations: list[str] | None = None,
) -> dict:
    user_msg = json.dumps(
        {
            "bucket_index": bucket_idx,
            "pooled_occupations": list(occupations or []),
            "n_pooled_occupations": len(occupations or []),
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
                    "top_occupations": list(m.get("top_occupations", []))[:10],
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
    data["operation_id"] = _snake(str(data.get("operation_id", f"bucket_{bucket_idx}")))
    data.setdefault(
        "operation_name", data["operation_id"].replace("_", " ").title()
    )
    data.setdefault("definition", "")
    data.setdefault("agent_doability", "medium")
    data.setdefault("agent_doability_rationale", "")
    data.setdefault("evaluation_artifact", "")
    data.setdefault("evaluation_method", "other")
    data.setdefault("minimum_task_contract", "")
    data.setdefault("bucket_coherence", "")
    data.setdefault("split_suggestion", [])
    return data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant",
        default="operation_only",
        choices=("task_only", "title_plus_task", "operation_only"),
    )
    p.add_argument("--in-json", default=None)
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument(
        "--tag-suffix",
        default="",
        help="Optional suffix appended to the variant tag for output filenames "
        "(e.g. '_r2' yields 'operation_only_r2__t0p55'). Leave empty to use the "
        "variant name alone.",
    )
    p.add_argument(
        "--clusters-parquet",
        default=None,
        help="Path to onet_task_clusters__<variant>.parquet used to compute the "
        "pooled-occupation count per bucket for the cross-job absorption pass. "
        "Defaults to data/derived/onet_task_clusters__<variant>.parquet.",
    )
    p.add_argument(
        "--min-pooled-occupations",
        type=int,
        default=10,
        help="Cross-job absorption: after agglomerative merge, any bucket whose "
        "pooled (unique) O*NET occupations across its member clusters falls below "
        "this threshold is iteratively merged into its nearest neighbour bucket "
        "(by theme-centroid cosine distance). Use --min-pooled-occupations 0 to "
        "disable.",
    )
    p.add_argument(
        "--min-sub-ops",
        type=int,
        default=2,
        help="Cross-job absorption: also merge any bucket with fewer sub-ops than "
        "this into its nearest neighbour, unless its pooled occupations already "
        "clearly exceed --min-pooled-occupations by a factor of >=2 (in which "
        "case a single cross-occupational sub-op is allowed to stand alone).",
    )
    args = p.parse_args()

    base_suffix = f"__{args.variant}{args.tag_suffix}"
    read_suffix = f"__{args.variant}"
    in_json = Path(
        args.in_json or DERIVED / f"work_pattern_labels_by_cluster{read_suffix}.json"
    )
    tag = f"{base_suffix}__t{args.threshold:.2f}".replace(".", "p", 1)
    out_json = DERIVED / f"operation_hierarchy{tag}.json"
    out_csv = DERIVED / f"operation_hierarchy{tag}.csv"
    out_tree = DERIVED / f"operation_hierarchy{tag}__tree.txt"

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY / OPENAI_API missing")

    per_cluster = json.loads(in_json.read_text())
    print(f"Input: {len(per_cluster)} cluster themes from {in_json}")
    client = OpenAI(api_key=api_key)

    # --- embed + threshold merge -----------------------------------
    cache = DERIVED / f"_tmp_theme_vecs{base_suffix}.npy"
    if cache.exists():
        vecs = np.load(cache)
        if len(vecs) != len(per_cluster):
            vecs = embed_themes(client, args.embed_model, per_cluster)
            np.save(cache, vecs)
    else:
        vecs = embed_themes(client, args.embed_model, per_cluster)
        np.save(cache, vecs)

    labels_arr = AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        distance_threshold=args.threshold,
        n_clusters=None,
    ).fit_predict(vecs)
    n_buckets = int(labels_arr.max()) + 1
    print(
        f"Threshold {args.threshold:.2f} -> {n_buckets} buckets "
        f"({(labels_arr == -1).sum()} unassigned)"
    )

    buckets: dict[int, list[dict]] = {}
    bucket_vec_idx: dict[int, list[int]] = {}
    for i, (r, lab) in enumerate(zip(per_cluster, labels_arr)):
        buckets.setdefault(int(lab), []).append(r)
        bucket_vec_idx.setdefault(int(lab), []).append(i)

    # --- load per-cluster occupation sets for the cross-job check --
    cluster_occupations = _load_cluster_occupations(
        args.clusters_parquet, args.variant
    )
    bucket_occupations = _bucket_occupation_sets(buckets, cluster_occupations)
    _log_bucket_occupation_counts(buckets, bucket_occupations, label="pre-absorb")

    # --- cross-job absorption pass ---------------------------------
    # Iteratively merge any bucket that is job-specific (too few pooled
    # occupations or too few sub-ops) into its nearest neighbour bucket by
    # theme-centroid cosine distance. Runs BEFORE the LLM describes the
    # bucket, so the LLM never sees a single-occupation bucket and cannot
    # emit a job-specific top-level operation.
    if args.min_pooled_occupations > 0 or args.min_sub_ops > 1:
        buckets, bucket_vec_idx, bucket_occupations = _absorb_job_specific_buckets(
            buckets=buckets,
            bucket_vec_idx=bucket_vec_idx,
            bucket_occupations=bucket_occupations,
            vecs=vecs,
            min_pooled_occupations=args.min_pooled_occupations,
            min_sub_ops=args.min_sub_ops,
        )
        _log_bucket_occupation_counts(
            buckets, bucket_occupations, label="post-absorb"
        )

    # --- LLM describe each bucket in parallel ----------------------
    descriptions: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                describe_bucket,
                client,
                args.model,
                bid,
                members,
                sorted(bucket_occupations.get(bid, set())),
            ): bid
            for bid, members in buckets.items()
        }
        for fut in as_completed(futures):
            bid = futures[fut]
            try:
                descriptions[bid] = fut.result()
            except Exception as e:
                print(f"[warn] bucket {bid} describe failed: {e}")
                descriptions[bid] = {
                    "operation_id": f"bucket_{bid}",
                    "operation_name": f"Bucket {bid}",
                    "definition": "",
                    "agent_doability": "medium",
                    "agent_doability_rationale": str(e),
                    "evaluation_artifact": "",
                    "evaluation_method": "other",
                    "minimum_task_contract": "",
                    "bucket_coherence": "",
                    "split_suggestion": [],
                }

    # --- assemble hierarchy ----------------------------------------
    hierarchy = []
    flat_rows = []
    for bid, members in sorted(
        buckets.items(), key=lambda kv: -sum(m["n_tasks"] for m in kv[1])
    ):
        desc = descriptions[bid]
        n_tasks = sum(int(m["n_tasks"]) for m in members)
        n_occs_max = max(int(m["n_occupations"]) for m in members)
        n_occs_sum_dedup = None  # fill later if we load task-clusters parquet
        sub_ops = []
        for m in sorted(members, key=lambda x: -int(x["n_tasks"])):
            sub_ops.append(
                {
                    "cluster_id": int(m["cluster_id"]),
                    "theme_label": m.get("theme_label", ""),
                    "definition": m.get("definition", ""),
                    "n_tasks": int(m["n_tasks"]),
                    "n_occupations": int(m["n_occupations"]),
                    "best_theme_task_text": m.get("best_theme_task_text", ""),
                    "minimum_task_contract": m.get("minimum_task_contract", ""),
                }
            )
            flat_rows.append(
                {
                    "operation_id": desc["operation_id"],
                    "operation_name": desc["operation_name"],
                    "agent_doability": desc["agent_doability"],
                    "evaluation_method": desc["evaluation_method"],
                    "cluster_id": int(m["cluster_id"]),
                    "sub_op_label": m.get("theme_label", ""),
                    "sub_op_definition": m.get("definition", ""),
                    "sub_op_n_tasks": int(m["n_tasks"]),
                    "sub_op_n_occupations": int(m["n_occupations"]),
                }
            )
        pooled_occs = sorted(bucket_occupations.get(bid, set()))
        hierarchy.append(
            {
                "bucket_id": bid,
                "operation_id": desc["operation_id"],
                "operation_name": desc["operation_name"],
                "definition": desc["definition"],
                "agent_doability": desc["agent_doability"],
                "agent_doability_rationale": desc["agent_doability_rationale"],
                "evaluation_artifact": desc["evaluation_artifact"],
                "evaluation_method": desc["evaluation_method"],
                "minimum_task_contract": desc["minimum_task_contract"],
                "bucket_coherence": desc["bucket_coherence"],
                "split_suggestion": desc["split_suggestion"],
                "n_sub_ops": len(sub_ops),
                "n_tasks": n_tasks,
                "max_occupations_in_any_sub_op": n_occs_max,
                "n_pooled_occupations": len(pooled_occs),
                "pooled_occupations": pooled_occs,
                "sub_operations": sub_ops,
            }
        )

    out_json.write_text(json.dumps(hierarchy, indent=2, ensure_ascii=False))
    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)

    # --- human-readable tree --------------------------------------
    lines: list[str] = []
    lines.append(f"Threshold: {args.threshold:.2f}")
    lines.append(f"Top-level operations: {len(hierarchy)}")
    lines.append(f"Sub-operations (= input clusters): {len(per_cluster)}")
    lines.append("=" * 78)
    for h in hierarchy:
        lines.append("")
        lines.append(
            f"[{h['agent_doability']:>6s}] {h['operation_name']}  "
            f"({h['n_tasks']} tasks, {h['n_sub_ops']} sub-ops, "
            f"{h.get('n_pooled_occupations', 0)} occs, "
            f"eval: {h['evaluation_method']})"
        )
        lines.append(f"    id: {h['operation_id']}")
        if h.get("definition"):
            lines.append("    def: " + h["definition"])
        if h.get("minimum_task_contract"):
            lines.append("    contract: " + h["minimum_task_contract"])
        if h.get("bucket_coherence"):
            lines.append("    coherence: " + h["bucket_coherence"])
        for s in h["sub_operations"]:
            lines.append(
                f"      - {s['theme_label']:<55s}  "
                f"({s['n_tasks']:>4d} tasks / {s['n_occupations']:>3d} occs)"
            )
    out_tree.write_text("\n".join(lines))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_tree}")


if __name__ == "__main__":
    main()
