#!/usr/bin/env python3
"""Map scoped ESCO items to the frozen 18-op taxonomy.

Produces two independent op assignments per ESCO item, plus a consensus
label and a disagreement inventory:

1. **Centroid assignment** (``method = "centroid"``)
   Cosine similarity of the ESCO item embedding to each op's centroid in
   the O*NET ``task_only`` embedding space. The centroid for op ``o`` is
   the mean of L2-normalised O*NET task embeddings whose cluster is a
   member of ``o`` (per
   ``data/derived/manual_labeling/operation_taxonomy.json``, with the
   ``assistance -> protocol_execution`` merge applied).

2. **LLM assignment** (``method = "llm"``)
   Single-shot call to ``gpt-5.4-mini`` with the frozen op definitions
   (``one_line``, ``input``, ``output``, ``distinct_from``) plus an
   explicit ``none_of_the_above`` option, returning ``op_id``,
   ``confidence in {high, medium, low}``, and a 1-sentence rationale.

Both methods see the same input string (default: ``preferredLabel``).
Consensus rule: if the two agree on ``op_id`` (ignoring the LLM's
``none_of_the_above`` branch), that is the final label; otherwise the
item goes on the disagreement review list and carries a
``consensus_op = null`` with ``review_reason`` populated.

Cached per-item LLM outputs are written append-only to a JSONL so the
pass is resumable.

Inputs (defaults):
  data/derived/esco/esco_items_scoped.parquet
  data/derived/esco/embeddings/esco_item_embeddings__label_only.npz
  data/derived/esco/embeddings/esco_item_embeddings__label_only_meta.parquet
  data/derived/embeddings/onet_task_embeddings__task_only.npz
  data/derived/onet_task_clusters__task_only.parquet
  data/derived/manual_labeling/operation_taxonomy.json

Outputs:
  data/derived/esco/esco_op_assignments.parquet
  data/derived/esco/esco_op_assignments.csv
  data/derived/esco/esco_disagreement_cases.csv
  data/derived/esco/op_centroid_cosine_reference.json  (per-op centroid
      self-similarity stats used to derive the ``unmappable_geometric``
      threshold)
  data/derived/esco/esco_llm_cache.jsonl  (append-only cache)
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived"
ESCO_DERIVED = DERIVED / "esco"

DEFAULT_MODEL = os.environ.get("ESCO_MAP_MODEL", "gpt-5.4-mini")

# Post-clustering merge applied in the paper (``assistance`` held a single
# cluster whose member tasks are clinical / surgical support procedures,
# folded into ``protocol_execution``).
MERGE_TOP_LEVEL: dict[str, str] = {"assistance": "protocol_execution"}


SYSTEM_PROMPT = textwrap.dedent(
    """
    You assign a single ESCO skill/competence item to at most ONE operation in
    a frozen 18-operation taxonomy of U.S. knowledge work (derived from O*NET
    task statements, Job Zones 3-5). Input is the ESCO item's preferred
    English label (and optionally its description). Output is the operation
    the item most directly realises.

    The frozen taxonomy (you MUST use exactly these ``op_id`` values, or the
    literal string ``none_of_the_above``):

    {ops_block}

    Decision rules.

      * Pick the operation whose ``one_line`` + ``input`` + ``output``
        best matches the ESCO item. Use ``distinct_from`` to separate
        adjacent operations.
      * An ESCO item that is itself not a work operation (e.g. a
        knowledge domain, a tool or instrument, a personal trait) must
        be returned as ``none_of_the_above`` rather than forced into an
        op.
      * An ESCO item that combines operations (e.g. "analyse X AND
        report findings") is assigned to its PRIMARY operation; the
        rationale must note the secondary.

    Return a strict JSON object with these keys:

      ``op_id``: one of the 18 op ids OR ``none_of_the_above``.
      ``confidence``: one of ``high``, ``medium``, ``low``.
      ``rationale``: one sentence, at most 30 words.

    No other keys. No markdown.
    """
).strip()


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def load_taxonomy() -> tuple[list[dict], dict[str, list[int]]]:
    tax = json.loads((DERIVED / "manual_labeling" / "operation_taxonomy.json").read_text())
    ops_raw = tax["top_level_operations"]
    # Apply merge: combine member_clusters from merged ops.
    merged: dict[str, dict] = {}
    for o in ops_raw:
        tgt_id = MERGE_TOP_LEVEL.get(o["operation_id"], o["operation_id"])
        if tgt_id not in merged:
            merged[tgt_id] = {
                "operation_id": tgt_id,
                "one_line": o.get("one_line", ""),
                "input": o.get("input", ""),
                "output": o.get("output", ""),
                "distinct_from": dict(o.get("distinct_from", {})),
                "member_clusters": list(o.get("member_clusters", [])),
            }
        else:
            merged[tgt_id]["member_clusters"].extend(o.get("member_clusters", []))
    # When the source op (assistance) is merged into the target, keep
    # the target's one_line/input/output (protocol_execution's); the merge
    # is a structural fold, not a semantic redefinition.
    ops = list(merged.values())
    op_clusters = {o["operation_id"]: o["member_clusters"] for o in ops}
    return ops, op_clusters


def compute_op_centroids(
    ops: list[dict],
    op_clusters: dict[str, list[int]],
    onet_emb: np.ndarray,
    onet_clusters: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Weighted-mean L2-normalised centroid per op in the O*NET embedding space."""

    onet_emb_n = l2_normalize(onet_emb.astype(np.float32))
    # Align by row_index (stored with the cluster parquet).
    onet_clusters = onet_clusters.sort_values("row_index").reset_index(drop=True)
    cluster_assign = onet_clusters["hdbscan_cluster_assigned"].to_numpy()

    op_ids = [o["operation_id"] for o in ops]
    centroids = np.zeros((len(op_ids), onet_emb_n.shape[1]), dtype=np.float32)
    for i, op_id in enumerate(op_ids):
        members = np.isin(cluster_assign, op_clusters[op_id])
        if not members.any():
            raise SystemExit(f"no O*NET tasks for op {op_id}")
        centroids[i] = onet_emb_n[members].mean(axis=0)
    # Re-normalise so cosine reduces to dot product.
    centroids = l2_normalize(centroids)
    return centroids, op_ids


def centroid_reference_stats(
    centroids: np.ndarray, onet_emb: np.ndarray, onet_clusters: pd.DataFrame,
    op_ids: list[str], op_clusters: dict[str, list[int]],
) -> dict[str, dict]:
    """For each op, report cosine-to-own-centroid percentiles over its
    member O*NET tasks.  Used to derive the ``unmappable_geometric``
    threshold: an ESCO item whose top-1 cosine is below the 5th percentile
    of within-op member cosines is flagged as unmappable by the geometric
    test.
    """

    onet_emb_n = l2_normalize(onet_emb.astype(np.float32))
    onet_clusters = onet_clusters.sort_values("row_index").reset_index(drop=True)
    cluster_assign = onet_clusters["hdbscan_cluster_assigned"].to_numpy()
    stats: dict[str, dict] = {}
    for i, op_id in enumerate(op_ids):
        members = np.isin(cluster_assign, op_clusters[op_id])
        cos = onet_emb_n[members] @ centroids[i]
        stats[op_id] = {
            "n_onet_tasks": int(members.sum()),
            "centroid_cosine_p05": float(np.percentile(cos, 5)),
            "centroid_cosine_p50": float(np.percentile(cos, 50)),
            "centroid_cosine_p95": float(np.percentile(cos, 95)),
        }
    return stats


# ---- LLM classification ---------------------------------------------------


def _format_ops_block(ops: list[dict]) -> str:
    lines = []
    for o in ops:
        lines.append(f"- ``{o['operation_id']}``")
        lines.append(f"    one_line: {o.get('one_line','').strip()}")
        lines.append(f"    input:    {o.get('input','').strip()}")
        lines.append(f"    output:   {o.get('output','').strip()}")
        df = o.get("distinct_from") or {}
        for k, v in list(df.items())[:3]:
            lines.append(f"    distinct_from.{k}: {v.strip()}")
    return "\n".join(lines)


def load_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if "esco_uri" in rec:
                        cache[rec["esco_uri"]] = rec
                except Exception:
                    continue
    return cache


def llm_classify_one(client: OpenAI, model: str, system_prompt: str, esco_uri: str,
                     item_text: str, valid_ids: set[str]) -> dict:
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": item_text},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            text = resp.choices[0].message.content
            data = json.loads(text)
            op_id = str(data.get("op_id") or "").strip()
            if op_id not in valid_ids and op_id != "none_of_the_above":
                op_id = "none_of_the_above"
            confidence = str(data.get("confidence") or "low").strip().lower()
            if confidence not in {"high", "medium", "low"}:
                confidence = "low"
            rationale = str(data.get("rationale") or "").strip()
            return {
                "esco_uri": esco_uri,
                "llm_op_id": op_id,
                "llm_confidence": confidence,
                "llm_rationale": rationale,
            }
        except Exception as exc:  # pragma: no cover
            if attempt == 3:
                return {
                    "esco_uri": esco_uri,
                    "llm_op_id": "none_of_the_above",
                    "llm_confidence": "low",
                    "llm_rationale": f"llm_error:{exc!r}",
                }
            continue
    # Unreachable but keep types happy.
    return {
        "esco_uri": esco_uri,
        "llm_op_id": "none_of_the_above",
        "llm_confidence": "low",
        "llm_rationale": "llm_error:exhausted_retries",
    }


# ---- Main -----------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--esco-parquet", default=str(ESCO_DERIVED / "esco_items_scoped.parquet"))
    p.add_argument(
        "--esco-emb-npz",
        default=str(ESCO_DERIVED / "embeddings" / "esco_item_embeddings__label_only.npz"),
    )
    p.add_argument(
        "--esco-emb-meta",
        default=str(ESCO_DERIVED / "embeddings" / "esco_item_embeddings__label_only_meta.parquet"),
    )
    p.add_argument(
        "--onet-emb-npz",
        default=str(DERIVED / "embeddings" / "onet_task_embeddings__task_only.npz"),
    )
    p.add_argument(
        "--onet-clusters-parquet",
        default=str(DERIVED / "onet_task_clusters__task_only.parquet"),
    )
    p.add_argument("--llm-model", default=DEFAULT_MODEL)
    p.add_argument(
        "--llm-input",
        choices=("label_only", "label_plus_description"),
        default="label_only",
    )
    p.add_argument(
        "--llm-workers", type=int, default=8,
        help="Concurrent LLM calls (ThreadPoolExecutor).",
    )
    p.add_argument(
        "--skip-llm", action="store_true",
        help="Only compute centroid assignment (useful for dev iteration).",
    )
    p.add_argument(
        "--cache", default=str(ESCO_DERIVED / "esco_llm_cache.jsonl"),
        help="Append-only JSONL cache of per-item LLM outputs.",
    )
    args = p.parse_args()

    ESCO_DERIVED.mkdir(parents=True, exist_ok=True)

    # Load taxonomy + build centroids.
    ops, op_clusters = load_taxonomy()
    onet_emb = np.load(args.onet_emb_npz)["embeddings"]
    onet_clusters = pd.read_parquet(args.onet_clusters_parquet)
    centroids, op_ids = compute_op_centroids(ops, op_clusters, onet_emb, onet_clusters)
    ref_stats = centroid_reference_stats(centroids, onet_emb, onet_clusters, op_ids, op_clusters)
    (ESCO_DERIVED / "op_centroid_cosine_reference.json").write_text(
        json.dumps(ref_stats, indent=2)
    )

    # Load ESCO items + embeddings.
    esco_meta = pd.read_parquet(args.esco_emb_meta).reset_index(drop=True)
    esco_emb = np.load(args.esco_emb_npz)["embeddings"]
    if len(esco_meta) != len(esco_emb):
        raise SystemExit("ESCO embedding / meta length mismatch")
    esco_emb_n = l2_normalize(esco_emb.astype(np.float32))

    # Centroid assignment.
    cos = esco_emb_n @ centroids.T  # (n_items, n_ops)
    top1 = cos.argmax(axis=1)
    top1_cos = cos[np.arange(len(cos)), top1]
    sorted_idx = np.argsort(-cos, axis=1)
    top2_idx = sorted_idx[:, 1]
    top2_cos = cos[np.arange(len(cos)), top2_idx]
    margin = top1_cos - top2_cos

    centroid_op_id = [op_ids[i] for i in top1]
    thresholds = np.array([ref_stats[o]["centroid_cosine_p05"] for o in centroid_op_id])
    unmappable_geom = top1_cos < thresholds

    rows = esco_meta.copy()
    rows["centroid_op_id"] = centroid_op_id
    rows["centroid_cos_top1"] = top1_cos.astype(np.float32)
    rows["centroid_cos_top2"] = top2_cos.astype(np.float32)
    rows["centroid_margin"] = margin.astype(np.float32)
    rows["unmappable_geometric"] = unmappable_geom
    rows["centroid_top2_op_id"] = [op_ids[i] for i in top2_idx]

    # LLM assignment (parallel, cached).
    if not args.skip_llm:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY / OPENAI_API env var not set.")
        client = OpenAI(api_key=api_key)
        system_prompt = SYSTEM_PROMPT.format(ops_block=_format_ops_block(ops))
        valid_ids = {o["operation_id"] for o in ops}

        cache_path = Path(args.cache)
        cache = load_cache(cache_path)
        print(f"LLM cache: {len(cache)} cached items at {cache_path}")

        # Build per-item input text.
        if args.llm_input == "label_only":
            texts = rows["preferred_label"].fillna("").astype(str).tolist()
        else:
            texts = [
                f"{r or ''}: {d or ''}".strip(": ").strip()
                for r, d in zip(rows["preferred_label"], rows["description"])
            ]

        to_run = [
            (uri, texts[i])
            for i, uri in enumerate(rows["esco_uri"].tolist())
            if uri not in cache
        ]
        print(f"LLM classify: {len(to_run)} to run (model={args.llm_model})")

        cache_lock = threading.Lock()
        with cache_path.open("a") as fh, ThreadPoolExecutor(max_workers=args.llm_workers) as ex:
            futures = {
                ex.submit(llm_classify_one, client, args.llm_model, system_prompt, uri, txt, valid_ids): uri
                for uri, txt in to_run
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="llm"):
                rec = fut.result()
                with cache_lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    cache[rec["esco_uri"]] = rec

        llm_op = [cache.get(u, {}).get("llm_op_id", "none_of_the_above") for u in rows["esco_uri"]]
        llm_conf = [cache.get(u, {}).get("llm_confidence", "low") for u in rows["esco_uri"]]
        llm_rat = [cache.get(u, {}).get("llm_rationale", "") for u in rows["esco_uri"]]
        rows["llm_op_id"] = llm_op
        rows["llm_confidence"] = llm_conf
        rows["llm_rationale"] = llm_rat
    else:
        rows["llm_op_id"] = None
        rows["llm_confidence"] = None
        rows["llm_rationale"] = None

    # Consensus.
    def _consensus(row) -> tuple[str | None, str]:
        c = row["centroid_op_id"]
        l = row.get("llm_op_id")
        if l is None:
            return (c if not row["unmappable_geometric"] else None,
                    "centroid_only" if not row["unmappable_geometric"] else "centroid_unmappable")
        if l == "none_of_the_above":
            return (None, "llm_none_of_the_above")
        if l == c:
            return (l, "agree")
        return (None, f"disagree:centroid={c};llm={l}")

    cons = rows.apply(_consensus, axis=1, result_type="expand")
    rows["consensus_op_id"] = cons[0]
    rows["review_reason"] = cons[1]

    # Write outputs.
    out_parquet = ESCO_DERIVED / "esco_op_assignments.parquet"
    rows.to_parquet(out_parquet, index=False)
    out_csv = ESCO_DERIVED / "esco_op_assignments.csv"
    rows.to_csv(out_csv, index=False)
    print(f"Wrote {out_parquet} ({len(rows)} rows)")
    print(f"Wrote {out_csv}")

    disagreements = rows[rows["consensus_op_id"].isna()].copy()
    disagreements.to_csv(ESCO_DERIVED / "esco_disagreement_cases.csv", index=False)
    print(f"Disagreements / unmappable: {len(disagreements)} / {len(rows)} "
          f"({100 * len(disagreements) / max(len(rows), 1):.1f}%)")

    # Quick console summary.
    by_op = (
        rows.groupby("consensus_op_id", dropna=False)
        .size()
        .reset_index(name="n_items")
        .sort_values("n_items", ascending=False)
    )
    print("\nConsensus assignment by op:")
    print(by_op.to_string(index=False))

    agree = (rows["centroid_op_id"] == rows["llm_op_id"]).sum() if "llm_op_id" in rows else 0
    print(f"\ncentroid == llm: {agree} / {len(rows)} "
          f"({100 * agree / max(len(rows), 1):.1f}%)")


if __name__ == "__main__":
    main()
