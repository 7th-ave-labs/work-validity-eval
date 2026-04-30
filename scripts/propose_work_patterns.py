#!/usr/bin/env python3
"""Draft work-pattern labels from O*NET task statements.

Sampling strategy: stratify by HDBSCAN cluster + job zone, weight by importance and
relevance so that high-salience tasks are more likely to surface.

The LLM is asked to propose *neutral work-pattern labels* grounded in job-task
language (not benchmark / AI terminology). Output is a JSON list of
{label_name, definition, inclusion_examples, exclusion_examples}.

We then merge + de-duplicate in a second pass and write a candidate CSV to
`data/derived/work_pattern_labels_candidates.csv`. A human curator (or the next
step of this pipeline) promotes the frozen subset to `data/work_pattern_labels.csv`.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"

MODEL_ID = os.environ.get("PROPOSE_WORK_PATTERNS_MODEL", "gpt-5.4-mini")


PROPOSE_PROMPT = textwrap.dedent(
    """
    You are helping a researcher draft a taxonomy of *work patterns* observed in
    U.S. O*NET occupation-specific task statements for Job Zones 3–5 (mostly
    knowledge / professional / technical work).

    A "work pattern" names the kind of operation a worker performs on
    information, documents, people, or physical/technical systems. It is
    expressed in neutral, job-task-side language (not benchmark or AI
    terminology). Good examples:

    - "Compare incoming records against standards or specifications"
    - "Draft or revise regulated documents with explicit structural rules"
    - "Diagnose faults in a physical system using tests and measurements"
    - "Translate stakeholder requirements into a technical specification"
    - "Teach or supervise others through structured curriculum or protocols"

    Bad examples (reject these framings):
    - "Question answering over documents"  (benchmark framing)
    - "Reasoning"                           (AI framing)
    - "Uses email"                          (tool-only, no operation)

    The goal is to produce ~25–40 candidate labels that together cover the
    observed tasks. Each label should be:
      * operation-first (not tool-first, not occupation-first)
      * applicable across at least 2–3 occupations where possible
      * distinguishable from its neighbours (don't merge 'diagnose' and 'treat')

    Below are stratified sample task statements (each line is:
    [cluster <n> | zone <z> | occupation] task text).
    Read them and emit a JSON array. Each element MUST have keys:
      label_name, definition, inclusion_examples (3 short), exclusion_examples (2 short)

    Return JSON only.
    """
).strip()


MERGE_PROMPT = textwrap.dedent(
    """
    You are consolidating multiple drafts of a work-pattern taxonomy for
    knowledge/professional occupations (O*NET Job Zones 3–5). Merge near
    duplicates, split when two ideas were bundled, and enforce operation-first
    (not tool-first, not occupation-first) naming. Keep 22–32 labels.

    For each output label:
      * label_id: snake_case (e.g. compare_against_standard)
      * label_name: concise title-case (e.g. "Compare Records Against Standards")
      * definition: 1–2 sentences, job-task-side language
      * inclusion_examples: 3 short O*NET-style strings
      * exclusion_examples: 2 short strings that look similar but are a different pattern
      * sibling_labels: list of other label_ids this is easy to confuse with

    Input is a JSON array of candidate label objects (possibly noisy). Return a
    JSON array of the cleaned labels only. No prose.
    """
).strip()


def stratified_sample(df: pd.DataFrame, n_per_cluster: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for cid, sub in df.groupby("hdbscan_cluster"):
        weight = sub["importance_mean"].fillna(3.0) * (sub["relevance_pct"].fillna(50.0) / 100.0)
        weight = weight.clip(lower=0.01)
        probs = (weight / weight.sum()).to_numpy()
        k = min(n_per_cluster, len(sub))
        idx = rng.choice(len(sub), size=k, replace=False, p=probs)
        parts.append(sub.iloc[idx])
    return pd.concat(parts, ignore_index=True)


def format_for_prompt(df: pd.DataFrame) -> str:
    lines = []
    for _, row in df.iterrows():
        cid = int(row["hdbscan_cluster"])
        zone = int(row["job_zone"]) if pd.notna(row["job_zone"]) else -1
        title = str(row["occupation_title"])[:80]
        text = str(row["task_text_clean"])[:400]
        lines.append(f"[cluster {cid:2d} | zone {zone} | {title}] {text}")
    return "\n".join(lines)


def call_llm_json(client: OpenAI, model: str, system: str, user: str) -> list[dict]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    txt = resp.choices[0].message.content
    data = json.loads(txt)
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
        raise ValueError(f"LLM returned dict without list value: keys={list(data)}")
    if isinstance(data, list):
        return data
    raise ValueError(f"LLM did not return a list: {type(data)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clusters", default=str(DERIVED / "onet_task_clusters.parquet"))
    p.add_argument("--out", default=str(DERIVED / "work_pattern_labels_candidates.csv"))
    p.add_argument("--json-out", default=str(DERIVED / "work_pattern_labels_candidates.json"))
    p.add_argument("--n-per-cluster", type=int, default=12)
    p.add_argument("--n-chunks", type=int, default=3, help="How many independent proposal passes.")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY / OPENAI_API missing")

    clusters = pd.read_parquet(args.clusters)
    clustered = clusters[clusters["hdbscan_cluster"] != -1].copy()
    noise = clusters[clusters["hdbscan_cluster"] == -1].copy()
    print(f"Clustered: {len(clustered)}  Noise: {len(noise)}  Total: {len(clusters)}")

    client = OpenAI(api_key=api_key)

    all_candidates: list[dict] = []
    for chunk in range(args.n_chunks):
        seed = args.seed + chunk
        sample = stratified_sample(clustered, n_per_cluster=args.n_per_cluster, seed=seed)
        if len(noise) > 0:
            noise_sample = noise.sample(min(len(noise), 60), random_state=seed)
            noise_sample = noise_sample.assign(hdbscan_cluster=-1)
            sample = pd.concat([sample, noise_sample], ignore_index=True)
        body = format_for_prompt(sample)
        print(f"[chunk {chunk}] samples={len(sample)}")
        proposals = call_llm_json(
            client,
            args.model,
            PROPOSE_PROMPT,
            "Here are the sampled tasks:\n\n" + body + "\n\nReturn JSON array.",
        )
        print(f"[chunk {chunk}] proposals={len(proposals)}")
        all_candidates.extend(proposals)

    merged = call_llm_json(
        client,
        args.model,
        MERGE_PROMPT,
        json.dumps(all_candidates, ensure_ascii=False),
    )
    print(f"Merged labels: {len(merged)}")

    rows = []
    for item in merged:
        rows.append(
            {
                "label_id": item.get("label_id", ""),
                "label_name": item.get("label_name", ""),
                "description": item.get("definition", ""),
                "inclusion_examples": " | ".join(item.get("inclusion_examples", [])[:3]),
                "exclusion_examples": " | ".join(item.get("exclusion_examples", [])[:2]),
                "sibling_labels": ",".join(item.get("sibling_labels", [])),
                "cluster_or_method": f"text-embedding-3-small; HDBSCAN(min_cluster={40}); LLM proposal + merge pass (model={args.model})",
                "frozen": False,
                "example_onetsoc_tasks": "",
                "notes": "",
            }
        )
    pd.DataFrame(rows).to_csv(args.out, index=False)
    Path(args.json_out).write_text(json.dumps(merged, indent=2))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
