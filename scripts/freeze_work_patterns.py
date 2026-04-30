#!/usr/bin/env python3
"""Second-pass consolidation of candidate work-pattern labels.

Input: data/derived/work_pattern_labels_candidates.json (from propose_work_patterns.py)
Output:
  data/work_pattern_labels.csv                (the frozen taxonomy; small curated set)
  data/derived/work_pattern_labels_frozen.json (full JSON with examples + siblings)
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"

TIGHTEN_PROMPT = textwrap.dedent(
    """
    You are tightening a draft work-pattern taxonomy for U.S. O*NET occupation
    task statements (Job Zones 3–5). The input contains {n_in} candidate labels
    that are partly duplicated or too fine-grained.

    Produce a CLEAN, FROZEN taxonomy of EXACTLY 24–28 labels that:
      * uses operation-first language (not tools, not occupations)
      * covers the concrete patterns visible in the input examples
      * collapses near-duplicates (e.g. "compare to standard" + "inspect for defects" -> one label if they share the minimum task contract)
      * keeps patterns meaningful across many occupations
      * preserves the most-distinctive technical/physical/process categories
        (e.g. diagnose-and-repair-physical-systems, design-physical-artefact,
        teach-structured-curriculum) alongside the document/analysis cluster

    Every output item MUST have:
      * label_id          snake_case, stable
      * label_name        title case
      * definition        1–2 sentences
      * inclusion_examples  3 short strings, drawn / paraphrased from the input
      * exclusion_examples  2 short strings that look similar but belong elsewhere
      * sibling_labels    list of label_ids that are easy to confuse with this one
      * minimum_task_contract    1 sentence: the properties a benchmark must
                                 preserve to claim it measures this pattern

    Return a JSON object of the form {{"labels": [...]}}.
    """
).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates-json", default=str(DERIVED / "work_pattern_labels_candidates.json"))
    p.add_argument("--out-csv", default=str(ROOT / "data" / "work_pattern_labels.csv"))
    p.add_argument("--out-json", default=str(DERIVED / "work_pattern_labels_frozen.json"))
    p.add_argument("--model", default=os.environ.get("PROPOSE_WORK_PATTERNS_MODEL", "gpt-5.4-mini"))
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY / OPENAI_API missing")

    candidates = json.loads(Path(args.candidates_json).read_text())
    n_in = len(candidates)
    print(f"Consolidating {n_in} candidate labels")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": TIGHTEN_PROMPT.format(n_in=n_in)},
            {"role": "user", "content": json.dumps(candidates, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    data = json.loads(resp.choices[0].message.content)
    labels = data.get("labels") if isinstance(data, dict) else data
    if not isinstance(labels, list):
        raise ValueError(f"Unexpected response: {data!r}")
    print(f"Frozen labels: {len(labels)}")

    Path(args.out_json).write_text(json.dumps(labels, indent=2))
    rows = []
    for item in labels:
        rows.append(
            {
                "label_id": item.get("label_id", ""),
                "label_name": item.get("label_name", ""),
                "description": item.get("definition", ""),
                "cluster_or_method": (
                    "text-embedding-3-small + HDBSCAN(min_cluster=40, min_samples=10); "
                    f"two-stage LLM consolidation (model={args.model})"
                ),
                "frozen": True,
                "example_onetsoc_tasks": " | ".join(item.get("inclusion_examples", [])[:3]),
                "minimum_task_contract": item.get("minimum_task_contract", ""),
                "sibling_labels": ",".join(item.get("sibling_labels", [])),
                "exclusion_examples": " | ".join(item.get("exclusion_examples", [])[:2]),
                "notes": "",
            }
        )
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
