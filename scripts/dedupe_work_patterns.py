#!/usr/bin/env python3
"""Force-dedupe the frozen taxonomy down to a target size.

Operates on data/derived/work_pattern_labels_frozen.json and rewrites
  data/work_pattern_labels.csv
  data/derived/work_pattern_labels_frozen.json
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

PROMPT = textwrap.dedent(
    """
    You will receive a JSON array of candidate "work pattern" labels for U.S.
    O*NET occupation task statements (Job Zones 3–5). The set contains obvious
    duplicates and near-duplicates that MUST be collapsed.

    Produce EXACTLY {target} labels. Hard rules:

    1. Operation-first naming. No tools, no occupations in the name.
    2. If two candidates share a minimum task contract, MERGE them; keep the
       broadest sensible name and fold the examples from both in.
    3. Keep meaningful breadth. Do not let the taxonomy collapse into
       "document work" + "physical work" only. Preserve distinct operations
       such as: diagnose / repair equipment; design artefact; conduct
       experiments; teach people; negotiate / mediate; review for compliance;
       extract and verify records; communicate with clients; operate / set
       parameters; collect / maintain data.
    4. Each label must be applicable across multiple occupations in the input.
    5. Output exactly {target} items. If the input has fewer clearly distinct
       ideas than {target}, SPLIT smartly to reach {target}; if more, MERGE
       ruthlessly.

    Each output item MUST have:
      label_id, label_name, definition,
      inclusion_examples (3),
      exclusion_examples (2),
      sibling_labels (list of label_ids),
      minimum_task_contract (1 sentence)

    Return JSON: {{"labels": [...]}}.
    """
).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", default=str(DERIVED / "work_pattern_labels_frozen.json"))
    ap.add_argument("--out-csv", default=str(ROOT / "data" / "work_pattern_labels.csv"))
    ap.add_argument("--out-json", default=str(DERIVED / "work_pattern_labels_frozen.json"))
    ap.add_argument("--target", type=int, default=26)
    ap.add_argument(
        "--model",
        default=os.environ.get("DEDUPE_WORK_PATTERNS_MODEL", "gpt-5.4-mini"),
    )
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY / OPENAI_API missing")

    candidates = json.loads(Path(args.in_json).read_text())
    print(f"Dedupe: {len(candidates)} -> target {args.target}")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": PROMPT.format(target=args.target)},
            {"role": "user", "content": json.dumps(candidates, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    data = json.loads(resp.choices[0].message.content)
    labels = data.get("labels") if isinstance(data, dict) else data
    print(f"Output labels: {len(labels)}")

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
                    f"two-stage LLM consolidation + dedupe (proposal + dedupe models; dedupe={args.model})"
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
