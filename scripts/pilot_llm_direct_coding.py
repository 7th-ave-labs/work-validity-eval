#!/usr/bin/env python3
"""Pilot LLM direct-coding sensitivity check (Step 1.10 in the plan).

On a small stratified sample, ask the LLM to propose a work-pattern label
WITHOUT the frozen taxonomy. Compare the returned labels with the frozen
taxonomy labels assigned by scripts/code_onet_tasks.py to see whether the
LLM drifts toward AI/benchmark language when it has no constraint.

Outputs:
  data/derived/pilot_llm_direct_coding.csv  (task x frozen label x free-form label x drift tag)
  prints a short drift summary.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"

SYSTEM = textwrap.dedent(
    """
    You are a research coder. For each short O*NET occupation task statement,
    propose a single short "work pattern" label (6 words max) that describes
    what kind of operation the worker performs. Use operation-first language
    and do not use benchmark, AI, or evaluation vocabulary. Also give a short
    definition (1 sentence).

    Return JSON: {"codings": [{"task_id": ..., "free_label": ..., "definition": ...}, ...]}
    """
).strip()


DRIFT_TOKENS = {
    # AI / ML framing
    "question answering", "qa", "chatbot", "inference", "prediction", "reasoning",
    "classification", "regression", "summarization", "generation", "retrieval",
    "embedding", "prompt", "agent", "llm", "rag", "benchmark", "dataset",
    "fine-tuning", "tokenization", "few-shot", "zero-shot", "tool use",
}


def has_drift(text: str) -> list[str]:
    import re
    s = (text or "").lower()
    return sorted({t for t in DRIFT_TOKENS if re.search(r'\b' + re.escape(t) + r'\b', s)})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--coded", default=str(DERIVED / "onet_tasks_coded.parquet"))
    p.add_argument("--n", type=int, default=150, help="Pilot sample size.")
    p.add_argument("--model", default=os.environ.get("CODING_MODEL", "gpt-5.4-mini"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=15)
    p.add_argument("--out", default=str(DERIVED / "pilot_llm_direct_coding.csv"))
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY / OPENAI_API missing")

    df = pd.read_parquet(args.coded)
    # Stratify by primary_label_id, take a handful per label.
    per_label = max(1, args.n // df["primary_label_id"].nunique())
    rng = random.Random(args.seed)
    picks = []
    for lid, sub in df.groupby("primary_label_id"):
        idx = list(sub.index)
        rng.shuffle(idx)
        picks.extend(sub.loc[idx[:per_label]].to_dict("records"))
    rng.shuffle(picks)
    picks = picks[: args.n]
    print(f"Pilot sample: {len(picks)} tasks")

    client = OpenAI(api_key=api_key)

    def run_batch(batch):
        user = "Tasks (JSON-lines):\n\n" + "\n".join(
            json.dumps(
                {
                    "task_id": r["task_id"],
                    "occupation_title": r["occupation_title"],
                    "task_text": r["task_text_clean"],
                },
                ensure_ascii=False,
            )
            for r in batch
        ) + "\n\nReturn JSON {\"codings\": [...]} in same order."
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                data = json.loads(resp.choices[0].message.content)
                c = data.get("codings") if isinstance(data, dict) else data
                if not isinstance(c, list) or len(c) != len(batch):
                    raise ValueError("shape mismatch")
                return batch, c
            except Exception as exc:
                time.sleep(2 ** attempt)
        return batch, None

    batches = [picks[i:i+args.batch_size] for i in range(0, len(picks), args.batch_size)]
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_batch, b) for b in batches]
        for fut in as_completed(futs):
            batch, codings = fut.result()
            if codings is None:
                continue
            for src, c in zip(batch, codings):
                free = (c.get("free_label") or "").strip()
                defn = (c.get("definition") or "").strip()
                drift = has_drift(free) + has_drift(defn)
                rows.append(
                    {
                        "task_id": src["task_id"],
                        "onetsoc_code": src["onetsoc_code"],
                        "occupation_title": src["occupation_title"],
                        "task_text": src["task_text_clean"],
                        "frozen_label": src["primary_label_id"],
                        "llm_free_label": free,
                        "llm_free_definition": defn,
                        "drift_tokens": ",".join(drift),
                    }
                )

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} rows={len(out)}")
    if len(out) == 0:
        return
    drift_rate = (out["drift_tokens"].str.len() > 0).mean()
    print(f"Drift rate (any AI/benchmark token in free label/definition): {drift_rate:.1%}")
    print("Top free-label phrases (lowercased):")
    counts = out["llm_free_label"].str.lower().value_counts().head(20)
    for k, v in counts.items():
        print(f"  {v:3d}  {k}")


if __name__ == "__main__":
    main()
