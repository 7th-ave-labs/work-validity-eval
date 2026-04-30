#!/usr/bin/env python3
"""Round-2 knowledge-work screen.

The initial ``knowledge_screen`` in ``prepare_onet_tasks.py`` was
occupation-level (gated the whole occupation on whether it is a
knowledge / professional / technical zone). Human review of a 100-row
stratified stripping sample showed that this lets through a
meaningful tail of *task-level* non-knowledge-work items — direct
manual / clinical / lab / installation / performance / routine-clerical
tasks that sit inside otherwise-knowledge occupations.

This script runs a second, *task-level* screen on top of the
previously-included rows (``knowledge_screen == "include"`` in
``data/derived/onet_tasks_prepared.parquet``). It does NOT re-consider
rows that were already excluded by the first screen.

Inputs:
  data/derived/onet_tasks_prepared.parquet

Outputs:
  data/derived/onet_tasks_prepared_round2.parquet  (adds columns:
    round2_knowledge_decision, round2_knowledge_reason,
    round2_knowledge_category)
  data/derived/round2_knowledge_screen_cache.jsonl  (append-only, resumable)

Decision values:
  include   -- target knowledge-work operation; keep.
  exclude   -- direct physical/manual/clinical/performance/routine
               task; drop from downstream taxonomy work.
  uncertain -- ambiguous hybrid; keep but tag.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
CACHE_PATH = DERIVED / "round2_knowledge_screen_cache.jsonl"
DEFAULT_MODEL = os.environ.get("ROUND2_KSCREEN_MODEL", "gpt-5.4-mini")

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are triaging a single O*NET occupational task statement for
    inclusion in a TASK-LEVEL "knowledge work" taxonomy. The job of the
    taxonomy is to characterise operations that are primarily COGNITIVE
    and whose output is a JUDGEMENT-bearing artifact an AI agent could
    plausibly produce or be evaluated on (report, plan, analysis,
    recommendation, written record, design, review, advisory, etc.).

    Exclude tasks that are NOT primarily cognitive knowledge work, even
    when they appear inside a knowledge-work occupation. Concretely,
    EXCLUDE if the task is:

    - a DIRECT PHYSICAL, MANUAL, CLINICAL, SURGICAL, OR LAB PROCEDURE
      (e.g. "operate on a patient", "take fingerprints", "stain slides",
      "fit orthopedic braces", "care for athletic injuries");
    - PHYSICAL INSTALLATION, WIRING, PLACING, MOUNTING, FITTING,
      ADJUSTING HARDWARE components, or REPAIR / MAINTENANCE of
      equipment, machinery, or facilities (e.g. "place conduit and pull
      wires", "adjust meter components to specifications", "inspect and
      repair automotive equipment", "install color filters");
    - PHYSICAL CLEANING, STERILIZATION, APPLICATION of coatings, or
      PHYSICAL MARKING of parts / templates (e.g. "wash and sterilize
      equipment", "apply UV protection to plumbing", "mark codes on
      patterns");
    - PHYSICAL INSPECTION / MEASUREMENT of objects using hand tools
      (e.g. "verify pattern dimensions using calipers");
    - PERFORMATIVE, RITUAL, OR MEDIA-PRODUCTION EXECUTION (e.g. "read
      from sacred texts", "perform narration or present announcements",
      "synchronize prerecorded audio with visual action");
    - ROUTINE CLERICAL or procurement EXECUTION (e.g. "order or
      requisition supplies", "file, answer telephones, handle routine
      correspondence");
    - SELF-training, MEETING ATTENDANCE as an attendee, or committee
      service framed purely as an obligation (e.g. "participate in
      additional training to keep skills up to date", "attend staff
      meetings and serve on committees, as required");
    - Pure SPECIALTY / SCOPE STATEMENTS that describe a focus area
      rather than an operation (e.g. "specialize in regulatory issues
      related to agriculture").

    INCLUDE if the task is primarily cognitive, judgement-bearing, and
    produces an artifact or a recommendation. Positive examples from
    the reviewer's accept set (keep this pattern):

    - "Record patients' health histories."  (operation = produce a
      judgement-bearing health-history document)
    - "Advise patients and their families on nutritional principles,
      dietary plans, and diet modifications."
    - "Determine the need for facility maintenance and repair, and
      notify appropriate personnel." (assessment + notification; not
      the repair itself)
    - "Conduct post-event evaluations to determine how future events
      could be improved." (evaluative knowledge work)
    - "Refer patients for specialty consultations or treatments."
      (judgement + referral decision; not the treatment itself)
    - "Recommend best fuel for specific sites or circumstances."
    - "Plan and prepare grant proposals to request program funding."
    - "Supervise and direct the work of correctional officers to
      ensure the safe custody, discipline, and welfare of inmates."
      (cognitive supervision)
    - "Oversee public health programs, including statistical analysis,
      health care planning, surveillance systems, and public health
      improvement."
    - "Review cases by analyzing autopsies, laboratory findings, or
      case investigation reports."
    - "Develop statistical models or simulations, using statistical or
      modeling software."
    - "Analyze market conditions to determine acreage allocations."
    - "Read trade journals and professional literature to stay informed
      on trends, innovations, and changes that affect media planning."
    - "Develop and manage work breakdown structure (WBS) of IT
      projects."

    UNCERTAIN: use sparingly. Only when the task is a genuine hybrid
    where the non-physical / judgement portion and the physical portion
    appear of roughly equal weight AND you cannot confidently predict
    the reviewer's call.

    Return JSON with exactly these keys:

      decision: "include" | "exclude" | "uncertain"
      category: one short label from {"physical_procedure",
        "clinical_procedure", "lab_procedure", "installation_or_repair",
        "maintenance_or_cleaning", "physical_measurement",
        "performance_or_ritual", "media_production_execution",
        "routine_clerical", "procurement_execution",
        "self_training_or_attendance", "specialty_statement",
        "hybrid_judgement_and_physical", "cognitive_knowledge_work",
        "other"}
      reason: one short sentence citing the decision rule that
        applied. Do NOT restate the task verbatim.

    Return JSON only.
    """
).strip()

USER_TEMPLATE = textwrap.dedent(
    """
    occupation: {occupation_title}
    task: {task_text}
    """
).strip()


def load_cache() -> dict[tuple[str, str], dict]:
    cache: dict[tuple[str, str], dict] = {}
    if not CACHE_PATH.exists():
        return cache
    with CACHE_PATH.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            key = (str(rec["onetsoc_code"]), str(rec["task_id"]))
            cache[key] = rec
    return cache


def call_llm(client: OpenAI, model: str, occupation: str, task: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    occupation_title=occupation, task_text=task
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)
    dec = str(data.get("decision", "uncertain")).lower().strip()
    if dec not in {"include", "exclude", "uncertain"}:
        dec = "uncertain"
    cat = str(data.get("category", "other")).lower().strip() or "other"
    reason = str(data.get("reason", ""))[:400]
    return {"decision": dec, "category": cat, "reason": reason}


def worker(
    client: OpenAI, model: str, row: dict, max_retries: int = 4
) -> dict:
    for attempt in range(max_retries):
        try:
            out = call_llm(
                client, model, row["occupation_title"], row["task_text_clean"]
            )
            return {
                "onetsoc_code": row["onetsoc_code"],
                "task_id": row["task_id"],
                "round2_knowledge_decision": out["decision"],
                "round2_knowledge_category": out["category"],
                "round2_knowledge_reason": out["reason"],
            }
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "onetsoc_code": row["onetsoc_code"],
                    "task_id": row["task_id"],
                    "round2_knowledge_decision": "uncertain",
                    "round2_knowledge_category": "other",
                    "round2_knowledge_reason": f"error: {e}",
                }
            time.sleep(2 ** attempt)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default=str(DERIVED / "onet_tasks_prepared.parquet"),
    )
    p.add_argument(
        "--output",
        default=str(DERIVED / "onet_tasks_prepared_round2.parquet"),
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=30)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY / OPENAI_API missing")
    client = OpenAI(api_key=api_key)

    df = pd.read_parquet(args.input)
    # Only re-evaluate previously-included rows.
    already_included = df[df["knowledge_screen"] == "include"].copy()
    already_included["onetsoc_code"] = already_included["onetsoc_code"].astype(str)
    already_included["task_id"] = already_included["task_id"].astype(str)

    if args.limit:
        already_included = already_included.head(args.limit)

    print(f"Input rows (knowledge_screen=include): {len(already_included)}")

    cache = load_cache()
    print(f"Cache: {len(cache)} prior decisions")

    pending = [
        r
        for r in already_included.to_dict("records")
        if (str(r["onetsoc_code"]), str(r["task_id"])) not in cache
    ]
    print(f"Pending tasks to screen: {len(pending)}")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_records: list[dict] = []

    with CACHE_PATH.open("a") as cache_file:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(worker, client, args.model, r): r for r in pending
            }
            done = 0
            for fut in as_completed(futures):
                rec = fut.result()
                new_records.append(rec)
                cache_file.write(json.dumps(rec) + "\n")
                cache_file.flush()
                done += 1
                if done % 250 == 0 or done == len(pending):
                    print(f"  screened {done}/{len(pending)}")

    cache = load_cache()
    # Build output table: all previously-included rows, enriched with
    # round2 columns. Previously-excluded rows get round2_decision =
    # "n/a".
    keys = list(zip(df["onetsoc_code"].astype(str), df["task_id"].astype(str)))
    decisions = []
    categories = []
    reasons = []
    for k in keys:
        if k in cache:
            decisions.append(cache[k]["round2_knowledge_decision"])
            categories.append(cache[k]["round2_knowledge_category"])
            reasons.append(cache[k]["round2_knowledge_reason"])
        else:
            decisions.append("n/a")
            categories.append("n/a")
            reasons.append("not previously included")
    df["round2_knowledge_decision"] = decisions
    df["round2_knowledge_category"] = categories
    df["round2_knowledge_reason"] = reasons

    df.to_parquet(args.output, index=False)

    kept = (df["round2_knowledge_decision"] == "include").sum()
    uncertain = (df["round2_knowledge_decision"] == "uncertain").sum()
    excluded = (df["round2_knowledge_decision"] == "exclude").sum()
    print()
    print(f"Wrote {args.output}")
    print(f"Round-2 decisions on previously-included corpus:")
    print(f"  include   : {kept}")
    print(f"  uncertain : {uncertain}")
    print(f"  exclude   : {excluded}")
    print()
    print("Breakdown by exclude category:")
    excl = df[df["round2_knowledge_decision"] == "exclude"]
    print(excl["round2_knowledge_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
