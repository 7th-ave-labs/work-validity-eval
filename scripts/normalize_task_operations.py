#!/usr/bin/env python3
"""Canonicalize O*NET task statements into domain-stripped operation phrases.

For each task in ``data/derived/onet_tasks_prepared.parquet``, call an LLM
to rewrite the task so only the OPERATION (verb + generic object type)
remains - the profession, industry, and domain-specific nouns are
stripped. The stripped version is what we embed for clustering; the
original text is preserved alongside for inspection.

Motivation: the current ``task_only`` embedding variant still clusters
by job / domain because task statements themselves contain domain nouns
(e.g. "Design vacuum tube collector systems for solar applications"
dominates on "solar"; "Debug robotics programs" dominates on
"robotics"). We want clustering to group by shared operation (e.g.
design a hardware system, debug software) across domains.

Output JSON per task:
  {
    "onetsoc_code": str,
    "task_id": int,
    "operation_phrase": str,  # domain-stripped, 3-10 words
    "operation_verbs": [str],  # 1-3 canonical verbs
    "operation_object_type": str,  # enum
    "domain_stripped": [str],  # terms removed from the original
    "notes": str
  }

Canonical object types (enum):
  code, hardware, material, document, plan, data, image_or_media,
  person, organization, environment, money, schedule, procedure, other

Parallel by default; resumable via an append-only JSONL cache that
tracks already-processed task ids.

Outputs:
  data/derived/task_operations_normalized.jsonl  (cache, append-only)
  data/derived/task_operations_normalized.parquet  (clean snapshot)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"

DEFAULT_MODEL = os.environ.get("NORMALIZE_TASK_MODEL", "gpt-5.4-mini")

OBJECT_TYPES = [
    "code",
    "hardware",
    "material",
    "document",
    "plan",
    "data",
    "image_or_media",
    "person",
    "organization",
    "environment",
    "money",
    "schedule",
    "procedure",
    "other",
]


SYSTEM_PROMPT = textwrap.dedent(
    f"""
    You rewrite U.S. O*NET task statements into DOMAIN-STRIPPED operation
    phrases. The goal is to preserve WHAT THE WORKER DOES
    (the cognitive / physical operation and the generic kind of object
    they operate on), while removing profession-, industry-, and
    domain-specific terminology so that tasks from different occupations
    doing the SAME operation end up with similar phrasing.

    Input: one O*NET task statement.

    Output JSON (exact keys):
      "operation_phrase": 3-10 word sentence-case phrase: <verb> <generic
          object> [<generic modifier>]. No proper nouns, no occupation
          names, no industry nouns, no product brands, no domain nouns
          (e.g. "robotic", "solar", "nanoscale", "curriculum", "patient",
          "legal"). Prefer common-noun generalisations: "robotic
          system" -> "hardware system"; "vacuum tube collector system
          for solar applications" -> "hardware system"; "treatment
          plan" -> "care plan for a person" or "individual plan for
          a person"; "nanomaterials" -> "material"; "patient" ->
          "person"; "students" -> "learners"; "court" -> "formal
          setting".
      "operation_verbs": array of 1-3 canonical verbs from the source
          task, lowercase, infinitive (without "to"). E.g. ["design"],
          ["diagnose", "treat"], ["supervise", "coordinate"].
      "operation_object_type": one of {OBJECT_TYPES} - the DOMINANT
          kind of object the operation acts on.
      "domain_stripped": array of specific terms removed from the
          original, lowercase, for inspection (e.g. ["solar", "vacuum
          tube", "collector"]).
      "notes": optional short note if the task is ambiguous or mixes
          operations; else "".

    Rules:
      * Keep it tight. Don't invent content that is not in the source.
      * If the source mentions multiple operations (e.g. "design AND
        test"), pick the DOMINANT one for ``operation_phrase`` and list
        the others in ``operation_verbs``.
      * Do NOT rename a person-focused operation to an artefact
        operation, or vice versa. "Advise student on coursework" is
        about advising a person, not about coursework.
      * The operation_phrase should still uniquely identify the kind of
        work, just without the domain. If stripping domain makes the
        phrase meaningless (rare), keep one generic domain hint.
      * **KEEP THE PURPOSE / CRITERION when it is load-bearing.** If
        the source says "analyze data to determine validity, quality,
        and scientific significance", the purpose ("determine
        validity / significance") is part of the operation's identity
        and must be preserved. Prefer
        "analyze data for validity and significance" over the naked
        "analyze data". Same for "review cash reserves to check
        internal control" -> "verify cash reserves for internal
        control" not "verify cash reserves".
      * **DO NOT BUNDLE distinct operations with a weak umbrella
        verb.** If the source lists multiple independent operations
        (e.g. "prepare or monitor project schedules, budgets, or cost
        control systems", "conduct research, compile data, and prepare
        papers", "interview and hire writers OR negotiate contracts,
        royalties, and payments", "design AND administer training
        courses"), pick the dominant one and return a specific
        operation_phrase for it; use ``notes`` to list the other
        operations.
      * **PREFER STRONG, ACTIVE VERBS over weak umbrella phrasings.**
        "monitor regulations" beats "maintain awareness of
        regulations"; "evaluate and grade learner work" beats "observe
        and evaluate work"; "manage staffing lifecycle" beats "manage
        personnel".
      * **PRESERVE THE CORE OPERATION, NOT A TANGENTIAL BYPRODUCT.**
        If the source says "design or use surveillance tools ... to
        identify health risks", the dominant operation is designing /
        using the surveillance instrument, not "identify risks". The
        operation_phrase must name the doing, not the summary.
      * Prefer sentence-case single phrase (no leading comma,
        semicolon, or "or"-joined alternatives). One operation, one
        phrase.

    Examples:
      source: "Design or develop vacuum tube collector systems for
        solar applications."
      -> {{"operation_phrase": "design hardware system",
           "operation_verbs": ["design", "develop"],
           "operation_object_type": "hardware",
           "domain_stripped": ["vacuum tube", "collector", "solar"],
           "notes": ""}}

      source: "Debug robotics programs."
      -> {{"operation_phrase": "debug software",
           "operation_verbs": ["debug"],
           "operation_object_type": "code",
           "domain_stripped": ["robotics"],
           "notes": ""}}

      source: "Develop and implement treatment plans."
      -> {{"operation_phrase": "develop and implement care plan for a person",
           "operation_verbs": ["develop", "implement"],
           "operation_object_type": "plan",
           "domain_stripped": ["treatment"],
           "notes": ""}}

      source: "Prepare and deliver lectures to undergraduate or
        graduate students on assigned topics."
      -> {{"operation_phrase": "prepare and deliver instruction to learners",
           "operation_verbs": ["prepare", "deliver"],
           "operation_object_type": "procedure",
           "domain_stripped": ["lectures", "undergraduate", "graduate",
             "students"],
           "notes": ""}}

      source: "Maintain student attendance records, grades, and other
        required records."
      -> {{"operation_phrase": "maintain records",
           "operation_verbs": ["maintain"],
           "operation_object_type": "document",
           "domain_stripped": ["student", "attendance", "grades"],
           "notes": ""}}

      source: "Testify in court or at administrative proceedings."
      -> {{"operation_phrase": "testify in a formal setting",
           "operation_verbs": ["testify"],
           "operation_object_type": "document",
           "domain_stripped": ["court", "administrative"],
           "notes": ""}}

      source: "Synthesize, process, or characterize nanomaterials,
        using advanced tools or techniques."
      -> {{"operation_phrase": "synthesize and characterize a material",
           "operation_verbs": ["synthesize", "characterize"],
           "operation_object_type": "material",
           "domain_stripped": ["nanomaterials", "advanced tools"],
           "notes": ""}}

    Return JSON only.
    """
).strip()


def normalize_one(client: OpenAI, model: str, task_text: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_text},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)
    # Defensive normalisation of shape.
    data["operation_phrase"] = str(data.get("operation_phrase", "")).strip()
    verbs = data.get("operation_verbs") or []
    if isinstance(verbs, str):
        verbs = [verbs]
    data["operation_verbs"] = [str(v).strip().lower() for v in verbs][:3]
    obj = str(data.get("operation_object_type", "other")).strip().lower()
    if obj not in OBJECT_TYPES:
        obj = "other"
    data["operation_object_type"] = obj
    stripped = data.get("domain_stripped") or []
    if isinstance(stripped, str):
        stripped = [stripped]
    data["domain_stripped"] = [str(x).strip() for x in stripped]
    data["notes"] = str(data.get("notes", "")).strip()
    return data


def load_cache(cache_path: Path) -> dict[tuple[str, int], dict]:
    cache: dict[tuple[str, int], dict] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = (row["onetsoc_code"], int(row["task_id"]))
                cache[key] = row
            except Exception:
                continue
    return cache


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default=str(DERIVED / "onet_tasks_prepared.parquet"))
    p.add_argument("--cache", default=str(DERIVED / "task_operations_normalized.jsonl"))
    p.add_argument("--out", default=str(DERIVED / "task_operations_normalized.parquet"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--limit", type=int, default=None, help="optional limit (for pilot)")
    p.add_argument("--sample", type=int, default=None, help="stratified sample size (pilot); overrides --limit")
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY / OPENAI_API missing")

    df = pd.read_parquet(args.inp)
    # Expect columns onetsoc_code, task_id, task_text_clean, occupation_title,
    # job_zone, importance_mean, relevance_pct
    must = ["onetsoc_code", "task_id", "task_text_clean"]
    for c in must:
        if c not in df.columns:
            raise SystemExit(f"missing column {c}")
    df = df.dropna(subset=must).reset_index(drop=True)
    print(f"Input: {len(df)} tasks from {args.inp}")

    if args.sample:
        df = df.sample(min(len(df), args.sample), random_state=args.sample_seed).reset_index(drop=True)
        print(f"Sample -> {len(df)} tasks")
    elif args.limit:
        df = df.head(args.limit).reset_index(drop=True)
        print(f"Limit -> {len(df)} tasks")

    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path)
    print(f"Cache: {len(cache)} pre-existing rows")

    todo_rows = [
        (int(i), r) for i, r in df.iterrows()
        if (r["onetsoc_code"], int(r["task_id"])) not in cache
    ]
    print(f"To process: {len(todo_rows)} tasks (skipping {len(df) - len(todo_rows)} cached)")

    if args.dry_run:
        print("--dry-run: skipping LLM calls")
        return

    client = OpenAI(api_key=api_key)
    lock = threading.Lock()

    def worker(job):
        i, r = job
        try:
            result = normalize_one(client, args.model, str(r["task_text_clean"]))
            result["onetsoc_code"] = r["onetsoc_code"]
            result["task_id"] = int(r["task_id"])
            result["task_text_clean"] = str(r["task_text_clean"])
            result["occupation_title"] = str(r.get("occupation_title", ""))
            return i, result, None
        except Exception as e:
            return i, None, repr(e)

    done = 0
    errs = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(worker, job) for job in todo_rows]
        for fut in as_completed(futs):
            i, result, err = fut.result()
            if err is not None:
                errs += 1
                if errs <= 10:
                    print(f"  [err {i}] {err}", file=sys.stderr)
                continue
            with lock:
                with cache_path.open("a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                cache[(result["onetsoc_code"], result["task_id"])] = result
            done += 1
            if done % 250 == 0:
                print(f"  ... {done}/{len(todo_rows)} done")

    print(f"Finished: {done} new, {errs} errors, {len(cache)} total in cache")

    # Snapshot final parquet from cache (only rows present in the input df,
    # so a stratified pilot run produces a pilot parquet).
    rows = []
    for _, r in df.iterrows():
        key = (r["onetsoc_code"], int(r["task_id"]))
        if key in cache:
            rows.append(cache[key])
    out = pd.DataFrame(rows)
    out.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
