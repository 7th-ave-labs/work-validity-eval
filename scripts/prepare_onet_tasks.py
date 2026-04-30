#!/usr/bin/env python3
"""Join O*NET task statements, ratings, occupation titles, and job zones.

Reads tab-delimited files from data/raw/onet/ (see scripts/download_onet_core.sh).

This preparation step also applies an inclusive prescreen for knowledge / expert
work. Rows that clearly look hands-on, routine, or physical are excluded. If the
heuristic is unsure, the row is kept for downstream clustering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "onet"
DERIVED = ROOT / "data" / "derived"

OCCUPATION_INCLUDE_TERMS = (
    "account",
    "analys",
    "architect",
    "attorney",
    "audit",
    "counsel",
    "consult",
    "compliance",
    "designer",
    "developer",
    "economist",
    "editor",
    "engineer",
    "examiner",
    "financial",
    "investig",
    "law",
    "manager",
    "medical",
    "nurse",
    "physician",
    "planner",
    "policy",
    "professor",
    "program",
    "project",
    "regulator",
    "research",
    "scientist",
    "specialist",
    "strateg",
    "supervisor",
    "teacher",
    "therap",
)

OCCUPATION_EXCLUDE_TERMS = (
    "assembler",
    "butcher",
    "cashier",
    "cleaner",
    "cook",
    "dishwasher",
    "driver",
    "farmworker",
    "fishing",
    "groundskeeper",
    "laborer",
    "maid",
    "mover",
    "operator",
    "packer",
    "painter",
    "roofer",
    "server",
    "stocker",
    "truck",
    "waiter",
    "welder",
)

TASK_INCLUDE_TERMS = (
    "analy",
    "assess",
    "audit",
    "budget",
    "calculate",
    "claim",
    "classif",
    "compare",
    "compliance",
    "contract",
    "coordinate",
    "counsel",
    "curric",
    "data",
    "design",
    "diagnos",
    "document",
    "draft",
    "evaluate",
    "examin",
    "forecast",
    "form",
    "investig",
    "manage",
    "measure",
    "medical record",
    "model",
    "monitor",
    "negot",
    "plan",
    "policy",
    "prepare report",
    "program",
    "proposal",
    "recommend",
    "reconcile",
    "record",
    "regulat",
    "report",
    "research",
    "review",
    "schedule",
    "specification",
    "standard",
    "submission",
    "summar",
    "supervis",
    "survey",
    "teach",
    "test",
    "train",
    "treat",
    "verify",
)

TASK_EXCLUDE_TERMS = (
    "assemble",
    "clean",
    "cook",
    "cut ",
    "deliver",
    "dig",
    "drill",
    "drive",
    "feed",
    "harvest",
    "install",
    "lift",
    "load",
    "mow",
    "pack",
    "paint",
    "plant",
    "pour",
    "repair",
    "scrub",
    "sew",
    "shovel",
    "stock",
    "trim",
    "unload",
    "wash",
    "weld",
    "wire",
)


def read_onet_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        low_memory=False,
        encoding="utf-8",
        encoding_errors="replace",
    )


def clean_task_text(s: pd.Series) -> pd.Series:
    return s.fillna("").str.strip().str.replace(r"\s+", " ", regex=True)


def is_naish(cat: object) -> bool:
    """O*NET encodes missing / not-applicable category as 'n/a' or an empty cell (read as NaN)."""
    if pd.isna(cat):
        return True
    s = str(cat).strip().lower()
    return s in ("n/a", "na", "")


def _normalize_text(*parts: object) -> str:
    text = " ".join(str(part) for part in parts if pd.notna(part))
    return f" {text.lower()} "


def _count_hits(text: str, terms: tuple[str, ...]) -> int:
    return sum(1 for term in terms if term in text)


def classify_occupation(title: object, description: object, job_zone: object) -> str:
    text = _normalize_text(title, description)
    include_hits = _count_hits(text, OCCUPATION_INCLUDE_TERMS)
    exclude_hits = _count_hits(text, OCCUPATION_EXCLUDE_TERMS)
    zone_bonus = 1 if pd.notna(job_zone) and float(job_zone) >= 4 else 0
    score = include_hits + zone_bonus - exclude_hits
    if score >= 2:
        return "include"
    if score <= -2:
        return "exclude"
    return "uncertain"


def classify_task(task_text: object) -> str:
    text = _normalize_text(task_text)
    include_hits = _count_hits(text, TASK_INCLUDE_TERMS)
    exclude_hits = _count_hits(text, TASK_EXCLUDE_TERMS)
    if include_hits >= exclude_hits + 2:
        return "include"
    if exclude_hits >= include_hits + 2:
        return "exclude"
    return "uncertain"


def combine_screen(occupation_screen: str, task_screen: str) -> str:
    if task_screen == "include":
        return "include"
    if task_screen == "exclude":
        return "exclude"
    if occupation_screen == "exclude":
        return "exclude"
    return "include"


def screen_reason(occupation_screen: str, task_screen: str) -> str:
    if task_screen == "exclude":
        return "task text looks primarily hands-on / routine"
    if occupation_screen == "exclude":
        return "occupation context looks primarily non-knowledge / manual"
    if occupation_screen == "include" or task_screen == "include":
        return "knowledge / expert cues present"
    return "borderline case kept for clustering"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--job-zones",
        type=int,
        nargs="+",
        default=[4, 5],
        help="Job zone numbers to keep (default: 4 5). Use e.g. 3 4 5 to add all zone 3.",
    )
    p.add_argument(
        "--disable-knowledge-screen",
        action="store_true",
        help="Skip the inclusive knowledge / expert prescreen and keep all zone-filtered rows.",
    )
    p.add_argument("--write-csv", action="store_true", help="Also write a CSV alongside parquet.")
    args = p.parse_args()
    zone_set = set(args.job_zones)

    ts_path = RAW / "Task_Statements.txt"
    tr_path = RAW / "Task_Ratings.txt"
    occ_path = RAW / "Occupation_Data.txt"
    jz_path = RAW / "Job_Zones.txt"

    for req in (ts_path, tr_path, occ_path, jz_path):
        if not req.exists():
            raise SystemExit(
                f"Missing {req}. Run scripts/download_onet_core.sh first."
            )

    tasks = read_onet_tsv(ts_path)
    tasks = tasks.rename(
        columns={
            "O*NET-SOC Code": "onetsoc_code",
            "Task ID": "task_id",
            "Task": "task_text",
            "Task Type": "task_type",
            "Incumbents Responding": "incumbents_responding",
            "Date": "date_updated",
            "Domain Source": "domain_source",
        }
    )
    tasks["task_id"] = tasks["task_id"].astype(str)
    tasks["task_text_clean"] = clean_task_text(tasks["task_text"])

    occ = read_onet_tsv(occ_path).rename(
        columns={
            "O*NET-SOC Code": "onetsoc_code",
            "Title": "occupation_title",
            "Description": "occupation_description",
        }
    )
    occ = occ[["onetsoc_code", "occupation_title", "occupation_description"]]

    jz = read_onet_tsv(jz_path).rename(
        columns={"O*NET-SOC Code": "onetsoc_code", "Job Zone": "job_zone"}
    )
    jz["job_zone"] = pd.to_numeric(jz["job_zone"], errors="coerce")
    jz = jz[["onetsoc_code", "job_zone"]]

    ratings = read_onet_tsv(tr_path).rename(
        columns={
            "O*NET-SOC Code": "onetsoc_code",
            "Task ID": "task_id",
            "Scale ID": "scale_id",
            "Category": "category",
            "Data Value": "data_value",
        }
    )
    ratings["task_id"] = ratings["task_id"].astype(str)
    ratings["data_value"] = pd.to_numeric(ratings["data_value"], errors="coerce")

    # Importance & relevance: single row per task (category not used for these scales)
    im = ratings[(ratings["scale_id"] == "IM") & (ratings["category"].map(is_naish))][
        ["onetsoc_code", "task_id", "data_value"]
    ].rename(columns={"data_value": "importance_mean"})
    rt = ratings[(ratings["scale_id"] == "RT") & (ratings["category"].map(is_naish))][
        ["onetsoc_code", "task_id", "data_value"]
    ].rename(columns={"data_value": "relevance_pct"})

    # Frequency: percent distribution across categories 1–7
    ft = ratings[ratings["scale_id"] == "FT"].copy()
    ft["cat"] = pd.to_numeric(ft["category"], errors="coerce")
    ft_pivot = ft.pivot_table(
        index=["onetsoc_code", "task_id"],
        columns="cat",
        values="data_value",
        aggfunc="first",
    )
    ft_pivot = ft_pivot.rename(
        columns={i: f"freq_pct_cat_{int(i)}" for i in ft_pivot.columns if pd.notna(i)}
    )
    ft_pivot = ft_pivot.reset_index()

    merged = tasks.merge(occ, on="onetsoc_code", how="left")
    merged = merged.merge(jz, on="onetsoc_code", how="left")
    merged = merged.merge(im, on=["onetsoc_code", "task_id"], how="left")
    merged = merged.merge(rt, on=["onetsoc_code", "task_id"], how="left")
    merged = merged.merge(ft_pivot, on=["onetsoc_code", "task_id"], how="left")

    merged = merged[merged["job_zone"].isin(zone_set)]

    if args.disable_knowledge_screen:
        merged["occupation_screen"] = "include"
        merged["task_screen"] = "include"
        merged["knowledge_screen"] = "include"
        merged["knowledge_screen_reason"] = "screen disabled"
    else:
        merged["occupation_screen"] = [
            classify_occupation(title, desc, zone)
            for title, desc, zone in zip(
                merged["occupation_title"],
                merged["occupation_description"],
                merged["job_zone"],
            )
        ]
        merged["task_screen"] = merged["task_text_clean"].map(classify_task)
        merged["knowledge_screen"] = [
            combine_screen(occ, task)
            for occ, task in zip(merged["occupation_screen"], merged["task_screen"])
        ]
        merged["knowledge_screen_reason"] = [
            screen_reason(occ, task)
            for occ, task in zip(merged["occupation_screen"], merged["task_screen"])
        ]
        keep_mask = merged["knowledge_screen"] == "include"
        print(
            "Knowledge / expert screen:",
            f"kept={int(keep_mask.sum())}",
            f"excluded={int((~keep_mask).sum())}",
        )
        merged = merged[keep_mask].copy()

    DERIVED.mkdir(parents=True, exist_ok=True)
    out_parquet = DERIVED / "onet_tasks_prepared.parquet"
    merged.to_parquet(out_parquet, index=False)
    print(f"Wrote {out_parquet} ({len(merged)} rows)")

    if args.write_csv:
        out_csv = DERIVED / "onet_tasks_prepared.csv"
        merged.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
