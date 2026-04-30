#!/usr/bin/env python3
"""Scope ESCO skills/competences to knowledge-work and write the item table.

Reads ``data/raw/esco/*.csv`` (produced by ``scripts/esco/download_esco.sh``)
and writes the scoped ESCO item table used by the downstream embedding and
classification steps:

    data/derived/esco/esco_items_scoped.parquet

Scope (ESCO analogue of the O*NET Job Zone 3-5 filter on the derivation side):

1. ``skills_en.csv`` rows with ``skillType == "skill/competence"``.
   ``knowledge`` rows describe subject-matter domains (anatomy, metallurgy,
   case law) rather than operations a worker performs; they sit at a different
   construct level and are excluded from the primary pass. A supplementary
   parquet with ``skillType == "knowledge"`` is also written for reference.
2. The skill appears as ``essential`` for at least one occupation whose
   ``iscoGroup`` begins with ``1``, ``2``, or ``3`` (Managers, Professionals,
   Technicians and Associate Professionals) per ISCO-08.

The output columns are stable and documented so downstream embedding /
classification scripts can consume the parquet without reading the raw CSVs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "esco"
DERIVED = ROOT / "data" / "derived" / "esco"

KNOWLEDGE_WORK_ISCO_PREFIX = ("1", "2", "3")

WS_RE = re.compile(r"\s+")


def clean_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().map(lambda t: WS_RE.sub(" ", t))


def build_scope(
    skills: pd.DataFrame,
    occupations: pd.DataFrame,
    rel: pd.DataFrame,
    isco_prefix: tuple[str, ...] = KNOWLEDGE_WORK_ISCO_PREFIX,
) -> pd.DataFrame:
    """Return a per-skill frame with knowledge-work essential-link counts."""

    occ_kw = occupations[
        occupations["iscoGroup"].astype(str).str.startswith(isco_prefix)
    ][["conceptUri", "iscoGroup", "preferredLabel"]].rename(
        columns={
            "conceptUri": "occupationUri",
            "iscoGroup": "occupation_isco_group",
            "preferredLabel": "occupation_label",
        }
    )

    essential = rel[rel["relationType"] == "essential"][
        ["occupationUri", "skillUri"]
    ]
    essential_kw = essential.merge(occ_kw, on="occupationUri", how="inner")

    per_skill = (
        essential_kw.groupby("skillUri")
        .agg(
            n_occupations_essential=("occupationUri", "nunique"),
            isco_major_groups=(
                "occupation_isco_group",
                lambda s: sorted({str(x)[0] for x in s if str(x)}),
            ),
            anchor_occupations=(
                "occupation_label",
                lambda s: "; ".join(sorted(set(s))[:3]),
            ),
        )
        .reset_index()
    )
    per_skill["isco_major_groups"] = per_skill["isco_major_groups"].map(
        lambda xs: ",".join(xs)
    )
    return per_skill


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--min-occupations",
        type=int,
        default=1,
        help=(
            "Keep skills linked as essential to at least this many"
            " knowledge-work occupations (default 1)."
        ),
    )
    p.add_argument("--write-csv", action="store_true")
    p.add_argument(
        "--include-knowledge",
        action="store_true",
        help="Also write a supplementary parquet of knowledge-type items.",
    )
    args = p.parse_args()

    DERIVED.mkdir(parents=True, exist_ok=True)

    skills = pd.read_csv(RAW / "skills_en.csv", low_memory=False)
    occupations = pd.read_csv(RAW / "occupations_en.csv", low_memory=False)
    rel = pd.read_csv(RAW / "occupationSkillRelations.csv", low_memory=False)

    for col in ("preferredLabel", "description", "altLabels"):
        if col in skills.columns:
            skills[col] = clean_text(skills[col])
    skills["skillType"] = skills["skillType"].fillna("").str.strip()
    skills["reuseLevel"] = skills["reuseLevel"].fillna("").str.strip()

    per_skill_kw = build_scope(skills, occupations, rel)

    scoped = skills.merge(
        per_skill_kw, left_on="conceptUri", right_on="skillUri", how="left"
    )
    scoped["n_occupations_essential"] = scoped["n_occupations_essential"].fillna(0).astype(int)
    scoped["isco_major_groups"] = scoped["isco_major_groups"].fillna("")
    scoped["anchor_occupations"] = scoped["anchor_occupations"].fillna("")

    primary = scoped[
        (scoped["skillType"] == "skill/competence")
        & (scoped["n_occupations_essential"] >= args.min_occupations)
    ].copy()

    primary = primary[
        [
            "conceptUri",
            "preferredLabel",
            "altLabels",
            "description",
            "reuseLevel",
            "skillType",
            "n_occupations_essential",
            "isco_major_groups",
            "anchor_occupations",
        ]
    ].rename(columns={"conceptUri": "esco_uri", "preferredLabel": "preferred_label"})
    primary = primary.drop_duplicates("esco_uri").reset_index(drop=True)

    out_parquet = DERIVED / "esco_items_scoped.parquet"
    primary.to_parquet(out_parquet, index=False)
    print(f"Wrote {out_parquet} ({len(primary)} skill/competence items)")
    if args.write_csv:
        out_csv = DERIVED / "esco_items_scoped.csv"
        primary.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")

    if args.include_knowledge:
        knowledge = scoped[
            (scoped["skillType"] == "knowledge")
            & (scoped["n_occupations_essential"] >= args.min_occupations)
        ].copy()
        knowledge = knowledge[
            [
                "conceptUri",
                "preferredLabel",
                "altLabels",
                "description",
                "reuseLevel",
                "skillType",
                "n_occupations_essential",
                "isco_major_groups",
                "anchor_occupations",
            ]
        ].rename(columns={"conceptUri": "esco_uri", "preferredLabel": "preferred_label"})
        knowledge = knowledge.drop_duplicates("esco_uri").reset_index(drop=True)
        out_k = DERIVED / "esco_items_knowledge_supplementary.parquet"
        knowledge.to_parquet(out_k, index=False)
        print(f"Wrote {out_k} ({len(knowledge)} knowledge items)")

    breakdown = (
        primary.groupby("isco_major_groups")
        .size()
        .reset_index(name="n_items")
        .sort_values("n_items", ascending=False)
    )
    print("\nISCO major-group distribution of scoped skill/competence items:")
    print(breakdown.to_string(index=False))


if __name__ == "__main__":
    main()
