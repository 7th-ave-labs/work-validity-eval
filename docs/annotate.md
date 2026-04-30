# Annotation codebook

Use this document to define labels, inclusion rules, edge cases, and adjudication so annotations stay comparable over time.

**Project plan:** the paper (`main.tex`, especially §3 and Appendix A) defines benchmark/O\*NET **freeze** timing, claim scope (O\*NET task text vs. full jobs; benchmark **sample** vs. all evaluations), optional **duplicate coding** on a slice of O\*NET rows, and **anchor examples** for validity arguments. Keep adjudication rules here so they stay stable after the freeze.

## Label definitions

- **label_value**: What each allowed value means (include examples of borderline cases).

## Task text (`benchmarks.csv`)

- How `task_text` is produced and any preprocessing (truncation, language, etc.).

## Provenance

- **source**: Where the benchmark row came from (e.g. internal run, public benchmark name).
- **source_ref**: Stable pointer (URL, dataset version, file hash) when applicable.

## Third-party data

When rows draw on **ONET**, **ESCO**, or other catalogs, copy their license/attribution requirements into release notes or a `data/NOTICE` file and keep `source_ref` specific enough to audit.
