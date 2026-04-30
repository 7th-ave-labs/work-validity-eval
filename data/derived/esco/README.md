# ESCO cross-taxonomy robustness artefacts

This directory holds the derived outputs of the ESCO validation study
described in `main.tex` §5 (External-taxonomy robustness: ESCO) and
Appendix C. The goal of the study is to test
whether the 18-operation taxonomy derived from O*NET task statements
is recoverable under a second, independently built occupational
ontology. O*NET stays the derivation dataset; ESCO is used only as a
test bed.

## Freeze metadata

| item | value |
| --- | --- |
| ESCO version | v1.1.1 (English CSV) |
| Source | `https://github.com/tabiya-tech/tabiya-livelihoods-classifier` mirror of the ESCO download package |
| Retrieval script | `scripts/esco/download_esco.sh` |
| Scope filter | `skillType == "skill/competence"` AND linked as `essential` to at least one ISCO-08 occupation in major groups 1, 2, 3 |
| Scoped count | 5,826 items |
| Embedding model | `text-embedding-3-small` (OpenAI), 1,536-d, L2-normalised |
| LLM model (assignment) | `gpt-5.4-mini`, temperature 0, JSON schema output |
| HDBSCAN params | `min_cluster_size=40`, `min_samples=10`, UMAP 15-d cosine, `n_neighbors=30`, seed 42 |
| Taxonomy ref | `data/derived/manual_labeling/operation_taxonomy.json` with merge `assistance -> protocol_execution` applied |
| Freeze date | 2026-04-22 |

## Files

| file | produced by | purpose |
| --- | --- | --- |
| `esco_items_scoped.parquet` | `scripts/esco/prepare_esco_items.py` | 5,826 scoped skill/competence items with ISCO linkage columns |
| `esco_items_scoped.csv` | same | human-readable export |
| `esco_items_knowledge_supplementary.parquet` | same | 1,977 `knowledge`-type items kept for reference; not used in the primary analysis |
| `embeddings/esco_item_embeddings__label_only.npz` | `scripts/esco/embed_esco_items.py` | primary embeddings (preferredLabel only) |
| `embeddings/esco_item_embeddings__label_only_meta.parquet` | same | row-aligned metadata |
| `embeddings/esco_item_embeddings__label_plus_description.npz` | same | sensitivity embeddings (preferredLabel + description) |
| `esco_op_assignments.parquet` | `scripts/esco/map_esco_to_ops.py` | centroid op, LLM op, consensus op, rationale, confidence |
| `esco_op_assignments.csv` | same | human-readable export |
| `esco_disagreement_cases.csv` | same | items where centroid and LLM disagree or LLM said `none_of_the_above` |
| `op_centroid_cosine_reference.json` | same | per-op centroid statistics (p05, p50, p95 within-op O*NET cosines) |
| `esco_llm_cache.jsonl` | same | append-only LLM response cache |
| `esco_hdbscan_clusters.parquet` | `scripts/esco/cluster_esco_items.py` | HDBSCAN labels, UMAP 2-d coords, and the per-item assignment columns from `esco_op_assignments.parquet` |
| `esco_structural_recoverability.json` | same | ARI/NMI against LLM labels (HDBSCAN and KMeans k=18), per-op homogeneity, per-cluster purity, kNN label consistency with permutation null, silhouette |
| `esco_coverage_by_op.csv` | `scripts/esco/build_esco_coverage.py` | per-op counts, centroid-agree pct, confidence breakdown |
| `esco_coverage_summary.json` | same | overall coverage and method-agreement stats |
| `esco_unmapped_sample.csv` | same | the 96 `none_of_the_above` items with centroid context and LLM rationale |

## Reproducing end to end

```
bash   scripts/esco/download_esco.sh
python scripts/esco/prepare_esco_items.py --write-csv --include-knowledge
python scripts/esco/embed_esco_items.py --variant label_only
python scripts/esco/embed_esco_items.py --variant label_plus_description
python scripts/esco/map_esco_to_ops.py --llm-input label_plus_description --llm-workers 16
python scripts/esco/cluster_esco_items.py
python scripts/esco/build_esco_coverage.py
python scripts/esco/build_esco_atlas.py
```

Running order matters only for steps that depend on prior outputs; the
LLM step is resumable via `esco_llm_cache.jsonl`.
