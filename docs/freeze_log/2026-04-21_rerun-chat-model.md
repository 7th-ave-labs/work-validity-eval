# Experiment: 2026-04-21_rerun-chat-model

## Purpose

Full rerun of the chat-model-dependent pipeline after the latest
pipeline additions (commit `459dc84`: knowledge prescreen, named
embedding variants, PCA+HDBSCAN clustering, cluster review export).
Previous embeddings are preserved for sensitivity analysis;
only the chat model passes (cluster labels, task coding, pilot
direct-coding sensitivity check) and downstream artifacts were
regenerated from scratch.

## Commands (exact)

```bash
export OPENAI_API_KEY=...

# prepare (now applies inclusive knowledge/expert prescreen)
python3 scripts/prepare_onet_tasks.py --job-zones 3 4 5 --write-csv

# embeddings
#  - title_plus_task: filtered from the legacy 12,611-row embedding matrix
#    to the 12,464-row prescreened set (bit-identical vectors); moved to
#    data/derived/embeddings/onet_task_embeddings__title_plus_task.*
#  - task_only: freshly generated (same embedding model)
python3 scripts/embed_onet_tasks.py --variant task_only

# clustering + cluster review (updated recipe, see "Clustering recipe update" below)
python3 scripts/cluster_onet_tasks.py --variant task_only
python3 scripts/cluster_onet_tasks.py --variant title_plus_task
python3 scripts/export_cluster_review.py

# auto-labels v2 (unsupervised, sensitivity; frozen taxonomy unchanged)
python3 scripts/auto_label_v2.py --variant task_only
python3 scripts/auto_label_v2.py --variant title_plus_task
python3 scripts/build_label_crosstab.py --variant task_only
python3 scripts/build_label_crosstab.py --variant title_plus_task

# chat-model rerun (full recode + downstream)
bash scripts/rerun_after_chat_model_change.sh

# sensitivity pilot (direct LLM coding, no taxonomy)
python3 scripts/pilot_llm_direct_coding.py
```

## Inputs

- O*NET release / download script: O*NET 30.2 (`scripts/download_onet_core.sh`,
  `db_30_2_text`); raw extracts already present under `data/raw/onet/`.
- Prepared tasks: `data/derived/onet_tasks_prepared.parquet`
  (12,464 rows; 147 rows excluded by the inclusive
  knowledge / expert prescreen out of 12,611 zone 3/4/5 rows).
- Embeddings:
  - `data/derived/embeddings/onet_task_embeddings__task_only.npz` (new default)
  - `data/derived/embeddings/onet_task_embeddings__title_plus_task.npz`
    (legacy template; vectors preserved from the previous run,
    re-indexed to the prescreened set for alignment)
- Clustering: `scripts/cluster_onet_tasks.py` with the updated recipe
  (see *Clustering recipe update* below). Original first-pass numbers
  here were with a `PCA → HDBSCAN` recipe that produced 14 clusters /
  88% noise on the `task_only` variant — degenerate as a sampling
  frame for the proposer. Recipe was replaced mid-rerun.

## Clustering recipe update (2026-04-21, second pass)

The first-pass `PCA → HDBSCAN` recipe collapsed `task_only` embeddings
into 14 clusters / 11,013 noise (88.4%). Root cause: PCA on 1536-d
L2-normalised text embeddings concentrates pairwise distances and
gives HDBSCAN very little contrast; the variant switch from
`title_plus_task` to operation-first `task_only` makes this worse
because occupational identity no longer provides extra anchoring.

Replaced with:

1. L2-normalise the selected embedding variant.
2. **UMAP** to 15 dims, `metric='cosine'`, `n_neighbors=30`,
   `min_dist=0.0`, seed 42 — the cluster space.
3. **HDBSCAN** on that UMAP space with `min_cluster_size=30`,
   `min_samples=10`, `cluster_selection_method='eom'`,
   `prediction_data=True`.
4. **Soft-assign noise** via `hdbscan.all_points_membership_vectors`:
   every task gets a `hdbscan_cluster_assigned` so downstream samplers
   / proposer / review have full corpus coverage.
5. **KMeans(k=40)** on the same UMAP space as a stability cross-check;
   report ARI / NMI vs HDBSCAN on non-noise points.
6. Per-variant outputs (both written; `task_only` is primary and also
   mirrored to the canonical unsuffixed paths for back-compat):
   - `data/derived/onet_task_clusters__<variant>.parquet`
   - `data/derived/cluster_summary__<variant>.csv`
   - `data/derived/cluster_stability__<variant>.json`
   - `data/derived/figs/{pca,umap,tsne}_clusters__<variant>.png`

Results under the new recipe:

| variant            | HDBSCAN clusters | noise (hard) | soft-assigned | ARI vs KMeans | NMI |
| ------------------ | ---------------- | ------------ | ------------- | ------------- | --- |
| `task_only`        | 108              | 4,487 (36.0%) | all           | 0.460         | 0.822 |
| `title_plus_task`  | 183              | 1,581 (12.7%) | all           | 0.319         | 0.797 |

`task_only` remains the primary clustering used by the proposer /
review step; `title_plus_task` is kept as a sensitivity view (it
carves clusters more occupationally, e.g. "Teach and develop courses"
dominating many frozen buckets because the occupation token leaks
into the embedding).

## Auto-label v2 (unsupervised-vs-frozen sensitivity)

New scripts:

- `scripts/auto_label_v2.py` — projects the LLM-proposed cluster
  label from `cluster_onet_tasks.py` (`cluster_label`,
  `cluster_summary`) onto every task using
  `hdbscan_cluster_assigned`, so every one of the 12,464 tasks has
  an operation-first auto-label independent of the frozen 28-label
  taxonomy. Output:
  `data/derived/auto_labels_v2__<variant>.csv`
  (unique auto-labels: 108 for `task_only`, 183 for `title_plus_task`).
- `scripts/build_label_crosstab.py` — joins `onet_tasks_coded.parquet`
  (frozen `primary_label_id`) with `auto_labels_v2__<variant>.csv`
  and writes:
  - `data/derived/label_crosstab__<variant>.csv` — long-format
    (frozen_label × auto_label) with `n_tasks`, `n_occupations`,
    `salience_weighted_n`, `mean_importance`, `share_of_frozen_label`.
  - `data/derived/label_crosstab__<variant>__diag.csv` — per frozen
    label: `n_tasks`, `n_auto_labels`, `dominant_auto_label`,
    `dominant_share`, `top2_auto_labels`.

**Important:** the 28-label taxonomy stays frozen and remains the
primary coding target. `auto_labels_v2` is sensitivity only — it
shows where the unsupervised pass over `task_only` agrees or
disagrees with the frozen taxonomy, and surfaces candidate
sub-structure (e.g., `consult_and_advise` fans out over 88 auto
labels; `negotiate_or_mediate` is the cleanest match at 45% dominant
share).

## Outputs (paths)

- Clusters / figures (per variant; `task_only` mirrored to the
  canonical unsuffixed paths):
  - `data/derived/onet_task_clusters__<variant>.parquet`
  - `data/derived/cluster_summary__<variant>.csv`
  - `data/derived/cluster_stability__<variant>.json`
  - `data/derived/figs/pca_clusters__<variant>.png`
  - `data/derived/figs/umap_clusters__<variant>.png`
  - `data/derived/figs/tsne_clusters__<variant>.png`
- Auto-labels v2 sensitivity:
  - `data/derived/auto_labels_v2__<variant>.csv`
  - `data/derived/label_crosstab__<variant>.csv`
  - `data/derived/label_crosstab__<variant>__diag.csv`
- Cluster review CSV: `data/derived/cluster_review_by_centroid.csv`
  (local review artifact; not committed).
- Coded tasks: `data/derived/onet_tasks_coded.parquet` (+ `.csv`),
  12,464 rows coded under the frozen 28-label taxonomy.
- Coverage + comparison (pre-pass-6 snapshot): `data/derived/coverage_by_label.csv`,
  `data/onet_benchmark_comparison.csv`, `data/derived/figs/coverage_vs_salience.png`,
  `docs/coverage_validity_synthesis.md` (auto-generated rollup).
  - **Post-pass-6 note (2026-04-21):** `data/onet_benchmark_comparison.csv`
    and `docs/coverage_validity_synthesis.md` have been **removed** (see the
    pass-6 section in `docs/research_paper.md` §3). The per-(benchmark, op)
    `PRESERVED` dict was written from benchmark-level literature-review
    summaries, not from reading actual benchmark items or rubric/scoring
    code, and was demoted on Levi's feedback that validity claims need
    item-level grounding. Item-level benchmark validity is a follow-up PR.
    Current coverage (bench-level only, no validity claims) is in
    `docs/coverage_synthesis.md`.
- Sensitivity pilot: `data/derived/pilot_llm_direct_coding.csv`
  (drift rate on 145-task stratified sample: 1.4%).

## Operation-first normalization pass (2026-04-21, third pass)

Spot-check of nearest neighbours in the `task_only` embedding space
showed the geometry was still domain-anchored, not operation-anchored:
*"Design or develop vacuum tube collector systems for solar
applications"* sits entirely among solar / wind / photonics tasks;
*"Debug robotics programs"* stays inside robotics and mechatronics;
*"Develop treatment plans"* stays inside mental-health / clinical
specialties. text-embedding-3-small was latching onto the
profession / industry / product nouns in the task statements.

Fix: introduce a new embedding variant `operation_only` whose input
is a domain-stripped operation phrase produced by
`scripts/normalize_task_operations.py`. For each of the 12,464
prescreened tasks, `gpt-5.4-mini` (temperature 0.0, JSON mode)
rewrites the task into `<verb> <generic object> [<generic modifier>]`
with no profession, industry, product, or domain nouns (see
prompt in the script). Examples:

| source | normalized |
| ------ | ---------- |
| "Debug robotics programs." | "debug software" |
| "Design robotic systems, such as automatic vehicle control, autonomous vehicles…" | "design hardware system" |
| "Design or develop vacuum tube collector systems for solar applications." | "design hardware system" |
| "Synthesize, process, or characterize nanomaterials, using advanced tools or techniques." | "synthesize and characterize a material" |
| "Prepare and deliver lectures to undergraduate or graduate students on assigned topics." | "prepare and deliver instruction to learners" |
| "Maintain student attendance records, grades, and other required records." | "maintain records" |

Alongside each normalized phrase the script stores
`operation_verbs`, `operation_object_type` (one of the 14-enum
{code, hardware, material, document, plan, data, image_or_media,
person, organization, environment, money, schedule, procedure,
other}), and `domain_stripped` (for inspection). Output:
`data/derived/task_operations_normalized.jsonl` (append-only cache)
and `data/derived/task_operations_normalized.parquet` (12,464 rows;
10,711 unique phrases — 14% exact collapse).

Pipeline rerun on the new variant:

```bash
python3 scripts/normalize_task_operations.py --workers 30
python3 scripts/embed_onet_tasks.py --variant operation_only
python3 scripts/cluster_onet_tasks.py --variant operation_only
python3 scripts/refresh_work_patterns.py --variant operation_only --workers 20
python3 scripts/consolidate_work_patterns.py --variant operation_only --target 28
```

Cluster geometry on `operation_only` vs the domain-contaminated
`task_only` run:

| metric | task_only (domain-contam.) | operation_only (normalized) |
| ------ | -------------------------- | --------------------------- |
| hard HDBSCAN clusters | 108 | **120** |
| noise (hard) | 4,487 (36.0%) | **3,922 (31.4%)** |
| min n_occupations per cluster | 3 | **10** |
| median n_occupations per cluster | 34.5 | **46** |
| clusters with <5 occupations | 2 (nanotech, robotics) | **0** |
| clusters with ≥30 occupations | 62 / 108 (57%) | **95 / 120 (79%)** |
| max n_occupations in a single cluster | 115 | **180** |

The narrow-domain artifacts disappeared: the old clusters
"Develop nanoscale systems" (3 occupations — nanosystems engineers
only) and "Build robotic systems" (3 occupations — robotics engineers
only) both dissolve into broad `design hardware system` / `develop
technical specifications` clusters that now span 37-44 occupations
(photonics, mechatronics, avionics, nuclear, solar, wind, water,
mechanical, microsystems, robotics, nano-, computer architects, etc.)
doing the same operation on different kinds of hardware.

Refreshed taxonomy (`data/derived/work_pattern_labels_refresh__operation_only.json`,
28 labels; same embedding-driven agglomerative-then-describe
consolidation as the 2026-04-21 second pass, but on the 120
operation-only clusters):

| label_id | n_tasks | n_clusters | max_occ |
| -------- | ------: | ---------: | ------: |
| develop_train_and_document_procedures | 1,694 | 23 | 93 |
| monitor_and_review_for_compliance_or_progress | 1,054 | 10 | 172 |
| develop_and_deliver_instruction | 734 | 13 | 72 |
| conduct_research_and_analyze_data | 568 | 6 | 108 |
| coordinate_and_confer_with_stakeholders | 563 | 6 | 180 |
| manage_administrative_operations | 494 | 6 | 129 |
| advise_or_counsel_people | 414 | 8 | 63 |
| provide_assistive_devices_and_support | 387 | 6 | 60 |
| assess_risk_or_impact | 316 | 4 | 59 |
| participate_in_meetings_and_committees | 238 | 5 | 58 |
| select_and_obtain_materials | 204 | 2 | 118 |
| perform_clerical_and_financial_office_duties | 187 | 2 | 59 |
| respond_to_emergencies | 172 | 3 | 56 |
| interview_to_gather_information | 157 | 2 | 70 |
| maintain_professional_knowledge | 142 | 3 | 62 |
| present_evidence_in_legal_proceedings | 123 | 1 | 54 |
| interpret_diagnostic_test_results | 119 | 2 | 61 |
| edit_and_assemble_media_and_images | 116 | 2 | 33 |
| analyze_and_design_energy_efficiency | 113 | 2 | 37 |
| improve_operational_processes | 112 | 2 | 55 |
| prepare_and_publish_information | 102 | 2 | 61 |
| direct_rehearsals_and_performances | 95 | 1 | 19 |
| estimate_project_costs | 92 | 1 | 73 |
| analyze_performance_data | 90 | 2 | 47 |
| write_grant_proposals | 89 | 2 | 39 |
| compile_reference_materials | 71 | 2 | 36 |
| compile_and_map_geospatial_data | 49 | 1 | 22 |
| measure_ocular_function | 47 | 1 | 10 |

**Status.** The refreshed 28 labels are still awaiting human review
before promotion. `data/work_pattern_labels.csv` and
`scripts/finalize_work_patterns.py` are **unchanged** at commit time;
`onet_tasks_coded.parquet` and downstream coverage / comparison
artifacts are still under the original frozen taxonomy. Promotion
(rewriting `finalize_work_patterns.py` and rerunning
`code_onet_tasks.py`) is deferred until the refreshed labels are
approved.

## Cross-job operation constraint (2026-04-22 follow-up)

Levi flagged that the promoted 24-label taxonomy still contained
several **job-specific** operations — i.e. top-level operations that
only one profession can do — e.g. `prescribe_treatments_and_medications`
(physicians only), `present_evidence_in_hearings` (lawyers only),
`compile_subject_bibliographies` (librarians only),
`facilitate_group_discussions` (counselors only), `assess_and_grade_work`
(teachers only), `conduct_scientific_research` (scientists only). These
survived the agglomerative merge because the sub-op clusters feeding
them were already occupation-homogeneous at the 0.55 cosine-distance
threshold, and the LLM describe-step happily labelled the single sub-op
in its native profession's jargon.

Fix applied to the merge + LLM-labeling code (no data regeneration in
this PR; rerun the pipeline to realise the effect):

1. `scripts/build_operation_hierarchy.py`:
   - New **cross-job absorption pass** after agglomerative clustering.
     Loads `onet_task_clusters__<variant>.parquet`, computes the
     pooled unique `onetsoc_code` set across each bucket's member
     sub-op clusters, and iteratively absorbs any bucket whose pooled
     occupation count falls below `--min-pooled-occupations` (default
     **10**) into its nearest neighbour bucket by theme-centroid cosine
     distance. Buckets with fewer than `--min-sub-ops` (default **2**)
     are absorbed too, unless pooled occupations already clearly exceed
     `2 * min_pooled_occupations` (a genuinely cross-occupational lone
     sub-op is allowed to stand). Runs BEFORE the LLM describe step so
     single-occupation buckets never reach the LLM.
   - Absorption log includes pre/post `pooled_occs` and `sub_ops`
     min/median/max for auditability.
   - Bucket's `pooled_occupations` set is also emitted on the LLM
     describe prompt and on every hierarchy record
     (`n_pooled_occupations`, `pooled_occupations`, and in the tree
     summary) so downstream review can sort by cross-job breadth.
   - `DESCRIBE_SYSTEM_PROMPT` extended with a HARD REQUIREMENT that
     the top-level operation must be cross-job, plus concrete BAD
     (job-specific) vs. GOOD (cross-job) examples and an "abstract
     UP" rule.

2. `scripts/refresh_work_patterns.py`:
   - Per-cluster `SYSTEM_PROMPT` now requires the `theme_label` to be
     a cross-job operation at an abstraction level that applies to at
     least 3-5 other O*NET occupations, with BAD/GOOD examples. The
     previous "should plausibly apply" soft guidance is replaced by
     an explicit "abstract up so the operation names the WORK, not
     the worker" rule.

3. `scripts/consolidate_work_patterns.py`:
   - `DESCRIBE_SYSTEM_PROMPT` upgraded in parallel to
     `build_operation_hierarchy.py` (hard cross-job requirement,
     BAD/GOOD examples, "abstract UP" rule).

Re-run command (expected next pipeline pass):

```bash
python3 scripts/refresh_work_patterns.py --variant operation_only --workers 20
python3 scripts/build_operation_hierarchy.py --variant operation_only \
    --tag-suffix _r2 --threshold 0.55 \
    --min-pooled-occupations 10 --min-sub-ops 2
python3 scripts/finalize_work_patterns.py
```

The absorption pass is idempotent and conservative: disabling it
reduces to the old behaviour (`--min-pooled-occupations 0
--min-sub-ops 1`).

## Notes

- API model ids / snapshot ids:
  - Embedding: `text-embedding-3-small`.
  - Chat (coding, cluster labels, pilot): `gpt-5.4-mini` (default);
    pin an exact provider snapshot here if a result feeds the paper.
  - Temperature 0.0 for coding / cluster labels / pilot; JSON-mode responses.
- Taxonomy (28 work-pattern labels in `data/work_pattern_labels.csv`)
  and benchmark list (`data/benchmarks.csv`, 12 scored units from 10 sources)
  are **unchanged** and still covered by the 2026-04-20 freeze note.
- Previous `onet_tasks_coded.parquet` backed up locally by the rerun
  script as `onet_tasks_coded.parquet.bak_20260421_172103` (not committed;
  regenerate via the rerun script if needed).
- The legacy unversioned embedding paths
  (`data/derived/onet_task_embeddings.npz` and `..._meta.parquet`)
  were superseded by the variant-suffixed paths under
  `data/derived/embeddings/` and are removed from version control.
- Anything that did not reproduce bit-for-bit: cluster assignments,
  LLM cluster labels, and LLM task codings vary across chat-model runs
  (non-deterministic API); the pipeline parameters and frozen taxonomy
  are deterministic.
