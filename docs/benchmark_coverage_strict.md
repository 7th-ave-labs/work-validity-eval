# Strict benchmark coverage against the pass-7 18-op taxonomy

Companion to `main.tex` §3 (audit of frontier suites under the strict
coverage rule). This file records a **strict** audit of whether
popular LLM / agent benchmarks cover any of the 18 cross-job
operations used by the atlas build. These are
the 19 ops in
`data/derived/manual_labeling/operation_taxonomy.json` with the
singleton `assistance` op merged into `protocol_execution` at the
clustering step (see `scripts/build_operation_atlas.py`
`MERGE_TOP_LEVEL`). Under the merge, the clinical / surgical /
nursing-support cluster (79, `support_execution_of_expert_procedures`)
sits inside `protocol_execution`, which widens the benchmark-coverage
picture for that op.

The earlier pass-6 `coverage_synthesis.md` is a *coverage* artifact
in the weakest sense: "at least one scored unit in this hand-linked
sample targets this op." That is not the same as "this benchmark
*exercises* this operation as it is actually practised across
professions." Downstream work on validity and RL-reward critique
hangs off the stricter claim. This doc records what that stricter
audit actually returns.

## Strict coverage rubric

A benchmark **strictly covers** an operation only if *all* of the
following are true:

1. **Full operation, not a primitive.** The benchmark must test the
   complete input → output of the operation as defined in
   `operation_taxonomy.json` (`input`, `output`, `one_line`). A
   benchmark that exercises only one NLP primitive inside the op
   (e.g. arithmetic reasoning inside `analysis`, single-fact
   retrieval inside `record_keeping`) does not count.
2. **Cross-job coverage.** The benchmark's items must span ≥3
   distinct professional families (same ≥3-family bar used inside the
   taxonomy). A benchmark whose items all come from one profession
   (e.g. SWE-bench = software engineers only; MedQA = physicians
   only) does not strictly cover an op, because the op is defined as
   cross-job and a single-profession benchmark cannot tell you
   whether the system generalises.
3. **Realistic output artifact.** The benchmark's scored output must
   match the op's declared `output` (report / spec / fix / record /
   lesson / artifact) in shape, not just in content. Multiple-choice
   proxies for operations whose real output is a document do not
   strictly cover the op.
4. **Realistic input horizon.** The benchmark must feed the model
   inputs whose length, structure, and noise profile resemble the
   real op's input (raw evidence, a client session, a broken system,
   a corpus, etc.). Synthetic or heavily-pre-digested inputs do not
   strictly cover the op.
5. **Graded on op-relevant criteria.** The scoring rubric must
   measure properties the real op cares about (correctness,
   faithfulness, safety, coverage, auditability, fitness-for-purpose)
   rather than a proxy like ROUGE, BLEU, or single-token accuracy
   when the real op is graded professionally.

A benchmark that satisfies rule 1 but fails rule 2 (single-domain) is
marked **partial / single-domain**. A benchmark that tests a proper
sub-primitive of an op is marked **primitive-only**. A benchmark that
bears only surface similarity is marked **absent**.

## Audit of popular benchmarks under the strict rubric

All entries below should be read as claims about the benchmark's
scope, not about the capability of any model that runs on it.

| benchmark | nearest op | strict verdict | why |
| --- | --- | --- | --- |
| MMLU | `self_study` / `analysis` | **primitive-only** | Exercises knowledge recall + one-step reasoning in multiple-choice form. Neither the `self_study` output (practitioner-current-with-field) nor the `analysis` output (open-ended report / interpretation) is produced. |
| ARC, HellaSwag, TriviaQA | `self_study` | **primitive-only** | Knowledge / commonsense recall. No learning-process is measured, which is what `self_study` actually is. |
| SQuAD, Natural Questions, BoolQ | `record_keeping` | **primitive-only** | Single-passage extractive QA. Covers the "given a relevant record, extract an answer" sub-step of `record_keeping` but not the retrieve-curate-translate-serve loop. |
| KILT / BEIR retrieval suites | `record_keeping` | **primitive-only** | Corpus-level retrieval, the cleanest LLM analogue of `record_keeping`. Still misses translation, curation, and cross-format record maintenance. |
| GSM8K, MATH, AIME, MMLU-STEM | `analysis` | **primitive-only** | Arithmetic / mathematical reasoning. A primitive inside `analysis` (and `appraisal`, `inspection`), not the operation itself. |
| FEVER, FacTool, AVeriTeC | `investigation` | **primitive-only** | Single-claim verification. `investigation` requires reconstructing a past event or uncovering wrongdoing from distributed evidence over time. FEVER-style claim checks are one evidence step inside that op. |
| CNN/DM, XSum, MultiNews | `analysis` | **primitive-only** | Single-source summarisation, ROUGE-graded. `analysis` requires synthesising across evidence and producing an interpretive finding graded by an expert; these benchmarks do neither. |
| HumanEval, MBPP, APPS | `design` | **partial / single-domain** | Covers software-artifact specification (output matches `design.output`: a spec that produces a working artifact). But items are single-profession (software engineers only). |
| BigCodeBench, LiveCodeBench | `design` | **partial / single-domain** | Same verdict as HumanEval; larger / fresher corpora, still single-profession. |
| SWE-bench, SWE-Lancer | `troubleshooting` | **partial / single-domain** | Cleanest existing instance of `troubleshooting` (observed malfunction → diagnose → fix → verified-by-test). But software-only; `troubleshooting` as defined is the same operation in clinical care, mechanical repair, IT, HVAC, and personal care, none of which are represented. |
| AgentBench, Mind2Web, WebArena, VisualWebArena | `coordination` / `administration` | **partial / single-domain** | Web-agent tasks with tool use and multi-step planning. Domain is narrow (consumer web / developer tools). No cross-job coordination coverage. |
| τ-bench | `administration` | **partial / single-domain** | Customer-service retail workflow with policy adherence. A proper fragment of `administration`, one domain. |
| WorkArena, OSWorld | `administration` / `coordination` | **partial / single-domain** | Enterprise-workstation tasks. Same fragment-of-`administration` verdict; desk-work profession slice only. |
| AgentClinic, MedQA clinical vignettes | `troubleshooting` (diagnostic) + `advising` | **partial / single-domain** | Exercises the diagnose-then-recommend loop in one profession (medicine). Domain is realistic but strict cross-job condition fails. |
| MedQA rote, USMLE step exams | `self_study` | **primitive-only** | Recall of clinical knowledge. The operation `self_study` as maintenance-of-competency is not measured; these are snapshots of a practitioner's stock of knowledge. |
| LegalBench (analysis) | `analysis` | **partial / single-domain** | Real legal analytical sub-tasks on real inputs. Strict cross-job fails (law only). |
| LegalBench (contract review tasks) | `inspection` | **partial / single-domain** | Compliance checks against rules; output shape matches `inspection`. Single-profession. |
| ContractNLI, CUAD | `inspection` | **partial / single-domain** | Clause classification / extraction inside a contract; covers one sub-step of the `inspection` loop in one domain. |
| BioASQ, PubMedQA | `analysis` (biomedical) | **partial / single-domain** | Biomedical evidence → finding. Domain-bound. |
| FinQA, TAT-QA | `analysis` (financial) | **partial / single-domain** | Tabular-plus-text numeric analysis. Domain-bound. |
| GAIA | `analysis` + `record_keeping` + tool-use | **partial** | Research-style open-ended tasks requiring retrieval + reasoning + tool use. Cross-domain inputs, but items are puzzle-shaped rather than real-op-shaped; the scored output is a short final answer, not the report / record artifact the ops actually produce. |
| AgentBench (SQL / filesystem / card games) | various | **primitive-only** | Tests tool use and planning in constrained environments, not any knowledge-work op end-to-end. |
| CICERO, Diplomacy, Rampage | `representation` | **partial / single-domain** | Negotiation / dialog in game settings. Touches the negotiation sub-step of `representation`; no real-stakes external-engagement coverage. |
| LMSYS Chatbot Arena | *none* | **absent** | Pairwise preference on free-form chat; not graded on any operation's output criteria. |
| TruthfulQA, HaluEval | *none* (primitive of many ops) | **primitive-only** | Faithfulness / hallucination. A cross-cutting property of most op outputs; not an op. |
| Winograd, WSC, SuperGLUE | *none* | **primitive-only** | Linguistic primitives. |
| Creative-writing evals (e.g. Reddit WritingPrompts, CREATIVE-WRIT) | `creative_production` | **partial / single-domain** | Scored creative writing. Covers one medium; `creative_production` as practiced across visual design, photo/video, edited media is not scored. |
| MusicCaps, AudioCaps | `creative_production` (criticism) | **primitive-only** | Captioning, not producing the artifact. |
| AssistantBench | `administration` | **primitive-only** | Consumer-web assistant tasks. Under the 18-op taxonomy `assistance` is merged into `protocol_execution`, but AssistantBench's actual task shape (help-a-user browse the web) is an administrative assistant primitive, not safety-critical procedure execution. |
| ClassEval | `design` | **partial / single-domain** | Class-level code generation. Single-profession (SWE); sibling of HumanEval / BigCodeBench. |
| HELM-instruct, InstructEval | *mixed* | **primitive-only** | Broad instruction-following evaluations; each scenario is a fragment of some op but no scenario tests a full operation's input/output. |

## What this means for the paper

- **Strict coverage is essentially empty.** Across the 18 operations,
  no benchmark I am aware of satisfies strict coverage (full op *and*
  cross-job *and* realistic artifact *and* realistic input *and*
  op-relevant grading). The most defensible existing instances are
  **partial / single-domain**: SWE-bench for `troubleshooting`,
  HumanEval for `design`, τ-bench for `administration`, LegalBench
  for `analysis` / `inspection`, AgentClinic for the
  diagnose-then-recommend fragment.
- **Several operations have no defensible benchmark at all**, even
  at the partial level: `fabrication`, `emergency_response`,
  `rule_enforcement`, `self_study` (as competency-maintenance rather
  than knowledge recall), `coordination` (as cross-functional
  alignment), `teaching` (as the learner-outcome op, not as Q&A).
  `protocol_execution` is no longer on this list: under the 18-op
  merge it absorbs the clinical-procedure cluster (cluster 79), and
  MedAgentBench's EHR-order / clinical-admin items exercise that
  slice directly — it is **partial / single-domain**, not absent.
- **NLP primitive layer vs. work-op layer.** Most existing benchmarks
  cleanly test NLP / agent *primitives* (retrieve, extract, summarise,
  classify, reason, tool-use). What they do not test is the
  *composition* of primitives into a full operation, graded on
  op-relevant criteria, across a realistic cross-section of
  professions. This is the dark matter the 19-op taxonomy makes
  visible. A natural next step for the paper is a 2-axis figure of
  (18 ops) × (≈10 primitives) with benchmark coverage dots, where
  the dominant pattern is density in the primitive columns and
  sparsity in the op rows.
- **Implication for RL / reward design.** Training signals that
  reward NLP-primitive performance do not rule out drift on the full
  op's grading criteria (e.g. a summariser optimised on ROUGE can
  drift from the evidence-synthesis `analysis.output`; a chatbot
  optimised on pairwise preference can drift from the
  personalised-recommendation criterion in `advising`). Under the
  strict rubric, almost none of the reward signals in standard
  post-training map onto full-op grading, so there is no way to
  diagnose such drift from benchmark scores alone.

## In-depth analysis of the 12 benchmarks in `data/benchmarks.csv`

These are the benchmarks explicitly named in the plan
(`data/benchmarks.csv`, 12 rows). For each, the table records:

- **Op (pass-7)** — nearest op in the 18-op taxonomy (pass-7
  consensus with the `assistance` → `protocol_execution` merge).
- **Scored unit** — what the benchmark actually grades one of.
- **Input horizon** — shape, length, noise profile of each item's input.
- **Produced artifact** — what the model's output looks like when graded.
- **Scoring rubric** — how correctness is decided.
- **Profession families** — how many distinct professional families the
  benchmark's items are drawn from (the ≥3 bar).
- **Strict verdict** — one of *strict-cover*, *partial / single-domain*,
  *primitive-only*, *absent*, per the rubric above.
- **Why** — the specific rule(s) the benchmark fails or satisfies.
- **Closest full-op gap** — what the benchmark would need to add to
  become a strict cover of the nearest op.

| benchmark | op (pass-7) | scored unit | input horizon | produced artifact | scoring rubric | profession families | strict verdict | why | closest full-op gap |
| --- | --- | --- | --- | --- | --- | :---: | --- | --- | --- |
| **LegalBench — issue spotting** | `inspection` | "does rule X apply to fact pattern Y?" | short single-document fact pattern | binary / multi-class label | exact match vs. expert key | 1 (law) | **primitive-only** | A single classification step inside the real `inspection` loop (rule-checking), graded by label match rather than by producing a findings document. | Multi-clause review with cited findings across ≥3 regulated professions (e.g. legal + CMC + clinical-trial protocols), graded by expert-style audit rubric. |
| **LegalBench — rule application** | `analysis` | "apply rule to facts and give the conclusion" | rule + fact pattern | short conclusion | exact match vs. expert key | 1 (law) | **primitive-only** | The IRAC application step in isolation; the real `analysis` op produces a memo / brief synthesising multiple sources and scored on coverage + faithfulness. | Open-ended legal memo output graded by experts on IRAC completeness, not a single exact-match conclusion. |
| **LegalBench — rhetorical analysis** | `analysis` | sentence / clause role in a legal text | single excerpt | span / sentence label | exact-match accuracy | 1 (law) | **primitive-only** | Linguistic classification of clause function; the full `analysis` op produces an interpretive document, not per-span labels. | Same as above — scored artifact must be the interpretive document. |
| **FinanceBench (core)** | `analysis` | "answer a question from this filing" | long 10-K / 10-Q / earnings filing | short answer + citation | answer correctness vs. human-annotated + citation check | 1 (finance / accounting) | **partial / single-domain** | Inputs realistic in length and noise; citation-grounded QA meaningfully exercises the "evidence → finding" loop for `analysis`. But single-domain (finance filings only) and scored unit is a short answer, not a full analytical memo. | Broaden to ≥3 evidence domains (finance + legal + scientific) and score analytical memos against expert rubrics, not short spans. |
| **BrowseComp** | `record_keeping` → `analysis` | "find and return the verified answer to a hard factual question" | open web via browser tool | single short answer | exact / string match vs. reference | mixed (questions are open-domain, corpus is the web; not a specific profession) | **primitive-only** | Tests retrieval-plus-verification for a single fact. The real `record_keeping` op produces a retrieved, curated, cited record that downstream work can use. | Require production of a citable, verifiable record artifact (retrieved passage + provenance) rather than a single token answer. |
| **GAIA** | `analysis` + `record_keeping` | multi-step tool-using task with one canonical final answer | web + files + images (heterogeneous) | single final-answer string | exact string match | mixed (puzzle / assistant tasks, not a profession cross-section) | **partial / primitive** | Genuinely end-to-end in *tool composition*, but the scored output is a final-answer string. The ops' real artifacts (report, record, decision memo) are not produced. | Require a delivered artifact (memo, record, briefing) scored by an expert rubric, with items drawn from ≥3 real professional workflows. |
| **WebArena (core)** | `administration` / `coordination` | "drive a browser to change the goal state of a sandboxed web app" | sandboxed consumer web apps (shopping, git, maps, CMS, reddit) | end-state of the web app | programmatic state check | consumer web / developer only | **partial / single-domain** | Realistically tests tool-using administrative workflows and is scored on artifact state, which matches `administration.output`. But items are consumer-web / dev-tools, not cross-sector knowledge-worker administration (HR, finance, procurement, clinical ops, legal ops, academic ops). | Add items from ≥3 enterprise functions (HR, finance, compliance, clinical) with state checks and auditable change logs. |
| **WorkArena (core)** | `administration` | "complete enterprise workflow tasks in ServiceNow" | ServiceNow sandbox | ticket / record end-state | programmatic goal state | 1 (ITSM workflows) | **partial / single-domain** | Honest slice of `administration` in one enterprise platform. Scored on state change, which matches the op's output shape. Single-domain (ITSM); does not cover HR / finance / procurement / compliance administration. | Multi-platform coverage of ≥3 enterprise functions with cross-function workflows. |
| **OSWorld (core)** | `administration` / `coordination` | "complete a desktop task with mouse + keyboard across apps" | full desktop OS | goal-state file / app state | programmatic checks | mixed (office worker tasks across apps) | **partial / primitive** | Tests computer-use primitive (GUI + file-system) more than any full work op. Items are heterogeneous but mostly office fragments. | Re-frame as end-to-end office workflows (prepare + send + archive a document package) scored on the delivered artifact, cross-sector. |
| **OfficeBench (core)** | `record_keeping` → `creative_production` (for slides) | "produce / edit a document to match a specification" | office files (Word / Excel / PPT) | target document | document-state diff vs. reference artifact | mixed (generic office) | **partial / primitive** | Exercises the artifact-production primitive inside several ops (records, reports, slides). Scored on document-state match to a reference — this is strict on primitive, but each item is a narrow spec rather than a real op ("produce a quarterly findings report across sources"). | Upgrade items from mechanical diffs to professionally-graded document outputs on realistic inputs, spanning ≥3 professions. |
| **MedAgentBench (core)** | `administration` (clinical ops) + `protocol_execution` (clinical procedures) | "complete a clinical admin / decision-support task using FHIR tools" | simulated EHR + FHIR API | resulting FHIR resources (orders, notes, referrals) | programmatic checks on resulting resources | 1 (medicine / clinical ops) | **partial / single-domain** | Very close to a strict-cover for clinical-ops `administration` *and* the clinical slice of `protocol_execution` (under the 18-op merge, cluster 79's order-placement / procedure-prep tasks — IV catheter placement, ventilator setup, suturing workflows — are in-scope). Input horizon is realistic and the scored artifact is the produced record. Fails cross-job because it's medicine-only. | Port the same artifact-scoring framework to ≥2 other professions' record systems (legal docket, financial ledger, academic registrar) with comparable tool surfaces. |
| **APEX-SWE (core)** | `troubleshooting` + `design` | "implement a change in a repo that passes tests + rubric" | real repo + issue + test environment | code change (diff) | hidden tests + rubric review | 1 (software engineering) | **partial / single-domain** | Strongest existing instance of end-to-end `troubleshooting` (observe malfunction → diagnose → fix → verify). Rubric layer moves it past pure test-pass. Single-profession. | Apply the same diagnose-then-fix format to ≥2 other troubleshooting domains (mechanical repair sims, clinical differential + treatment, IT incident response) with analogous ground-truth validation. |

### Roll-up of the 12 scored units (10 benchmark families)

- **Strict-cover (all 5 rules satisfied):** 0 of 12.
- **Partial / single-domain** (rule 1 satisfied, rule 2 fails): 6 of 12 — FinanceBench, WebArena, WorkArena, MedAgentBench, APEX-SWE, LegalBench (as a whole collection, if you grade the suite rather than the individual tasks).
- **Partial / primitive** (rule 1 partially, primitive-heavy): 3 of 12 — GAIA, OSWorld, OfficeBench.
- **Primitive-only** (rule 1 fails — benchmark tests a sub-step of the op): 3 of 12 — LegalBench issue-spotting, LegalBench rhetorical analysis, BrowseComp.
- **Absent** (no meaningful op match): 0 of 12.

### What these 12 scored units do NOT touch

No benchmark in the plan targets any of the following operations:
`advising` (1:1 personalised recommendation to a client),
`teaching` (learner-outcome),
`appraisal` (rubric-graded scoring of a submission),
`investigation` (reconstruct a past event / uncover wrongdoing across distributed evidence),
`fabrication`, `emergency_response`, `rule_enforcement`, `self_study`
(as competency-maintenance),
`coordination` (as cross-functional alignment outside the WebArena / WorkArena slice),
`representation` (external engagement with real stakes),
`creative_production` (original artistic output).

`design` is partially covered by APEX-SWE / ClassEval (SWE-only) and
`protocol_execution` is partially covered by MedAgentBench
(clinical-only, under the 18-op merge); neither reaches strict coverage.

That is **10 of 18 operations with no benchmark in the plan** at any
level, plus a further 2 (`design`, `protocol_execution`) covered only
in a single profession. These are the directions where new benchmarks
are most needed if the taxonomy is to be turned into a validity
instrument. The code-review benchmarks highlighted in the position
paper draft (SWE-PRBench, CodeFuse-CR-Bench,
withmartian/code-review-benchmark) are the cleanest existing
candidates for `appraisal` and `inspection` — still SWE-only, but
their scored output is "findings on someone else's work" rather than
"author a correct patch", which matches the op shape.

## Relation to `data/labels.csv` and `docs/coverage_synthesis.md`

The existing `data/labels.csv` records 15 hand-linked (benchmark, op)
pairs under the pass-6 24-op taxonomy. These links are
*coverage-weak*: they assert that at least one scored unit in each
sample targets the op. They are explicitly not strict-coverage
claims. Under the strict rubric above, most of those same links
reduce to **partial / single-domain** or **primitive-only**. Before
any validity claim is made, each hand-linked pair needs to be
re-audited with the rubric in this file, and the pass-6 24-op labels
need to be re-aligned to the pass-7 18-op operations (with the
`assistance` → `protocol_execution` merge applied).
