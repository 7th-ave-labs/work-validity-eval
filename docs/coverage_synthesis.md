# Benchmark coverage summary (auto-generated)

**Scope.** This document reports *coverage only*: for each of the 24
promoted operations, how many scored units in the benchmark sample
(`data/benchmarks.csv`) have been hand-linked to this operation in
`data/labels.csv` (source: `manual-literature-review`, bench-level).

**Not in scope.** Item-level validity — i.e., whether a scored unit's
*scoring* actually tests the operation's `minimum_task_contract` on
its items — is **not** produced here. That requires pulling benchmark
items + rubric/scoring code and coding each item against the contract,
which is the next PR. Treat coverage tiers below as an upper bound on
what the sample claims to measure, not on what it meaningfully tests.

**Reference set.** O\*NET 30.2 task statements, Job Zones 3-5, filtered
through the pass-0 knowledge-work prescreen and the pass-5 round-2
task-level knowledge screen (see `main.tex` §3 and Appendix A on
taxonomy construction). 8,338 tasks, deterministically coded into 24
operations via the cluster→label join
(`data/derived/cluster_to_label__operation_only_r2.csv`).

## Per-operation task counts and salience share

| op | tasks | occupations | salience share | n scored units | tier |
| -- | ----: | ----------: | -------------: | -------------: | ---- |
| Inspect, Evaluate, and Verify Compliance and Performance | 1365 | 464 | 16.6% | 2 | coverage_moderate |
| Develop and Maintain Instructional and Operational Documents | 1248 | 421 | 16.0% | 1 | coverage_weak |
| Consult and Coordinate With Others | 588 | 352 | 7.4% | 0 | coverage_missing |
| Supervise and Coordinate Work | 654 | 328 | 7.3% | 0 | coverage_missing |
| Advise and Counsel People | 555 | 300 | 7.2% | 0 | coverage_missing |
| Analyze and Interpret Data for Decisions | 546 | 281 | 7.0% | 5 | coverage_broad |
| Maintain and Record Operational Information | 382 | 285 | 5.3% | 4 | coverage_broad |
| Prepare Reports and Presentations | 341 | 263 | 4.2% | 1 | coverage_weak |
| Create and Update Visual and Technical Drawings | 333 | 145 | 3.9% | 0 | coverage_missing |
| Design And Develop Technical Systems And Applications | 380 | 144 | 3.8% | 1 | coverage_weak |
| Assess Eligibility or Financial Status | 297 | 114 | 3.2% | 0 | coverage_missing |
| Assess and Restore Environmental Conditions | 268 | 82 | 2.6% | 0 | coverage_missing |
| Estimate Costs and Manage Budgets | 201 | 146 | 2.2% | 0 | coverage_missing |
| Develop and Execute Outreach Campaigns for External Support | 191 | 118 | 1.9% | 0 | coverage_missing |
| Recommend Process Improvements | 152 | 115 | 1.7% | 0 | coverage_missing |
| Select Materials by Properties | 151 | 89 | 1.6% | 0 | coverage_missing |
| Conduct Scientific Research | 148 | 104 | 1.6% | 0 | coverage_missing |
| Select and Place People Into Roles | 130 | 74 | 1.4% | 0 | coverage_missing |
| Negotiate Contracts and Terms | 133 | 73 | 1.4% | 0 | coverage_missing |
| Assess and Grade Work | 82 | 50 | 1.3% | 0 | coverage_missing |
| Present Evidence in Hearings | 81 | 45 | 0.9% | 0 | coverage_missing |
| Prescribe Treatments and Medications | 49 | 36 | 0.7% | 1 | coverage_weak |
| Facilitate Group Discussions | 31 | 31 | 0.5% | 0 | coverage_missing |
| Compile Subject Bibliographies | 32 | 29 | 0.2% | 0 | coverage_missing |

## Operations with no scored unit in the sample

- **Consult and Coordinate With Others** (`consult_and_coordinate_with_others`) — 588 O*NET tasks across 352 occupations, salience share ≈ 7.4%.
- **Supervise and Coordinate Work** (`supervise_and_coordinate_work`) — 654 O*NET tasks across 328 occupations, salience share ≈ 7.3%.
- **Advise and Counsel People** (`advise_and_counsel_people`) — 555 O*NET tasks across 300 occupations, salience share ≈ 7.2%.
- **Create and Update Visual and Technical Drawings** (`create_and_update_visual_and_technical_drawings`) — 333 O*NET tasks across 145 occupations, salience share ≈ 3.9%.
- **Assess Eligibility or Financial Status** (`assess_eligibility_or_financial_status`) — 297 O*NET tasks across 114 occupations, salience share ≈ 3.2%.
- **Assess and Restore Environmental Conditions** (`assess_and_restore_environmental_conditions`) — 268 O*NET tasks across 82 occupations, salience share ≈ 2.6%.
- **Estimate Costs and Manage Budgets** (`estimate_costs_and_manage_budgets`) — 201 O*NET tasks across 146 occupations, salience share ≈ 2.2%.
- **Develop and Execute Outreach Campaigns for External Support** (`develop_and_execute_outreach_campaigns_for_external_support`) — 191 O*NET tasks across 118 occupations, salience share ≈ 1.9%.
- **Recommend Process Improvements** (`recommend_process_improvements`) — 152 O*NET tasks across 115 occupations, salience share ≈ 1.7%.
- **Select Materials by Properties** (`select_materials_by_properties`) — 151 O*NET tasks across 89 occupations, salience share ≈ 1.6%.
- **Conduct Scientific Research** (`conduct_scientific_research`) — 148 O*NET tasks across 104 occupations, salience share ≈ 1.6%.
- **Select and Place People Into Roles** (`select_and_place_people_into_roles`) — 130 O*NET tasks across 74 occupations, salience share ≈ 1.4%.
- **Negotiate Contracts and Terms** (`negotiate_contracts_and_terms`) — 133 O*NET tasks across 73 occupations, salience share ≈ 1.4%.
- **Assess and Grade Work** (`assess_and_grade_work`) — 82 O*NET tasks across 50 occupations, salience share ≈ 1.3%.
- **Present Evidence in Hearings** (`present_evidence_in_hearings`) — 81 O*NET tasks across 45 occupations, salience share ≈ 0.9%.
- **Facilitate Group Discussions** (`facilitate_group_discussions`) — 31 O*NET tasks across 31 occupations, salience share ≈ 0.5%.
- **Compile Subject Bibliographies** (`compile_subject_bibliographies`) — 32 O*NET tasks across 29 occupations, salience share ≈ 0.2%.

## Operations covered by a single scored unit

- **Develop and Maintain Instructional and Operational Documents** (`develop_and_maintain_instructional_and_operational_documents`) — salience share 16.0%; covered by `bench_officebench_core`.
- **Prepare Reports and Presentations** (`prepare_reports_and_presentations`) — salience share 4.2%; covered by `bench_officebench_core`.
- **Design And Develop Technical Systems And Applications** (`design_and_develop_technical_systems_and_applications`) — salience share 3.8%; covered by `bench_apex_swe_core`.
- **Prescribe Treatments and Medications** (`prescribe_treatments_and_medications`) — salience share 0.7%; covered by `bench_medagentbench_core`.

## Operations covered by 2+ scored units

- **Inspect, Evaluate, and Verify Compliance and Performance** (`inspect_evaluate_and_verify_compliance_and_performance`) — salience share 16.6%; covered by 2 units: `bench_legalbench_issue_spotting;bench_legalbench_rule_application`.
- **Analyze and Interpret Data for Decisions** (`analyze_and_interpret_data_for_decisions`) — salience share 7.0%; covered by 5 units: `bench_browsecomp_core;bench_financebench_core;bench_gaia_core;bench_legalbench_rhetorical_analysis;bench_legalbench_rule_application`.
- **Maintain and Record Operational Information** (`maintain_and_record_operational_information`) — salience share 5.3%; covered by 4 units: `bench_medagentbench_core;bench_osworld_core;bench_webarena_core;bench_workarena_core`.
