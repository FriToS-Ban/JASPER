# Design decisions

This records the main choices behind JASPER's evidence-grounded research agent, what they buy,
and what they cost.

## 1. Deterministic numbers, LLM language

All figures come from deterministic functions over a registry of stored data. The LLM may choose
which analytical tool to call next, propose extra expectations or critique issues, and write prose.
It never computes a number that enters the evidence registry, and it cannot set the verdict.

*Why:* the original pipeline executed LLM-written Python and printed whatever came back. When that
code returned an unexpected schema (as in the stored `analysis_metrics.json` from earlier runs, with
empty `steps` and `aggregated`), the report had nothing traceable to rely on. Deterministic tools
are testable, reproducible, and lineage-preserving.

*Cost:* the analytical vocabulary is limited to the catalogue in `research/metrics.py`. Questions
outside it end in `INSUFFICIENT_EVIDENCE`, which is the intended behaviour, not a crash. The legacy
code-generation path remains available behind `ENABLE_CODEGEN_ANALYSIS`, and its outputs are
labelled unverified and excluded from the verdict.

## 2. Rule-based verdict

The verdict is a function of expectation statuses (`SUPPORTS`=1, `MIXED_POSITIVE`=0.65,
`NEUTRAL`=0.5, `MIXED_NEGATIVE`=0.35, `CONTRADICTS`=0, averaged within each concept group and then
across groups; >=0.8 supported, <=0.2 not supported). Any unmeasurable core expectation gives
`INSUFFICIENT_EVIDENCE`.

*Why:* a verdict that an LLM writes cannot be audited or reproduced. A rule can be challenged
line by line, and the claim verifier flags any narrative that states a different verdict.

*Cost:* thresholds (2% relative change for amounts, 0.5 percentage points for ratios, PEG 1/2) are
conventions. The report says so, and assessments carry caveats when readings sit near a threshold.

## 3. "Contradicted" requires an unambiguous referent

A mismatch becomes `CONTRADICTED` only when the claim cites an evidence ID, names a period, or
refers to a single-valued metric. Otherwise it is `UNSUPPORTED`, explicitly "not proof the claim
is false".

*Why:* the task requires that verification failure never be read as falsity. Conservative
contradiction keeps that line clean; unsupported claims are still surfaced prominently.

## 4. Evidence IDs are content-addressed

Snapshot IDs hash the call and payload; evidence IDs hash ticker, metric, period, snapshot and
locator (raw) or operation, inputs and parameters (derived). Re-registration of identical evidence
is idempotent; a different value for the same metric and period from another snapshot is recorded
as a conflict, which the critique can escalate.

*Why:* identical inputs produce identical IDs across runs, so reports and benchmark results are
diffable and claims can cite stable identifiers.

## 5. A router over stages, not a new framework

The existing orchestrator already had stage execution, milestones, artifact storage and a
human-review gate. The agent loop is a small router (`research/agent.py:next_stage`) driven by
recorded state, and both the async workflow client and the synchronous CLI/benchmark runner follow
it. Stage names and the `run_stage` contract are unchanged; `critique` and `claim_verification`
were added.

*Why:* the loop is explicit, bounded (iteration limit, reanalysis limit, narrative-attempt limit,
tool-call budget, stage-execution cap) and testable stage by stage, and existing callers keep working.

*Note:* despite its name, the original code did not use LangGraph's runtime. Adopting it now would
have meant rewriting the milestone and review machinery for no gain in auditability.

## 6. Critique-triggered human review reuses the existing gate

When the critique decides `HUMAN_REVIEW` (high-severity source conflict, LLM-flagged high-impact
ambiguity that names real metrics, or calculations that stay inconsistent after reanalysis), the
agent still finishes the report, then pauses before delivery with the reason attached.

*Why:* reviewers get a complete report to judge rather than a half-finished state, and the
existing resume API works unchanged.

## 7. Frozen fixtures for evaluation

Benchmark cases run against payloads captured through the same `YFinanceTool` code path the live
system uses, including captured failures (the news endpoint was down at capture time, and that gap
is replayed). Snapshot metadata records `frozen_fixture` mode and the capture time.

*Why:* expected conclusions must stay valid while Yahoo's data changes, and two runs of the suite
must be comparable.

## 8. Independent, disk-based scoring

The evaluator reloads the run's files, checks reproducibility, recomputes the verdict from recorded
assessments and re-verifies every claim. Disagreement with the run's own verification is a failure
category. Rubric judges are reported alongside deterministic scores, never instead of them.

## Known limitations

- The rule-based hypothesis parser covers common phrasings only. The blind probes measured this:
  on the paraphrase probe, 4 of 6 verdicts were acceptable before vocabulary fixes (one of those
  four for the wrong reason). On the held-out set, 6 of 8 were acceptable, and both misses were
  `INSUFFICIENT_EVIDENCE` abstentions rather than wrong directional calls. The LLM planner can add
  expectations; it was not separately benchmarked at scale because of endpoint latency.
- Expected benchmark conclusions were written by the same developer who built the rules, from the
  same frozen data. In-sample offline accuracy is therefore an optimistic upper bound; the
  held-out result is the honest estimate.
- Claim extraction is pattern-based. Unusual phrasing can produce `NOT_VERIFIABLE` claims or
  attach a number to the wrong metric; citations reduce this because they pin the comparison.
- Yahoo Finance offers roughly four annual periods, no segment data, and undisclosed
  methodology for TTM ratios. These limits are stated in every report.
- No causal controls (peers, macro, FX, M&A) are applied; associations are descriptive.
