# JASPER benchmark v1

## Contents

| File | Purpose |
|---|---|
| `cases.json` | 18 core hypotheses with expected and acceptable verdicts, expected evidence categories, important caveats, evaluation criteria and the figures behind each expectation (`notes`) |
| `paraphrases.json` | 6 reworded hypotheses over the same data. A robustness probe for the hypothesis parser |
| `holdout.json` | 8 held-out hypotheses written after development and never used for tuning |
| `fixtures/<TICKER>.json` | Frozen Yahoo Finance payloads for 18 tickers, captured on 2026-11-26 through the same `YFinanceTool` code the live agent uses (company info, annual and quarterly statements, 5 years of daily prices, dividends; news captured as an outage error) |

Categories in the core set: supported (5), contradicted (4), partially supported (2), conflicting
metrics (2), period-dependent (3), insufficient evidence (2).

## How expected conclusions were set

Each expected verdict was written from the frozen fixture figures, recorded in the case's `notes`
(for example `Operating margin 17.0% (FY2022) -> 9.2% -> 7.9% -> 5.1% (FY2025)`). Where a careful
analyst could reasonably land on a neighbouring verdict, `acceptable_verdicts` lists both. Expected
answers live only in these files: the runner passes the agent nothing but the hypothesis text and
the ticker, and a test asserts that no expected verdict leaks into the research plan.

`important_caveats` include caveats the data can reveal (latest-period reversals, negative
equity, non-operating gains) and some it cannot (Ford Credit's captive-finance debt, Netflix's
content-spend timing). The latter measure whether the agent overstates what it knows. Caveat
recall below 100% is expected.

## Interpreting results honestly

The rule-based parser, assessment rules and these expected conclusions were developed in the same
cycle, so **core-set offline accuracy is in-sample** and should be read as an upper bound. Two
probes estimate generalisation:

| Run | Acceptable verdicts | Notes |
|---|---|---|
| Paraphrase probe, first (blind) run | 4/6 | 2 abstentions (`INSUFFICIENT_EVIDENCE`) on phrasings the parser did not know; 1 correct verdict for the wrong reason ("cheap valuation" parsed as earnings growth). Parser vocabulary was then extended, so later paraphrase runs are no longer blind (6/6). |
| Held-out set (blind; never tuned) | 6/8 | Both misses are `INSUFFICIENT_EVIDENCE` abstentions ("less profitable", "carrying more debt"), not wrong directional calls. |
| Core set, offline (in-sample) | 18/18 | 0 contradicted or unsupported figures; 18/18 runs reproducible from disk; 7 plan revisions; caveat recall 89%. |

### Live-LLM subset (6 cases, one per category)

Generator `nvidia/nemotron-3.5-lightning-30b-a3b` (planning proposals, up to 2 tool choices per
iteration, critique, narrative); blind rubric judge `minimaxai/minimax-m3.1-flash`; commit `b8b1167`.
Cases: NVDA-01, TSLA-01, AAPL-01, META-01, AMZN-01, DIS-01. Each case took 4.6-8.8 minutes.

| Measure | Result |
|---|---|
| Verdicts within acceptable range | 6/6 (all exact) |
| Claims extracted from LLM text | 325: 280 verified, 12 partially verified, 7 not verifiable, 26 flagged |
| Flagged claims after manual triage | **4 genuine LLM errors**: one unit error repeated twice (AMZN wrote "-105.93%" for a ratio of -105.93, i.e. -10,593%), and two verdict overstatements (AMZN and DIS wrote "not supported" where the evidence verdict was partially supported / insufficient evidence). The other 22 were checker misreads, fixed in commit `f46c9f0` and re-checked against the saved evidence. |
| LLM-chosen tool calls | 26, all validated and logged |
| Plan revisions / human-review escalations | 7 / 3 (AAPL, META, AMZN: the LLM critic raised high-severity issues) |
| Blind LLM judge (1-5) | logical coherence 4 in all cases; verdict justification 4-5; metric relevance 3-5 |

`results.json` for this run was scored with the pre-fix checker (hence 26 flags and lower coherence
scores). Runs from `f46c9f0` onward also save `research_state.json`, so later checker changes can
re-score them without new LLM calls.

### Live-LLM, remaining 12 core cases

Same generator, judge and limits as above; commit `7cacd04` (clean), run as three parallel batches
of four and merged into `eval_results/core12-llm-nemotron/`. Cases: MSFT, NFLX, PFE, WMT, INTC,
NKE, GOOGL, F, KO, BA, SBUX, MRNA. Each case took 4.5-9.4 minutes.

| Measure | Result |
|---|---|
| Verdicts within acceptable range | 12/12 (all exact) |
| Claims extracted from LLM text | 577: 515 verified, 20 partially verified, 2 not verifiable, 40 flagged (35 contradicted, 5 unsupported) |
| Flagged claims after manual triage | **0 genuine LLM errors**: all 40 are checker misreads (`eval_results/core12-llm-nemotron/flagged_claims.json` and `triage.md`). The largest groups: a citation bound to the figure in the next clause (9), a stated decrease whose sign was not applied (5), a formula in parentheses read as the metric (5), "critique issue 1" read as a value (5), and a multi-year range read as one year's growth (4). |
| Text problems the checker did not flag | 4 wording slips found while triaging: "five fiscal years" for four (BA), both PEGs attributed to the computed EPS CAGR (GOOGL), a one-year margin rise folded into "declined" (INTC), and "earnings contraction" without saying it is the latest quarter (WMT) |
| LLM-chosen tool calls / LLM errors | 52, all validated and logged / 0 |
| Plan revisions / human-review escalations | 15 / 5 (MSFT, INTC, NKE, GOOGL, BA: LLM critic high-severity issues) |
| Narrative attempts | 22 across 12 cases: 10 cases used their one retry, triggered by misread claims |
| Blind LLM judge (1-5) | logical coherence 3-4; verdict justification 3-5; metric relevance 4-5 |

Because every flag here is a misread, the deterministic scores understate this run: numerical
accuracy (91%) and logical coherence (88%) are pulled down by checker errors, and the narrative
retry was spent correcting sentences that were already right (in one KO case the model replaced
"115.5%" with "115.47%" and was flagged again). Some misreads are regressions or incomplete fixes
from `f46c9f0` (the decline sign, stated ranges, the widened window reaching "critique issue N");
`triage.md` lists them with code locations. The checker, not the LLM, is the weak link on this run.

Takeaways across all 18 live cases: the rule-based verdict held under an LLM in the loop (18/18);
the verifier caught real LLM mistakes that read plausibly in the first subset (a unit error, a
verdict overstated) but produces false flags on free-form prose that cost a retry; and the LLM
critic escalated to human review in 8 of 18 cases, more often than the deterministic critic.

### LLM-critic escalation rule (replayed, no new LLM calls)

After the first subset, escalation on LLM-critic issues was restricted to issues that challenge
evidence behind a supporting assessment and that no deterministic check already flagged
(`docs/design-decisions.md`, section 6). `scripts/replay_critic_escalation.py` re-applies the rule
to each run's final critique; severity only matters at the last branch of the decision policy, so
the replayed decision is exact.

| Runs | LLM-only escalations recorded | Under the rule | Notes |
|---|---|---|---|
| First 6 (rule written after reading them: in-sample) | 3 (AAPL, META, AMZN) | 0 | All three restated a contradicted or reversed expectation, a deterministic flag, or an unassessed metric. |
| Remaining 12 (rule fixed before inspection) | 5 (MSFT, INTC, NKE, GOOGL, BA) | 0 | Manual reading: four restate recorded findings; INTC (FY versus TTM gap called a "source conflict") is borderline. |

The rule still kept high severity on challenges to supporting evidence (WMT: two PEG values that
disagree; SBUX: a margin "recovery" framed without its earlier peak; DIS: segment attribution), but
in those runs the iteration limit, diminishing returns or insufficient evidence decided first, so a
live LLM-triggered escalation under the rule has not occurred yet.

The pattern in the offline probes matters too: when the deterministic parser does not understand a
hypothesis, it abstains rather than guessing a direction. The LLM planner (`--llm nvidia`) is the intended remedy for
coverage; its proposals are validated before use.

## Reproducing

```bash
jasper eval run --suite benchmarks/v1 --llm offline --label core
jasper eval run --suite benchmarks/v1 --llm offline --include-holdout --only-extra --label holdout
jasper eval run --suite benchmarks/v1 --llm offline --include-paraphrases --only-extra --label paraphrases
```

Offline runs are deterministic: the same commit and suite (`content_sha256` in `results.json`)
produce identical verdicts, scores and claim counts. Live-LLM runs (`--llm nvidia`) vary between
runs and models; record the model in the label.

To refresh the data, run `python scripts/freeze_benchmark_fixtures.py <TICKERS>` into a new suite
version and re-derive the expected conclusions from the new figures. Do not edit v1 in place, or
results stop being comparable.
