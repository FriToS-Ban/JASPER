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
re-score them without new LLM calls. The remaining 12 core cases have not been run with the LLM.

Takeaways: the rule-based verdict held under an LLM in the loop; the verifier caught real LLM
mistakes that read plausibly (a unit error, a verdict overstated); and the LLM critic escalates
to human review more often than the deterministic critic (3 of 6 cases), which may need a
stricter threshold.

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
