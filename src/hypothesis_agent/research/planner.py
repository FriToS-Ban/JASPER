"""Hypothesis parsing and revisable, hypothesis-driven research plans.

A plan states *what the hypothesis predicts* (expectations such as "operating_margin increases")
and *which evidence is needed* to test it. Plans are versioned: the critique stage can add
metrics or datasets, and every revision records its trigger and rationale.

Expectations come from a deterministic phrase parser and, optionally, an LLM planner. LLM
proposals are validated against the metric catalogue and can only add expectations for concepts
the parser did not cover, so the LLM cannot silently rewrite what is being tested.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, ValidationError

from hypothesis_agent.research.evidence import utc_now_iso
from hypothesis_agent.research.metrics import (
    dataset_for_metric,
    is_known_metric,
    metric_category,
    raw_dependencies,
)

Direction = Literal[
    "increase", "decrease", "above", "below", "positive", "negative",
    "overvalued", "undervalued", "base_effect", "unmeasurable",
]


class Expectation(BaseModel):
    expectation_id: str
    group: str = Field(description="Concept group; composite concepts expand to several metrics in one group.")
    concept: str
    metric: Optional[str] = None
    direction: Direction
    threshold: Optional[float] = None
    target_year: Optional[int] = None
    window_start_year: Optional[int] = None
    window_end_year: Optional[int] = None
    window_last_n_years: Optional[int] = None
    weight: Literal["primary", "secondary"] = "primary"
    measurable: bool = True
    rationale: str
    source: Literal["rule_parser", "llm_planner"] = "rule_parser"
    phrase: str = ""

    @property
    def has_explicit_window(self) -> bool:
        return any(v is not None for v in (self.window_start_year, self.window_end_year, self.window_last_n_years))


class PlannedMetric(BaseModel):
    metric: str
    category: str
    frequency: Literal["annual", "quarterly"] = "annual"
    rationale: str
    origin: Literal["expectation", "baseline", "critique", "llm"] = "baseline"
    added_in_version: int = 1


class PlanRevision(BaseModel):
    version: int
    iteration: int
    reason: str
    triggered_by: List[str] = Field(default_factory=list)
    added_metrics: List[str] = Field(default_factory=list)
    added_datasets: List[str] = Field(default_factory=list)
    at: str = Field(default_factory=utc_now_iso)


class ResearchPlan(BaseModel):
    version: int = 1
    hypothesis: str
    ticker: str
    objective: str
    expectations: List[Expectation] = Field(default_factory=list)
    metrics: List[PlannedMetric] = Field(default_factory=list)
    datasets: List[str] = Field(default_factory=list)
    price_period: str = "1y"
    forward_looking: bool = False
    notes: List[str] = Field(default_factory=list)
    revisions: List[PlanRevision] = Field(default_factory=list)

    def categories(self) -> List[str]:
        return sorted({m.category for m in self.metrics})

    def metric_keys(self, frequency: Optional[str] = None) -> List[str]:
        return [m.metric for m in self.metrics if frequency is None or m.frequency == frequency]

    def has_metric(self, metric: str, frequency: str = "annual") -> bool:
        return any(m.metric == metric and m.frequency == frequency for m in self.metrics)


# ----------------------------------------------------------------------------------------------
# Lexicon
# ----------------------------------------------------------------------------------------------
HEALTH_COMPONENTS: List[Tuple[str, Direction]] = [
    ("operating_margin", "increase"),
    ("net_income", "increase"),
    ("free_cash_flow", "increase"),
    ("total_debt", "decrease"),
]
PROFITABILITY_COMPONENTS = ["operating_margin", "net_income"]

# (phrase regex, metric | composite key). Longer phrases first so "free cash flow" wins over "cash flow".
CONCEPTS: List[Tuple[str, str]] = [
    (r"financially (?:healthier|stronger|weaker)|financial (?:health|position|strength)|balance[- ]sheet health", "@health"),
    (r"free[- ]cash[- ]flow(?: generation)?|\bfcf\b", "free_cash_flow"),
    (r"cash conversion|(?:its )?(?:net income|earnings|profits?) into (?:operating )?cash", "cash_conversion"),
    (r"operating cash flows?|cash from operations", "operating_cash_flow"),
    (r"gross margins?", "gross_margin"),
    (r"net (?:profit )?margins?|profit margins?", "net_margin"),
    (r"operating (?:profit )?margins?|operating profitability|ebit margins?", "operating_margin"),
    (r"(?:revenue|sales|top[- ]line) growth", "total_revenue_growth_yoy"),
    (r"(?:earnings|eps) growth", "diluted_eps_growth_yoy"),
    (r"operating income growth", "operating_income_growth_yoy"),
    (r"operating income", "operating_income"),
    (r"net income|net profit|net earnings|bottom line", "net_income"),
    (r"earnings per share|\beps\b", "diluted_eps"),
    (r"capital (?:expenditures?|spending)|\bcapex\b", "capex_intensity"),
    (r"working capital", "change_in_working_capital"),
    (r"receivables?", "receivables_days"),
    (r"balance[- ]sheet leverage|\bleverage\b|debt[- ]to[- ]equity", "debt_to_equity"),
    (r"\bdebt\b|borrowings", "total_debt"),
    (r"liquidity", "current_ratio"),
    (r"profitability|profits?\b|profitable", "@profitability"),
    (r"\bmargins?\b", "operating_margin"),
    (r"revenues?|\bsales\b|top[- ]line", "total_revenue"),
    (r"\bearnings\b", "net_income"),
    (r"share price|stock price|\bshares\b|\bstock\b", "price_return"),
    (r"volatility", "annualized_volatility"),
]

# Concepts JASPER has no structured data source for. A hypothesis resting on these cannot be
# settled by the collected evidence, which must be stated rather than papered over.
UNMEASURABLE = [
    r"\bsegments?\b", r"streaming", r"services revenue", r"market share", r"pipeline", r"\bguidance\b",
    r"management", r"\bbrand\b", r"subscribers?", r"data[- ]center demand", r"\bmoat\b", r"competitive",
    r"regulat\w+", r"customer\w*", r"product launch", r"\bai demand\b", r"geographic", r"\bchina\b",
]

INCREASE_WORDS = [
    r"improv\w*", r"expan\w+", r"ris(?:e|es|ing)", r"increas\w*", r"grow(?:s|ing|n)?\b", r"stronger",
    r"strengthen\w*", r"higher", r"recover\w*", r"accelerat\w*", r"healthier", r"scal(?:e|es|ing)\b",
    r"\bstrong\b", r"robust", r"solid", r"climb\w*", r"\bup\b", r"widen\w*", r"\bmore of\b", r"\bheavy\b",
]
DECREASE_WORDS = [
    r"deteriorat\w*", r"declin\w*", r"fall(?:s|ing)?\b", r"fell", r"shrink\w*", r"contract\w*", r"weaker",
    r"weakening", r"lower", r"slow\w*", r"decelerat\w*", r"compress\w*", r"worsen\w*", r"\bdown\b",
    r"narrow\w*", r"erod\w*", r"drop\w*", r"decreas\w*", r"under pressure", r"pressured", r"squeez\w*",
    r"\bless of\b",
]
OVERVALUED_WORDS = r"\b(?:over[- ]?valued|over[- ]?priced|expensive|richly valued|stretched valuation)\b"
UNDERVALUED_WORDS = r"\b(?:under[- ]?valued|under[- ]?priced|cheap|inexpensive|attractively valued|bargain)\b"
LEVEL_ABOVE = [(r"double[- ]digit", 0.10), (r"\bstrong\b|robust|rapid|high", 0.10)]
NEGATIONS = r"(?:not|no longer|never|isn't|aren't|hasn't|haven't)\s+(?:\w+\s+)?$"
NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
FY = r"(?:fy\s?|fiscal(?: year)?\s?)?"


@dataclass
class _Window:
    start: Optional[int] = None
    end: Optional[int] = None
    last_n: Optional[int] = None


def _parse_window(text: str) -> Tuple[_Window, str]:
    """Extract an explicit analysis window and return text with the window phrase removed."""

    lowered = text.lower()
    patterns = [
        (rf"between {FY}((?:19|20)\d\d) and {FY}((?:19|20)\d\d)", "range"),
        (rf"from {FY}((?:19|20)\d\d) (?:to|through|until) {FY}((?:19|20)\d\d)", "range"),
        (rf"(?:since|after|following)(?: the)? {FY}((?:19|20)\d\d)(?:\s+\w+)?", "start"),
        (r"(?:over|in|during|across) the (?:last|past) (\w+) (?:fiscal )?years?", "last_n"),
    ]
    for pattern, kind in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        window = _Window()
        if kind == "range":
            window.start, window.end = int(match.group(1)), int(match.group(2))
        elif kind == "start":
            window.start = int(match.group(1))
        else:
            token = match.group(1)
            window.last_n = int(token) if token.isdigit() else NUMBER_WORDS.get(token)
        stripped = lowered[: match.start()] + " " + lowered[match.end():]
        return window, stripped
    return _Window(), lowered


_CONNECTOR = re.compile(
    r"\s*(?:;|,?\s+\b(but|while|whereas|although|though|yet|despite|even as|even though|and|as|with)\b)\s+"
)


def _split_clauses(text: str) -> List[Tuple[str, Optional[str]]]:
    parts: List[Tuple[str, Optional[str]]] = []
    last = 0
    connector: Optional[str] = None
    for match in _CONNECTOR.finditer(text):
        parts.append((text[last: match.start()], connector))
        connector = (match.group(1) or ";").lower()
        last = match.end()
    parts.append((text[last:], connector))
    return [(clause.strip(" .,"), conn) for clause, conn in parts if clause.strip(" .,")]


def _find_all(patterns: Iterable[str], text: str) -> List[Tuple[int, int, str]]:
    spans = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            spans.append((match.start(), match.end(), match.group(0)))
    return spans


def _concept_spans(clause: str) -> List[Tuple[int, int, str, str]]:
    taken: List[Tuple[int, int]] = []
    spans: List[Tuple[int, int, str, str]] = []
    for pattern, target in CONCEPTS:
        for match in re.finditer(pattern, clause):
            start, end = match.span()
            if any(start < t_end and end > t_start for t_start, t_end in taken):
                continue
            taken.append((start, end))
            spans.append((start, end, match.group(0), target))
    return sorted(spans)


def _assign_directions(clause: str, concepts: List[Tuple[int, int, str, str]]) -> Dict[int, str]:
    """Assign each direction word to at most one concept, closest (pre-modifier preferred) first."""

    words = [(s, e, "increase") for s, e, _ in _find_all(INCREASE_WORDS, clause)]
    words += [(s, e, "decrease") for s, e, _ in _find_all(DECREASE_WORDS, clause)]
    pairs = []
    for c_index, (c_start, c_end, _, _) in enumerate(concepts):
        for w_index, (w_start, w_end, direction) in enumerate(words):
            if c_start <= w_start < c_end:
                continue
            if w_end <= c_start:
                penalty = c_start - w_end
            else:
                penalty = w_start - c_end + 6  # post-concept words ("margins are improving") rank second
            if penalty < 0:
                continue
            negated = re.search(NEGATIONS, clause[:w_start]) is not None
            resolved = direction if not negated else ("decrease" if direction == "increase" else "increase")
            pairs.append((penalty, c_index, w_index, resolved))
    assigned: Dict[int, str] = {}
    used_words: set = set()
    for penalty, c_index, w_index, resolved in sorted(pairs):
        if c_index in assigned or w_index in used_words:
            continue
        assigned[c_index] = resolved
        used_words.add(w_index)
    return assigned


def parse_hypothesis(text: str) -> Tuple[List[Expectation], Dict[str, Any]]:
    """Deterministically translate a hypothesis into testable expectations."""

    window, working = _parse_window(text)
    meta: Dict[str, Any] = {"window": window.__dict__, "forward_looking": False, "unmeasurable": []}
    expectations: List[Expectation] = []
    counter = 0

    def add(**kwargs: Any) -> None:
        nonlocal counter
        counter += 1
        kwargs.setdefault("window_start_year", window.start)
        kwargs.setdefault("window_end_year", window.end)
        kwargs.setdefault("window_last_n_years", window.last_n)
        expectations.append(Expectation(expectation_id=f"H{counter}", **kwargs))

    if re.search(r"\bwill\b|\bby (?:fy\s?)?20\d\d\b|\bgoing forward\b|\bnext (?:\w+ )?(?:years?|quarters?|months?)\b", working):
        meta["forward_looking"] = True

    # Valuation relative to growth.
    valuation = re.search(OVERVALUED_WORDS, working) or re.search(UNDERVALUED_WORDS, working)
    if valuation:
        direction: Direction = "overvalued" if re.fullmatch(OVERVALUED_WORDS, valuation.group(0)) else "undervalued"
        add(group="G-valuation", concept="valuation relative to earnings growth", metric="peg_computed",
            direction=direction, threshold=None,
            rationale="Compare P/E with historical EPS growth (PEG); >2 suggests rich valuation, <1 cheap.",
            phrase=valuation.group(0))
        working = working.replace(valuation.group(0), " ")
        working = re.sub(r"\bvaluation\b|\blooks\b|\bappears\b", " ", working)
        working = re.sub(r"relative to (?:how fast )?(?:its )?(?:historical )?(?:earnings|eps)(?: growth| have grown| has grown)?", " ", working)

    # Favourable comparison base.
    base = re.search(r"comparison (?:base|period)|base effect|easy comp\w*|weak (?:(?:fy\s?)?\d{4} )?(?:base|comparison)", working)
    if base:
        year = re.search(rf"{FY}((?:19|20)\d\d)\s+(?:revenue|sales)\s+growth|(?:revenue|sales) growth in {FY}((?:19|20)\d\d)", working)
        target = int(next(g for g in year.groups() if g)) if year else None
        add(group="G-base", concept="growth explained by comparison base", metric="total_revenue",
            direction="base_effect", target_year=target,
            rationale="Growth after a prior-year decline that leaves revenue below its earlier peak indicates a base effect.",
            phrase=base.group(0))
        meta["window"] = window.__dict__
        return expectations, meta

    unmeasurable_hits = [m.group(0) for p in UNMEASURABLE for m in [re.search(p, working)] if m]
    if unmeasurable_hits:
        meta["unmeasurable"] = unmeasurable_hits
        add(group="G-unmeasurable", concept=", ".join(unmeasurable_hits), metric=None, direction="unmeasurable",
            measurable=False,
            rationale=("The hypothesis rests on data JASPER cannot collect from consolidated financial "
                       f"statements ({', '.join(unmeasurable_hits)})."),
            phrase=", ".join(unmeasurable_hits))
    if meta["forward_looking"] and not expectations:
        add(group="G-forecast", concept="forward-looking prediction", metric=None, direction="unmeasurable",
            measurable=False, rationale="Historical data cannot confirm a forecast; only trend context can be provided.",
            phrase="forecast")

    group_index = 0
    for clause, connector in _split_clauses(working):
        weight: Literal["primary", "secondary"] = "secondary" if connector in {"as", "with"} or unmeasurable_hits else "primary"
        concepts = _concept_spans(clause)
        directions = _assign_directions(clause, concepts)
        percent = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:%|percent)", clause)
        for c_index, (start, end, phrase, target) in enumerate(concepts):
            direction = directions.get(c_index)
            level = next((thr for pat, thr in LEVEL_ABOVE if re.search(pat, clause)), None)
            group_index += 1
            group = f"G{group_index}"
            if target == "@health":
                direction_word = direction or "increase"
                for metric, healthy in HEALTH_COMPONENTS:
                    resolved = healthy if direction_word == "increase" else ("decrease" if healthy == "increase" else "increase")
                    add(group=group, concept=phrase, metric=metric, direction=resolved, weight=weight,
                        rationale=f"Financial health implies {metric.replace('_', ' ')} should {resolved}.", phrase=clause)
                continue
            growth_target = target if target.endswith("_growth_yoy") else (
                f"{target}_growth_yoy" if target in {"total_revenue", "net_income", "diluted_eps"} and percent else None)
            if growth_target and percent and direction in (None, "increase"):
                threshold = float(percent.group(1)) / 100.0
                add(group=group, concept=phrase, metric=growth_target, direction="above", threshold=threshold,
                    weight=weight, rationale=f"Hypothesis states growth of about {percent.group(0)}; tested as latest growth at or above it.",
                    phrase=clause)
                continue
            if direction is None and target.endswith("_growth_yoy"):
                add(group=group, concept=phrase, metric=target, direction="above", threshold=0.0, weight=weight,
                    rationale="Hypothesis presupposes positive growth.", phrase=clause)
                continue
            if direction is None:
                continue
            strong_words = re.search(r"double[- ]digit|\bstrong\b|robust|rapid", clause)
            if target.endswith("_growth_yoy") and level is not None and direction == "increase" and strong_words:
                add(group=group, concept=phrase, metric=target, direction="above", threshold=level, weight=weight,
                    rationale=f"Hypothesis describes growth as strong; tested as latest growth above {level:.0%}.",
                    phrase=clause)
                continue
            if target == "total_revenue" and re.search(r"double[- ]digit", clause):
                add(group=group, concept=phrase, metric="total_revenue_growth_yoy", direction="above", threshold=0.10,
                    weight=weight, rationale="Double-digit growth tested as latest YoY revenue growth above 10%.", phrase=clause)
                continue
            metrics = PROFITABILITY_COMPONENTS if target == "@profitability" else [target]
            for metric in metrics:
                add(group=group, concept=phrase, metric=metric, direction=direction, weight=weight,
                    rationale=f"'{clause}' implies {metric.replace('_', ' ')} should {direction}.", phrase=clause)
    return expectations, meta


# ----------------------------------------------------------------------------------------------
# Plan construction and revision
# ----------------------------------------------------------------------------------------------
BASELINE_METRICS: List[Tuple[str, str]] = [
    ("total_revenue", "Scale and growth context for any thesis."),
    ("total_revenue_growth_yoy", "Growth context."),
    ("operating_margin", "Core profitability context."),
    ("net_income", "Bottom-line context."),
    ("free_cash_flow", "Cash generation context."),
    ("trailing_pe", "Valuation context."),
    ("price_return", "Market reaction context."),
]
EXPECTATION_SUPPORT_METRICS: Dict[str, List[str]] = {
    "peg_computed": ["diluted_eps", "diluted_eps_growth_yoy", "trailing_pe", "forward_pe", "peg_ratio", "current_price"],
    "cash_conversion": ["operating_cash_flow", "net_income"],
    "free_cash_flow": ["operating_cash_flow", "capital_expenditure"],
    "total_debt": ["debt_to_equity", "stockholders_equity"],
    "debt_to_equity": ["total_debt", "stockholders_equity"],
    "capex_intensity": ["capital_expenditure", "total_revenue"],
}


def build_plan(
    hypothesis: str,
    ticker: str,
    llm_proposal: Optional[Dict[str, Any]] = None,
) -> ResearchPlan:
    expectations, meta = parse_hypothesis(hypothesis)
    notes: List[str] = []
    if meta.get("forward_looking"):
        notes.append("Hypothesis is forward-looking; historical evidence can only test whether trends are consistent with it.")
    if llm_proposal:
        added, rejected = _merge_llm_expectations(expectations, llm_proposal)
        if added:
            notes.append(f"LLM planner added {added} expectation(s) for concepts the phrase parser did not cover.")
        if rejected:
            notes.append(f"LLM planner proposals rejected during validation: {'; '.join(rejected[:5])}")
    if not expectations:
        notes.append("No testable financial claim could be parsed from the hypothesis.")

    plan = ResearchPlan(
        hypothesis=hypothesis,
        ticker=ticker,
        objective=_objective(hypothesis, expectations),
        expectations=expectations,
        forward_looking=bool(meta.get("forward_looking")),
        notes=notes,
    )
    for expectation in expectations:
        if expectation.metric is None:
            continue
        _add_metric(plan, expectation.metric, "expectation", f"Tests {expectation.expectation_id}: {expectation.concept} should {expectation.direction}.")
        for support in EXPECTATION_SUPPORT_METRICS.get(expectation.metric, []):
            _add_metric(plan, support, "expectation", f"Supports evaluation of {expectation.metric}.")
    for metric, rationale in BASELINE_METRICS:
        _add_metric(plan, metric, "baseline", rationale)
    if llm_proposal:
        for metric in llm_proposal.get("additional_metrics", []) or []:
            if isinstance(metric, str) and is_known_metric(metric):
                _add_metric(plan, metric, "llm", "Requested by LLM planner.")
    needs_long_history = any(e.has_explicit_window or e.metric in {"price_return", "annualized_volatility"} for e in expectations)
    plan.price_period = "5y" if needs_long_history else "1y"
    plan.datasets = required_datasets(plan)
    return plan


def _objective(hypothesis: str, expectations: Sequence[Expectation]) -> str:
    measurable = [e for e in expectations if e.measurable]
    if not expectations:
        return f"Determine whether '{hypothesis}' states a testable financial claim."
    tests = "; ".join(f"{e.metric or e.concept} {e.direction}" for e in measurable[:6])
    return f"Test whether the evidence shows: {tests or 'the stated (unmeasurable) concepts'}."


def _add_metric(plan: ResearchPlan, metric: str, origin: str, rationale: str, frequency: str = "annual") -> bool:
    if plan.has_metric(metric, frequency) or not is_known_metric(metric):
        return False
    plan.metrics.append(PlannedMetric(metric=metric, category=metric_category(metric), frequency=frequency,
                                      rationale=rationale, origin=origin, added_in_version=plan.version))
    return True


def required_datasets(plan: ResearchPlan) -> List[str]:
    datasets = ["company_info", "price_history", "income_statement_annual"]
    for planned in plan.metrics:
        for raw in raw_dependencies(planned.metric):
            dataset = dataset_for_metric(raw, planned.frequency)
            if dataset and dataset not in datasets:
                datasets.append(dataset)
    return datasets


def revise_plan(
    plan: ResearchPlan,
    *,
    iteration: int,
    reason: str,
    triggered_by: List[str],
    metrics: Sequence[Tuple[str, str]] = (),
) -> Optional[PlanRevision]:
    """Add (metric, frequency) pairs to the plan; returns the revision or None if nothing changed."""

    before_datasets = list(plan.datasets)
    new_version = plan.version + 1
    added: List[str] = []
    for metric, frequency in metrics:
        if plan.has_metric(metric, frequency) or not is_known_metric(metric):
            continue
        plan.metrics.append(PlannedMetric(metric=metric, category=metric_category(metric), frequency=frequency,
                                          rationale=reason, origin="critique", added_in_version=new_version))
        added.append(metric if frequency == "annual" else f"{metric} ({frequency})")
    if not added:
        return None
    plan.version = new_version
    plan.datasets = required_datasets(plan)
    revision = PlanRevision(
        version=new_version, iteration=iteration, reason=reason, triggered_by=triggered_by,
        added_metrics=added, added_datasets=[d for d in plan.datasets if d not in before_datasets],
    )
    plan.revisions.append(revision)
    return revision


class _LLMExpectation(BaseModel):
    concept: str
    metric: Optional[str] = None
    direction: Direction
    threshold: Optional[float] = None
    rationale: str = ""


def _merge_llm_expectations(expectations: List[Expectation], proposal: Dict[str, Any]) -> Tuple[int, List[str]]:
    covered = {e.metric for e in expectations}
    rejected: List[str] = []
    added = 0
    for raw in proposal.get("expectations", []) or []:
        try:
            item = _LLMExpectation.model_validate(raw)
        except ValidationError as exc:
            rejected.append(f"invalid schema ({exc.errors(include_url=False)[0]['msg']})")
            continue
        if item.metric is not None and not is_known_metric(item.metric):
            rejected.append(f"unknown metric '{item.metric}'")
            continue
        if item.metric in covered:
            continue
        covered.add(item.metric)
        index = len(expectations) + 1
        expectations.append(Expectation(
            expectation_id=f"H{index}", group=f"L{index}", concept=item.concept, metric=item.metric,
            direction=item.direction, threshold=item.threshold,
            weight="secondary" if any(e.source == "rule_parser" for e in expectations) else "primary",
            measurable=item.metric is not None and item.direction != "unmeasurable",
            rationale=item.rationale or "Proposed by LLM planner.", source="llm_planner", phrase=item.concept,
        ))
        added += 1
    return added, rejected
