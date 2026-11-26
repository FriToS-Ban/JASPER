"""Claim extraction from generated text and deterministic verification against evidence.

Statuses:
    VERIFIED            -- matches collected/recomputed evidence within rounding tolerance
    PARTIALLY_VERIFIED  -- approximately right, or right for a different/unstated period or source
    UNSUPPORTED         -- no evidence found that could confirm it (NOT proof that it is false)
    CONTRADICTED        -- the metric and period are unambiguous and evidence shows a different value
    NOT_VERIFIABLE      -- qualitative or unidentifiable assertion with no structured evidence

A claim is only CONTRADICTED when JASPER can pin down exactly which number the text refers to
(a cited evidence ID, an explicit period, or a metric with a single value). Otherwise a mismatch
is UNSUPPORTED: absence of a match is not evidence of falsity.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from hypothesis_agent.research.analytics import AnalyticsError, PayloadLoader, classify_steps, recompute
from hypothesis_agent.research.evidence import EvidenceRecord, EvidenceRegistry, PeriodType
from hypothesis_agent.research.metrics import METRICS, RATIO_RECIPES, get_metric, metric_category

ClaimStatus = Literal["VERIFIED", "PARTIALLY_VERIFIED", "UNSUPPORTED", "CONTRADICTED", "NOT_VERIFIABLE"]
EVIDENCE_ID = re.compile(r"EV-[0-9a-f]{8}")
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

# Alternative metrics that may legitimately back a claim, in preference order.
METRIC_FAMILY: Dict[str, List[str]] = {
    "operating_margin": ["operating_margin", "ttm_operating_margin"],
    "net_margin": ["net_margin", "ttm_profit_margin"],
    "gross_margin": ["gross_margin", "ttm_gross_margin"],
    "total_revenue": ["total_revenue", "ttm_revenue"],
    "free_cash_flow": ["free_cash_flow", "free_cash_flow_computed", "ttm_free_cash_flow"],
    "operating_cash_flow": ["operating_cash_flow", "ttm_operating_cash_flow"],
    "trailing_pe": ["trailing_pe", "pe_computed"],
    "debt_to_equity": ["debt_to_equity", "yahoo_debt_to_equity"],
    "return_on_equity": ["return_on_equity", "return_on_equity_computed"],
    "diluted_eps": ["diluted_eps", "trailing_eps"],
    "peg_ratio": ["peg_ratio", "peg_computed"],
    "peg_computed": ["peg_computed", "peg_ratio"],
}
TTM_VARIANT = {"operating_margin": "ttm_operating_margin", "net_margin": "ttm_profit_margin",
               "gross_margin": "ttm_gross_margin", "total_revenue": "ttm_revenue",
               "free_cash_flow": "ttm_free_cash_flow", "operating_cash_flow": "ttm_operating_cash_flow"}
ALWAYS_ABSOLUTE = {"capital_expenditure", "share_repurchases", "dividends_paid"}

CATEGORY_BY_METRIC_CATEGORY = {
    "growth": "growth", "profitability": "margin", "cash_flow": "cash_flow", "leverage": "leverage",
    "liquidity": "leverage", "valuation": "valuation", "price_performance": "price",
    "working_capital": "cash_flow", "scale": "other",
}

GROWTH_WORDS = r"grow|grew|growth|increas|rose|rise|up\b|declin|fell|fall|down\b|drop|yoy|year[- ]over[- ]year|expand|contract|jump|surg|slid|shrank"
NEGATIVE_WORDS = r"loss|negative|deficit|outflow|burn"
APPROX_WORDS = r"(?:about|approximately|roughly|nearly|around|almost|~|circa|some)\s*$"
GREATER_WORDS = r"(?:more than|over|above|exceed\w*|greater than|at least)\s*$"
LESS_WORDS = r"(?:less than|under|below|at most)\s*$"
FRAMING = r"\b(?:hypothes\w+|thesis|we test\w*|tested whether|claims? that|predicts?|would|could|might|may|if)\b"
QUALITATIVE = (
    r"market share|competitive|competition|moat|management|brand|demand|pipeline|guidance|strategy|"
    r"customers?|regulat\w+|macro\w*|innovation|ecosystem|leadership|execution|sentiment|outlook"
)
INCREASE = r"\b(?:improv\w*|expand\w*|rose|ris(?:e|es|ing)|increas\w*|grew|grow\w*|strengthen\w*|higher|recover\w*|accelerat\w*|widen\w*|climb\w*)\b"
DECREASE = r"\b(?:deteriorat\w*|declin\w*|fell|fall\w*|shr[iu]nk\w*|shrank|contract\w*|weaken\w*|lower|slow\w*|decelerat\w*|compress\w*|worsen\w*|narrow\w*|erod\w*|drop\w*|decreas\w*)\b"

VERDICT_PATTERNS = [
    (r"\b(?:not supported|unsupported|contradicted|rejected|refuted|does not hold|is not borne out)\b", "NOT_SUPPORTED"),
    (r"\bpartially supported|partly supported|mixed support|partial support\b", "PARTIALLY_SUPPORTED"),
    (r"\binsufficient evidence|inconclusive|cannot be (?:confirmed|determined|evaluated)\b", "INSUFFICIENT_EVIDENCE"),
    (r"\b(?:is|are|remains|appears) (?:well[- ]|strongly |fully )?supported\b|\bconfirmed\b", "SUPPORTED"),
]


class Claim(BaseModel):
    claim_id: str
    section: str
    text: str
    category: str
    metric: Optional[str] = None
    reported_text: Optional[str] = None
    reported_value: Optional[float] = None
    reported_unit: Optional[str] = None
    comparator: Literal["eq", "approx", "gt", "lt"] = "eq"
    period_year: Optional[int] = None
    direction: Optional[str] = None
    cited_evidence: List[str] = Field(default_factory=list)
    expected_value: Optional[float] = None
    expected_display: Optional[str] = None
    matched_evidence: List[str] = Field(default_factory=list)
    calculation: Optional[str] = None
    status: ClaimStatus = "NOT_VERIFIABLE"
    reason: str = ""
    missing_evidence: Optional[str] = None
    tolerance: Optional[float] = None


class ClaimReport(BaseModel):
    claims: List[Claim] = Field(default_factory=list)
    counts: Dict[str, int] = Field(default_factory=dict)
    numeric_claims: int = 0
    numeric_verified: int = 0
    unresolved_citations: List[str] = Field(default_factory=list)

    def problems(self) -> List[Claim]:
        return [c for c in self.claims if c.status in {"UNSUPPORTED", "CONTRADICTED", "PARTIALLY_VERIFIED", "NOT_VERIFIABLE"}]


# ----------------------------------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------------------------------
NUMBER = re.compile(
    r"(?P<sign>[-−–])?(?P<cur>US\$|\$|USD\s?)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s*(?P<scale>trillion|billion|million|thousand|tn\b|bn\b|mn\b|[TBMK]\b))?"
    r"(?:\s*(?P<unit>%|percent\b|percentage points?\b|pp\b|bps\b|basis points\b|x\b|times\b))?",
    re.IGNORECASE,
)
YEAR = re.compile(r"\b(?:(?:FY|fiscal(?: year)?)\s?'?(?P<fy>\d{4}|\d{2})|(?P<plain>(?:19|20)\d{2}))\b", re.IGNORECASE)
SCALES = {"trillion": 1e12, "tn": 1e12, "t": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9,
          "million": 1e6, "mn": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}


def _synonym_index() -> List[Tuple[str, str]]:
    pairs: Dict[str, str] = {}
    for definition in METRICS.values():
        for synonym in definition.synonyms:
            pairs.setdefault(synonym.lower(), definition.key)
    for extra, key in (("p/e ratio", "trailing_pe"), ("p/e multiple", "trailing_pe"), ("operating profit margin", "operating_margin"),
                       ("free-cash-flow", "free_cash_flow"), ("capex intensity", "capex_intensity"), ("peg", "peg_computed"),
                       ("debt/equity", "debt_to_equity"), ("cash conversion", "cash_conversion")):
        pairs.setdefault(extra, key)
    return sorted(pairs.items(), key=lambda item: -len(item[0]))


SYNONYMS = _synonym_index()


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[\(\"'])", text)
    return [p.strip() for p in parts if p.strip()]


@dataclass
class _Mention:
    start: int
    end: int
    metric: str


def _metric_mentions(sentence: str) -> List[_Mention]:
    lowered = sentence.lower()
    taken: List[Tuple[int, int]] = []
    mentions: List[_Mention] = []
    for phrase, metric in SYNONYMS:
        for match in re.finditer(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", lowered):
            span = match.span()
            if any(span[0] < e and span[1] > s for s, e in taken):
                continue
            taken.append(span)
            mentions.append(_Mention(span[0], span[1], metric))
    return sorted(mentions, key=lambda m: m.start)


def _year_mentions(sentence: str) -> List[Tuple[int, int, int]]:
    years = []
    for match in YEAR.finditer(sentence):
        raw = match.group("fy") or match.group("plain")
        year = int(raw) if len(raw) == 4 else 2000 + int(raw)
        years.append((match.start(), match.end(), year))
    return years


def _numbers(sentence: str, years: Sequence[Tuple[int, int, int]]) -> List[Dict[str, Any]]:
    found = []
    year_spans = [(s, e) for s, e, _ in years]
    for match in NUMBER.finditer(sentence):
        start, end = match.span("num")
        if any(s <= start < e for s, e in year_spans):
            continue
        prefix = sentence[max(0, match.start() - 2): match.start()]
        if re.search(r"[A-Za-z\-]$", prefix) and not match.group("cur"):
            continue  # part of an identifier such as EV-1234, Q3 or 10-K
        if re.match(r"\s*-?[A-Za-z]", sentence[match.end(): match.end() + 2]) and not (match.group("scale") or match.group("unit")):
            if not re.match(r"\s*(?:per|in|on|of|to|from|and|vs|at|for|over|versus|is|was|were)\b", sentence[match.end():], re.IGNORECASE):
                continue
        number_text = match.group("num").replace(",", "")
        try:
            value = float(number_text)
        except ValueError:
            continue
        unit_raw = (match.group("unit") or "").lower()
        scale_raw = (match.group("scale") or "").lower()
        currency = bool(match.group("cur"))
        if not (unit_raw or scale_raw or currency) and value.is_integer() and 1900 <= value <= 2100:
            continue
        decimals = len(number_text.split(".")[1]) if "." in number_text else 0
        sign = -1.0 if match.group("sign") else 1.0
        if unit_raw in {"%", "percent"}:
            kind, normalized, step = "percent", value / 100.0, 10 ** -decimals / 100.0
        elif unit_raw.startswith("percentage") or unit_raw == "pp":
            kind, normalized, step = "points", value / 100.0, 10 ** -decimals / 100.0
        elif unit_raw in {"bps", "basis points"}:
            kind, normalized, step = "points", value / 10000.0, 10 ** -decimals / 10000.0
        elif unit_raw in {"x", "times"}:
            kind, normalized, step = "multiple", value, 10 ** -decimals
        elif scale_raw:
            scale = SCALES[scale_raw]
            kind, normalized, step = "currency", value * scale, 10 ** -decimals * scale
        elif currency:
            kind, normalized, step = "currency_small", value, 10 ** -decimals
        else:
            kind, normalized, step = "plain", value, 10 ** -decimals
        found.append({"start": match.start(), "end": match.end(), "text": match.group(0).strip(), "kind": kind,
                      "value": sign * normalized, "step": step})
    return found


def _compatible(kind: str, metric: str) -> bool:
    definition = get_metric(metric)
    unit = definition.unit
    if kind == "percent":
        return unit in {"ratio", "percent"} or unit.startswith("USD") or unit == "shares"
    if kind == "points":
        return unit in {"ratio", "percent"}
    if kind == "currency":
        return unit == "USD"
    if kind == "currency_small":
        return unit in {"USD/share", "USD"}
    if kind == "multiple":
        return unit in {"x", "ratio"}
    return unit in {"x", "USD/share", "ratio", "days"}


def _nearest(items: Sequence[Any], position: int, *, before_window: int, after_window: int, key=lambda m: (m.start, m.end)) -> Optional[Any]:
    best, best_distance = None, None
    for item in items:
        start, end = key(item)
        if end <= position:
            distance = position - end
            if distance > before_window:
                continue
        else:
            distance = start - position + 15  # bias towards mentions preceding the number
            if start - position > after_window:
                continue
        if best_distance is None or distance < best_distance:
            best, best_distance = item, distance
    return best


def extract_claims(sections: Dict[str, str]) -> List[Claim]:
    claims: List[Claim] = []

    def new_id() -> str:
        return f"C{len(claims) + 1:02d}"

    seen_sentences: set = set()
    for section, text in sections.items():
        for sentence in split_sentences(text):
            if sentence in seen_sentences:
                continue  # the same sentence repeated in another section is verified once
            seen_sentences.add(sentence)
            citations = EVIDENCE_ID.findall(sentence)
            clean = EVIDENCE_ID.sub(lambda m: " " * len(m.group(0)), sentence)
            clean = ISO_DATE.sub(lambda m: " " * len(m.group(0)), clean)
            years = _year_mentions(clean)
            mentions = _metric_mentions(clean)
            numbers = _numbers(clean, years)
            lowered = clean.lower()
            framing = re.search(FRAMING, lowered) is not None

            for verdict_pattern, verdict in VERDICT_PATTERNS:
                if re.search(r"\b(?:hypothes\w+|thesis|verdict|conclusion)\b", lowered) and re.search(verdict_pattern, lowered):
                    claims.append(Claim(claim_id=new_id(), section=section, text=sentence, category="verdict",
                                        reported_text=verdict, direction=verdict))
                    break

            numeric_metrics: List[str] = []
            for number in numbers:
                compatible = [m for m in mentions if _compatible(number["kind"], m.metric)]
                mention = _nearest(compatible, number["start"], before_window=90, after_window=40)
                local_cites = _citations_for(sentence, number["end"], numbers, citations)
                if mention is None and local_cites and len(numbers) > 1:
                    # No metric name nearby, but the figure carries its own citation: the cited
                    # record defines the metric (resolved during verification).
                    year = _nearest(years, number["start"], before_window=60, after_window=45, key=lambda y: (y[0], y[1]))
                    claims.append(Claim(
                        claim_id=new_id(), section=section, text=sentence, category="cited", metric=None,
                        reported_text=number["text"], reported_value=number["value"], reported_unit=number["kind"],
                        period_year=year[2] if year else None, cited_evidence=local_cites, tolerance=number["step"],
                    ))
                    continue
                if mention is None:
                    if number["kind"] in {"percent", "currency", "points", "multiple"}:
                        claims.append(Claim(claim_id=new_id(), section=section, text=sentence, category="other",
                                            reported_text=number["text"], reported_value=number["value"],
                                            cited_evidence=citations, status="NOT_VERIFIABLE",
                                            reason="Number could not be linked to a known financial metric."))
                    continue
                metric = mention.metric
                context = lowered[max(0, mention.start - 5): number["end"] + 25]
                if number["kind"] == "percent" and get_metric(metric).unit.startswith("USD"):
                    metric = f"{metric}_cagr" if "cagr" in context or "compound" in context else f"{metric}_growth_yoy"
                    if not re.search(GROWTH_WORDS + "|cagr|compound", context):
                        metric = metric  # percent attached to an amount is read as growth
                if number["kind"] == "points":
                    metric = f"{metric}_change"
                if re.search(r"\bttm\b|trailing twelve|trailing 12", lowered) and metric in TTM_VARIANT:
                    metric = TTM_VARIANT[metric]
                year = _nearest(years, number["start"], before_window=60, after_window=45, key=lambda y: (y[0], y[1]))
                prefix = lowered[max(0, number["start"] - 25): number["start"]]
                comparator = "eq"
                if re.search(APPROX_WORDS, prefix):
                    comparator = "approx"
                elif re.search(GREATER_WORDS, prefix):
                    comparator = "gt"
                elif re.search(LESS_WORDS, prefix):
                    comparator = "lt"
                cited = _citations_for(sentence, number["end"], numbers, citations)
                value = number["value"]
                if metric.split("_growth")[0] in METRICS and re.search(DECREASE, context) and value > 0 and metric.endswith("_growth_yoy"):
                    value = -value  # "revenue fell 5%" -> growth of -5%
                claims.append(Claim(
                    claim_id=new_id(), section=section, text=sentence,
                    category=CATEGORY_BY_METRIC_CATEGORY.get(metric_category(metric), "other"),
                    metric=metric, reported_text=number["text"], reported_value=value,
                    reported_unit=number["kind"], comparator=comparator,
                    period_year=year[2] if year else None, cited_evidence=cited, tolerance=number["step"],
                ))
                numeric_metrics.append(mention.metric)

            if not framing:
                for mention in mentions:
                    if mention.metric in numeric_metrics:
                        continue
                    window = lowered[max(0, mention.start - 30): mention.end + 40]
                    direction = None
                    if re.search(INCREASE, window):
                        direction = "increase"
                    if re.search(DECREASE, window):
                        direction = "decrease" if direction is None else "ambiguous"
                    if direction in {None, "ambiguous"}:
                        continue
                    year = _nearest(years, mention.start, before_window=60, after_window=45, key=lambda y: (y[0], y[1]))
                    claims.append(Claim(
                        claim_id=new_id(), section=section, text=sentence, category="trend",
                        metric=mention.metric, direction=direction, period_year=year[2] if year else None,
                        cited_evidence=citations,
                    ))
                    numeric_metrics.append(mention.metric)

            if not numbers and not mentions and re.search(QUALITATIVE, lowered) and not framing:
                claims.append(Claim(claim_id=new_id(), section=section, text=sentence, category="qualitative",
                                    status="NOT_VERIFIABLE",
                                    reason="Qualitative assertion; JASPER collected no structured evidence on this topic."))
    return claims


def _citations_for(sentence: str, number_end: int, numbers: Sequence[Dict[str, Any]], citations: List[str]) -> List[str]:
    """Citations between this number and the next number belong to it.

    A sentence-level citation is only attributed to a number when the sentence has a single number;
    otherwise a citation placed after one figure must not vouch for a different figure.
    """

    next_start = min((n["start"] for n in numbers if n["start"] > number_end), default=len(sentence))
    local = EVIDENCE_ID.findall(sentence[number_end:next_start])
    if local:
        return local
    return citations if len(numbers) == 1 else []


# ----------------------------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------------------------
@dataclass
class _Candidate:
    value: float
    unit: str
    period_year: Optional[int]
    period: str
    evidence_ids: List[str]
    metric: str
    calculation: str
    is_latest: bool = False
    note: Optional[str] = None


def _display(value: float, unit: str) -> str:
    if unit in {"ratio", "ratio_points"}:
        return f"{value * 100:.2f}%" if unit == "ratio" else f"{value * 100:+.2f}pp"
    if unit == "percent":
        return f"{value:.2f}%"
    if unit in {"x", "correlation"}:
        return f"{value:.2f}x" if unit == "x" else f"{value:.2f}"
    if abs(value) >= 1e9:
        return f"{value / 1e9:,.3f}B"
    if abs(value) >= 1e6:
        return f"{value / 1e6:,.2f}M"
    return f"{value:,.4g}"


class ClaimVerifier:
    def __init__(self, registry: EvidenceRegistry, ticker: str, verdict: Optional[str] = None,
                 loader: Optional[PayloadLoader] = None) -> None:
        self.registry = registry
        self.ticker = ticker
        self.verdict = verdict
        self.loader = loader

    # -- candidates ----------------------------------------------------------------------------
    def _record_candidate(self, record: EvidenceRecord) -> _Candidate:
        value = record.value
        note = None
        if record.kind == "derived":
            try:
                fresh = recompute(self.registry, record, self.loader)
            except (AnalyticsError, KeyError, ZeroDivisionError) as exc:
                fresh, note = record.value, f"could not recompute ({exc})"
            if not math.isclose(fresh, record.value, rel_tol=1e-9, abs_tol=1e-12):
                note = f"stored value {record.value:g} did not reproduce; recomputed {fresh:g} used"
                value = fresh
        calc = record.derivation.formula if record.derivation else f"{record.source_function}{record.field_path or ''}"
        return _Candidate(value=value, unit=record.unit, period_year=record.period_year, period=record.period,
                          evidence_ids=[record.evidence_id], metric=record.metric, calculation=calc, note=note)

    def _derived_on_the_fly(self, metric: str) -> List[_Candidate]:
        """Recompute growth/ratio candidates from raw inputs when the analysis never registered them."""

        out: List[_Candidate] = []
        if metric.endswith("_growth_yoy"):
            base = metric[: -len("_growth_yoy")]
            series = self._ratio_series(base)
            for prev, cur in zip(series, series[1:]):
                if prev.value == 0:
                    continue
                out.append(_Candidate(value=(cur.value - prev.value) / abs(prev.value), unit="ratio",
                                      period_year=cur.period_year, period=cur.period,
                                      evidence_ids=prev.evidence_ids + cur.evidence_ids, metric=metric,
                                      calculation=f"({base} FY{cur.period_year} - FY{prev.period_year}) / |FY{prev.period_year}| (recomputed during verification)"))
        elif metric.endswith("_change"):
            base = metric[: -len("_change")]
            series = self._ratio_series(base)
            pairs = list(zip(series, series[1:]))
            if len(series) > 2:
                pairs.append((series[0], series[-1]))
            for a, b in pairs:
                out.append(_Candidate(value=b.value - a.value, unit="ratio_points", period_year=b.period_year,
                                      period=f"FY{a.period_year} to FY{b.period_year}", evidence_ids=a.evidence_ids + b.evidence_ids,
                                      metric=metric, calculation=f"{base} FY{b.period_year} - FY{a.period_year} (recomputed)"))
        elif metric in RATIO_RECIPES:
            out = self._ratio_series(metric)
        return out

    def _ratio_series(self, metric: str) -> List[_Candidate]:
        """Annual series as candidates: registered records, or a ratio recomputed from raw inputs."""

        if metric not in RATIO_RECIPES:
            return [self._record_candidate(r) for r in self.registry.series(self.ticker, metric, PeriodType.ANNUAL)]
        existing = self.registry.series(self.ticker, metric, PeriodType.ANNUAL)
        if existing:
            return [self._record_candidate(r) for r in existing]
        num_key, den_key, scale, sign = RATIO_RECIPES[metric]
        nums = {r.period_end: r for r in self.registry.series(self.ticker, num_key, PeriodType.ANNUAL)}
        dens = {r.period_end: r for r in self.registry.series(self.ticker, den_key, PeriodType.ANNUAL)}
        out = []
        for period_end in sorted(set(nums) & set(dens)):
            if dens[period_end].value == 0:
                continue
            out.append(_Candidate(value=sign * nums[period_end].value / dens[period_end].value * scale, unit=METRICS[metric].unit,
                                  period_year=nums[period_end].period_year, period=nums[period_end].period,
                                  evidence_ids=[nums[period_end].evidence_id, dens[period_end].evidence_id], metric=metric,
                                  calculation=f"{num_key} / {den_key} (recomputed during verification)"))
        return out

    def _candidates(self, metric: str, cited: Sequence[EvidenceRecord]) -> Tuple[List[_Candidate], bool]:
        family = METRIC_FAMILY.get(metric, [metric])
        cited_match = [r for r in cited if r.metric in family or r.metric == metric]
        if cited_match:
            return [self._record_candidate(r) for r in cited_match], True
        if cited:
            # The citation pins the number to a record of a different metric than the wording suggests.
            others = []
            for record in cited:
                candidate = self._record_candidate(record)
                candidate.note = (candidate.note + "; " if candidate.note else "") + (
                    f"cited evidence is {record.metric}, while the wording suggests {metric}")
                others.append(candidate)
            return others, True
        candidates: List[_Candidate] = []
        for alt in family:
            records = [r for r in self.registry.find(ticker=self.ticker, metric=alt)
                       if r.period_type in {PeriodType.ANNUAL, PeriodType.TTM, PeriodType.POINT_IN_TIME, PeriodType.RANGE}]
            for record in records:
                candidate = self._record_candidate(record)
                if alt != metric:
                    candidate.note = (candidate.note + "; " if candidate.note else "") + f"matched alternative source metric {alt}"
                candidates.append(candidate)
        if not any(c.metric == metric for c in candidates):
            candidates = self._derived_on_the_fly(metric) + candidates
        annual = [c for c in candidates if c.metric == metric and c.period_year is not None]
        if annual:
            latest_year = max(c.period_year for c in annual)  # type: ignore[type-var]
            for c in annual:
                c.is_latest = c.period_year == latest_year
        elif candidates:
            candidates[-1].is_latest = True
        return candidates, False

    @staticmethod
    def _sentence_names_metric(text: str, metric: str) -> bool:
        lowered = text.lower()
        try:
            definition = get_metric(metric)
        except KeyError:
            return False
        words = [definition.label.lower()] + [s.lower() for s in definition.synonyms]
        base = metric.split("_growth")[0].split("_cagr")[0]
        if base != metric and base in METRICS:
            words += [METRICS[base].label.lower()] + [s.lower() for s in METRICS[base].synonyms]
        return any(w and w in lowered for w in words)

    # -- comparisons ---------------------------------------------------------------------------
    @staticmethod
    def _comparable(claim: Claim, candidate: _Candidate) -> Tuple[float, float]:
        reported = claim.reported_value or 0.0
        expected = candidate.value
        if candidate.unit == "percent" and claim.reported_unit in {"percent", "points"}:
            expected = candidate.value / 100.0
        if candidate.metric in ALWAYS_ABSOLUTE:
            return abs(reported), abs(expected)
        if expected < 0 < reported and re.search(NEGATIVE_WORDS, claim.text.lower()):
            return reported, abs(expected)
        return reported, expected

    @staticmethod
    def _match_level(claim: Claim, reported: float, expected: float) -> str:
        if claim.comparator == "gt":
            return "exact" if expected > reported else "none"
        if claim.comparator == "lt":
            return "exact" if expected < reported else "none"
        step = claim.tolerance or 0.0
        diff = abs(reported - expected)
        exact_tol = step / 2 + 1e-9 + abs(expected) * 1e-6
        if claim.comparator == "approx":
            exact_tol = max(exact_tol, abs(expected) * 0.025)
        if diff <= exact_tol:
            return "exact"
        if diff <= max(step * 1.01, abs(expected) * 0.05):
            return "approx"
        return "none"

    def verify(self, claim: Claim) -> Claim:
        if claim.category == "verdict":
            return self._verify_verdict(claim)
        if claim.category == "cited" and claim.metric is None:
            resolved = next((self.registry.records[e] for e in claim.cited_evidence if e in self.registry.records), None)
            if resolved is None:
                claim.status = "UNSUPPORTED"
                claim.reason = f"Cited evidence {', '.join(claim.cited_evidence)} does not exist in the registry."
                claim.missing_evidence = "The cited evidence record"
                return claim
            claim.metric = resolved.metric
            claim.category = CATEGORY_BY_METRIC_CATEGORY.get(metric_category(resolved.metric), "other")
        if claim.status == "NOT_VERIFIABLE" and claim.metric is None:
            return claim
        cited_records = [self.registry.records[e] for e in claim.cited_evidence if e in self.registry.records]
        unknown = [e for e in claim.cited_evidence if e not in self.registry.records]
        if claim.category == "trend":
            return self._verify_trend(claim, unknown)
        assert claim.metric is not None
        candidates, from_citation = self._candidates(claim.metric, cited_records)
        prefix = f"Cited evidence {', '.join(unknown)} does not exist in the registry. " if unknown else ""
        if not candidates:
            missing = [m.reason for m in self.registry.missing if m.ticker == self.ticker and m.metric in {claim.metric, claim.metric.split('_growth')[0]}]
            claim.status = "UNSUPPORTED"
            claim.reason = prefix + f"No collected evidence for {get_metric(claim.metric).label}." + (f" Recorded gap: {missing[-1]}" if missing else "")
            claim.missing_evidence = f"{get_metric(claim.metric).label} ({'FY' + str(claim.period_year) if claim.period_year else 'any period'})"
            return claim
        pool = candidates
        if claim.period_year is not None and not from_citation:
            in_period = [c for c in candidates if c.period_year == claim.period_year]
            if not in_period:
                claim.status = "UNSUPPORTED"
                available = sorted({c.period_year for c in candidates if c.period_year})
                claim.reason = prefix + f"No {get_metric(claim.metric).label} evidence for FY{claim.period_year}; available years: {available}."
                claim.missing_evidence = f"{get_metric(claim.metric).label} for FY{claim.period_year}"
                return claim
            pool = in_period
        scored = []
        for candidate in pool:
            reported, expected = self._comparable(claim, candidate)
            scored.append((candidate, self._match_level(claim, reported, expected), abs(reported - expected), expected))
        rank = {"exact": 0, "approx": 1, "none": 2}
        primary_metric = claim.metric
        scored.sort(key=lambda s: (rank[s[1]], s[0].metric != primary_metric, not s[0].is_latest, s[2]))
        best, level, _, expected = scored[0]
        claim.expected_value = expected
        claim.expected_display = _display(best.value, best.unit)
        claim.matched_evidence = best.evidence_ids
        claim.calculation = best.calculation
        notes = [n for n in [best.note] if n]
        family = METRIC_FAMILY.get(primary_metric, [primary_metric])
        cited_other_metric = from_citation and best.metric not in family and best.metric != primary_metric
        if cited_other_metric and self._sentence_names_metric(claim.text, best.metric):
            cited_other_metric = False  # e.g. "PEG ratio" wording citing a peg_computed record
        exact_primary = level == "exact" and best.metric == primary_metric
        if level == "exact" and (best.is_latest or claim.period_year is not None or from_citation or len(pool) == 1):
            claim.status = "VERIFIED" if ((exact_primary or from_citation) and not cited_other_metric) else "PARTIALLY_VERIFIED"
            claim.reason = prefix + ("Matches evidence within rounding tolerance." if claim.status == "VERIFIED"
                                     else f"Matches a related source ({best.metric}) rather than the stated metric.")
        elif level == "exact":
            claim.status = "PARTIALLY_VERIFIED"
            claim.reason = prefix + f"Matches the {best.period} value, but the text does not state that period (latest period differs)."
        elif level == "approx":
            claim.status = "PARTIALLY_VERIFIED"
            claim.reason = prefix + f"Approximately consistent: reported {claim.reported_text} vs evidence {claim.expected_display} ({best.period})."
        else:
            unambiguous = from_citation or claim.period_year is not None or len({c.period for c in pool if c.metric == primary_metric}) == 1
            if unambiguous:
                claim.status = "CONTRADICTED"
                claim.reason = prefix + f"Evidence shows {claim.expected_display} for {best.period}; text reports {claim.reported_text}."
            else:
                claim.status = "UNSUPPORTED"
                claim.reason = prefix + (f"No collected period matches {claim.reported_text}; closest is {claim.expected_display} ({best.period}). "
                                         "Period not stated, so this is not treated as a contradiction.")
                claim.missing_evidence = "Explicit period or evidence citation for the stated figure"
        if notes:
            claim.reason += " Note: " + "; ".join(notes) + "."
        if unknown and claim.status == "VERIFIED":
            claim.status = "PARTIALLY_VERIFIED"
        return claim

    def _verify_trend(self, claim: Claim, unknown: List[str]) -> Claim:
        assert claim.metric is not None
        records = self.registry.series(self.ticker, claim.metric, PeriodType.ANNUAL)
        series = [self._record_candidate(r) for r in records] or self._derived_on_the_fly(claim.metric)
        if len(series) < 2:
            claim.status = "UNSUPPORTED"
            claim.reason = f"Fewer than two periods of {get_metric(claim.metric).label} collected; direction cannot be checked."
            claim.missing_evidence = f"{get_metric(claim.metric).label} history"
            return claim
        is_ratio = series[0].unit in {"ratio", "x", "days", "ratio_points"}
        wanted = "increasing" if claim.direction == "increase" else "decreasing"
        opposite = "decreasing" if wanted == "increasing" else "increasing"
        claim.matched_evidence = [e for c in series for e in c.evidence_ids]
        if claim.period_year is not None:
            by_year = {c.period_year: c for c in series}
            if claim.period_year not in by_year or (claim.period_year - 1) not in by_year:
                claim.status = "UNSUPPORTED"
                claim.reason = f"Need FY{claim.period_year - 1} and FY{claim.period_year} {get_metric(claim.metric).label} to check direction."
                claim.missing_evidence = f"{get_metric(claim.metric).label} FY{claim.period_year - 1}-FY{claim.period_year}"
                return claim
            steps, overall = classify_steps([by_year[claim.period_year - 1].value, by_year[claim.period_year].value], is_ratio)
            claim.expected_display = f"{overall} in FY{claim.period_year}"
            claim.status = "VERIFIED" if overall == wanted else ("CONTRADICTED" if overall == opposite else "PARTIALLY_VERIFIED")
            claim.reason = f"FY{claim.period_year} vs FY{claim.period_year - 1}: {overall}."
            return claim
        steps, overall = classify_steps([c.value for c in series], is_ratio)
        latest = steps[-1]
        first_label, last_label = f"FY{series[0].period_year}", f"FY{series[-1].period_year}"
        claim.expected_display = f"{overall} {first_label}-{last_label}; latest step {latest}"
        if overall == wanted and latest != opposite:
            claim.status, claim.reason = "VERIFIED", f"{first_label} to {last_label} was {overall} and the latest period did not reverse."
        elif overall == opposite and latest != wanted:
            claim.status, claim.reason = "CONTRADICTED", f"{first_label} to {last_label} was {overall}; latest period {latest}."
        else:
            claim.status = "PARTIALLY_VERIFIED"
            claim.reason = f"Depends on the period: {first_label} to {last_label} was {overall} but the latest period was {latest}."
        if unknown:
            claim.reason = f"Cited evidence {', '.join(unknown)} does not exist. " + claim.reason
            if claim.status == "VERIFIED":
                claim.status = "PARTIALLY_VERIFIED"
        return claim

    def _verify_verdict(self, claim: Claim) -> Claim:
        if self.verdict is None:
            claim.status = "NOT_VERIFIABLE"
            claim.reason = "No deterministic verdict was available to compare against."
            return claim
        claim.expected_display = self.verdict
        if claim.direction == self.verdict:
            claim.status, claim.reason = "VERIFIED", "Matches the deterministic evidence-based verdict."
        else:
            claim.status = "CONTRADICTED"
            claim.reason = f"Text states {claim.direction}; the evidence-based verdict is {self.verdict}."
        return claim


def verify_text(
    sections: Dict[str, str],
    registry: EvidenceRegistry,
    ticker: str,
    *,
    verdict: Optional[str] = None,
    loader: Optional[PayloadLoader] = None,
) -> ClaimReport:
    verifier = ClaimVerifier(registry, ticker, verdict, loader)
    claims = [verifier.verify(c) for c in extract_claims(sections)]
    counts: Dict[str, int] = {}
    for claim in claims:
        counts[claim.status] = counts.get(claim.status, 0) + 1
    numeric = [c for c in claims if c.reported_value is not None and c.metric is not None]
    unresolved = sorted({e for c in claims for e in c.cited_evidence if e not in registry.records})
    return ClaimReport(claims=claims, counts=counts, numeric_claims=len(numeric),
                       numeric_verified=sum(1 for c in numeric if c.status == "VERIFIED"),
                       unresolved_citations=unresolved)


def narrative_feedback(report: ClaimReport, limit: int = 8) -> List[str]:
    """Plain-language corrections fed back to the narrator on a retry."""

    lines = []
    for claim in report.claims:
        if claim.status in {"CONTRADICTED", "UNSUPPORTED"} and claim.category != "qualitative":
            lines.append(f"{claim.claim_id} [{claim.status}] '{claim.text}' -> {claim.reason}")
    return lines[:limit]
