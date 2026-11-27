"""Benchmark suite schema and loading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from hypothesis_agent.research.metrics import CATEGORIES

VerdictLabel = Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED", "INSUFFICIENT_EVIDENCE"]
CaseCategory = Literal[
    "supported", "contradicted", "partially_supported", "insufficient_evidence", "conflicting_metrics", "period_dependent",
]


class Caveat(BaseModel):
    description: str
    keywords: List[str] = Field(default_factory=list, description="Any keyword found in the run's audit text counts as covered.")


class BenchmarkCase(BaseModel):
    id: str
    category: CaseCategory
    ticker: str
    hypothesis: str
    research_objective: str
    expected_evidence_categories: List[str]
    expected_verdict: VerdictLabel
    acceptable_verdicts: List[VerdictLabel]
    important_caveats: List[Caveat] = Field(default_factory=list)
    evaluation_criteria: List[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _check(self) -> "BenchmarkCase":
        unknown = [c for c in self.expected_evidence_categories if c not in CATEGORIES]
        if unknown:
            raise ValueError(f"{self.id}: unknown evidence categories {unknown}")
        if self.expected_verdict not in self.acceptable_verdicts:
            raise ValueError(f"{self.id}: expected_verdict must be one of acceptable_verdicts")
        return self


class BenchmarkSuite(BaseModel):
    suite: str
    version: str
    description: str = ""
    verdict_scale: List[str] = Field(default_factory=lambda: ["NOT_SUPPORTED", "PARTIALLY_SUPPORTED", "SUPPORTED"])
    cases: List[BenchmarkCase]
    root: Optional[str] = None
    fixture_dir: Optional[str] = None
    content_sha256: Optional[str] = None

    @model_validator(mode="after")
    def _unique(self) -> "BenchmarkSuite":
        ids = [c.id for c in self.cases]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"Duplicate case ids: {sorted(duplicates)}")
        return self


def load_suite(
    suite_dir: str | Path,
    *,
    include_paraphrases: bool = False,
    include_holdout: bool = False,
    core: bool = True,
    case_ids: Optional[List[str]] = None,
) -> BenchmarkSuite:
    """Load cases.json plus optional paraphrases.json / holdout.json; verify every case has a fixture."""

    root = Path(suite_dir).resolve()
    payload = json.loads((root / "cases.json").read_text(encoding="utf-8"))
    files = [root / "cases.json"]
    if not core:
        payload["cases"] = []
    for enabled, name in ((include_paraphrases, "paraphrases.json"), (include_holdout, "holdout.json")):
        if enabled and (root / name).exists():
            extra = json.loads((root / name).read_text(encoding="utf-8"))
            payload["cases"] = payload["cases"] + extra["cases"]
            files.append(root / name)
    suite = BenchmarkSuite.model_validate(payload)
    fixture_dir = root / "fixtures"
    missing = [c.ticker for c in suite.cases if not (fixture_dir / f"{c.ticker.upper()}.json").exists()]
    if missing:
        raise FileNotFoundError(f"Missing fixtures for tickers: {sorted(set(missing))}")
    if case_ids:
        wanted = set(case_ids)
        unknown = wanted - {c.id for c in suite.cases}
        if unknown:
            raise KeyError(f"Unknown case ids: {sorted(unknown)}")
        suite.cases = [c for c in suite.cases if c.id in wanted]
    digest = hashlib.sha256()
    for path in files + sorted({fixture_dir / f"{c.ticker.upper()}.json" for c in suite.cases}):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    suite.root = str(root)
    suite.fixture_dir = str(fixture_dir)
    suite.content_sha256 = digest.hexdigest()
    return suite
