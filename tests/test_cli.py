"""CLI entry points: research on frozen fixtures, then reproduce the run from disk."""
from __future__ import annotations

from pathlib import Path

import pytest

from hypothesis_agent.cli import main

FIXTURES = Path(__file__).resolve().parents[1] / "benchmarks" / "v1" / "fixtures"


def test_cli_research_then_reproduce(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main([
        "research", "--ticker", "TSLA", "--llm", "offline", "--fixtures", str(FIXTURES), "--out", str(tmp_path),
        "--workflow-id", "cli-test", "--hypothesis", "Tesla's operating margins are expanding as revenue scales.",
    ])
    output = capsys.readouterr().out
    assert code == 0
    assert "Verdict: NOT_SUPPORTED" in output and "Report:" in output and "Route:" in output
    assert main(["reproduce", str(tmp_path / "cli-test")]) == 0
    assert '"reproducible": true' in capsys.readouterr().out
