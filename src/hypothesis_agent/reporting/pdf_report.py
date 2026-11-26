"""Evidence-backed PDF research report.

The report is built from recorded research state only. Every figure in the quantitative sections
is printed next to its evidence ID; the source register maps evidence to snapshots and exact
yfinance calls; narrative sentences carry claim-verification markers.
"""
from __future__ import annotations

import io
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple
from xml.sax.saxutils import escape

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from hypothesis_agent.models.hypothesis import HypothesisRequest
from hypothesis_agent.research.agent import display_value
from hypothesis_agent.research.claims import EVIDENCE_ID, split_sentences
from hypothesis_agent.research.evidence import EvidenceRegistry
from hypothesis_agent.storage.artifact_store import ArtifactStore

PRIMARY = HexColor("#1a1a2e")
ACCENT = HexColor("#0f3460")
HIGHLIGHT = HexColor("#e94560")
GRID = HexColor("#d0d4dc")
SOFT = HexColor("#f3f5f9")
STATUS_COLORS = {
    "VERIFIED": "#0a7d32", "PARTIALLY_VERIFIED": "#b07200", "UNSUPPORTED": "#b03a2e",
    "CONTRADICTED": "#c0002a", "NOT_VERIFIABLE": "#5b6270",
}
VERDICT_COLORS = {"SUPPORTED": "#0a7d32", "PARTIALLY_SUPPORTED": "#b07200", "NOT_SUPPORTED": "#c0002a",
                  "INSUFFICIENT_EVIDENCE": "#5b6270"}
SECTION_TITLES = [
    "1. Investment Hypothesis",
    "2. Executive Summary",
    "3. Final Verdict",
    "4. Supporting Evidence",
    "5. Contradictory Evidence",
    "6. Quantitative Analysis",
    "7. Charts",
    "8. Research Methodology",
    "9. Evidence Coverage and Data Completeness",
    "10. Structured Uncertainty Assessment",
    "11. Unsupported / Unverified Claims",
    "12. Research Iterations and Plan Revisions",
    "13. Analytical Limitations",
    "14. Source Register and Data Snapshots",
]


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("RTitle", parent=base["Heading1"], fontSize=24, textColor=PRIMARY, alignment=TA_CENTER, spaceAfter=4),
        "subtitle": ParagraphStyle("RSub", parent=base["Normal"], fontSize=10, textColor=ACCENT, alignment=TA_CENTER, spaceAfter=14),
        "h2": ParagraphStyle("RH2", parent=base["Heading2"], fontSize=13.5, textColor=PRIMARY, spaceBefore=14, spaceAfter=6),
        "h3": ParagraphStyle("RH3", parent=base["Heading3"], fontSize=10.5, textColor=ACCENT, spaceBefore=6, spaceAfter=3),
        "body": ParagraphStyle("RBody", parent=base["Normal"], fontSize=9.5, leading=13.5, alignment=TA_JUSTIFY, spaceAfter=5),
        "bullet": ParagraphStyle("RBullet", parent=base["Normal"], fontSize=9.5, leading=13, leftIndent=12, spaceAfter=3),
        "cell": ParagraphStyle("RCell", parent=base["Normal"], fontSize=7.6, leading=9.6),
        "cellb": ParagraphStyle("RCellB", parent=base["Normal"], fontSize=7.6, leading=9.6, fontName="Helvetica-Bold"),
        "caption": ParagraphStyle("RCap", parent=base["Normal"], fontSize=7.2, leading=9, textColor=ACCENT, spaceAfter=10),
        "verdict": ParagraphStyle("RVerdict", parent=base["Normal"], fontSize=15, leading=19, alignment=TA_CENTER),
        "footer": ParagraphStyle("RFoot", parent=base["Normal"], fontSize=7.5, textColor=ACCENT, alignment=TA_CENTER),
    }


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text)), style)


def _table(rows: List[List[Any]], widths: Sequence[float], styles: Dict[str, ParagraphStyle], header: bool = True) -> Table:
    data = []
    for r_index, row in enumerate(rows):
        cells = []
        for cell in row:
            if isinstance(cell, (Paragraph, Image)):
                cells.append(cell)
            else:
                cells.append(_p(cell if cell is not None else "", styles["cellb"] if header and r_index == 0 else styles["cell"]))
        data.append(cells)
    table = Table(data, colWidths=list(widths), repeatRows=1 if header else 0)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.3, GRID),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), SOFT))
    table.setStyle(TableStyle(commands))
    return table


def _bullets(items: Sequence[str], styles: Dict[str, ParagraphStyle], story: List[Any], empty: str) -> None:
    if not items:
        story.append(_p(empty, styles["body"]))
        return
    for item in items:
        story.append(Paragraph(f"&bull; {escape(str(item))}", styles["bullet"]))


def _period_short(period: str) -> str:
    return period.split(" (")[0]


def _annotate(text: str, claims_by_sentence: Dict[str, List[Dict[str, Any]]]) -> str:
    """Escape narrative text and append claim-status markers after sentences that carry claims."""

    out = []
    for sentence in split_sentences(text):
        marked = escape(sentence)
        tags = []
        for claim in claims_by_sentence.get(sentence, []):
            color = STATUS_COLORS.get(claim["status"], "#5b6270")
            tags.append(f'<font color="{color}" size="7">[{claim["claim_id"]} {claim["status"].lower().replace("_", " ")}]</font>')
        out.append(marked + (" " + " ".join(tags) if tags else ""))
    return " ".join(out)


def render_research_report(
    *,
    request: HypothesisRequest,
    workflow_id: str,
    state: Dict[str, Any],
    artifact_store: ArtifactStore,
    compress: bool = True,
) -> Tuple[str, List[str]]:
    """Render the PDF; returns (file URI, list of section titles rendered)."""

    styles = _styles()
    registry = EvidenceRegistry.from_dict(state.get("registry"))
    plan = state.get("plan") or {}
    verdict = state.get("verdict") or {"verdict": "INSUFFICIENT_EVIDENCE", "rationale": "No assessment was produced."}
    assessments: List[Dict[str, Any]] = state.get("assessments", [])
    claims_report: Dict[str, Any] = state.get("claims") or {"claims": [], "counts": {}}
    claims: List[Dict[str, Any]] = claims_report.get("claims", [])
    uncertainty: Dict[str, Any] = state.get("uncertainty") or {}
    narratives = state.get("narratives", [])
    latest = narratives[-1] if narratives else {"narrative": {}, "report_payload": {}}
    narrative = latest.get("narrative") or {}
    payload = latest.get("report_payload") or {}
    claims_by_sentence: Dict[str, List[Dict[str, Any]]] = {}
    for claim in claims:
        claims_by_sentence.setdefault(claim["text"], []).append(claim)
    cited_ids: List[str] = []

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=20 * mm,
                            bottomMargin=18 * mm, pageCompression=1 if compress else 0, invariant=0 if compress else 1,
                            title=f"JASPER report {workflow_id}", author="JASPER")
    width = A4[0] - 36 * mm
    story: List[Any] = []
    rendered: List[str] = []

    def section(index: int) -> None:
        title = SECTION_TITLES[index]
        rendered.append(title)
        story.append(_p(title, styles["h2"]))

    story.append(_p("JASPER", styles["title"]))
    story.append(_p("Evidence-Grounded Investment Research Report", styles["subtitle"]))

    # 1. Hypothesis ---------------------------------------------------------------------------
    section(0)
    snapshots = list(registry.snapshots.values())
    modes = sorted({s.source_mode for s in snapshots}) or ["none"]
    story.append(_table([
        ["Field", "Value"],
        ["Hypothesis", request.hypothesis_text],
        ["Ticker", plan.get("ticker", ", ".join(request.entities) or "N/A")],
        ["Research objective", plan.get("objective", "")],
        ["Workflow ID", workflow_id],
        ["Data mode", ", ".join(modes)],
        ["Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")],
    ], [32 * mm, width - 32 * mm], styles))

    # 2. Executive summary --------------------------------------------------------------------
    section(1)
    summary_text = narrative.get("executive_summary") or payload.get("executive_summary") or "No executive summary was generated."
    story.append(Paragraph(_annotate(str(summary_text), claims_by_sentence), styles["body"]))
    body = narrative.get("narrative")
    if body:
        story.append(_p("Analysis narrative (generated; claims checked against evidence)", styles["h3"]))
        for paragraph in str(body).split("\n\n"):
            story.append(Paragraph(_annotate(paragraph, claims_by_sentence), styles["body"]))
    if payload.get("key_findings"):
        story.append(_p("Key findings", styles["h3"]))
        for finding in payload["key_findings"]:
            story.append(Paragraph("&bull; " + _annotate(str(finding), claims_by_sentence), styles["bullet"]))

    # 3. Verdict ------------------------------------------------------------------------------
    section(2)
    label = verdict.get("verdict", "INSUFFICIENT_EVIDENCE")
    color = VERDICT_COLORS.get(label, "#5b6270")
    story.append(Paragraph(f'<font color="{color}"><b>{escape(label.replace("_", " "))}</b></font>', styles["verdict"]))
    story.append(Spacer(1, 4))
    story.append(_p(f"Rule-based verdict (not set by the LLM): {verdict.get('rationale', '')}", styles["body"]))
    human = state.get("human_review") or {}
    if human.get("requested"):
        story.append(Paragraph('<font color="#e94560"><b>Human review requested:</b></font> '
                               f"{escape(str(human.get('reason', '')))}", styles["body"]))
    decision = state.get("decision") or {}
    if decision:
        story.append(_p(f"Final research decision: {decision.get('status')} - {decision.get('rationale')}", styles["body"]))

    # 4/5. Supporting and contradicting evidence ----------------------------------------------
    def evidence_rows(selected: List[Dict[str, Any]]) -> List[List[Any]]:
        rows: List[List[Any]] = [["ID", "Expectation", "Finding", "Evidence"]]
        for a in selected:
            cited_ids.extend(a.get("evidence_ids", []))
            finding = a["summary"] + ("" if not a.get("caveats") else " Caveat: " + " ".join(a["caveats"]))
            rows.append([a["expectation_id"], f"{a['concept']} ({a['metric'] or 'n/a'}: {a['direction']})", finding,
                         ", ".join(a.get("evidence_ids", [])[:6]) or "none"])
        return rows

    section(3)
    supporting = [a for a in assessments if a["status"] in {"SUPPORTS", "MIXED_POSITIVE"}]
    if supporting:
        story.append(_table(evidence_rows(supporting), [12 * mm, 40 * mm, 80 * mm, width - 132 * mm], styles))
    else:
        story.append(_p("No tested expectation was supported by the collected evidence.", styles["body"]))
    section(4)
    contradicting = [a for a in assessments if a["status"] in {"CONTRADICTS", "MIXED_NEGATIVE", "MIXED_POSITIVE", "NEUTRAL"}]
    if contradicting:
        story.append(_table(evidence_rows(contradicting), [12 * mm, 40 * mm, 80 * mm, width - 132 * mm], styles))
    else:
        story.append(_p("No collected evidence contradicted the tested expectations.", styles["body"]))
    insufficient = [a for a in assessments if a["status"] == "INSUFFICIENT"]
    if insufficient:
        story.append(_p("Expectations that could not be evaluated", styles["h3"]))
        _bullets([f"{a['expectation_id']} ({a['concept']}): {a['summary']}" for a in insufficient], styles, story, "")

    # 6. Quantitative analysis ----------------------------------------------------------------
    section(5)
    rows: List[List[Any]] = [["Expectation", "Metric", "Test", "Window", "Status (score)"]]
    from hypothesis_agent.research.assessment import STATUS_SCORE
    for a in assessments:
        score = STATUS_SCORE.get(a["status"])
        rows.append([f"{a['expectation_id']} [{a['weight']}]", a["metric"] or "unmeasurable", a["direction"],
                     a.get("window") or "-", f"{a['status']}" + (f" ({score:.2f})" if score is not None else "")])
    story.append(_table(rows, [24 * mm, 40 * mm, 22 * mm, 34 * mm, width - 120 * mm], styles))
    story.append(Spacer(1, 6))
    story.append(_p("Key metrics, latest three periods (every value links to an evidence record in section 14)", styles["h3"]))
    metric_rows: List[List[Any]] = [["Metric", "Values by period [evidence ID]", "Source / derivation"]]
    for planned in plan.get("metrics", [])[:24]:
        if planned.get("frequency") != "annual":
            continue
        series = registry.series(plan.get("ticker", ""), planned["metric"]) or registry.find(ticker=plan.get("ticker"), metric=planned["metric"])
        if not series:
            continue
        values = []
        for record in series[-3:]:
            cited_ids.append(record.evidence_id)
            values.append(f"{_period_short(record.period)}: {display_value(record.value, record.unit)} [{record.evidence_id}]")
        last = series[-1]
        derivation = last.derivation.formula if last.derivation else f"{last.source_function}{last.field_path or ''}"
        metric_rows.append([last.label, "; ".join(values), derivation])
    if len(metric_rows) > 1:
        story.append(_table(metric_rows, [34 * mm, 82 * mm, width - 116 * mm], styles))
    else:
        story.append(_p("No metrics could be computed from the collected data.", styles["body"]))

    # 7. Charts -------------------------------------------------------------------------------
    section(6)
    charts = state.get("charts", [])
    if not charts:
        story.append(_p("No charts were produced (fewer than two periods of data for every tested metric).", styles["body"]))
    for chart in charts:
        path = chart.get("artifact_path")
        cited_ids.extend(chart.get("evidence_ids", []))
        try:
            image = Image(path, width=width * 0.92, height=width * 0.92 * 0.39)
        except Exception as exc:  # chart file missing: say so rather than dropping the attribution
            story.append(_p(f"{chart['chart_id']} could not be embedded ({exc}).", styles["body"]))
        else:
            story.append(KeepTogether([image, _p(chart.get("caption", ""), styles["caption"])]))

    # 8. Methodology --------------------------------------------------------------------------
    section(7)
    tool_calls = state.get("tool_calls", [])
    by_tool = Counter(c["tool"] for c in tool_calls)
    by_origin = Counter(c["origin"] for c in tool_calls)
    limits = state.get("limits") or {}
    _bullets([
        f"Plan: {len(plan.get('expectations', []))} expectation(s) parsed from the hypothesis "
        f"(sources: {', '.join(sorted({e['source'] for e in plan.get('expectations', [])})) or 'none'}); "
        f"{len(plan.get('metrics', []))} metrics planned; final plan version {plan.get('version', 1)}.",
        "Data: datasets retrieved through validated tools from Yahoo Finance (yfinance); each raw payload is stored and hashed (section 14).",
        "Calculations: all ratios, growth rates, trends and valuation checks are deterministic Python functions; the LLM never computes figures.",
        f"Tool calls: {len(tool_calls)} total ({', '.join(f'{k} x{v}' for k, v in by_tool.most_common())}); "
        f"by origin: {', '.join(f'{k} {v}' for k, v in by_origin.items())}.",
        "Trend rule: change below 2% (amounts) or 0.5 percentage points (ratios) is flat; with no explicit window, a latest-period "
        "reversal marks the expectation as MIXED.",
        "Verdict rule: concept-group mean of SUPPORTS=1, MIXED_POSITIVE=0.65, NEUTRAL=0.5, MIXED_NEGATIVE=0.35, CONTRADICTS=0; "
        ">=0.8 SUPPORTED, <=0.2 NOT SUPPORTED, otherwise PARTIALLY; any unmeasurable core expectation gives INSUFFICIENT EVIDENCE.",
        f"Bounds: max {limits.get('max_iterations', '?')} research iterations, {limits.get('max_tool_calls', '?')} tool calls, "
        f"{limits.get('max_narrative_attempts', '?')} narrative attempts.",
        "Claim verification: numbers in generated text are extracted and compared with evidence (recomputing derived values) using "
        "rounding-aware tolerances; see section 11.",
    ], styles, story, "")

    # 9. Coverage -----------------------------------------------------------------------------
    section(8)
    completeness = uncertainty.get("data_completeness") or {}
    requested = completeness.get("metrics_requested", [])
    collected = completeness.get("metrics_collected", [])
    story.append(_p(f"Metrics requested: {len(requested)}; collected or computed: {len(collected)} "
                    f"(coverage {completeness.get('coverage_ratio', 0):.0%}, a count ratio, not a probability).", styles["body"]))
    unavailable = completeness.get("metrics_unavailable", [])
    if unavailable:
        story.append(_table([["Unavailable metric", "Reason"]] + [[u["label"], u["reason"]] for u in unavailable],
                            [45 * mm, width - 45 * mm], styles))
    failed = completeness.get("failed_datasets", [])
    if failed:
        story.append(_p("Failed retrievals", styles["h3"]))
        _bullets([f"{f['dataset']}: {f['error']}" for f in failed], styles, story, "")
    if completeness.get("missing_periods"):
        story.append(_p("Periods returned empty by the source: " + ", ".join(completeness["missing_periods"]), styles["body"]))
    _bullets(completeness.get("limitations", []), styles, story, "No coverage limitations recorded.")

    # 10. Uncertainty -------------------------------------------------------------------------
    section(9)
    balance = uncertainty.get("evidence_balance") or {}
    story.append(_table([
        ["Evidence balance", "Count", "Expectations"],
        ["Supporting", len(balance.get("supporting", [])), ", ".join(e["expectation_id"] for e in balance.get("supporting", []))],
        ["Contradicting", len(balance.get("contradicting", [])), ", ".join(e["expectation_id"] for e in balance.get("contradicting", []))],
        ["Mixed / neutral", len(balance.get("mixed_or_neutral", [])), ", ".join(e["expectation_id"] for e in balance.get("mixed_or_neutral", []))],
        ["Insufficient evidence", len(balance.get("insufficient", [])), ", ".join(e["expectation_id"] for e in balance.get("insufficient", []))],
    ], [45 * mm, 18 * mm, width - 63 * mm], styles))
    quality = uncertainty.get("source_quality") or {}
    story.append(_p("Source quality", styles["h3"]))
    freshness = (f"Latest financial period ends {quality.get('latest_period_end')} "
                 f"({quality.get('staleness_days')} days before report date)." if quality.get("latest_period_end") else "No dated financial periods.")
    _bullets([f"Sources: {', '.join(quality.get('sources', [])) or 'none'}.", freshness]
             + quality.get("limitations", []) + [f"Inconsistency: {i}" for i in quality.get("inconsistencies", [])], styles, story, "")
    story.append(_p("JASPER does not report a numeric confidence score; the items above are the uncertainty assessment.", styles["body"]))

    # 11. Claims ------------------------------------------------------------------------------
    section(10)
    counts = claims_report.get("counts", {})
    story.append(_p("Claim verification summary: " + (", ".join(f"{k.lower().replace('_', ' ')} {v}" for k, v in sorted(counts.items())) or "no claims extracted")
                    + f". Numeric claims: {claims_report.get('numeric_claims', 0)}, verified: {claims_report.get('numeric_verified', 0)}.", styles["body"]))
    story.append(_p("Legend: CONTRADICTED = evidence for the same metric and period shows a different value. UNSUPPORTED = no evidence "
                    "found to confirm it; this is not proof the claim is false. NOT VERIFIABLE = qualitative or unlinked statement. "
                    "PARTIALLY VERIFIED = approximately right, or right for a different/unstated period or source.", styles["caption"]))
    problems = [c for c in claims if c["status"] != "VERIFIED"]
    if not problems:
        story.append(_p("All checked claims passed verification." if claims else "No verifiable claims were found in the generated text.", styles["body"]))
    else:
        rows = [["Claim", "Status", "Statement", "Why verification failed", "Missing evidence / source"]]
        for c in problems:
            source = c.get("missing_evidence") or ""
            if c.get("matched_evidence"):
                source = (source + "; " if source else "") + "closest: " + ", ".join(c["matched_evidence"][:3])
                cited_ids.extend(c["matched_evidence"][:3])
            if c.get("expected_display"):
                source += f" (evidence value {c['expected_display']})"
            rows.append([c["claim_id"], Paragraph(f'<font color="{STATUS_COLORS.get(c["status"], "#000")}">{c["status"]}</font>', styles["cell"]),
                         c["text"], c["reason"], source or "-"])
        story.append(_table(rows, [11 * mm, 24 * mm, 50 * mm, 52 * mm, width - 137 * mm], styles))

    # 12. Iterations --------------------------------------------------------------------------
    section(11)
    iterations = state.get("iterations", [])
    if iterations:
        rows = [["Iter.", "Plan v", "Evidence (+new)", "Tool calls", "Verdict", "Decision and rationale", "Plan revision"]]
        for it in iterations:
            revision = it.get("revision")
            revision_text = "-"
            if revision:
                revision_text = f"v{revision['version']}: +{', '.join(revision['added_metrics'])}"
                if revision.get("added_datasets"):
                    revision_text += f"; datasets +{', '.join(revision['added_datasets'])}"
            rows.append([it["iteration"], it["plan_version"], f"{it['evidence_records']} (+{it['new_evidence']})", it["tool_calls"],
                         it["verdict"], f"{it['decision']}: {it['rationale']}", revision_text])
        story.append(_table(rows, [10 * mm, 11 * mm, 20 * mm, 14 * mm, 25 * mm, 55 * mm, width - 135 * mm], styles))
    else:
        story.append(_p("No critique iterations were recorded.", styles["body"]))
    reported_issues: set = set()
    for critique in state.get("critiques", []):
        fresh = [i for i in critique.get("issues", []) if i["description"] not in reported_issues]
        reported_issues.update(i["description"] for i in critique.get("issues", []))
        if fresh:
            story.append(_p(f"Critique iteration {critique['iteration']}: new issues", styles["h3"]))
            _bullets([f"[{i['type']}, {i['severity']}, {i['source']}] {i['description']}" for i in fresh[:10]], styles, story, "")
        elif critique.get("issues"):
            story.append(_p(f"Critique iteration {critique['iteration']}: no new issues ({len(critique['issues'])} carried over).", styles["body"]))
    if len(narratives) > 1:
        story.append(_p("Narrative revisions", styles["h3"]))
        _bullets([f"Attempt {n['attempt']}: claim results {n.get('claim_counts', {})}" + (" - regenerated with verification feedback" if n.get("feedback") else "")
                  for n in narratives], styles, story, "")
    errors = state.get("errors", [])
    if errors:
        story.append(_p("Recorded errors (non-fatal)", styles["h3"]))
        _bullets([e["error"] for e in errors[:8]], styles, story, "")

    # 13. Limitations -------------------------------------------------------------------------
    section(12)
    limitations = list(uncertainty.get("analytical_limitations", []))
    limitations += [str(x) for x in narrative.get("limitations", []) if str(x) not in limitations]
    _bullets(limitations, styles, story, "No analytical limitations recorded.")

    # 14. Source register ---------------------------------------------------------------------
    section(13)
    rows = [["Snapshot", "Dataset", "Retrieval call", "Retrieved (UTC)", "Coverage", "Rows", "SHA-256 (prefix)", "Status"]]
    for s in snapshots:
        rows.append([s.snapshot_id, s.dataset, s.retrieval_function, s.retrieved_at, f"{s.coverage_start or '-'} to {s.coverage_end or '-'}",
                     s.record_count, (s.content_sha256 or "")[:16], s.status if not s.error else f"{s.status}: {s.error[:60]}"])
    story.append(_table(rows, [24 * mm, 22 * mm, 36 * mm, 23 * mm, 23 * mm, 10 * mm, 20 * mm, width - 158 * mm], styles))
    story.append(_p("Raw payloads are stored next to this report as snapshot_<ticker>_<dataset>_<snapshot>.json; "
                    "reproducibility_manifest.json and evidence_registry.json hold the full lineage.", styles["caption"]))
    story.append(_p("Evidence referenced in this report", styles["h3"]))
    seen = list(dict.fromkeys(e for e in cited_ids if registry.has(e)))
    for claim in claims:
        seen.extend(e for e in EVIDENCE_ID.findall(claim["text"]) if registry.has(e) and e not in seen)
    rows = [["Evidence ID", "Metric", "Period", "Value", "Lineage"]]
    for eid in seen[:120]:
        record = registry.get(eid)
        if record.kind == "raw":
            lineage = f"{record.snapshot_id} {record.field_path}"
        else:
            lineage = f"{record.derivation.formula} <- {', '.join(record.derivation.inputs or record.derivation.input_snapshots)}"
        rows.append([eid, record.label, record.period, display_value(record.value, record.unit), lineage])
    story.append(_table(rows, [23 * mm, 31 * mm, 29 * mm, 24 * mm, width - 107 * mm], styles))
    if len(seen) > 120:
        story.append(_p(f"{len(seen) - 120} further evidence records are listed in evidence_registry.json.", styles["caption"]))

    story.append(Spacer(1, 16))
    story.append(_p(f"Generated by JASPER - workflow {workflow_id}", styles["footer"]))
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    uri = artifact_store.write_bytes(workflow_id, "validation_report.pdf", pdf_bytes).resolve().as_uri()
    return uri, rendered
