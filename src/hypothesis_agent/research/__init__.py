"""Evidence-grounded research components for JASPER.

Modules:
    evidence     -- evidence registry, data snapshots, derivation lineage
    extraction   -- raw yfinance payload -> evidence records
    metrics      -- canonical metric catalogue shared by every component
    analytics    -- deterministic financial calculations (outside the LLM)
    tools        -- validated, budgeted tool interface the agent uses
    planner      -- hypothesis parsing and revisable research plans
    assessment   -- deterministic evaluation of hypothesis expectations and verdicts
    critique     -- structured critique and branching decisions
    claims       -- claim extraction and deterministic verification
    uncertainty  -- structured uncertainty assessment (no fabricated confidence)
"""
