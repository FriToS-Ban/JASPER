"""Benchmark evaluation harness for JASPER.

benchmark -- suite schema and loading (expected answers live only in the suite files)
runner    -- runs JASPER per case and writes results.json + summary.md
scoring   -- deterministic, disk-based independent scoring
judge     -- rubric definitions, deterministic and blind LLM rubric judges
human     -- blind human review export/import with agreement statistics
compare   -- version-to-version comparison of results files
"""
