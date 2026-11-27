"""``python -m hypothesis_agent.evaluation ...`` is shorthand for ``jasper eval ...``."""
import sys

from hypothesis_agent.cli import main

raise SystemExit(main(["eval", *sys.argv[1:]]))
