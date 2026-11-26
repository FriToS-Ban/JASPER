"""Allow ``python -m hypothesis_agent ...`` as an alias for the ``jasper`` CLI."""
from hypothesis_agent.cli import main

raise SystemExit(main())
