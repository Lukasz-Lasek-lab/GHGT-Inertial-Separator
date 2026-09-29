"""
Unified Command-Line Interface (CLI) for MsCO2limit:
'Hybrid AI-driven Approach for Optimization and Geometric Analysis of Inertial Separators in Chemical Looping Systems'

Thin entry point delegating to src.cli.main().
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Delegates directly to src.cli.main()."""
    from src.cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
