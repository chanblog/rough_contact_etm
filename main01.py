"""Backward-compatible launcher for the load-sweep simulation.

Prefer ``python -m rough_contact_etm.run_simulation`` after installing the
package, but this wrapper keeps the original ``python main01.py`` workflow.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rough_contact_etm.run_simulation import main


if __name__ == "__main__":
    main()
