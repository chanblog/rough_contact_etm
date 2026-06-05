"""Backward-compatible launcher for post-processing."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rough_contact_etm.postprocessing import main


if __name__ == "__main__":
    main()
