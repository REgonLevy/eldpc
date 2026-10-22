#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the eldpc verification suite.

    python verify/run_verify.py --quick [--out DIR]

With the package installed (or its build directory on PYTHONPATH), this generates the suite's
corpora from public seeds, builds the reference decoder into a cache directory, decodes every shot
with the reference and with eldpc (single shot and batch), and writes DIR/summary.json and
DIR/summary.md. The exit status is 0 if and only if every row has 0 mismatches against the
reference. See docs/VERIFY.md.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eldpc_verify.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
