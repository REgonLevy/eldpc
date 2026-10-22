#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""eldpc-tune: the knob tuner.

Finds the fastest decoder configuration (ms_scaling_factor, max_iter, decoder, keep-warm) that
meets a logical error rate target and a latency budget for a code and its noise, and writes a
config file the package loads.  The procedure is fixed; see README.md in this directory.

Usage:
    python tuning/eldpc_tune.py --problem bb144xyz --p 0.001 --ler-target 1e-4 \
        --p99-budget-us 60 --auto --out tune-bb144xyz
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eldpctune.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
