#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rewrite DIR/pareto.md from DIR/results.json with the tuner's own renderer.

The tuner writes both files in one run; this tool re-renders the Markdown from an existing
results file, so a committed result can get the current table layout without re-running the
measurement.  It computes nothing: every number comes from results.json.

    python tuning/tools/render_pareto.py tuning/results/<name>/results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TUNING = os.path.dirname(_HERE)
if _TUNING not in sys.path:
    sys.path.insert(0, _TUNING)

from eldpctune import report  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="path to a tuner results.json")
    args = parser.parse_args(argv)
    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)
    if not str(results.get("schema", "")).startswith("eldpc.tune."):
        print(f"{args.results}: not a tuner results file", file=sys.stderr)
        return 2
    out = os.path.join(os.path.dirname(os.path.abspath(args.results)), "pareto.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report.render_pareto_md(results))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
