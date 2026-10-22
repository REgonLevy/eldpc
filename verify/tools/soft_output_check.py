#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Optional check: eldpc's soft output against the reference's posterior LLRs.

The verification suite's pass criterion is the correction, the observables and (against a separately built
library, when one is given) the status word. This tool measures a stronger, informational property: whether the
posterior LLRs that ``eldpc.Decoder.soft_output()`` returns are bit-identical to the reference
decoder's ``dbench_get_soft_output()`` values for the same shot, on the suite's corpora.

    python verify/tools/soft_output_check.py --configs a03125c3,ms30 --out soft.json

For every (problem, preset) it decodes the shot set single-shot with both libraries, compares the
soft outputs entry by entry (both-NaN counts as equal), and reports the number of equal/differing
shots, differing entries, the maximum absolute difference and any hard-decision mismatch. Exit
status 0 when every soft output is bit-identical, 1 otherwise. This is an information-only check:
the exactness contract does not require float LLR identity, so a difference here is reported, not
treated as a suite failure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_VERIFY = str(Path(__file__).resolve().parents[1])      # verify/tools/x.py -> verify
if _VERIFY not in sys.path:
    sys.path.insert(0, _VERIFY)

from eldpc_verify.abi import AbiDecoder  # noqa: E402
from eldpc_verify.config import default_cache_root, load_config  # noqa: E402
from eldpc_verify.corpus import ensure_corpus  # noqa: E402
from eldpc_verify.reference import build_reference  # noqa: E402
from eldpc_verify.runner import config_string  # noqa: E402


def _split(text):
    return [x.strip() for x in text.split(",") if x.strip()] if text else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--problems", help="comma-separated problem ids (default: all)")
    parser.add_argument("--configs", default="a03125c3,ms30",
                        help="comma-separated preset names (default: a03125c3,ms30)")
    parser.add_argument("--shots", type=int, default=None,
                        help="shots per problem (default: the suite's quick shot count)")
    parser.add_argument("--cache", default=None,
                        help="corpus and reference cache (default: the suite's cache)")
    parser.add_argument("--out", help="write the JSON result here as well as to stdout")
    args = parser.parse_args(argv)

    import eldpc

    cfg = load_config()
    by_id = {p["id"]: p for p in cfg["problems"]}
    problems = _split(args.problems) or [p["id"] for p in cfg["problems"]]
    unknown = [p for p in problems if p not in by_id]
    if unknown:
        parser.error(f"unknown problem ids {unknown}")
    configs = _split(args.configs) or ["a03125c3", "ms30"]
    presets = eldpc.presets()
    unknown = [c for c in configs if c not in presets]
    if unknown:
        parser.error(f"unknown presets {unknown}")

    cache = Path(args.cache) if args.cache else default_cache_root()
    t0 = time.perf_counter()
    ref_info = build_reference(cache)
    ref_build_seconds = time.perf_counter() - t0

    rows = []
    totals = {"shots": 0, "equal_shots": 0, "differing_shots": 0, "differing_entries": 0,
              "total_entries": 0, "hard_mismatches": 0, "nan_entries": 0}
    max_abs_diff = 0.0
    for pid in problems:
        problem = by_id[pid]
        shots = int(args.shots) if args.shots else int(cfg["shots"]["quick"][pid])
        parent = parent_shots = None
        if problem["shape"]["kind"] == "window":
            parent = by_id[problem["shape"]["parent"]]
            parent_shots = int(cfg["shots"]["quick"][parent["id"]])
        corpus = ensure_corpus(cache, problem, cfg["base_seed"], shots,
                               parent=parent, parent_shots=parent_shots)
        for name in configs:
            cs = config_string(presets[name])
            reference = AbiDecoder(ref_info.path, cs, corpus.H, corpus.O, corpus.priors,
                                   corpus.dem_text)
            decoder = eldpc.Decoder(corpus.H, corpus.O, corpus.priors, config=name)
            row = {"problem": pid, "config": name, "shots": corpus.shots,
                   "equal_shots": 0, "differing_shots": 0, "differing_entries": 0,
                   "total_entries": 0, "nan_entries": 0, "hard_mismatches": 0,
                   "max_abs_diff": 0.0, "first_differing_shots": []}
            t = time.perf_counter()
            try:
                for i in range(corpus.shots):
                    ref_obs, ref_corr, ref_status = reference.decode(corpus.dets[i])
                    soft_ref = reference.soft_output()
                    corr, obs, status = decoder.decode_full(corpus.dets[i])
                    soft_eldpc = decoder.soft_output()
                    if soft_ref.shape != soft_eldpc.shape:
                        raise SystemExit(f"{pid}/{name}: soft output shape {soft_ref.shape} != "
                                         f"{soft_eldpc.shape}")
                    both_nan = np.isnan(soft_ref) & np.isnan(soft_eldpc)
                    differs = (soft_ref != soft_eldpc) & ~both_nan
                    differing = int(np.count_nonzero(differs))
                    row["total_entries"] += int(soft_ref.size)
                    row["differing_entries"] += differing
                    row["nan_entries"] += int(np.count_nonzero(np.isnan(soft_ref)))
                    if differing:
                        row["differing_shots"] += 1
                        if len(row["first_differing_shots"]) < 20:
                            row["first_differing_shots"].append(int(i))
                        delta = np.abs(soft_ref[~both_nan] - soft_eldpc[~both_nan])
                        if delta.size:
                            row["max_abs_diff"] = max(row["max_abs_diff"],
                                                      float(np.max(delta)))
                    else:
                        row["equal_shots"] += 1
                    if not (np.array_equal(ref_corr, corr) and np.array_equal(ref_obs, obs)):
                        row["hard_mismatches"] += 1
            finally:
                reference.close()
                decoder.close()
            row["seconds"] = round(time.perf_counter() - t, 3)
            max_abs_diff = max(max_abs_diff, row["max_abs_diff"])
            for key in ("shots", "equal_shots", "differing_shots", "differing_entries",
                        "total_entries", "hard_mismatches", "nan_entries"):
                totals[key] += row[key]
            rows.append(row)
            print(f"{pid} / {name}: {row['equal_shots']}/{row['shots']} equal, "
                  f"{row['differing_entries']} differing entries, "
                  f"max |diff| {row['max_abs_diff']:.3g}", flush=True)

    result = {
        "schema": "eldpc.verify.soft_output.v1",
        "reference_build_info": ref_info.build_info,
        "reference_build_seconds": round(ref_build_seconds, 3),
        "reference_source_sha256": ref_info.source_sha256,
        "rows": rows,
        "totals": totals,
        "max_abs_diff": max_abs_diff,
        "bit_identical": totals["differing_entries"] == 0 and totals["hard_mismatches"] == 0,
    }
    text = json.dumps(result, sort_keys=True)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0 if result["bit_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
