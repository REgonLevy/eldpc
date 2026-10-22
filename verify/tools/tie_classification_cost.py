#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Optional check: the cost of the OSD tie-free classification per shot.

The suite classifies an upstream-wheel difference only for the shots where the wheel differs. The
classification needs the reference's posterior LLRs and a per-shot pillar walk over the columns of
H (up to 75,600 columns for the joint X+Z problem), so doing it for every shot is expensive. This
tool measures that cost against the cost of the upstream decode on the same shots and prints the
numbers behind classifying only the differing shots.

    python verify/tools/tie_classification_cost.py \
        --problems bb144_unif_p1e-3_r12_z,bb144_unif_p1e-3_r12_xyz --shots 5

Uses the suite's corpus builder, the packaged reference and the installed ldpc wheel; it needs
the `eldpc` package for its preset table but no GPU. Exit status 0 when the measurement completes.
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

from eldpc_verify import upstream as upstream_mod  # noqa: E402
from eldpc_verify.abi import ST_BP_CONVERGED, AbiDecoder  # noqa: E402
from eldpc_verify.config import default_cache_root, load_config  # noqa: E402
from eldpc_verify.corpus import ensure_corpus  # noqa: E402
from eldpc_verify.reference import build_reference  # noqa: E402
from eldpc_verify.runner import config_string  # noqa: E402


def _split(text):
    return [x.strip() for x in text.split(",") if x.strip()] if text else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--problems", default="bb144_unif_p1e-3_r12_z,bb144_unif_p1e-3_r12_xyz")
    parser.add_argument("--config", default="a03125c3")
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    import eldpc

    ok, description = upstream_mod.availability()
    if not ok:
        print(f"error: {description}", file=sys.stderr)
        return 2
    cfg = load_config()
    by_id = {p["id"]: p for p in cfg["problems"]}
    problems = _split(args.problems) or ["bb144_unif_p1e-3_r12_z"]
    unknown = [p for p in problems if p not in by_id]
    if unknown:
        parser.error(f"unknown problem ids {unknown}")
    params = eldpc.presets()[args.config]
    cs = config_string(params)
    cache = Path(args.cache) if args.cache else default_cache_root()
    ref_info = build_reference(cache)

    rows = []
    for pid in problems:
        problem = by_id[pid]
        if problem["shape"]["kind"] == "window":
            parser.error("this check uses full problems only")
        corpus = ensure_corpus(cache, problem, cfg["base_seed"], int(args.shots))
        reference = AbiDecoder(ref_info.path, cs, corpus.H, corpus.O, corpus.priors,
                               corpus.dem_text)
        upstream = upstream_mod.UpstreamDecoder(corpus.H, corpus.O, corpus.priors, params)
        t0 = time.perf_counter()
        masks = upstream_mod.column_masks(corpus.H)
        masks_seconds = time.perf_counter() - t0
        classify_seconds = 0.0
        classified_shots = 0
        soft = None
        try:
            for i in range(corpus.shots):
                _, _, status = reference.decode(corpus.dets[i])
                if not (int(status) & ST_BP_CONVERGED):
                    soft = reference.soft_output()
                    t = time.perf_counter()
                    upstream_mod.shot_cause(int(status), soft, corpus.dets[i], params, masks)
                    classify_seconds += time.perf_counter() - t
                    classified_shots += 1
            t = time.perf_counter()
            for i in range(corpus.shots):
                upstream.decode(corpus.dets[i])
            upstream_seconds = time.perf_counter() - t
        finally:
            reference.close()
        rows.append({
            "problem": pid, "config": args.config, "shots": corpus.shots,
            "num_mechanisms": corpus.num_mechanisms,
            "classified_shots": classified_shots,
            "masks_seconds": round(masks_seconds, 4),
            "classify_seconds": round(classify_seconds, 4),
            "classify_ms_per_shot": round(1000.0 * classify_seconds / max(classified_shots, 1), 3),
            "upstream_seconds": round(upstream_seconds, 4),
            "upstream_ms_per_shot": round(1000.0 * upstream_seconds / corpus.shots, 3),
        })
        print(f"{pid}: n={corpus.num_mechanisms}, classified {classified_shots}/{corpus.shots} "
              f"shots in {rows[-1]['classify_ms_per_shot']} ms/shot; upstream decode "
              f"{rows[-1]['upstream_ms_per_shot']} ms/shot", flush=True)

    result = {"schema": "eldpc.verify.tie_classification_cost.v1",
              "reference_build_info": ref_info.build_info, "rows": rows,
              "upstream": description}
    text = json.dumps(result, sort_keys=True)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
