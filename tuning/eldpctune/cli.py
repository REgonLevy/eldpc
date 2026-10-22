# SPDX-License-Identifier: Apache-2.0
"""Command line of eldpc-tune."""
from __future__ import annotations

import argparse
import sys

from . import __version__, report
from .runner import NO_QUALIFIER_EXIT, TuneConfig, run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eldpc-tune",
        description="Find the fastest eldpc decoder configuration that meets a logical error "
                    "rate target and a latency budget, and write a config file for it.",
        epilog="Exit status: 0 = a choice was written (config.json); 3 = no configuration "
               "qualifies (results.json, pareto.md and config.nearest.json, no config.json); "
               "any other non-zero = an error or an abort (1 run error or a declined prompt, 2 "
               "bad arguments, 130 interrupted).")
    src = p.add_argument_group("problem (give exactly one)")
    src.add_argument("--problem", metavar="NAME",
                     help="built-in problem: bb72, bb144, bb144xyz, bb288, hgp98, sc5, sc9, "
                          "sc13, sc17")
    src.add_argument("--dem", metavar="FILE", help="stim detector error model file")
    src.add_argument("--circuit", metavar="FILE", help="stim circuit file (shots sampled from it)")
    src.add_argument("--h", metavar="FILE",
                     help="parity-check matrix for a code-capacity problem (text 'm n' + rows, "
                          "or the binary DBCSC001 format). Without --o the observables are "
                          "derived as a basis of ker(H)/row(H), which needs H to be the full "
                          "commuting stabilizer matrix (H H^T = 0); that is NOT a CSS logical "
                          "basis: for a CSS half (e.g. Hx) pass the observable matrix with --o")
    src.add_argument("--o", metavar="FILE",
                     help="observable matrix for --h (same formats), k x n; required for a CSS "
                          "half such as Hx (its rows are the logical operators of the other "
                          "type, not stabilizers); derived from H if omitted")
    p.add_argument("--p", metavar="P", type=float, nargs="+",
                   help="physical error rate(s); one point must meet the target at every rate. "
                        "Defaults to 1e-3 for --problem; required for --h")
    p.add_argument("--rounds", metavar="N", type=int,
                   help="override the number of rounds of a built-in problem")
    p.add_argument("--noise-model", choices=("uniform", "si1000"),
                   help="override the noise model of a built-in problem: surface codes default "
                        "to si1000 and also support uniform depolarizing noise; bb and hgp "
                        "problems always use uniform noise")
    p.add_argument("--p50-budget-us", metavar="US", type=float,
                   help="latency budget: p50 of a single-shot call")
    p.add_argument("--p99-budget-us", metavar="US", type=float,
                   help="latency budget: p99 of a single-shot call")
    p.add_argument("--ler-target", metavar="P", type=float,
                   help="upper bound on the logical error rate per shot that must be met with "
                        "99%% one-sided confidence")
    p.add_argument("--weight", metavar="W", type=float,
                   help="trade-off weight between log LER and latency: choose the smallest "
                        "log10(LER upper bound) + W * log10(p50_us). Without --ler-target only "
                        "points whose screen LER is no worse than stock ms30 on the same shots "
                        "are choosable (see --max-ler), and the choice must pass its own latency "
                        "confirmation")
    p.add_argument("--max-ler", metavar="X", type=float,
                   help="without --ler-target: replace the default LER floor (stock ms30's "
                        "screen LER on the same shots) by X; a point is choosable only when its "
                        "screen LER is at most X at every case. A cap on the estimate, not a "
                        "proven bound (--ler-target proves one); --max-ler 1 disables the floor")
    p.add_argument("--confirm-shots", metavar="N", type=int,
                   help="override the confirmation shot count: N fresh shots per candidate and "
                        "case, declared before the first confirmation shot and never stopped "
                        "early (default: derived from the target, the confidence and a planned "
                        "margin; --quick caps it)")
    p.add_argument("--confirm-candidates", metavar="K", type=int, default=None,
                   help="cap the confirmation at K candidates. By default there is no cap: the "
                        "in-budget candidates are confirmed one at a time, in the order the "
                        "choice ranks them, until one is accepted (it is the choice) or the "
                        "list is exhausted. A cap that stops the confirmation with in-budget "
                        "candidates unconfirmed is warned about and recorded")
    p.add_argument("--no-prune", action="store_true",
                   help="disable dominance pruning: evaluate every point of every row instead of "
                        "skipping a larger cap after a passing smaller one")
    p.add_argument("--decompose", dest="decompose", action="store_true",
                   help="derive the DEM of --circuit or of a built-in recipe with "
                        "decompose_errors=True (default: undecomposed, the model the decoder is "
                        "given)")
    p.add_argument("--no-decompose", dest="decompose", action="store_false",
                   help="derive the DEM undecomposed (the default)")
    p.set_defaults(decompose=False)
    p.add_argument("--cadence-us", metavar="US", type=float, default=0.0,
                   help="application call spacing in microseconds (0 = back to back)")
    p.add_argument("--seed", metavar="N", type=int, default=20261006,
                   help="seed of every sampled shot set (each stage derives its own seed)")
    p.add_argument("--out", metavar="DIR", required=True,
                   help="output directory (config.json, pareto.md, results.json, raw/)")
    p.add_argument("--auto", action="store_true",
                   help="run the whole procedure without prompts (there is no language model "
                        "anywhere in the procedure)")
    p.add_argument("--quick", action="store_true",
                   help="small shot counts and a smaller grid: a smoke test, not a tuning run")
    p.add_argument("--resume", action="store_true",
                   help="reload the accuracy progress saved in DIR/raw/checkpoint.json and "
                        "continue where an interrupted run stopped; latency is always measured "
                        "again")
    p.add_argument("--max-shots", metavar="N", type=int,
                   help="advanced: accuracy shots per point per case (default 131072; 2048 with "
                        "--quick; a full run with --ler-target below the default's zero-failure "
                        "floor of about 3.5e-5 derives the limit from the target, about "
                        "4.7/target). The smallest provable LER bound with zero failures is about "
                        "4.6 / max_shots")
    p.add_argument("--workers", metavar="N", type=int, default=1,
                   help="evaluate N grid points concurrently in the accuracy stages (default 1). "
                        "Accuracy decisions and shot records are identical to a sequential run; "
                        "only wall time changes, and by how much depends on the problem and the "
                        "machine")
    p.add_argument("--version", action="version", version=f"eldpc-tune {__version__}")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = TuneConfig(
        problem=args.problem, dem=args.dem, circuit=args.circuit, h=args.h, o=args.o,
        ps=list(args.p or []), rounds=args.rounds, noise_model=args.noise_model,
        p50_budget_us=args.p50_budget_us, p99_budget_us=args.p99_budget_us,
        ler_target=args.ler_target, max_ler=args.max_ler, weight=args.weight,
        cadence_us=args.cadence_us,
        seed=args.seed, out=args.out, auto=args.auto, quick=args.quick,
        max_shots=args.max_shots, workers=args.workers, resume=args.resume,
        confirm_shots=args.confirm_shots, confirm_candidates=args.confirm_candidates,
        no_prune=args.no_prune, decompose=args.decompose)
    cfg.command = ["eldpc-tune"] + list(sys.argv[1:] if argv is None else argv)
    try:
        cfg.validate()
    except ValueError as exc:
        print(f"eldpc-tune: error: {exc}", file=sys.stderr)
        return 2
    if not cfg.auto and sys.stdin.isatty():
        print("eldpc-tune will run the fixed procedure: coarse grid, refinement around the "
              "accuracy frontier, paired latency measurement, choice.")
        print(f"  output: {cfg.out}")
        answer = input("Continue? [Y/n] ").strip().lower()
        if answer not in ("", "y", "yes"):
            print("aborted")
            return 1
    import eldpc

    expected = (ValueError, RuntimeError, OSError)
    error_type = getattr(eldpc, "EldpcError", None)
    if isinstance(error_type, type) and not issubclass(error_type, expected):
        expected = expected + (error_type,)
    try:
        results = run(cfg, log=print)
        written = report.write_outputs(results, cfg.out)
    except expected as exc:
        print(f"eldpc-tune: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("eldpc-tune: interrupted", file=sys.stderr)
        return 130
    report.print_summary(results)
    print("")
    print("Files written:")
    for name in written["files"]:
        print(f"  {name}")
    if results["choice"].get("config") is None:
        # A run that finds nothing must not look like a success: no config.json was written
        # and the exit status says so (both paths).
        return NO_QUALIFIER_EXIT
    return 0
