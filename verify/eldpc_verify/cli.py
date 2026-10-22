# SPDX-License-Identifier: Apache-2.0
"""Command line of the eldpc verification suite (``verify/run_verify.py``)."""
from __future__ import annotations

import argparse
import sys

from .config import ConfigError
from .corpus import CorpusError
from .reference import ReferenceBuildError
from .runner import VerifyError, run


def _split_list(text):
    if text is None:
        return None
    values = [part.strip() for part in text.split(",") if part.strip()]
    return values or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_verify.py",
        description="Verify that eldpc's corrections and observables are bit-identical to the "
                    "reference decoder (ldpc 2.4.1 with an explicit tie order) on corpora "
                    "generated from public seeds. The problems, shot counts and presets of each "
                    "mode are defined in the suite config (verify/config.json).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                      help="the quick suite: the configured problems with the shots.quick counts and the configs.quick presets (default)")
    mode.add_argument("--full", action="store_true",
                      help="the full suite: the configured problems with the shots.full counts and the configs.full presets")
    parser.add_argument("--configs", metavar="A,B",
                        help="comma-separated preset names (default: the mode's list from the "
                             "suite config)")
    parser.add_argument("--problems", metavar="A,B",
                        help="comma-separated problem ids (default: every configured problem)")
    parser.add_argument("--out", metavar="DIR", default="eldpc-verify-out",
                        help="directory for summary.json and summary.md (default: "
                             "eldpc-verify-out)")
    parser.add_argument("--device", choices=("gpu", "cpu", "auto"), default="gpu",
                        help="the engine whose outputs are compared: gpu (the GPU core, default), cpu "
                             "(the CPU core, for a machine without a supported GPU; it must return the "
                             "same outputs) or auto (the GPU when it is usable, else the CPU core)")
    parser.add_argument("--lib", metavar="PATH",
                        help="path passed to eldpc.load() (the package's shared library); by "
                             "default the package chooses its own")
    parser.add_argument("--reference-lib", metavar="PATH",
                        help="a separately built library of the same decoder sources (decoder ABI v1); "
                             "when given, eldpc's status words are compared with it too. "
                             "Default: $ELDPC_REFERENCE_LIB when set (an empty value "
                             "disables the comparison). It must not be eldpc's own library")
    parser.add_argument("--config", metavar="PATH",
                        help="suite configuration file (default: verify/config.json)")
    parser.add_argument("--cache", metavar="DIR",
                        help="corpus and reference build cache (default: "
                             "$ELDPC_VERIFY_CACHE, else ~/.cache/eldpc/verify)")
    parser.add_argument("--jobs", type=int, metavar="N",
                        help="parallel jobs for the reference build (default: CPU count)")
    parser.add_argument("--no-upstream", action="store_true",
                        help="skip the informational comparison with the installed ldpc wheel")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.configs = _split_list(args.configs)
    args.problems = _split_list(args.problems)
    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be >= 1")
    try:
        return run(args, log=print)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except (VerifyError, ConfigError, ReferenceBuildError, CorpusError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
