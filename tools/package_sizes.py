#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Report the size of built eldpc wheels and of the native libraries inside them.

    python tools/package_sizes.py --wheel after=dist/eldpc-0.1.0-py3-none-linux_x86_64.whl \
        [--wheel before=old/eldpc-0.1.0-py3-none-linux_x86_64.whl] [--out tools/results/wheel-sizes.json]

Each ``--wheel NAME=PATH`` is measured: the file size, its SHA-256, and the uncompressed and
compressed size of every ``.so`` it contains. With two wheels it also reports the difference (the
cost of the extra core libraries) in bytes and MB (10^6 bytes). The JSON result records the
command, the git commit and the target platform, so a size quoted in a document can be traced to
the build it came from.

Wheels are built with ``pip wheel --no-build-isolation --no-deps -w DIR .`` from the package root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path


def git_commit(repo: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def wheel_info(name: str, path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        libraries = {}
        for entry in archive.infolist():
            if entry.filename.endswith(".so"):
                libraries[entry.filename] = {
                    "uncompressed_bytes": entry.file_size,
                    "compressed_bytes": entry.compress_size,
                }
    size = path.stat().st_size
    return {
        "name": name,
        # The file name and the SHA-256 identify the artifact; the directory it was built in is
        # not recorded, so a committed result names no machine path.
        "file": path.name,
        "bytes": size,
        "mb": round(size / 1e6, 3),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "libraries": libraries,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wheel", action="append", required=True, metavar="NAME=PATH",
                    help="a wheel to measure; repeat for a before/after pair")
    ap.add_argument("--out", default="", help="write the JSON result here too")
    args = ap.parse_args(argv)

    wheels = {}
    for item in args.wheel:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            ap.error(f"--wheel needs NAME=PATH, got {item!r}")
        wheels[name] = wheel_info(name, Path(path))

    # The command is recorded with each wheel path replaced by the wheel's file name (the same
    # information the wheel entry carries), so the result holds no machine path.
    command_args = []
    argv = sys.argv[1:]
    index = 0
    while index < len(argv):
        arg = argv[index]
        value = None
        if arg == "--wheel" and index + 1 < len(argv) and "=" in argv[index + 1]:
            value = argv[index + 1]
            index += 2
            command_args.append("--wheel")
        elif arg.startswith("--wheel=") and "=" in arg[len("--wheel="):]:
            value = arg[len("--wheel="):]
            index += 1
        if value is not None:
            wheel_name, _, wheel_path = value.partition("=")
            command_args.append(f"{wheel_name}={Path(wheel_path).name}")
            continue
        command_args.append(arg)
        index += 1
    result = {
        "probe": "eldpc wheel and native-library sizes",
        "command": " ".join([Path(sys.argv[0]).name, *command_args]),
        "commit": git_commit(Path(__file__).resolve().parents[2]),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "wheels": wheels,
    }
    if len(wheels) == 2:
        (a_name, a), (b_name, b) = wheels.items()
        delta = b["bytes"] - a["bytes"]
        native_bytes = sum(entry["uncompressed_bytes"] for entry in b["libraries"].values())
        result["deltas"] = {
            f"{b_name}_minus_{a_name}": {
                "bytes": delta,
                "mb": round(delta / 1e6, 3),
                "native_libraries_uncompressed_bytes": native_bytes,
                "native_libraries_uncompressed_mb": round(native_bytes / 1e6, 3),
            }
        }
    line = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(line + "\n", encoding="utf-8")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
