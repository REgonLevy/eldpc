#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check src/core/TOKENS.sha256 against a fastbpgpu6g source tarball.

    python tools/check_core_tarball.py <tarball>

The tarball is the pinned snapshot of the decoder sources; the package keeps the six library
files of the frozen source directory (fastbpgpu6g) with the development-history comments neutralised, and lists the
SHA-256 of their text without comments (code tokens and line structure) in src/core/TOKENS.sha256.
This tool reads the tarball's members (nothing is extracted), takes the copy of each listed file at
the frozen source directory's ``<name>`` and compares the SHA-256 of its text without comments (see
tools/check_comment_only.py) with the manifest. Exit code 0 iff every file is present once and
matches.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_comment_only import stripped_sha256  # noqa: E402

CORE_DIR = Path(__file__).resolve().parents[1] / "src" / "core"
MANIFEST = CORE_DIR / "TOKENS.sha256"
# The archive the package sources were taken from; reported for information.
PINNED_ARCHIVE_SHA256 = "1edb657aea0620cc56d038ee5ed068284ae20a54e39967c08f47bac3b680ac48"


def read_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2 or len(parts[0]) != 64:
            raise SystemExit(f"{path}:{lineno}: not a 'sha256  file' line: {line!r}")
        digest, name = parts
        if name in entries:
            raise SystemExit(f"{path}:{lineno}: duplicate entry for {name}")
        entries[name] = digest.lower()
    if not entries:
        raise SystemExit(f"{path}: no entries")
    return entries


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tarball", type=Path, help="the fastbpgpu6g source tarball")
    args = ap.parse_args(argv)

    manifest = read_manifest(MANIFEST)
    try:
        archive = tarfile.open(args.tarball, "r:*")
    except (OSError, tarfile.TarError) as exc:
        print(f"FAIL cannot open {args.tarball}: {exc}")
        return 2

    with archive:
        members: dict[str, tarfile.TarInfo] = {}
        for member in archive.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            # Only the library files of the decoder directory, at any archive prefix.
            if len(parts) >= 3 and parts[-3] == "candidates" and parts[-2] == "fastbpgpu6g":
                if parts[-1] in members:
                    print(f"FAIL {member.name}: the archive holds {parts[-1]} more than once")
                    return 1
                members[parts[-1]] = member

        failed = 0
        for name in sorted(manifest):
            if name not in members:
                print(f"FAIL {name}: not found in {args.tarball} (candidates/fastbpgpu6g/{name})")
                failed += 1
                continue
            extracted = archive.extractfile(members[name])
            assert extracted is not None
            digest = stripped_sha256(extracted.read().decode("utf-8"))
            if digest == manifest[name]:
                print(f"OK   {name} {digest}")
            else:
                print(f"FAIL {name}: tarball has {digest}, TOKENS.sha256 has {manifest[name]}")
                failed += 1
        extra = sorted(set(members) - set(manifest))
        for name in extra:
            print(f"note: the archive also holds {name}; TOKENS.sha256 does not list it")

    archive_hash = hashlib.sha256(args.tarball.read_bytes()).hexdigest()
    note = "matches the pinned archive" if archive_hash == PINNED_ARCHIVE_SHA256 else \
        f"differs from the pinned archive {PINNED_ARCHIVE_SHA256}; the member contents above are what is checked"
    print(f"archive sha256 {archive_hash} ({note})")
    print("PASS" if failed == 0 else f"{failed} file(s) failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
