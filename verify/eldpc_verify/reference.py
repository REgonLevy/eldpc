# SPDX-License-Identifier: Apache-2.0
"""Build and load the reference decoder from the package tree.

The reference is a small CMake target (``verify/reference``): ldpc 2.4.1's C++ headers with
the deterministic tie-order patch of the reference construction, behind the same C ABI shim the
reference construction's reference library uses. It is built on first use into a cache directory
keyed by the SHA-256 of every reference source file, so a package upgrade that changes any of them
builds a fresh library while an unchanged tree reuses the old one.

The build needs CMake >= 3.16 and a C++17 compiler. Their identity is recorded in the library's
``dbench_decoder_build_info()`` string together with the build flags and the patch SHA-256.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .abi import AbiLibrary

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
LIBRARY_NAME = "libeldpc_verify_reference.so"
_KEY_FILE = "reference.key"


class ReferenceBuildError(RuntimeError):
    """The reference decoder could not be built or loaded."""


@dataclass(frozen=True)
class ReferenceInfo:
    path: Path
    source_sha256: str
    build_info: str | None
    cached: bool

    def as_dict(self) -> dict:
        return {"path": str(self.path), "source_sha256": self.source_sha256,
                "build_info": self.build_info, "cached": self.cached}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_sha256(reference_dir: Path = REFERENCE_DIR) -> str:
    """SHA-256 over every file of the reference target, path and content.

    The digest depends on nothing but the files themselves (not on their location, timestamps or
    permissions), so it is a stable cache key across checkouts and installs."""
    reference_dir = Path(reference_dir)
    if not reference_dir.is_dir():
        raise ReferenceBuildError(f"reference sources not found at {reference_dir}")
    h = hashlib.sha256()
    for path in sorted(p for p in reference_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(reference_dir).as_posix()
        h.update(rel.encode("utf-8") + b"\0")
        h.update(_sha256_file(path).encode("ascii") + b"\n")
    return h.hexdigest()


def _find_cmake() -> str:
    explicit = os.environ.get("ELDPC_VERIFY_CMAKE")
    if explicit:
        return explicit
    found = shutil.which("cmake")
    if found:
        return found
    raise ReferenceBuildError("cmake was not found on PATH; set ELDPC_VERIFY_CMAKE to its path")


def _run(cmd, log_file, cwd=None):
    with open(log_file, "ab") as log:
        log.write(("\n$ " + " ".join(cmd) + "\n").encode())
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=cwd)
    return proc.returncode


def _tail(path: Path, lines: int = 40) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(no build log)"
    return "\n".join(text[-lines:])


def build_reference(cache_root, *, jobs=None, log=None) -> ReferenceInfo:
    """Build (or reuse) the reference library; returns its path and build description."""
    cache_root = Path(cache_root)
    key = source_sha256()
    dest = cache_root / "reference" / key
    library = dest / "lib" / LIBRARY_NAME
    marker = dest / _KEY_FILE
    if library.is_file() and marker.is_file():
        try:
            if marker.read_text(encoding="utf-8").strip() == key:
                info = ReferenceInfo(library, key, AbiLibrary(library).build_info, cached=True)
                if log:
                    log(f"reference: reusing {library}")
                return info
        except OSError:
            pass
    if log:
        log(f"reference: building {REFERENCE_DIR} (cache key {key[:16]})")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f"{dest.name}.build.{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    build_log = tmp / "build.log"
    cmake = _find_cmake()
    if jobs is None:
        jobs = max(1, os.cpu_count() or 1)
    build_dir = tmp
    try:
        rc = _run([cmake, "-S", str(REFERENCE_DIR), "-B", str(build_dir),
                   "-DCMAKE_BUILD_TYPE=Release"], build_log)
        if rc == 0:
            rc = _run([cmake, "--build", str(build_dir), "-j", str(int(jobs))], build_log)
        built = build_dir / "lib" / LIBRARY_NAME
        if rc != 0 or not built.is_file():
            last = dest.parent / "last-build.log"
            shutil.copyfile(build_log, last)
            raise ReferenceBuildError(
                f"building the reference decoder failed (exit {rc}); "
                f"the build log is {last}:\n{_tail(build_log)}")
        build_info = AbiLibrary(built).build_info
        (tmp / _KEY_FILE).write_text(key + "\n", encoding="utf-8")
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if log:
        log(f"reference: built {dest / 'lib' / LIBRARY_NAME}")
    return ReferenceInfo(dest / "lib" / LIBRARY_NAME, key, build_info, cached=False)
