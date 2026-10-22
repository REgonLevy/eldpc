#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check that the decoder-core sources differ from a reference revision in comments only.

The tool contains a small C/C++/CUDA lexer. ``blank_comments`` replaces every comment character
except newlines with a space and leaves everything else (code, string and character literals,
preprocessor lines, whitespace) untouched. ``strip_comments`` is the same with one difference: each
comment becomes a single space followed by the newlines the comment contained, so the result does
not depend on how long a comment is (a reworded comment keeps the file's tokens, whitespace outside
comments and line structure, but not its length). The hashes and the comparison use
``strip_comments``: two files give the same stripped text exactly when they differ only in the
text of their comments and have the same number of lines.

  python3 tools/check_comment_only.py                  print the SHA-256 of the stripped text of
                                                       every file in src/core and src/cpu
  python3 tools/check_comment_only.py --compare REV    compare the stripped text and the line
                                                       count of every file against ``git show
                                                       REV:<path>``; exit 1 on any difference
  python3 tools/check_comment_only.py --write          rewrite src/core/TOKENS.sha256 and
                                                       src/cpu/TOKENS.sha256
  python3 tools/check_comment_only.py --check          check the working tree against the two
                                                       TOKENS.sha256 manifests

The tool only reads source text; it never compiles or imports the package.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = ("src/core", "src/cpu")
SOURCE_SUFFIXES = (".c", ".h", ".cc", ".cpp", ".cu", ".cuh", ".hpp")
MANIFEST = "TOKENS.sha256"


def _is_ident(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _skip_continuations(text: str, i: int) -> int:
    """Return the index of the first character at or after ``i`` that is not a backslash-newline."""
    n = len(text)
    while i + 1 < n and text[i] == "\\" and text[i + 1] == "\n":
        i += 2
    return i


def comment_spans(text: str) -> list[tuple[int, int, str]]:
    """Return ``(start, end, kind)`` for every comment; ``kind`` is ``"line"`` or ``"block"``.

    A ``//`` comment continues over a backslash-newline, as in the C++ translation phases.
    String literals (with escapes, and raw strings), character literals (with escapes) and
    C++14 digit separators are recognised so that ``//`` and ``/*`` inside them are not comments.
    """
    spans: list[tuple[int, int, str]] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            j = i + 2
            while j < n:
                if text[j] == "\\" and j + 1 < n and text[j + 1] == "\n":
                    j += 2
                    continue
                if text[j] == "\n":
                    break
                j += 1
            spans.append((i, j, "line"))
            i = j
        elif ch == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            spans.append((i, j, "block"))
            i = j
        elif ch == '"':
            # raw string literal: R"delim( ... )delim" (optionally with an encoding prefix)
            if i > 0 and text[i - 1] == "R" and (i < 2 or not _is_ident(text[i - 2]) or
                                                 text[i - 2:i] in ("LR", "uR", "UR") or
                                                 text[max(0, i - 3):i] == "u8R"):
                k = text.find("(", i + 1)
                if k > 0 and k - i - 1 <= 16:
                    delim = text[i + 1:k]
                    end = text.find(")" + delim + '"', k + 1)
                    i = n if end < 0 else end + len(delim) + 2
                    continue
            i += 1
            while i < n and text[i] != '"':
                if text[i] == "\\":
                    i += 1
                if i < n and text[i] == "\n" and text[i - 1] != "\\":
                    break  # unterminated string: stop at the newline
                i += 1
            i += 1
        elif ch == "'":
            # a digit separator (1'000) is not the start of a character literal
            if i > 0 and i + 1 < n and _is_ident(text[i - 1]) and _is_ident(text[i + 1]):
                k = i - 1
                while k >= 0 and (_is_ident(text[k]) or text[k] == "'" or text[k] == "."):
                    k -= 1
                if text[k + 1].isdigit():
                    i += 1
                    continue
            i += 1
            while i < n and text[i] != "'":
                if text[i] == "\\":
                    i += 1
                if i < n and text[i] == "\n":
                    break
                i += 1
            i += 1
        else:
            i += 1
    return spans


def blank_comments(text: str) -> str:
    """Replace every comment character except newlines with a space."""
    out: list[str] = []
    last = 0
    for start, end, _kind in comment_spans(text):
        out.append(text[last:start])
        out.append("".join(c if c == "\n" else " " for c in text[start:end]))
        last = end
    out.append(text[last:])
    return "".join(out)


def strip_comments(text: str) -> str:
    """Replace every comment by one space plus the newlines it contained (length independent)."""
    out: list[str] = []
    last = 0
    for start, end, _kind in comment_spans(text):
        out.append(text[last:start])
        out.append(" " + "\n" * text.count("\n", start, end))
        last = end
    out.append(text[last:])
    return "".join(out)


def stripped_sha256(text: str) -> str:
    return hashlib.sha256(strip_comments(text).encode("utf-8")).hexdigest()


def tracked_sources(directory: str) -> list[str]:
    """The C/C++/CUDA source files of ``directory`` (paths relative to the package root)."""
    found = (p for p in (ROOT / directory).iterdir() if p.is_file() and p.suffix in SOURCE_SUFFIXES)
    return sorted(f"{directory}/{p.name}" for p in found)


def git_show(rev: str, path: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:{path}"], check=True,
                          capture_output=True).stdout.decode("utf-8")


def read(path: str) -> str:
    return (ROOT / path).read_bytes().decode("utf-8")


def manifest_lines(directory: str) -> list[str]:
    return [f"{stripped_sha256(read(p))}  {Path(p).name}" for p in tracked_sources(directory)]


def cmd_print() -> int:
    for directory in DIRS:
        for p in tracked_sources(directory):
            print(f"{stripped_sha256(read(p))}  {p}")
    return 0


def cmd_compare(rev: str) -> int:
    bad = 0
    for directory in DIRS:
        for p in tracked_sources(directory):
            new, old = read(p), git_show(rev, p)
            problems = []
            if new.count("\n") != old.count("\n"):
                problems.append(f"line count {new.count(chr(10))} != {old.count(chr(10))}")
            if strip_comments(new) != strip_comments(old):
                problems.append("stripped text differs")
            n_changed = changed_comment_count(old, new)
            status = "FAIL " + "; ".join(problems) if problems else "ok"
            print(f"{stripped_sha256(new)}  {p}  comment-lines-changed={n_changed}  {status}")
            bad += bool(problems)
    print("RESULT: " + ("FAIL, a code token or the line structure differs" if bad else
                        f"OK, only comments differ from {rev}"))
    return 1 if bad else 0


def changed_comment_count(old: str, new: str) -> int:
    """Number of comment lines whose text differs (same line structure assumed)."""
    a, b = old.split("\n"), new.split("\n")
    return sum(1 for x, y in zip(a, b) if x != y)


def cmd_write() -> int:
    for directory in DIRS:
        (ROOT / directory / MANIFEST).write_text("\n".join(manifest_lines(directory)) + "\n")
    return 0


def cmd_check() -> int:
    bad = 0
    for directory in DIRS:
        want = (ROOT / directory / MANIFEST).read_text().splitlines()
        have = manifest_lines(directory)
        if sorted(want) != sorted(have):
            print(f"FAIL {directory}/{MANIFEST} does not match the stripped text of the sources")
            bad += 1
        else:
            print(f"ok   {directory}/{MANIFEST}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compare", metavar="REV", help="compare against git REV")
    ap.add_argument("--write", action="store_true", help="rewrite the TOKENS.sha256 manifests")
    ap.add_argument("--check", action="store_true", help="check the TOKENS.sha256 manifests")
    args = ap.parse_args(argv)
    if args.compare:
        return cmd_compare(args.compare)
    if args.write:
        return cmd_write()
    if args.check:
        return cmd_check()
    return cmd_print()


if __name__ == "__main__":
    sys.exit(main())
