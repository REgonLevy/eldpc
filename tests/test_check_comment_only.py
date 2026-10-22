# SPDX-License-Identifier: Apache-2.0
"""tools/check_comment_only.py: the comment lexer and the TOKENS.sha256 manifests of the core sources."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("check_comment_only",
                                                PACKAGE / "tools" / "check_comment_only.py")
cco = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cco)


def blank(text: str) -> str:
    return cco.blank_comments(text)


def test_line_and_block_comments_are_blanked_and_newlines_kept():
    src = "int a; // tail\n/* two\nlines */ int b;\n"
    out = blank(src)
    assert out.count("\n") == src.count("\n")
    assert out == "int a;" + " " * 8 + "\n" + " " * 6 + "\n" + " " * 9 + "int b;\n"
    assert len(out) == len(src)


def test_comment_markers_inside_string_literals_are_not_comments():
    src = 'const char *u = "http://x/*y*/"; // real\nint c;\n'
    out = blank(src)
    assert out.startswith('const char *u = "http://x/*y*/"; ')
    assert "real" not in out
    assert out.endswith("\nint c;\n")


def test_escaped_quotes_do_not_end_a_string():
    src = r'const char *s = "a\"// not a comment\"b"; // comment' + "\nint d;\n"
    out = blank(src)
    assert r'"a\"// not a comment\"b";' in out
    assert "comment\n" not in out.split("\n")[0].split(";")[-1] and out.split("\n")[0].rstrip().endswith(";")
    assert out.endswith("int d;\n")


def test_escaped_backslash_before_closing_quote():
    src = 'const char *p = "dir\\\\"; // c\nint e;\n'
    out = blank(src)
    assert 'const char *p = "dir\\\\";' in out
    assert "// c" not in out


def test_character_literals():
    src = "char q = '\"'; // c1\nchar s = '/'; /* c2 */ char t = '\\'';\n"
    out = blank(src)
    assert "char q = '\"';" in out
    assert "c1" not in out and "c2" not in out
    assert "char s = '/';" in out and "char t = '\\'';" in out


def test_digit_separator_is_not_a_character_literal():
    src = "int big = 1'000'000; // c\nint z;\n"
    out = blank(src)
    assert out.startswith("int big = 1'000'000; ")
    assert "c\n" not in out.split("\n")[0][-3:] and out.endswith("int z;\n")


def test_comment_inside_a_macro_line():
    src = "#define K(x) ((x) + 1)  // adds one\n#define L 2 /* two */ + 0\nint f;\n"
    out = blank(src)
    assert out.startswith("#define K(x) ((x) + 1)  ")
    assert "adds" not in out and "two" not in out
    assert "#define L 2 " in out and " + 0\n" in out


def test_line_comment_continues_over_a_backslash_newline():
    src = "int a; // first \\\nstill comment\nint b;\n"
    out = blank(src)
    assert out.count("\n") == src.count("\n")
    assert "still" not in out and out.endswith("int b;\n")


def test_unterminated_block_comment_runs_to_the_end():
    out = blank("int a; /* open\nforever")
    assert out.startswith("int a; ") and "open" not in out and "forever" not in out


def test_raw_string_literal():
    src = 'auto r = R"x(a // b /* c )" )x"; // tail\nint g;\n'
    out = blank(src)
    assert 'R"x(a // b /* c )" )x";' in out
    assert "tail" not in out


def test_strip_comments_is_independent_of_the_comment_text():
    a = "int a; // short\n/* x\ny */ int b;\n"
    b = "int a; // a much longer rewritten comment\n/* other\ntext */ int b;\n"
    assert cco.strip_comments(a) == cco.strip_comments(b)
    assert cco.stripped_sha256(a) == cco.stripped_sha256(b)
    # a changed token is detected, and so is a changed line count
    assert cco.stripped_sha256(a) != cco.stripped_sha256(a.replace("int b", "int c"))
    assert cco.stripped_sha256(a) != cco.stripped_sha256(a + "\n")
    assert cco.stripped_sha256(a) != cco.stripped_sha256("int a; // short\n/* x y */ int b;\n")


@pytest.mark.parametrize("directory", ["src/core", "src/cpu"])
def test_tokens_manifest_matches_the_stripped_text_of_the_sources(directory):
    manifest = {}
    for line in (PACKAGE / directory / "TOKENS.sha256").read_text().splitlines():
        digest, name = line.split()
        assert len(digest) == 64 and name not in manifest
        manifest[name] = digest
    sources = {Path(p).name for p in cco.tracked_sources(directory)}
    assert set(manifest) == sources
    for name, digest in manifest.items():
        text = (PACKAGE / directory / name).read_bytes().decode("utf-8")
        assert cco.stripped_sha256(text) == digest, f"{directory}/{name}: code tokens changed"
