# SPDX-License-Identifier: Apache-2.0
"""The committed-result tools: RUN.md notes name their script and never guess a commit."""
import importlib.util
import json
import os

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(TOOLS, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _results(path):
    data = {"schema": "eldpc.tune.v2",
            "tool": {"command": ["eldpc-tune", "--problem", "bb72", "--out", "x"],
                     "started_utc": "2026-01-01T00:00:00Z", "duration_seconds": 12.5,
                     "quick": False},
            "environment": {"machine": {"cpu_model": "CPU X", "cpu_count": 8,
                                        "gpu": {"devices": [{"name": "GPU X",
                                                             "compute_capability": "9.9",
                                                             "sm_count": 42, "l2_bytes": 1}]}},
                            "eldpc": {"version": "0.1.0", "isa": "avx2"},
                            "python": {"version": "3.12"},
                            "commands": [{"command": "nvcc --version", "stdout": "release 13.3"}]}}
    path.write_text(json.dumps(data))


def test_run_note_names_the_script_and_the_given_commit(tmp_path):
    note_tool = _load("make_run_note")
    _results(tmp_path / "results.json")
    assert note_tool.main(["--results", str(tmp_path / "results.json"), "--commit", "abc1234",
                           "--label", "screen"]) == 0
    text = (tmp_path / "RUN.md").read_text()
    assert "- command: `eldpc-tune --problem bb72 --out x`" in text
    assert "- commit: `abc1234` (the code that produced the run)" in text
    assert "`tuning/tools/make_run_note.py`" in text
    assert "| Wall time (s) | 12.5 |" in text


def test_run_note_without_a_commit_says_not_recorded(tmp_path):
    note_tool = _load("make_run_note")
    _results(tmp_path / "results.json")
    assert note_tool.main(["--results", str(tmp_path / "results.json")]) == 0
    text = (tmp_path / "RUN.md").read_text()
    assert "- commit: not recorded" in text
