# SPDX-License-Identifier: Apache-2.0
"""The corpus generator is deterministic: same seed and problem, same bytes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]      # the package root (tests/verify/x.py -> root)
if str(ROOT / "verify") not in sys.path:
    sys.path.insert(0, str(ROOT / "verify"))

from eldpc_verify import corpus as corpus_mod  # noqa: E402
from eldpc_verify.config import load_config  # noqa: E402
from eldpc_verify.corpus import corpus_dir, ensure_corpus, load_entry  # noqa: E402
from eldpc_testlib import requires_qldpc  # noqa: E402

pytestmark = requires_qldpc  # the corpora are built with qLDPC

PROBLEM_ID = "bb72_unif_p1e-3_r6_z"
SHOTS = 8
# The qLDPC revision of the corpus sizes in docs/VERIFY.md (the `qldpc` extra of pyproject.toml);
# other revisions can build other circuits, e.g. qLDPC 0.4.1 gives bb72 2664 mechanisms, not 2592.
TESTED_QLDPC_COMMIT = "4c4f80109215fa4a28b4d94e2c285e4683487bc1"


def _problem():
    cfg = load_config()
    return cfg, next(p for p in cfg["problems"] if p["id"] == PROBLEM_ID)


def _images(paths):
    return {p.relative_to(paths[0]).as_posix(): p.read_bytes()
            for p in sorted(paths[0].rglob("*")) if p.is_file()}


def test_same_seed_same_bytes(tmp_path):
    cfg, problem = _problem()
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    first = ensure_corpus(root_a, problem, cfg["base_seed"], SHOTS)
    second = ensure_corpus(root_b, problem, cfg["base_seed"], SHOTS)
    assert first.id == second.id == PROBLEM_ID
    assert first.shots == second.shots == SHOTS
    dir_a = corpus_dir(root_a, problem, cfg["base_seed"], SHOTS)
    dir_b = corpus_dir(root_b, problem, cfg["base_seed"], SHOTS)
    files_a, files_b = _images([dir_a]), _images([dir_b])
    assert files_a.keys() == files_b.keys()
    for name in files_a:
        assert files_a[name] == files_b[name], f"{name} differs between two builds"
    # The arrays themselves, read back through the cache loader.
    n = first.H.shape[1]
    if corpus_mod.generator_versions()["qldpc_commit"] == TESTED_QLDPC_COMMIT:
        assert n == 2592            # the corpora table of docs/VERIFY.md
    for corpus in (first, second):
        assert corpus.H.shape == (252, n)
        assert corpus.O.shape == (12, n)
        assert corpus.priors.shape == (n,)
        assert corpus.dets.shape == (SHOTS, 252)
        assert corpus.obs.shape == (SHOTS, 12)


def test_cache_reuse_returns_identical_entry(tmp_path):
    cfg, problem = _problem()
    root = tmp_path / "cache"
    first = ensure_corpus(root, problem, cfg["base_seed"], SHOTS)
    assert not first.from_cache
    second = ensure_corpus(root, problem, cfg["base_seed"], SHOTS)
    assert second.from_cache
    assert np.array_equal(second.dets, first.dets)
    assert np.array_equal(second.priors, first.priors)
    assert (second.H != first.H).nnz == 0
    assert second.manifest == first.manifest


def test_different_shots_make_a_different_entry(tmp_path):
    cfg, problem = _problem()
    root = tmp_path / "cache"
    a = ensure_corpus(root, problem, cfg["base_seed"], SHOTS)
    b = ensure_corpus(root, problem, cfg["base_seed"], SHOTS + 1)
    assert corpus_dir(root, problem, cfg["base_seed"], SHOTS) != \
        corpus_dir(root, problem, cfg["base_seed"], SHOTS + 1)
    assert a.dets.shape[0] == SHOTS and b.dets.shape[0] == SHOTS + 1
    # Both entries stay independently loadable (a shot count is part of the cache key).
    assert load_entry(corpus_dir(root, problem, cfg["base_seed"], SHOTS)).dets.shape[0] == SHOTS
    assert load_entry(corpus_dir(root, problem, cfg["base_seed"], SHOTS + 1)).dets.shape[0] == SHOTS + 1


def test_window_entry_is_deterministic_and_slices_its_parent(tmp_path):
    cfg = load_config()
    window = next(p for p in cfg["problems"] if p["shape"]["kind"] == "window")
    parent = next(p for p in cfg["problems"] if p["id"] == window["shape"]["parent"])
    first = ensure_corpus(tmp_path / "a", window, cfg["base_seed"], SHOTS,
                          parent=parent, parent_shots=SHOTS)
    second = ensure_corpus(tmp_path / "b", window, cfg["base_seed"], SHOTS,
                           parent=parent, parent_shots=SHOTS)
    parent_corpus = ensure_corpus(tmp_path / "a", parent, cfg["base_seed"], SHOTS)
    files_a = _images([corpus_dir(tmp_path / "a", window, cfg["base_seed"], SHOTS)])
    files_b = _images([corpus_dir(tmp_path / "b", window, cfg["base_seed"], SHOTS)])
    assert files_a.keys() == files_b.keys()
    for name in files_a:
        assert files_a[name] == files_b[name], f"{name} differs between two window builds"
    rows = np.asarray(first.manifest["window"]["rows"], dtype=np.int64)
    assert first.shots == SHOTS and first.H.shape[0] == rows.shape[0]
    assert np.array_equal(first.dets, parent_corpus.dets[:SHOTS][:, rows])
    assert np.array_equal(first.obs, parent_corpus.obs[:SHOTS])
    assert first.det_round.shape == (first.H.shape[0],)


def test_cached_entry_matches_its_manifest(tmp_path):
    cfg, problem = _problem()
    root = tmp_path / "cache"
    ensure_corpus(root, problem, cfg["base_seed"], SHOTS)
    loaded = load_entry(corpus_dir(root, problem, cfg["base_seed"], SHOTS))
    assert loaded.manifest["problem_id"] == PROBLEM_ID
    assert loaded.manifest["shots"] == SHOTS
    assert loaded.manifest["shot_seed"] >= 0
    assert loaded.dem_text.startswith("error(")


def test_generator_versions_are_part_of_the_key(tmp_path, monkeypatch):
    """Switching qLDPC or stim never reuses a corpus entry another version generated."""
    cfg, problem = _problem()
    seed = cfg["base_seed"]
    installed = corpus_mod.generator_versions()
    assert set(installed) == {"qldpc", "qldpc_commit", "stim", "stim_simd"}
    changes = ({}, {"qldpc": installed["qldpc"] + ".other"}, {"qldpc_commit": "other"},
               {"stim": installed["stim"] + ".other"}, {"stim_simd": installed["stim_simd"] + "-other"})
    keys = {corpus_mod.corpus_key(problem, seed, SHOTS, dict(installed, **change))
            for change in changes}
    assert len(keys) == len(changes)
    assert corpus_mod.corpus_key(problem, seed, SHOTS) == \
        corpus_mod.corpus_key(problem, seed, SHOTS, installed)

    root = tmp_path / "cache"
    first = ensure_corpus(root, problem, seed, SHOTS)
    assert not first.from_cache and first.manifest["generator"] == installed

    # Another qLDPC version: a fresh entry in its own directory; the old entry is left alone.
    other = dict(installed, qldpc=installed["qldpc"] + ".other")
    monkeypatch.setattr(corpus_mod, "generator_versions", lambda: dict(other))
    second = ensure_corpus(root, problem, seed, SHOTS)
    assert not second.from_cache
    assert second.path != first.path and second.manifest["generator"] == other
    assert load_entry(first.path).manifest["generator"] == installed
    third = ensure_corpus(root, problem, seed, SHOTS)
    assert third.from_cache and third.path == second.path

    # An entry whose manifest names another generator is rebuilt, never reused.
    manifest_path = second.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generator"] = installed
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    logs = []
    fourth = ensure_corpus(root, problem, seed, SHOTS, log=logs.append)
    assert not fourth.from_cache and fourth.manifest["generator"] == other
    assert any(line.startswith("corpus: rebuilding") for line in logs)
