# SPDX-License-Identifier: Apache-2.0
"""Fixtures of the eldpc test suite; the shared helpers are in eldpc_testlib.py."""

from __future__ import annotations

import pytest

from eldpc_testlib import REFERENCE_LIB  # noqa: F401  (also triggers the ELDPC_REQUIRE_GPU gate)


@pytest.fixture(scope="session")
def reference_lib():
    """The separately built reference library (decoder ABI v1) of the same sources."""
    from refabi import RefLibrary

    return RefLibrary(REFERENCE_LIB)


def pytest_configure(config):
    """Register the CPU-core marker here too (pyproject.toml registers it as well), so that it is
    known whichever configuration file pytest picks up for the run."""
    config.addinivalue_line(
        "markers",
        'cpu_core: the test needs no CUDA device; with none visible it runs on the CPU core (device="cpu")',
    )
