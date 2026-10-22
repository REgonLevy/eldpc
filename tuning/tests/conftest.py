# SPDX-License-Identifier: Apache-2.0
"""Test path setup: the tuner modules live one directory above.

With ``ELDPC_REQUIRE_GPU=1`` (as for the package tests) a test that skips because no GPU decoder
could be created fails instead, so a GPU run cannot pass by skipping the decoder tests.
"""
import os
import sys

import pytest

GPU_SKIP_REASON = "no GPU decoder available"


def pytest_configure(config):  # noqa: ARG001
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # noqa: ARG001
    outcome = yield
    report = outcome.get_result()
    if not report.skipped or os.environ.get("ELDPC_REQUIRE_GPU") != "1":
        return
    longrepr = report.longrepr
    reason = str(longrepr[2]) if isinstance(longrepr, tuple) and len(longrepr) == 3 else str(longrepr)
    if GPU_SKIP_REASON in reason:
        report.outcome = "failed"
        report.longrepr = f"ELDPC_REQUIRE_GPU=1, but the test skipped: {reason}"
