# SPDX-License-Identifier: Apache-2.0
"""The tie-free classification used to explain upstream-wheel differences.

The functions are pure (no ldpc import), so these tests pin their semantics on small matrices:
an equal-soft-value group that mixes pivot and non-pivot columns is where the OSD tie order can
change the result; a shot that BP converged on is always tie-free.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]      # the package root (tests/verify/x.py -> root)
if str(ROOT / "verify") not in sys.path:
    sys.path.insert(0, str(ROOT / "verify"))

from eldpc_verify.abi import ST_BP_CONVERGED, ST_POSTPROC_INVOKED  # noqa: E402
from eldpc_verify.dem import canonical_csc  # noqa: E402
from eldpc_verify.upstream import (bposd_tie_free, column_masks, equal_value_groups,  # noqa: E402
                                   osd_column_order, osd_pivot_walk, shot_cause)

# column 0 and column 1 are identical, column 2 is independent.
H = canonical_csc(np.array([[1, 1, 0], [0, 0, 1]], dtype=np.uint8))
PARAMS = {"decoder": "bposd", "osd_method": "osd0", "osd_order": 0}


def test_column_masks():
    assert column_masks(H) == [1, 1, 2]


def test_order_and_groups():
    soft = [0.0, 0.0, 1.0]
    assert osd_column_order(soft) == [0, 1, 2]
    assert equal_value_groups(osd_column_order(soft), soft) == [[0, 1], [2]]


def test_nan_sorts_last_and_groups_together():
    soft = [float("nan"), 0.0, float("nan")]
    order = osd_column_order(soft)
    assert order == [1, 0, 2]
    groups = equal_value_groups(order, soft)
    assert groups == [[1], [0, 2]]


def test_equal_group_splitting_pivots_is_not_tie_free():
    masks = column_masks(H)
    tie_free, detail = osd_pivot_walk(masks, [[0, 1], [2]])
    assert not tie_free
    assert detail == {"group_size": 2, "pivots_in_group": 1}
    assert bposd_tie_free(masks, [0.0, 0.0, 1.0], "osd0", 0)[0] is False


def test_distinct_soft_values_are_tie_free():
    masks = column_masks(H)
    assert osd_pivot_walk(masks, [[0], [1], [2]]) == (True, None)
    assert bposd_tie_free(masks, [3.0, 2.0, 1.0], "osd0", 0)[0] is True


def test_converged_shots_are_always_tie_free():
    masks = column_masks(H)
    syndrome = np.array([1, 0], dtype=np.uint8)
    # even with a splitting group, a converged shot never reaches OSD
    assert shot_cause(ST_BP_CONVERGED, [0.0, 0.0, 1.0], syndrome, PARAMS, masks) == (True, None)


def test_postprocessed_shot_uses_the_group_rule():
    masks = column_masks(H)
    syndrome = np.array([1, 0], dtype=np.uint8)
    tie_free, detail = shot_cause(ST_POSTPROC_INVOKED, [0.0, 0.0, 1.0], syndrome, PARAMS, masks)
    assert tie_free is False and detail["group_size"] == 2


def test_bp_decoder_never_uses_osd_order():
    masks = column_masks(H)
    params = {"decoder": "bp"}
    assert shot_cause(ST_POSTPROC_INVOKED, [0.0, 0.0, 1.0],
                      np.array([1, 0], dtype=np.uint8), params, masks) == (True, None)


def test_nan_values_count_as_equal():
    masks = column_masks(H)
    soft = [float("nan"), float("nan"), 1.0]
    tie_free, detail = bposd_tie_free(masks, soft, "osd0", 0)
    assert tie_free is False
    assert detail["group_size"] == 2
    assert math.isnan(soft[0])
