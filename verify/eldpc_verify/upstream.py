# SPDX-License-Identifier: Apache-2.0
"""Informational comparison with the installed upstream ``ldpc`` 2.4.1 wheel.

The pass criterion of the suite is the reference decoder; this column reports, for the same shots,
where the upstream Python wheel's ``BpOsdDecoder`` produces a different correction or observable
prediction. A difference is explained by the OSD tie order when the documented total order of the
reference decoder can change the OSD base of that shot: the reference orders columns by ascending
soft value with
ascending column index as the explicit tie-break (NaN after every number), while upstream's sort
leaves the order of equal values unspecified. The test is the reference construction's
tie-free rule:

- a shot that BP converged on (and any ``decoder=bp`` shot) never reaches post-processing, so it is
  tie-free;
- otherwise, walking the OSD column order in groups of equal soft value, a group that contributes
  both pivot and non-pivot columns is where the tie order can change the result (for osd_order > 0
  the exhaustive / combination-sweep conditions of the reference construction rule are applied too).

A differing shot that is tie-free is reported as "unexplained". This column never decides pass or
fail.
"""
from __future__ import annotations

import math

import numpy as np

from .abi import ST_BP_CONVERGED
from .dem import canonical_csc

_BP_METHOD = {"ms": "minimum_sum", "ps": "product_sum"}
_OSD_METHOD = {"osd0": "OSD_0", "osd_e": "OSD_E", "osd_cs": "OSD_CS"}


class UpstreamError(RuntimeError):
    """The upstream wheel is missing or a configuration cannot be mapped onto it."""


def availability() -> tuple[bool, str]:
    """(usable, description) of the installed upstream wheel."""
    try:
        import ldpc
    except Exception as exc:  # noqa: BLE001 - any import failure disables the column
        return False, f"ldpc is not importable: {exc}"
    version = getattr(ldpc, "__version__", None)
    return True, f"ldpc {version}" if version else "ldpc (version unknown)"


class UpstreamDecoder:
    """The pip ``ldpc`` decoder for one problem and one preset configuration."""

    def __init__(self, H, O, priors, params: dict):
        import ldpc

        pcm = canonical_csc(H).tocsr()
        self.n = int(pcm.shape[1])
        self.O = canonical_csc(O).astype(np.int64)
        self.k = int(self.O.shape[0])
        channel = [float(x) for x in np.asarray(priors, dtype=np.float64).reshape(-1)]
        common = {
            "error_channel": channel,
            "max_iter": int(params["max_iter"]),
            "bp_method": _BP_METHOD[params["bp_method"]],
            "ms_scaling_factor": float(params.get("ms_scaling_factor", 0.625)),
            "schedule": params.get("schedule", "parallel"),
        }
        decoder = params["decoder"]
        if decoder == "bposd":
            self.decoder = ldpc.BpOsdDecoder(
                pcm, osd_method=_OSD_METHOD[params["osd_method"]],
                osd_order=int(params.get("osd_order", 0)), **common)
        elif decoder == "bp":
            self.decoder = ldpc.BpDecoder(pcm, input_vector_type="syndrome", **common)
        else:
            raise UpstreamError(f"the upstream comparison does not support decoder={decoder!r}")
        self.zeros = np.zeros(self.n, dtype=np.uint8)

    def decode(self, syndrome) -> tuple[np.ndarray, np.ndarray]:
        """(correction (n,), observables (k,)) of one shot, in the ABI's unpacked convention."""
        syndrome = np.asarray(syndrome, dtype=np.uint8).reshape(-1)
        if not syndrome.any():
            corr = self.zeros.copy()
        else:
            corr = np.ascontiguousarray(self.decoder.decode(syndrome), dtype=np.uint8).reshape(-1)
        obs = ((self.O @ corr.astype(np.int64)) % 2).astype(np.uint8) if self.k else \
            np.zeros(0, dtype=np.uint8)
        return corr, obs


def osd_column_order(soft) -> list:
    """Column indices in the reference's OSD order: value ascending, ties by index, NaNs last by index."""
    def key(j):
        v = float(soft[j])
        if math.isnan(v):
            return (1, 0.0, j)
        return (0, v, j)

    return sorted(range(len(soft)), key=key)


def equal_value_groups(order, soft) -> list:
    """``order`` split into groups of equal soft value; all NaNs form one group."""
    groups = []
    for j in order:
        v = float(soft[j])
        if groups:
            prev = float(groups[-1][0])
            if (math.isnan(v) and math.isnan(prev)) or \
                    (not math.isnan(v) and not math.isnan(prev) and v == prev):
                groups[-1][1].append(j)
                continue
        groups.append((v, [j]))
    return [cols for _, cols in groups]


def column_masks(H) -> list:
    """One GF(2) bitmask per column of H (bit i = detector i)."""
    Hc = canonical_csc(H)
    masks = []
    for j in range(Hc.shape[1]):
        mask = 0
        for i in Hc.indices[Hc.indptr[j]:Hc.indptr[j + 1]]:
            mask |= 1 << int(i)
        masks.append(mask)
    return masks


def osd_pivot_walk(h_masks, groups):
    """Greedy GF(2) elimination of the columns in OSD order, group by group.

    Returns (ok, detail): ok is False at the first equal-value group that contributes both pivot
    and non-pivot columns, which is exactly where the tie order can change the pivot set; detail
    names that group and its counts."""
    basis = {}
    for cols in groups:
        added = 0
        for j in cols:
            mask = h_masks[j]
            while mask:
                pivot = mask.bit_length() - 1
                if pivot in basis:
                    mask ^= basis[pivot]
                else:
                    break
            if mask:
                basis[mask.bit_length() - 1] = mask
                added += 1
        if 0 < added < len(cols):
            return False, {"group_size": len(cols), "pivots_in_group": added}
    return True, None


def bposd_tie_free(h_masks, soft, osd_method, osd_order: int):
    """(tie_free, detail) for an OSD shot under the reference construction's tie-free rule."""
    groups = equal_value_groups(osd_column_order(soft), soft)
    ok, detail = osd_pivot_walk(h_masks, groups)
    if not ok:
        return False, detail
    if osd_order <= 0:
        return True, None
    # Higher-order OSD rules of the reference construction. The suite's presets all use
    # osd0/osd_order 0; the rules are kept so an osd_order > 0 preset is classified correctly.
    basis = {}
    pivots, non_pivots = [], []
    for cols in groups:
        for j in cols:
            mask = h_masks[j]
            while mask:
                pivot = mask.bit_length() - 1
                if pivot in basis:
                    mask ^= basis[pivot]
                else:
                    break
            if mask:
                basis[mask.bit_length() - 1] = mask
                pivots.append(j)
            else:
                non_pivots.append(j)
    non_pivot_set = set(non_pivots)
    if osd_method == "osd_e":
        first_w = set(non_pivots[:osd_order])
        for cols in groups:
            group_nps = [j for j in cols if j in non_pivot_set]
            if len(group_nps) >= 2 and any(j in first_w for j in group_nps):
                return False, {"reason": "osd_e window straddles an equal-value group"}
    elif osd_method == "osd_cs":
        for cols in groups:
            if sum(1 for j in cols if j in non_pivot_set) >= 2:
                return False, {"reason": "osd_cs sweeps an equal-value group"}
    return True, None


def shot_cause(status: int, soft, syndrome, params: dict, h_masks) -> tuple[bool, dict | None]:
    """(tie_free, detail) for one shot: can the documented tie order have changed its result?"""
    if int(status) & ST_BP_CONVERGED:
        return True, None
    decoder = params.get("decoder")
    if decoder == "bp":
        return True, None
    if decoder == "bposd":
        return bposd_tie_free(h_masks, soft, params.get("osd_method"), int(params.get("osd_order", 0)))
    return False, {"reason": f"no tie-free rule for decoder={decoder!r}"}
