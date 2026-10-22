# SPDX-License-Identifier: Apache-2.0
"""The device-class table: which core variant a decoder handle gets on a device.

The native front ("libeldpc.so") makes the actual choice at create time, from its own copy of
this table in ``src/port/core_variant.h``. This module is the Python side of the same table: it
lets :func:`eldpc.devices` report, for a device, the variant a handle on it would get without
loading the native library. ``tests/test_core_variant.py`` compares this table with the C one
(the exported ``eldpc_core_variant_for_device``) over a grid of device properties, so the two
cannot drift apart unnoticed.

A row is ``(compute capability major, minor, sm_min, sm_max, variant)``: the first row whose
compute capability matches and whose SM-count range contains the device's SM count wins; a
device that matches no row gets ``"default"``. ``sm_max`` is inclusive, or None for "no upper
bound". Keep this table in step with ``kDeviceClassTable`` in ``src/port/core_variant.h``.
"""

from __future__ import annotations

DEFAULT = "default"
ADA64 = "ada64"

#: Compute capability 8.9 with at least 128 SMs is the Ada 142-SM class (RTX 6000 Ada, L40S).
DEVICE_CLASS_TABLE = (
    (8, 9, 128, None, ADA64),
)


def core_variant_for_device(cc_major: int, cc_minor: int, sm_count: int) -> str:
    """The table's variant for a device of this compute capability and SM count.

    Invalid properties -- a negative compute capability, or no SMs -- match no row and give
    ``"default"``.
    """
    cc_major, cc_minor, sm_count = int(cc_major), int(cc_minor), int(sm_count)
    if cc_major < 0 or cc_minor < 0 or sm_count <= 0:
        return DEFAULT
    for row_major, row_minor, sm_min, sm_max, variant in DEVICE_CLASS_TABLE:
        if (row_major == cc_major and row_minor == cc_minor and sm_count >= sm_min
                and (sm_max is None or sm_count <= sm_max)):
            return variant
    return DEFAULT


def effective_core_variant(cc_major: int, cc_minor: int, sm_count: int, override=None) -> str:
    """The variant a handle would get, with a valid ``ELDPC_DEVICE_CLASS`` override applied.

    An override that is not ``"default"`` or ``"ada64"`` is ignored here; the native front
    rejects it and every create fails with a clear error instead.
    """
    if override in (DEFAULT, ADA64):
        return override
    return core_variant_for_device(cc_major, cc_minor, sm_count)
