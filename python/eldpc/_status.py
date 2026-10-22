# SPDX-License-Identifier: Apache-2.0
"""Decoding status words (``eldpc.h``): flags in the low 16 bits, BP iterations above."""

from __future__ import annotations

ST_BP_CONVERGED = 0x0001
ST_POSTPROC_INVOKED = 0x0002
ST_HIGHER_ORDER_WIN = 0x0004
ST_ABSTAIN = 0x0008
ST_SYNDROME_UNMATCHED = 0x0010
ST_FLAGS_MASK = 0xFFFF
ST_ITER_SHIFT = 16
ST_ITER_MAX = 0xFFFF


class status:  # noqa: N801 - a namespace, as in the API
    """Helpers and constants for the status word returned by :meth:`eldpc.Decoder.decode_full`."""

    ST_BP_CONVERGED = ST_BP_CONVERGED
    ST_POSTPROC_INVOKED = ST_POSTPROC_INVOKED
    ST_HIGHER_ORDER_WIN = ST_HIGHER_ORDER_WIN
    ST_ABSTAIN = ST_ABSTAIN
    ST_SYNDROME_UNMATCHED = ST_SYNDROME_UNMATCHED
    ST_FLAGS_MASK = ST_FLAGS_MASK
    ST_ITER_SHIFT = ST_ITER_SHIFT

    @staticmethod
    def converged(st: int) -> bool:
        """True when BP's output satisfied the syndrome and no post-processing ran."""
        return bool(int(st) & ST_BP_CONVERGED)

    @staticmethod
    def postprocessed(st: int) -> bool:
        """True when the OSD post-processing stage ran for this shot."""
        return bool(int(st) & ST_POSTPROC_INVOKED)

    @staticmethod
    def iterations(st: int) -> int:
        """The number of BP iterations actually run."""
        return (int(st) >> ST_ITER_SHIFT) & ST_ITER_MAX
