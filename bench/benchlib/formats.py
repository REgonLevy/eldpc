# SPDX-License-Identifier: Apache-2.0
"""File formats shared by bench.py and the C driver (bench/c/bench_driver.c).

All integers are little-endian; all arrays are unpacked (one byte per bit, value 0 or 1).

problem file
    8s  magic "ELDPCPB1"
    u32 version (1), m, n, k, nnzh, nnzo
    u32 h_colptr[n+1], u32 h_rowind[nnzh]      H (m x n) in CSC
    u32 o_colptr[n+1], u32 o_rowind[nnzo]      O (k x n) in CSC
    f64 priors[n]
    u32 dem_len, bytes dem_text[dem_len]       optional stim DEM text ("" if absent)

shots file
    8s  magic "ELDPCSH1"
    u32 version (1), shots, m, k
    u8  dets[shots*m]                          syndromes, shot-major
    u8  obs[shots*k]                           true observable flips (used by the LER check only)

C driver output file
    8s  magic "ELDPCOUT"
    u32 version (1), records_per_variant (shots in the single-shot modes, batches in batch mode),
        m, n, k, nvariants (2), mode (0 back-to-back, 1 cadence, 2 batch),
        gap_us, keep_warm_a, keep_warm_b, warmup_calls, cpu (0xffffffff = not pinned)
    u32 lib0_len, bytes lib0, u32 lib1_len, bytes lib1
    u32 cfg0_len, bytes cfg0, u32 cfg1_len, bytes cfg1
    u64 session_t0_ns                          CLOCK_MONOTONIC at the start of the measured loop
    2*records_per_variant records, 32 bytes each:
        u64 t0_ns, u64 t1_ns                   CLOCK_MONOTONIC around the call (one eldpc_decode, or one
                                               eldpc_decode_batch in batch mode)
        u64 rc                                  return code, sign-extended
        u64 status | (variant << 32)            status word, variant 0 or 1; in batch mode
                                               shots_in_the_batch | (variant << 32) instead of the status
    trailer:
        u64 keep_n, keep_err, keep_sum_ns, keep_max_ns, keep_final_sum_ns, keep_final_max_ns
        u64 checksum_all, checksum_variant0, checksum_variant1   FNV-1a over (corr, obs, status) per shot
        u64 marker 0x314c494152544e42              "BNTRAIL1" read as bytes
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import scipy.sparse as sp

PROBLEM_MAGIC = b"ELDPCPB1"
SHOTS_MAGIC = b"ELDPCSH1"
OUTPUT_MAGIC = b"ELDPCOUT"
OUTPUT_MARKER = 0x314C494152544E42


def canonical_csc(H) -> sp.csc_matrix:
    """Canonical form of a GF(2) matrix: sorted indices, no explicit zeros, uint8."""
    m = sp.csc_matrix(H, dtype=np.uint8, copy=True)
    m.sum_duplicates()
    m.eliminate_zeros()
    m.sort_indices()
    return m


def write_problem(path, H, O, priors, dem_text: str | None = None) -> None:
    H = canonical_csc(H)
    O = canonical_csc(O)
    m, n = H.shape
    if O.shape[1] != n:
        raise ValueError("O must have as many columns as H")
    k = O.shape[0]
    priors = np.asarray(priors, dtype=np.float64).reshape(-1)
    if priors.shape != (n,):
        raise ValueError(f"priors has shape {priors.shape}, expected ({n},)")
    if not np.all(np.isfinite(priors)) or not np.all((priors > 0) & (priors < 1)):
        raise ValueError("priors must be finite probabilities in (0, 1)")
    dem = (dem_text or "").encode("utf-8")
    with open(path, "wb") as f:
        f.write(PROBLEM_MAGIC)
        f.write(struct.pack("<6I", 1, m, n, k, H.nnz, O.nnz))
        np.asarray(H.indptr, dtype="<u4").tofile(f)
        np.asarray(H.indices, dtype="<u4").tofile(f)
        np.asarray(O.indptr, dtype="<u4").tofile(f)
        np.asarray(O.indices, dtype="<u4").tofile(f)
        np.asarray(priors, dtype="<f8").tofile(f)
        f.write(struct.pack("<I", len(dem)))
        f.write(dem)


def read_problem(path):
    """Return (H, O, priors, dem_text) with H/O as scipy CSC matrices."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != PROBLEM_MAGIC:
            raise ValueError(f"{path}: not an eldpc bench problem file")
        version, m, n, k, nnzh, nnzo = struct.unpack("<6I", f.read(24))
        if version != 1:
            raise ValueError(f"{path}: unsupported problem version {version}")
        hcp = np.fromfile(f, dtype="<u4", count=n + 1)
        hri = np.fromfile(f, dtype="<u4", count=nnzh)
        ocp = np.fromfile(f, dtype="<u4", count=n + 1)
        ori = np.fromfile(f, dtype="<u4", count=nnzo)
        priors = np.fromfile(f, dtype="<f8", count=n)
        (dem_len,) = struct.unpack("<I", f.read(4))
        dem_text = f.read(dem_len).decode("utf-8") if dem_len else ""
    if len(hcp) != n + 1 or len(ocp) != n + 1 or len(priors) != n:
        raise ValueError(f"{path}: truncated problem file")
    H = sp.csc_matrix((np.ones(len(hri), dtype=np.uint8), hri, hcp), shape=(m, n))
    O = sp.csc_matrix((np.ones(len(ori), dtype=np.uint8), ori, ocp), shape=(k, n))
    return H, O, priors, dem_text


def write_shots(path, dets, obs=None) -> None:
    dets = np.ascontiguousarray(dets, dtype=np.uint8)
    if dets.ndim != 2:
        raise ValueError("dets must be a (shots, m) array")
    shots, m = dets.shape
    obs = np.zeros((shots, 0), dtype=np.uint8) if obs is None else np.ascontiguousarray(obs, dtype=np.uint8)
    if obs.ndim != 2 or obs.shape[0] != shots:
        raise ValueError("obs must be a (shots, k) array")
    k = obs.shape[1]
    with open(path, "wb") as f:
        f.write(SHOTS_MAGIC)
        f.write(struct.pack("<4I", 1, shots, m, k))
        dets.tofile(f)
        obs.tofile(f)


def read_shots(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != SHOTS_MAGIC:
            raise ValueError(f"{path}: not an eldpc bench shots file")
        version, shots, m, k = struct.unpack("<4I", f.read(16))
        if version != 1:
            raise ValueError(f"{path}: unsupported shots version {version}")
        dets = np.fromfile(f, dtype=np.uint8, count=shots * m).reshape(shots, m)
        obs = np.fromfile(f, dtype=np.uint8, count=shots * k).reshape(shots, k) if k else np.zeros((shots, 0), np.uint8)
    if dets.shape != (shots, m) or obs.shape != (shots, k):
        raise ValueError(f"{path}: truncated shots file")
    return dets, obs


def read_driver_output(path):
    """Parse a bench_driver output file into a dict of metadata, per-shot records and keep-warm stats."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 8 or data[:8] != OUTPUT_MAGIC:
        raise ValueError(f"{path}: not a bench_driver output file")
    pos = 8

    def u32():
        nonlocal pos
        v = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        return v

    def u64():
        nonlocal pos
        v = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        return v

    def blob():
        nonlocal pos
        n = u32()
        s = data[pos:pos + n].decode("utf-8", "replace")
        pos += n
        return s

    version = u32()
    if version != 1:
        raise ValueError(f"{path}: unsupported driver output version {version}")
    meta = {
        "version": version,
        "records_per_variant": u32(),
        "m": u32(),
        "n": u32(),
        "k": u32(),
        "nvariants": u32(),
        "mode": u32(),
        "gap_us": u32(),
        "keep_warm_a": u32(),
        "keep_warm_b": u32(),
        "warmup_calls": u32(),
        "cpu": u32(),
    }
    meta["libraries"] = [blob(), blob()]
    meta["configs"] = [blob(), blob()]
    meta["session_t0_ns"] = u64()
    nrec = meta["records_per_variant"] * meta["nvariants"]
    if len(data) < pos + nrec * 32 + 10 * 8:
        raise ValueError(f"{path}: truncated driver output")
    rec = np.frombuffer(data, dtype="<u8", count=nrec * 4, offset=pos).reshape(nrec, 4)
    pos += nrec * 32
    variant = (rec[:, 3] >> np.uint64(32)).astype(np.int64)
    low = (rec[:, 3] & np.uint64(0xFFFFFFFF)).astype(np.int64)
    if meta["mode"] == 2:  # batch mode: the low word is the number of shots in the batch
        status, shots_per_record = None, low
    else:
        status, shots_per_record = low.astype(np.uint64), None
    rc = np.ascontiguousarray(rec[:, 2]).view(np.int64)  # the driver stores it sign-extended
    t0 = rec[:, 0].astype(np.int64)
    t1 = rec[:, 1].astype(np.int64)
    trailer = np.frombuffer(data, dtype="<u8", count=10, offset=pos)
    if int(trailer[9]) != OUTPUT_MARKER:
        raise ValueError(f"{path}: missing trailer marker")
    out = {
        "meta": meta,
        "t0_ns": t0,
        "t1_ns": t1,
        "dur_ns": t1 - t0,
        "rc": rc,
        "status": status,
        "shots_per_record": shots_per_record,
        "variant": variant,
        "keep": {
            "n": int(trailer[0]), "errors": int(trailer[1]),
            "periodic_sum_ns": int(trailer[2]), "periodic_max_ns": int(trailer[3]),
            "final_sum_ns": int(trailer[4]), "final_max_ns": int(trailer[5]),
        },
        "checksums": {"all": int(trailer[6]), "variant0": int(trailer[7]), "variant1": int(trailer[8])},
    }
    if np.any(variant < 0) or np.any(variant > 1):
        raise ValueError(f"{path}: bad variant tag")
    return out
