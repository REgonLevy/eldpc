/*
 * decoder_abi.h -- decoder-bench native decoder ABI, version 1.
 *
 * FROZEN. Do not edit. Acceptance overwrites this file with the pristine copy,
 * so any change you make here is silently discarded and your code will be
 * compiled against the original.
 *
 * A "decoder library" is a shared object (.so) that exports every function
 * declared below with C linkage and default visibility. The runner
 * (dbench-run, C++) loads it with dlopen(); the Python side loads it with
 * ctypes (python/dbench/abi.py). Both call ONLY the functions below.
 *
 * ---------------------------------------------------------------------------
 * Data conventions
 * ---------------------------------------------------------------------------
 * - A decoding problem has m = num_detectors rows, n = num_mechanisms columns
 *   and k = num_observables observables.
 * - H (m x n) and O (k x n) are GF(2) matrices in compressed-sparse-column
 *   form: column j has row indices rowind[colptr[j] .. colptr[j+1]-1],
 *   strictly increasing, all < m (resp. < k). colptr[0] == 0, colptr is
 *   non-decreasing, colptr[n] == nnz. Values are implicitly 1.
 * - priors[j] is the prior probability that mechanism j fires, a finite double
 *   with 0 < priors[j] < 1.
 * - LLRs are natural-log ratios llr = ln((1-p)/p). They must be finite.
 * - Syndromes and all bit outputs are UNPACKED: one uint8_t per bit, value 0
 *   or 1. A syndrome has m bytes, a correction has n bytes, an observable
 *   prediction has k bytes. Batches are row-major (shot-major).
 * - out_obs MUST equal O * out_correction (mod 2) whenever the decoder has
 *   DBENCH_CAP_CORRECTION and the shot is not abstained.
 *
 * ---------------------------------------------------------------------------
 * Behavioural contract (every decoder library must satisfy all of these)
 * ---------------------------------------------------------------------------
 * 1. Copy-in: dbench_create() copies everything it needs from the problem and
 *    config. The caller may free or overwrite them as soon as it returns.
 * 2. Stateless per shot: the outputs for a shot depend only on (problem,
 *    config, current priors, syndrome). They never depend on previously
 *    decoded shots, on the handle's history, on time, on the thread, on the
 *    address of buffers, or on any random source other than a seed given in
 *    the config. (Internal caches are fine if they do not change outputs.)
 * 3. Determinism: identical inputs give bit-identical outputs (correction,
 *    obs, status flags and iteration count) across calls, handles, processes
 *    and threads, and dbench_decode_batch() gives exactly what calling
 *    dbench_decode() on each shot in order would give.
 * 4. Threads: distinct handles may be used concurrently from different
 *    threads. One handle is never used by two threads at once (the caller
 *    guarantees this). Library-global state must therefore be thread-safe.
 * 5. No I/O on the decode path (no stdout/stderr, no files, no environment
 *    lookups), no allocation that grows without bound, no exceptions or
 *    signals escaping the ABI.
 * 6. Config: dbench_create() receives a config string of the form
 *    "key=value;key=value" (keys [a-z0-9_]+, values without ';' or '=',
 *    surrounding ASCII whitespace of keys and values ignored, empty segments
 *    ignored, a key may appear at most once). Unknown keys, duplicate keys,
 *    missing required keys and unparsable values make dbench_create() fail.
 *    The empty string is a valid config for decoders whose keys all have
 *    defaults.
 * 7. Everything inside dbench_decode() is timed. It does only the work needed
 *    to produce that shot's correction, observables and status: compute
 *    shot-independent data at create (or in dbench_set_priors), produce
 *    optional outputs (soft output) lazily in the call that requests them, and
 *    do not repeat checks whose result is known by construction.
 *
 * Return codes: functions returning int return DBENCH_OK (0) on success or one
 * of the negative DBENCH_E* codes below. dbench_create() returns NULL on
 * failure and, if err != NULL and err_len > 0, writes a NUL-terminated,
 * human-readable message (truncated to err_len-1 bytes) into err.
 */
#ifndef DBENCH_DECODER_ABI_H
#define DBENCH_DECODER_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DBENCH_ABI_VERSION 1u

#if defined(__GNUC__)
#define DBENCH_EXPORT __attribute__((visibility("default")))
#else
#define DBENCH_EXPORT
#endif

/* ---- return codes ------------------------------------------------------ */
#define DBENCH_OK 0
#define DBENCH_EINVAL (-1)     /* bad argument (NULL handle, wrong length...)  */
#define DBENCH_EINTERNAL (-2)  /* decoder-internal failure                     */
#define DBENCH_ENOTSUP (-3)    /* operation not supported by this decoder      */

/* ---- per-shot status word -----------------------------------------------
 * out_status[i] = (flags & 0xFFFF) | (min(iterations, 0xFFFF) << 16)
 * "iterations" is the number of message-passing iterations actually run (0
 * for decoders without iterations, and 0 for an all-zero syndrome when the
 * decoder short-circuits it).                                              */
#define DBENCH_ST_BP_CONVERGED 0x0001u       /* BP output satisfied the syndrome; no post-processing ran */
#define DBENCH_ST_POSTPROC_INVOKED 0x0002u   /* OSD / LSD / other post-processing ran */
#define DBENCH_ST_HIGHER_ORDER_WIN 0x0004u   /* a higher-order post-processing candidate replaced the order-0 solution */
#define DBENCH_ST_ABSTAIN 0x0008u            /* decoder declines this shot; obs/correction are all zero; shot counts as discarded */
#define DBENCH_ST_SYNDROME_UNMATCHED 0x0010u /* returned correction does not reproduce the syndrome (only with CAP_CORRECTION) */
#define DBENCH_ST_FLAGS_MASK 0x0000FFFFu
#define DBENCH_ST_ITER_SHIFT 16
#define DBENCH_ST_ITER_MAX 0xFFFFu

/* ---- capability bits returned by dbench_capabilities() ------------------- */
#define DBENCH_CAP_CORRECTION 0x01u   /* fills out_correction                     */
#define DBENCH_CAP_SET_PRIORS 0x02u   /* dbench_set_priors() is implemented      */
#define DBENCH_CAP_SOFT_OUTPUT 0x04u  /* dbench_get_soft_output() is implemented */
#define DBENCH_CAP_NATIVE_BATCH 0x08u /* decode_batch is more than a loop (informational only) */

/* ---- problem description --------------------------------------------------*/
typedef struct dbench_problem {
    uint32_t abi_version;       /* must be DBENCH_ABI_VERSION; decoders reject other values */
    uint32_t num_detectors;     /* m */
    uint32_t num_mechanisms;    /* n */
    uint32_t num_observables;   /* k (may be 0) */
    const uint32_t *h_colptr;   /* n+1 entries */
    const uint32_t *h_rowind;   /* h_colptr[n] entries */
    const uint32_t *o_colptr;   /* n+1 entries */
    const uint32_t *o_rowind;   /* o_colptr[n] entries */
    const double *priors;       /* n entries, 0 < p < 1 */
    const char *dem_text;       /* NUL-terminated stim DEM text of the same problem, or NULL */
} dbench_problem;

/* Opaque decoder handle. */
typedef struct dbench_decoder dbench_decoder;

/* Must return DBENCH_ABI_VERSION. Loaders refuse libraries returning anything else. */
DBENCH_EXPORT uint32_t dbench_abi_version(void);

/* Static, NUL-terminated, human-readable decoder name, e.g. "ldpc-ref". */
DBENCH_EXPORT const char *dbench_decoder_name(void);

/* Bitwise OR of DBENCH_CAP_* flags. Constant for the library. */
DBENCH_EXPORT uint32_t dbench_capabilities(void);

/* Build a decoder for `problem` with `config`. NULL on failure (see err). */
DBENCH_EXPORT dbench_decoder *dbench_create(const char *config, const dbench_problem *problem,
                                            char *err, size_t err_len);

/* Replace the priors used by subsequent decodes on this handle with the given
 * LLRs (n = num_mechanisms entries). Sticky until the next call. Decoders
 * without DBENCH_CAP_SET_PRIORS return DBENCH_ENOTSUP. After set_priors(llr),
 * outputs must equal those of a fresh handle created with priors
 * p_j = 1 / (1 + exp(llr_j)) (up to floating-point evaluation of that formula,
 * i.e. bit-identical when the decoder converts priors to LLRs with
 * llr = log((1-p)/p) and the given llr round-trips). */
DBENCH_EXPORT int dbench_set_priors(dbench_decoder *h, const double *llr, uint32_t n);

/* Decode one shot. syndrome: m bytes. out_correction: n bytes or NULL.
 * out_obs: k bytes (must be non-NULL when k > 0). out_status: 1 word or NULL.
 * Decoders without DBENCH_CAP_CORRECTION never write out_correction. */
DBENCH_EXPORT int dbench_decode(dbench_decoder *h, const uint8_t *syndrome, uint8_t *out_correction,
                                uint8_t *out_obs, uint32_t *out_status);

/* Decode num_shots shots: syndromes is num_shots*m bytes, out_corrections is
 * num_shots*n bytes or NULL, out_obs is num_shots*k bytes, out_status is
 * num_shots words or NULL. Results must equal a loop of dbench_decode(). */
DBENCH_EXPORT int dbench_decode_batch(dbench_decoder *h, uint32_t num_shots, const uint8_t *syndromes,
                                      uint8_t *out_corrections, uint8_t *out_obs, uint32_t *out_status);

/* Copy the decoder's soft output for the most recent dbench_decode() on this
 * handle (n doubles). For BP-family decoders this is the posterior LLR vector
 * that the post-processing stage ordered columns by. Decoders without
 * DBENCH_CAP_SOFT_OUTPUT return DBENCH_ENOTSUP. Undefined before the first
 * decode and after dbench_decode_batch(). */
DBENCH_EXPORT int dbench_get_soft_output(dbench_decoder *h, double *out_llr, uint32_t n);

/* Free the handle. NULL is a no-op. */
DBENCH_EXPORT void dbench_destroy(dbench_decoder *h);

/* OPTIONAL (not one of the required symbols above; the ABI version stays 1).
 * Static, NUL-terminated, free-form description of how the library was built,
 * e.g. "gcc 13.3.0; -O2 -ffp-contract=off -fno-fast-math; commit 1a2b3c4".
 * Loaders look it up with dlsym()/ctypes and record null when it is absent. */
DBENCH_EXPORT const char *dbench_decoder_build_info(void);

#ifdef __cplusplus
}
#endif

#endif /* DBENCH_DECODER_ABI_H */
