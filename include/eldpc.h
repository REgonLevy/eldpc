/* SPDX-License-Identifier: Apache-2.0 */
/*
 * eldpc.h -- public C API of the eldpc package.
 *
 * eldpc decodes qLDPC detector-error models with the exact fastbpgpu6g decoder: min-sum BP on the
 * GPU and an exact bit-packed OSD-0 on the host. libeldpc.so is a thin front over one of four
 * builds of the GPU decoder core: the default build and the "ada64" build of each host ISA
 * (libeldpc_core_avx2.so, libeldpc_core_avx512.so, libeldpc_core_avx2_ada64.so,
 * libeldpc_core_avx512_ada64.so), and over the optional CPU core, one build per host ISA
 * (libeldpc_cpu_avx2.so, libeldpc_cpu_avx512.so; the fastbpcpu6g sources, no CUDA code and no CUDA
 * library dependency). The host ISA is chosen at load time from the CPU's capabilities and the
 * OS's vector state; the GPU core variant is chosen per handle at create time from the target
 * device (eldpc_core_variant()), or forced with ELDPC_DEVICE_CLASS. The two GPU builds of one ISA
 * differ in one compile definition only: DBENCH_GPU_SM_BUDGET=64 in the ada64 build caps the SM
 * count the create path plans for. The problem struct, the return codes, the status word and the
 * semantics of every call are those of the decoder-bench ABI v1 (decoder_abi.h), renamed with the
 * eldpc_ prefix.
 *
 * Threads: distinct handles may be used concurrently from different threads; one handle must not
 * be used by two threads at once. A CPU handle computes each shot with `threads` threads: the
 * calling thread and `threads - 1` threads it starts at create, which busy-wait between decodes
 * until the handle is destroyed.
 */
#ifndef ELDPC_H
#define ELDPC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ELDPC_API_VERSION 1u

/* Return codes. */
#define ELDPC_OK 0
#define ELDPC_EINVAL (-1)     /* bad argument (NULL handle, wrong length, invalid value) */
#define ELDPC_EINTERNAL (-2)  /* decoder-internal failure                         */
#define ELDPC_ENOTSUP (-3)    /* operation not supported by this decoder          */

/* Per-shot status word: out_status = (flags & 0xFFFF) | (iterations << 16). */
#define ELDPC_ST_BP_CONVERGED 0x0001u        /* BP output satisfied the syndrome; no post-processing ran */
#define ELDPC_ST_POSTPROC_INVOKED 0x0002u    /* OSD ran */
#define ELDPC_ST_HIGHER_ORDER_WIN 0x0004u    /* a higher-order post-processing candidate won */
#define ELDPC_ST_ABSTAIN 0x0008u             /* the decoder declined the shot */
#define ELDPC_ST_SYNDROME_UNMATCHED 0x0010u  /* the correction does not reproduce the syndrome */
#define ELDPC_ST_FLAGS_MASK 0x0000FFFFu
#define ELDPC_ST_ITER_SHIFT 16
#define ELDPC_ST_ITER_MAX 0xFFFFu

/* Flags of eldpc_keep_warm(). */
#define ELDPC_KEEP_WARM_GPU 1u

#if defined(__GNUC__)
#define ELDPC_EXPORT __attribute__((visibility("default")))
#else
#define ELDPC_EXPORT
#endif

/* A decoding problem: m = num_detectors rows, n = num_mechanisms columns, k = num_observables
 * observables. H (m x n) and O (k x n) are GF(2) matrices in compressed sparse column form with
 * strictly increasing row indices; priors are n error probabilities in (0, 1); dem_text is the
 * optional stim detector error model of the same problem, or NULL.
 *
 * Field for field the same as dbench_problem of decoder-bench ABI v1. */
typedef struct eldpc_problem {
    uint32_t abi_version;              /* must be ELDPC_API_VERSION */
    uint32_t num_detectors;            /* m */
    uint32_t num_mechanisms;           /* n */
    uint32_t num_observables;          /* k (may be 0) */
    const uint32_t *h_colptr;          /* n+1 entries */
    const uint32_t *h_rowind;          /* h_colptr[n] entries */
    const uint32_t *o_colptr;          /* n+1 entries */
    const uint32_t *o_rowind;          /* o_colptr[n] entries */
    const double *priors;              /* n entries, 0 < p < 1 */
    const char *dem_text;              /* NUL-terminated stim DEM text, or NULL */
} eldpc_problem;

typedef struct eldpc_decoder eldpc_decoder;

/* Constant of this front library: ELDPC_API_VERSION. */
ELDPC_EXPORT uint32_t eldpc_api_version(void);

/* Package version, e.g. "0.1.0". */
ELDPC_EXPORT const char *eldpc_version(void);

/* How the loaded core and this front were built: compiler, CUDA toolkit, GPU architectures,
 * host ISA flags and the core identity. The core named here is the process's core: the default GPU
 * core, or the CPU core in a CPU-only build (a build made with ELDPC_WITHOUT_CUDA, which has no
 * GPU cores). */
ELDPC_EXPORT const char *eldpc_build_info(void);

/* The core this process loaded: "avx512" or "avx2" ("" when no core could be loaded). */
ELDPC_EXPORT const char *eldpc_isa(void);

/* The core variant of a handle: "default", "ada64" or "cpu" ("" when h is NULL). A GPU handle's
 * variant is chosen at eldpc_create() time: ELDPC_DEVICE_CLASS=default|ada64 when that variable is
 * set in the calling process (it is read at every create, so handles of both variants can coexist
 * in one process), otherwise from the target device through the device-class table
 * (eldpc_core_variant_for_device()). A CPU handle (config key device=cpu) is always "cpu". */
ELDPC_EXPORT const char *eldpc_core_variant(const eldpc_decoder *h);

/* The device-class table: the core variant a handle created for a device of the given compute
 * capability (major.minor) and SM count would get without an ELDPC_DEVICE_CLASS override.
 * "default" or "ada64"; never NULL. Invalid properties (negative, or zero SMs) match no row and
 * give "default". The initial table selects "ada64" for compute capability 8.9 with at least 128
 * SMs (the 142-SM Ada class) and "default" for every other device. */
ELDPC_EXPORT const char *eldpc_core_variant_for_device(int cc_major, int cc_minor, int sm_count);

/* Build a decoder for `problem` with `config` (ABI config string, "key=value;key=value").
 *
 * The config string may hold the front's own key device=gpu|cpu|auto (default gpu). This front
 * reads and removes it before the core parses the string, so a core never sees it. device=cpu creates a
 * CPU handle: the CPU core of the host ISA is loaded (RTLD_LOCAL) when the first CPU handle is
 * created, the config keys threads= (1..64, default 1: the threads that compute one shot, the
 * calling thread included) and cpus= (exactly `threads` CPU ids or ranges; the started threads are
 * pinned to its first `threads - 1` entries; default: the creating thread's affinity mask in
 * ascending order) go to it, and no CUDA call is made (no device query, no driver or runtime
 * call), so it works with no usable NVIDIA driver and with CUDA_VISIBLE_DEVICES set empty.
 * ELDPC_DEVICE_CLASS applies to GPU handles only. The GPU core rejects threads= and cpus=.
 * device=auto is device=gpu when device 0 is usable by this build's GPU cores (an NVIDIA driver
 * and a visible device, a driver that supports the cores' CUDA major version, and a compute
 * capability the cores carry SASS or PTX for) and device=cpu otherwise, including in a CPU-only
 * build; threads= and cpus= apply only when it chooses the CPU core (they are dropped for a GPU
 * handle). It does not say why it chose the CPU: eldpc_core_variant() returns "cpu" for such a
 * handle, and a device=gpu create on the same machine fails with the reason. Any other device
 * value fails the create with a message.
 *
 * NULL on failure; if err != NULL and err_len > 0 a NUL-terminated message is written to err. */
ELDPC_EXPORT eldpc_decoder *eldpc_create(const char *config, const eldpc_problem *problem, char *err,
                                         size_t err_len);

/* Replace the priors of subsequent decodes on this handle with the given LLRs, ln((1-p)/p). */
ELDPC_EXPORT int eldpc_set_priors(eldpc_decoder *h, const double *llr, uint32_t n);

/* Decode one shot. syndrome: m bytes (0/1). out_correction: n bytes or NULL. out_obs: k bytes
 * (non-NULL when k > 0). out_status: one word or NULL. */
ELDPC_EXPORT int eldpc_decode(eldpc_decoder *h, const uint8_t *syndrome, uint8_t *out_correction,
                              uint8_t *out_obs, uint32_t *out_status);

/* Decode num_shots shots, row-major: syndromes num_shots*m bytes, out_corrections num_shots*n
 * bytes or NULL, out_obs num_shots*k bytes, out_status num_shots words. The result of every shot
 * equals what eldpc_decode would return for it. */
ELDPC_EXPORT int eldpc_decode_batch(eldpc_decoder *h, uint32_t num_shots, const uint8_t *syndromes,
                                    uint8_t *out_corrections, uint8_t *out_obs, uint32_t *out_status);

/* Copy the soft output (posterior LLRs, n doubles) of the most recent single-shot decode. */
ELDPC_EXPORT int eldpc_get_soft_output(eldpc_decoder *h, double *out_llr, uint32_t n);

/* Enqueue one empty kernel on the handle's GPU stream to keep its clocks up while idle
 * (ELDPC_KEEP_WARM_GPU). Never changes a decode result. On a CPU handle this returns ELDPC_OK and
 * does nothing (a CPU handle has no GPU stream and no idle device work). */
ELDPC_EXPORT int eldpc_keep_warm(eldpc_decoder *h, uint32_t flags);

/* Free the handle. NULL is a no-op. */
ELDPC_EXPORT void eldpc_destroy(eldpc_decoder *h);

#ifdef __cplusplus
}
#endif

#endif /* ELDPC_H */
