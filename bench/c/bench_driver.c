/* SPDX-License-Identifier: Apache-2.0 */
/*
 * bench_driver.c -- timed single-shot C-API driver of the eldpc benchmark.
 *
 * One process measures two decoder handles (two configs, two builds, or the same decoder twice for the
 * A/A control) with the eldpc C API.  It reads a problem and a set of shots from files written by
 * bench.py, warms both handles up, and then calls eldpc_decode once per shot in alternating order
 * (A, B, A, B ... or B, A, B, A ...), timing every call with CLOCK_MONOTONIC around exactly the call.
 * Before each call it can wait for a fixed idle gap (a busy-wait, no system call) and, during that gap,
 * keep the handle's GPU warm through eldpc_keep_warm().  Per shot it records the call's start and end
 * time, the return code and the status word; a checksum over every correction, observable vector and
 * status word makes two runs comparable.  Outputs go to a compact binary file (layout in bench.py's
 * benchlib/formats.py and in docs/BENCH.md).
 *
 *   cc -O2 -std=c11 $(PKG_CONFIG_PATH=<site-packages>/eldpc/lib/pkgconfig pkg-config --cflags eldpc) \
 *      -o bench_driver bench_driver.c -ldl
 *   (bench.py runs this command itself, with -I taken from pkg-config or <site-packages>/eldpc/include.)
 *   bench_driver --problem P --shots S --out O --lib-a LA --lib-b LB --config-a CA --config-b CB \
 *                --order ab [--create-order ba] --mode b2b|cadence --gap-us G --keep-warm 0|1 \
 *                --shots-per-variant N --warmup M --cpu C [--warmup-ms T] [--handles both|sequential]
 *
 * --handles sequential (bench.py passes it when a handle runs on the CPU core) times one handle at a time:
 * the first handle of the call order is created, warmed up, timed for all its calls and destroyed before
 * the second is created, so the two handles never hold the CPU at the same time.  The CPU core's worker
 * threads busy-wait between decodes for the life of their handle, so two live CPU handles with N threads
 * each would put 2N spinning threads on the cores the run is pinned to.  With `sequential` the create
 * order is the call order, and handle v decodes shots 0, 1, 2, ... in order (the same shots as in the
 * interleaved design, where call pair i decodes shot i).  The default, `both`, is the interleaved design.
 *
 * Variant 0 (config A, library A) and variant 1 (config B, library B) are separate handles; comparing two
 * builds passes two library paths, comparing two configs of one build passes the same path twice.  The
 * libraries (<site-packages>/eldpc/_libs/libeldpc.so) are opened with dlopen so that two builds, which
 * export the same symbol names, can live in one process; the declarations come from eldpc.h.
 *
 * Exit: 0 all calls returned ELDPC_OK; 2 argument, file, library or create error; 3 a decode error.
 * The problem/shots/output formats are little-endian.  The driver never prints to stdout; it writes
 * to stderr one line per handle naming the handle's core variant ("bench_driver: core_variant V NAME",
 * from eldpc_core_variant(), "unknown" for a library without that call) and its one-line summary.
 */
#define _GNU_SOURCE
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <dlfcn.h>

#include "eldpc.h"  /* the installed public header: <site-packages>/eldpc/include/eldpc.h */

#define PROBLEM_MAGIC "ELDPCPB1"
#define SHOTS_MAGIC "ELDPCSH1"
#define OUTPUT_MAGIC "ELDPCOUT"
#define OUTPUT_VERSION 1u
#define RECORD_SIZE 32u
/* Keep-warm cadence of this benchmark: a call at most 0.5 ms apart during the idle gap, and a last call
 * shortly before the decode. */
#define KEEP_PERIOD_NS 500000ull
#define KEEP_FINAL_NS 25000ull

typedef eldpc_decoder *(*create_fn)(const char *, const eldpc_problem *, char *, size_t);
typedef int (*decode_fn)(eldpc_decoder *, const uint8_t *, uint8_t *, uint8_t *, uint32_t *);
typedef int (*decode_batch_fn)(eldpc_decoder *, uint32_t, const uint8_t *, uint8_t *, uint8_t *,
                               uint32_t *);
typedef int (*keep_warm_fn)(eldpc_decoder *, uint32_t);
typedef void (*destroy_fn)(eldpc_decoder *);

typedef struct {
    uint32_t m, n, k, nnzh, nnzo;
    uint32_t *h_colptr, *h_rowind, *o_colptr, *o_rowind;
    double *priors;
    char *dem_text;
} Problem;

typedef struct {
    uint32_t shots, m, k;
    uint8_t *dets;
    uint8_t *obs;
} Shots;

typedef struct {
    uint64_t n, err, sum_ns, max_ns, final_sum_ns, final_max_ns;
} KeepStats;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void die(const char *what, const char *detail) {
    fprintf(stderr, "bench_driver: %s%s%s\n", what, detail && *detail ? ": " : "", detail ? detail : "");
    exit(2);
}

static void busy_wait_ns(uint64_t ns) {
    const uint64_t t0 = now_ns();
    while (now_ns() - t0 < ns) {
    }
}

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) die("out of memory", NULL);
    return p;
}

static void xread(FILE *f, void *p, size_t bytes, const char *what) {
    if (bytes && fread(p, 1, bytes, f) != bytes) die("short read", what);
}

static uint32_t read_u32(FILE *f, const char *what) {
    uint32_t v;
    xread(f, &v, 4, what);
    return v;
}

static char *read_blob(FILE *f, uint32_t len, const char *what) {
    char *p = xmalloc((size_t)len + 1);
    xread(f, p, len, what);
    p[len] = '\0';
    return p;
}

static void read_magic(FILE *f, const char *want, const char *what) {
    char got[8];
    xread(f, got, 8, what);
    if (memcmp(got, want, 8) != 0) die("bad magic in", what);
}

static void read_problem(const char *path, Problem *p) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open problem file", path);
    read_magic(f, PROBLEM_MAGIC, path);
    uint32_t version = read_u32(f, path);
    if (version != 1u) die("unsupported problem file version", path);
    p->m = read_u32(f, path);
    p->n = read_u32(f, path);
    p->k = read_u32(f, path);
    p->nnzh = read_u32(f, path);
    p->nnzo = read_u32(f, path);
    p->h_colptr = xmalloc(4ull * (p->n + 1));
    p->h_rowind = xmalloc(4ull * p->nnzh);
    p->o_colptr = xmalloc(4ull * (p->n + 1));
    p->o_rowind = xmalloc(4ull * p->nnzo);
    p->priors = xmalloc(8ull * p->n);
    xread(f, p->h_colptr, 4ull * (p->n + 1), path);
    xread(f, p->h_rowind, 4ull * p->nnzh, path);
    xread(f, p->o_colptr, 4ull * (p->n + 1), path);
    xread(f, p->o_rowind, 4ull * p->nnzo, path);
    xread(f, p->priors, 8ull * p->n, path);
    uint32_t dem_len = read_u32(f, path);
    p->dem_text = dem_len ? read_blob(f, dem_len, path) : NULL;
    fclose(f);
    if (p->h_colptr[0] != 0 || p->h_colptr[p->n] != p->nnzh) die("bad H column pointers", path);
    if (p->o_colptr[0] != 0 || p->o_colptr[p->n] != p->nnzo) die("bad O column pointers", path);
}

static void read_shots(const char *path, Shots *s) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open shots file", path);
    read_magic(f, SHOTS_MAGIC, path);
    uint32_t version = read_u32(f, path);
    if (version != 1u) die("unsupported shots file version", path);
    s->shots = read_u32(f, path);
    s->m = read_u32(f, path);
    s->k = read_u32(f, path);
    if (s->shots == 0) die("no shots in", path);
    s->dets = xmalloc((size_t)s->shots * s->m);
    s->obs = xmalloc((size_t)s->shots * (s->k ? s->k : 1));
    xread(f, s->dets, (size_t)s->shots * s->m, path);
    if (s->k) xread(f, s->obs, (size_t)s->shots * s->k, path);
    fclose(f);
}

static void write_u32(FILE *f, uint32_t v) {
    if (fwrite(&v, 4, 1, f) != 1) die("write failed", NULL);
}

static void write_u64(FILE *f, uint64_t v) {
    if (fwrite(&v, 8, 1, f) != 1) die("write failed", NULL);
}

static void write_blob(FILE *f, const char *s) {
    uint32_t len = (uint32_t)strlen(s);
    write_u32(f, len);
    if (len && fwrite(s, 1, len, f) != len) die("write failed", NULL);
}

static uint64_t fnv1a(uint64_t h, const void *data, size_t bytes) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < bytes; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

/* A word-wise XOR fold, for the large correction arrays of a batch (a byte-wise FNV over 300 MB would
 * cost more than the decode it checks). */
static uint64_t fold_words(const uint8_t *p, size_t bytes) {
    uint64_t acc = 0;
    size_t i = 0;
    for (; i + 8 <= bytes; i += 8) {
        uint64_t w;
        memcpy(&w, p + i, 8);
        acc ^= w;
    }
    for (; i < bytes; ++i) acc ^= (uint64_t)p[i] << (8 * (i & 7));
    return acc;
}

/* Keep the handle's GPU warm for the remaining part of a gap: every KEEP_PERIOD_NS from the start of the
 * gap and once KEEP_FINAL_NS before its end.  The decode call itself is made by the caller afterwards, so
 * no keep-warm call is inside the timed region. */
static void keep_warm_gap(uint64_t gap_ns, keep_warm_fn kw, eldpc_decoder *h, KeepStats *st) {
    const uint64_t gs = now_ns(), ge = gs + gap_ns;
    const uint64_t fin = ge > KEEP_FINAL_NS ? ge - KEEP_FINAL_NS : gs;
    uint64_t next = gs;
    int final_done = 0;
    for (uint64_t t = now_ns(); t < ge; t = now_ns()) {
        if (!final_done && t >= fin) {
            const uint64_t a = now_ns();
            const int rc = kw(h, ELDPC_KEEP_WARM_GPU);
            const uint64_t b = now_ns();
            ++st->n;
            st->err += rc != ELDPC_OK;
            st->final_sum_ns += b - a;
            if (b - a > st->final_max_ns) st->final_max_ns = b - a;
            final_done = 1;
            continue;
        }
        if (t >= next) {
            const uint64_t a = now_ns();
            const int rc = kw(h, ELDPC_KEEP_WARM_GPU);
            const uint64_t b = now_ns();
            ++st->n;
            st->err += rc != ELDPC_OK;
            st->sum_ns += b - a;
            if (b - a > st->max_ns) st->max_ns = b - a;
            next = t + KEEP_PERIOD_NS;
        }
    }
    if (!final_done) { /* a gap shorter than KEEP_FINAL_NS: still keep warm once, right before the call */
        const uint64_t a = now_ns();
        const int rc = kw(h, ELDPC_KEEP_WARM_GPU);
        const uint64_t b = now_ns();
        ++st->n;
        st->err += rc != ELDPC_OK;
        st->final_sum_ns += b - a;
        if (b - a > st->final_max_ns) st->final_max_ns = b - a;
    }
}

/* The output header, up to and including the session start time. */
static void write_header(FILE *o, int is_batch, long nshots, long batch_reps, const Problem *p,
                             const char *mode, long gap_us, const int keep_warm[2], long warmup, int cpu,
                             const char *const lib_path[2], const char *const cfg[2]) {
    if (fwrite(OUTPUT_MAGIC, 1, 8, o) != 8) die("write failed", NULL);
    write_u32(o, OUTPUT_VERSION);
    /* Records per variant: shots in the single-shot modes, batches in batch mode. */
    write_u32(o, (uint32_t)(is_batch ? batch_reps : nshots));
    write_u32(o, p->m);
    write_u32(o, p->n);
    write_u32(o, p->k);
    write_u32(o, 2);
    write_u32(o, (uint32_t)(is_batch ? 2 : (strcmp(mode, "cadence") == 0)));
    write_u32(o, (uint32_t)gap_us);
    write_u32(o, (uint32_t)(keep_warm[0] ? 1 : 0));
    write_u32(o, (uint32_t)(keep_warm[1] ? 1 : 0));
    write_u32(o, (uint32_t)warmup);
    write_u32(o, (uint32_t)(cpu < 0 ? 0xffffffffu : (uint32_t)cpu));
    write_blob(o, lib_path[0]);
    write_blob(o, lib_path[1]);
    write_blob(o, cfg[0]);
    write_blob(o, cfg[1]);
    write_u64(o, now_ns()); /* the session start time */
}

/* The output trailer: keep-warm statistics, checksums and the end marker. */
static void write_trailer(FILE *o, const KeepStats *ks, uint64_t cksum_all, const uint64_t cksum_var[2]) {
    write_u64(o, ks->n);
    write_u64(o, ks->err);
    write_u64(o, ks->sum_ns);
    write_u64(o, ks->max_ns);
    write_u64(o, ks->final_sum_ns);
    write_u64(o, ks->final_max_ns);
    write_u64(o, cksum_all);
    write_u64(o, cksum_var[0]);
    write_u64(o, cksum_var[1]);
    write_u64(o, 0x314c494152544e42ull); /* "BNTRAIL1" little-endian trailer marker */
}

static void usage(void) {
    fprintf(stderr,
            "usage: bench_driver --problem P --shots S --out O --lib-a LA --lib-b LB\n"
            "                    --config-a CA --config-b CB --order ab|ba [--create-order ab|ba]\n"
            "                    --mode b2b|cadence|batch --gap-us G --keep-warm 0|1\n"
            "                    --shots-per-variant N --warmup M --warmup-ms T --cpu C\n"
            "                    [--batch-shots B --batch-reps R]   (batch mode)\n"
            "                    [--handles both|sequential]        (sequential: one live handle at a time)\n");
    exit(2);
}

int main(int argc, char **argv) {
    const char *problem_path = NULL, *shots_path = NULL, *out_path = NULL;
    const char *lib_a_path = NULL, *lib_b_path = NULL;
    const char *config_a = NULL, *config_b = NULL;
    const char *order = "ab", *create_order = NULL, *mode = "b2b", *handles = "both";
    long gap_us = 0, nshots = 1000, warmup = 1000, warmup_ms = 1500;
    long batch_shots = 1024, batch_reps = 16;
    int keep_warm[2] = {0, 0}, cpu = -1;

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
#define NEED_VALUE(name)                                                                     \
    if (!v) {                                                                                \
        fprintf(stderr, "bench_driver: %s needs a value\n", name);                            \
        usage();                                                                              \
    }
        if (!strcmp(a, "--problem")) { NEED_VALUE(a); problem_path = v; ++i; }
        else if (!strcmp(a, "--shots")) { NEED_VALUE(a); shots_path = v; ++i; }
        else if (!strcmp(a, "--out")) { NEED_VALUE(a); out_path = v; ++i; }
        else if (!strcmp(a, "--lib-a")) { NEED_VALUE(a); lib_a_path = v; ++i; }
        else if (!strcmp(a, "--lib-b")) { NEED_VALUE(a); lib_b_path = v; ++i; }
        else if (!strcmp(a, "--config-a")) { NEED_VALUE(a); config_a = v; ++i; }
        else if (!strcmp(a, "--config-b")) { NEED_VALUE(a); config_b = v; ++i; }
        else if (!strcmp(a, "--order")) { NEED_VALUE(a); order = v; ++i; }
        else if (!strcmp(a, "--create-order")) { NEED_VALUE(a); create_order = v; ++i; }
        else if (!strcmp(a, "--mode")) { NEED_VALUE(a); mode = v; ++i; }
        else if (!strcmp(a, "--handles")) { NEED_VALUE(a); handles = v; ++i; }
        else if (!strcmp(a, "--gap-us")) { NEED_VALUE(a); gap_us = atol(v); ++i; }
        else if (!strcmp(a, "--shots-per-variant")) { NEED_VALUE(a); nshots = atol(v); ++i; }
        else if (!strcmp(a, "--batch-shots")) { NEED_VALUE(a); batch_shots = atol(v); ++i; }
        else if (!strcmp(a, "--batch-reps")) { NEED_VALUE(a); batch_reps = atol(v); ++i; }
        else if (!strcmp(a, "--warmup")) { NEED_VALUE(a); warmup = atol(v); ++i; }
        else if (!strcmp(a, "--warmup-ms")) { NEED_VALUE(a); warmup_ms = atol(v); ++i; }
        else if (!strcmp(a, "--cpu")) { NEED_VALUE(a); cpu = atoi(v); ++i; }
        else if (!strcmp(a, "--keep-warm")) { NEED_VALUE(a); keep_warm[0] = keep_warm[1] = atoi(v); ++i; }
        else if (!strcmp(a, "--keep-warm-a")) { NEED_VALUE(a); keep_warm[0] = atoi(v); ++i; }
        else if (!strcmp(a, "--keep-warm-b")) { NEED_VALUE(a); keep_warm[1] = atoi(v); ++i; }
        else usage();
#undef NEED_VALUE
    }
    if (!problem_path || !shots_path || !out_path || !lib_a_path || !lib_b_path || !config_a || !config_b) usage();
    if (strcmp(order, "ab") != 0 && strcmp(order, "ba") != 0) die("--order must be ab or ba", order);
    if (create_order == NULL) create_order = order;  /* default: create and call in the same order */
    if (strcmp(create_order, "ab") != 0 && strcmp(create_order, "ba") != 0)
        die("--create-order must be ab or ba", create_order);
    if (strcmp(mode, "b2b") != 0 && strcmp(mode, "cadence") != 0 && strcmp(mode, "batch") != 0)
        die("--mode must be b2b, cadence or batch", mode);
    if (strcmp(handles, "both") != 0 && strcmp(handles, "sequential") != 0)
        die("--handles must be both or sequential", handles);
    const int sequential = strcmp(handles, "sequential") == 0;
    if (nshots <= 0 || nshots > 100000000L) die("--shots-n out of range", NULL);
    if (gap_us < 0) die("--gap-us must be >= 0", NULL);
    if ((keep_warm[0] || keep_warm[1]) && strcmp(mode, "cadence") != 0)
        die("keep-warm needs --mode cadence", NULL);

    Problem p;
    Shots s;
    read_problem(problem_path, &p);
    read_shots(shots_path, &s);
    if (s.m != p.m || s.k != p.k) die("shots file does not match the problem", shots_path);

    if (cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0) die("sched_setaffinity failed", NULL);
    }

    const char *lib_path[2] = {lib_a_path, lib_b_path};
    void *so[2] = {NULL, NULL};
    create_fn create[2] = {NULL, NULL};
    decode_fn decode[2] = {NULL, NULL};
    decode_batch_fn decode_batch[2] = {NULL, NULL};
    destroy_fn destroy[2] = {NULL, NULL};
    keep_warm_fn kw[2] = {NULL, NULL};
    for (int v = 0; v < 2; ++v) {
        so[v] = dlopen(lib_path[v], RTLD_NOW | RTLD_LOCAL);
        if (!so[v]) die("dlopen failed", dlerror());
        create[v] = (create_fn)dlsym(so[v], "eldpc_create");
        decode[v] = (decode_fn)dlsym(so[v], "eldpc_decode");
        decode_batch[v] = (decode_batch_fn)dlsym(so[v], "eldpc_decode_batch");
        destroy[v] = (destroy_fn)dlsym(so[v], "eldpc_destroy");
        kw[v] = (keep_warm_fn)dlsym(so[v], "eldpc_keep_warm");
        if (!create[v] || !decode[v] || !destroy[v])
            die("missing eldpc_create/eldpc_decode/eldpc_destroy in", lib_path[v]);
        if (!strcmp(mode, "batch") && !decode_batch[v])
            die("batch mode requested but the library has no eldpc_decode_batch", lib_path[v]);
        if (keep_warm[v] && !kw[v])
            die("keep-warm requested but the library has no eldpc_keep_warm", lib_path[v]);
        uint32_t (*api_version)(void) = (uint32_t (*)(void))dlsym(so[v], "eldpc_api_version");
        if (api_version && api_version() != ELDPC_API_VERSION) die("library API version mismatch", lib_path[v]);
    }

    eldpc_problem ep;
    memset(&ep, 0, sizeof ep);
    ep.abi_version = ELDPC_API_VERSION;
    ep.num_detectors = p.m;
    ep.num_mechanisms = p.n;
    ep.num_observables = p.k;
    ep.h_colptr = p.h_colptr;
    ep.h_rowind = p.h_rowind;
    ep.o_colptr = p.o_colptr;
    ep.o_rowind = p.o_rowind;
    ep.priors = p.priors;
    ep.dem_text = p.dem_text;

    char err[1024];
    err[0] = '\0';
    eldpc_decoder *h[2] = {NULL, NULL};
    const char *cfg[2] = {config_a, config_b};
    int order_idx[2] = {0, 1};
    if (!strcmp(order, "ba")) { order_idx[0] = 1; order_idx[1] = 0; }
    int create_idx[2] = {0, 1};
    if (!strcmp(create_order, "ba")) { create_idx[0] = 1; create_idx[1] = 0; }
    if (sequential) {
        /* One live handle at a time (see the top of this file): create, warm up, time and destroy the
         * first handle of the call order, then the second.  The records are written in that order, each
         * tagged with its variant like the interleaved ones. */
        const int seq_batch = strcmp(mode, "batch") == 0;
        if (seq_batch && (batch_shots < 1 || batch_reps < 1 || batch_shots > (long)s.shots))
            die("batch mode needs 1 <= --batch-shots <= shots and --batch-reps >= 1", NULL);
        uint8_t *scorr = xmalloc(p.n);
        uint8_t *sobs = xmalloc(p.k ? p.k : 1);
        uint8_t *bcorr = NULL, *bobs = NULL;
        uint32_t *bst = NULL;
        if (seq_batch) {
            bcorr = xmalloc((size_t)batch_shots * (p.n ? p.n : 1));
            bobs = xmalloc((size_t)batch_shots * (p.k ? p.k : 1));
            bst = xmalloc((size_t)batch_shots * sizeof(uint32_t));
        }
        uint32_t sstatus = 0;
        int src_all = 0;
        FILE *so_out = fopen(out_path, "wb");
        if (!so_out) die("cannot write", out_path);
        write_header(so_out, seq_batch, nshots, batch_reps, &p, mode, gap_us, keep_warm, warmup, cpu, lib_path,
                     cfg);
        const uint64_t sgap_ns = (uint64_t)gap_us * 1000ull;
        KeepStats sks;
        memset(&sks, 0, sizeof sks);
        uint64_t sck_all = 1469598103934665603ull;
        uint64_t sck_var[2] = {1469598103934665603ull, 1469598103934665603ull};
        for (int k = 0; k < 2; ++k) {
            const int v = order_idx[k];
            typedef const char *(*variant_fn)(const eldpc_decoder *);
            err[0] = '\0';
            h[v] = create[v](cfg[v], &ep, err, sizeof err);
            if (!h[v]) die("eldpc_create failed", err);
            const variant_fn variant_of = (variant_fn)dlsym(so[v], "eldpc_core_variant");
            const char *name = variant_of ? variant_of(h[v]) : NULL;
            fprintf(stderr, "bench_driver: core_variant %d %s\n", v, (name && *name) ? name : "unknown");
            const uint64_t w0 = now_ns();
            for (long w = 0; w < warmup; ++w) {
                if (w >= 10 && now_ns() - w0 > (uint64_t)warmup_ms * 1000000ull) break;
                const uint32_t shot = (uint32_t)(w % (long)s.shots);
                if (decode[v](h[v], s.dets + (size_t)shot * s.m, scorr, sobs, &sstatus) != ELDPC_OK) {
                    fprintf(stderr, "bench_driver: warm-up decode failed (variant %d, shot %u)\n", v, shot);
                    return 3;
                }
            }
            if (seq_batch) {
                for (long r = 0; r < batch_reps; ++r) {
                    long start = (r * batch_shots) % (long)s.shots;
                    if (start + batch_shots > (long)s.shots) start = 0;
                    const uint64_t t0 = now_ns();
                    const int rc = decode_batch[v](h[v], (uint32_t)batch_shots,
                                                   s.dets + (size_t)start * s.m, bcorr, bobs, bst);
                    const uint64_t t1 = now_ns();
                    if (rc != ELDPC_OK) src_all = 3;
                    const uint64_t rec[4] = {t0, t1, (uint64_t)(int64_t)rc,
                                             (uint64_t)batch_shots | ((uint64_t)v << 32)};
                    if (fwrite(rec, 8, 4, so_out) != 4) die("write failed", NULL);
                    const uint64_t fold = fold_words(bcorr, (size_t)batch_shots * p.n);
                    const size_t nst = (size_t)batch_shots * sizeof(uint32_t);
                    sck_var[v] = fnv1a(sck_var[v], bobs, (size_t)batch_shots * p.k);
                    sck_var[v] = fnv1a(sck_var[v], bst, nst);
                    sck_var[v] = fnv1a(sck_var[v], &fold, 8);
                    sck_all = fnv1a(sck_all, bobs, (size_t)batch_shots * p.k);
                    sck_all = fnv1a(sck_all, bst, nst);
                    sck_all = fnv1a(sck_all, &fold, 8);
                }
            } else {
                for (long j = 0; j < nshots; ++j) {
                    const uint32_t shot = (uint32_t)(j % (long)s.shots);
                    if (strcmp(mode, "cadence") == 0) {
                        if (keep_warm[v]) keep_warm_gap(sgap_ns, kw[v], h[v], &sks);
                        else busy_wait_ns(sgap_ns);
                    }
                    const uint64_t t0 = now_ns();
                    const int rc = decode[v](h[v], s.dets + (size_t)shot * s.m, scorr, sobs, &sstatus);
                    const uint64_t t1 = now_ns();
                    if (rc != ELDPC_OK) src_all = 3;
                    const uint64_t rec[4] = {t0, t1, (uint64_t)(int64_t)rc,
                                             (uint64_t)sstatus | ((uint64_t)v << 32)};
                    if (fwrite(rec, 8, 4, so_out) != 4) die("write failed", NULL);
                    sck_var[v] = fnv1a(sck_var[v], scorr, p.n);
                    sck_var[v] = fnv1a(sck_var[v], sobs, p.k);
                    sck_var[v] = fnv1a(sck_var[v], &sstatus, 4);
                    sck_all = fnv1a(sck_all, scorr, p.n);
                    sck_all = fnv1a(sck_all, sobs, p.k);
                    sck_all = fnv1a(sck_all, &sstatus, 4);
                }
            }
            destroy[v](h[v]);
            h[v] = NULL;
        }
        write_trailer(so_out, &sks, sck_all, sck_var);
        if (fclose(so_out) != 0) die("write failed", out_path);
        if (seq_batch)
            fprintf(stderr,
                    "bench_driver: %ld batches of %ld shots per variant, mode=batch, handles=sequential, "
                    "rc!=0: %d, checksum %016llx\n",
                    batch_reps, batch_shots, src_all ? 1 : 0, (unsigned long long)sck_all);
        else
            fprintf(stderr,
                    "bench_driver: %ld shots per variant, mode=%s gap=%ldus keep_warm=%d/%d, "
                    "handles=sequential, rc!=0: %d, checksum %016llx\n",
                    nshots, mode, gap_us, keep_warm[0], keep_warm[1], src_all ? 1 : 0,
                    (unsigned long long)sck_all);
        return src_all;
    }

    for (int k = 0; k < 2; ++k) {  /* --create-order selects which handle is created first */
        const int v = create_idx[k];
        err[0] = '\0';
        h[v] = create[v](cfg[v], &ep, err, sizeof err);
        if (!h[v]) die("eldpc_create failed", err);
    }
    for (int v = 0; v < 2; ++v) {  /* the core variant of each timed handle, for the results */
        typedef const char *(*variant_fn)(const eldpc_decoder *);
        const variant_fn variant_of = (variant_fn)dlsym(so[v], "eldpc_core_variant");
        const char *name = variant_of ? variant_of(h[v]) : NULL;
        fprintf(stderr, "bench_driver: core_variant %d %s\n", v, (name && *name) ? name : "unknown");
    }

    const int is_batch = strcmp(mode, "batch") == 0;
    if (is_batch && (batch_shots < 1 || batch_reps < 1 || batch_shots > (long)s.shots))
        die("batch mode needs 1 <= --batch-shots <= shots and --batch-reps >= 1", NULL);

    uint8_t *corr = xmalloc(p.n);
    uint8_t *obs = xmalloc(p.k ? p.k : 1);
    uint32_t status = 0;
    int rc_all = 0;

    /* Warm both handles in create order: `warmup` calls each, or `warmup_ms` per handle, whichever
     * comes first. */
    for (int k = 0; k < 2; ++k) {
        const int v = create_idx[k];
        const uint64_t w0 = now_ns();
        for (long w = 0; w < warmup; ++w) {
            if (w >= 10 && now_ns() - w0 > (uint64_t)warmup_ms * 1000000ull) break;
            const uint32_t shot = (uint32_t)(w % (long)s.shots);
            if (decode[v](h[v], s.dets + (size_t)shot * s.m, corr, obs, &status) != ELDPC_OK) {
                fprintf(stderr, "bench_driver: warm-up decode failed (variant %d, shot %u)\n", v, shot);
                return 3;
            }
        }
    }

    FILE *o = fopen(out_path, "wb");
    if (!o) die("cannot write", out_path);
    write_header(o, is_batch, nshots, batch_reps, &p, mode, gap_us, keep_warm, warmup, cpu, lib_path,
                       cfg);

    const uint64_t gap_ns = (uint64_t)gap_us * 1000ull;
    KeepStats ks;
    memset(&ks, 0, sizeof ks);
    uint64_t cksum_all = 1469598103934665603ull;
    uint64_t cksum_var[2] = {1469598103934665603ull, 1469598103934665603ull};

    if (is_batch) {
        uint8_t *bcorr = xmalloc((size_t)batch_shots * (p.n ? p.n : 1));
        uint8_t *bobs = xmalloc((size_t)batch_shots * (p.k ? p.k : 1));
        uint32_t *bst = xmalloc((size_t)batch_shots * sizeof(uint32_t));
        for (long r = 0; r < batch_reps; ++r) {
            for (int k = 0; k < 2; ++k) {
                const int v = order_idx[k];
                long start = (r * batch_shots) % (long)s.shots;
                if (start + batch_shots > (long)s.shots) start = 0;
                const uint64_t t0 = now_ns();
                const int rc = decode_batch[v](h[v], (uint32_t)batch_shots,
                                               s.dets + (size_t)start * s.m, bcorr, bobs, bst);
                const uint64_t t1 = now_ns();
                if (rc != ELDPC_OK) rc_all = 3;
                const uint64_t rec[4] = {t0, t1, (uint64_t)(int64_t)rc,
                                         (uint64_t)batch_shots | ((uint64_t)v << 32)};
                if (fwrite(rec, 8, 4, o) != 4) die("write failed", NULL);
                const uint64_t fold = fold_words(bcorr, (size_t)batch_shots * p.n);
                const size_t nst = (size_t)batch_shots * sizeof(uint32_t);
                cksum_var[v] = fnv1a(cksum_var[v], bobs, (size_t)batch_shots * p.k);
                cksum_var[v] = fnv1a(cksum_var[v], bst, nst);
                cksum_var[v] = fnv1a(cksum_var[v], &fold, 8);
                cksum_all = fnv1a(cksum_all, bobs, (size_t)batch_shots * p.k);
                cksum_all = fnv1a(cksum_all, bst, nst);
                cksum_all = fnv1a(cksum_all, &fold, 8);
            }
        }
    } else {
        for (long i = 0; i < 2 * nshots; ++i) {
            const int v = order_idx[i & 1];
            const uint32_t shot = (uint32_t)((i >> 1) % (long)s.shots);
            if (strcmp(mode, "cadence") == 0) {
                if (keep_warm[v]) keep_warm_gap(gap_ns, kw[v], h[v], &ks);
                else busy_wait_ns(gap_ns);
            }
            const uint64_t t0 = now_ns();
            const int rc = decode[v](h[v], s.dets + (size_t)shot * s.m, corr, obs, &status);
            const uint64_t t1 = now_ns();
            if (rc != ELDPC_OK) rc_all = 3;
            const uint64_t rec[4] = {t0, t1, (uint64_t)(int64_t)rc,
                                     (uint64_t)status | ((uint64_t)v << 32)};
            if (fwrite(rec, 8, 4, o) != 4) die("write failed", NULL);
            cksum_var[v] = fnv1a(cksum_var[v], corr, p.n);
            cksum_var[v] = fnv1a(cksum_var[v], obs, p.k);
            cksum_var[v] = fnv1a(cksum_var[v], &status, 4);
            cksum_all = fnv1a(cksum_all, corr, p.n);
            cksum_all = fnv1a(cksum_all, obs, p.k);
            cksum_all = fnv1a(cksum_all, &status, 4);
        }
    }

    write_trailer(o, &ks, cksum_all, cksum_var);
    if (fclose(o) != 0) die("write failed", out_path);

    if (is_batch)
        fprintf(stderr,
                "bench_driver: %ld batches of %ld shots per variant, mode=batch, rc!=0: %d, "
                "checksum %016llx\n",
                batch_reps, batch_shots, rc_all ? 1 : 0, (unsigned long long)cksum_all);
    else
        fprintf(stderr,
                "bench_driver: %ld shots per variant, mode=%s gap=%ldus keep_warm=%d/%d, rc!=0: %d, "
                "checksum %016llx\n",
                nshots, mode, gap_us, keep_warm[0], keep_warm[1], rc_all ? 1 : 0,
                (unsigned long long)cksum_all);
    for (int v = 0; v < 2; ++v) destroy[v](h[v]);
    return rc_all;
}
