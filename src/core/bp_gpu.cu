// fastbpgpu6g: the GPU half of the BP engine -- device code and the host side of the handle's GPU
// resources (see bp_engine.hpp for the interface).
//
// What the device side contains: the fused single-barrier schedule, with the partials
// double-buffered by iteration parity and a __syncthreads() at the end of the fused merge;
// block 0 carrying the degree-0 rows; the interleaved column layout;
// the shot's syndrome in the launch parameters, read through syn_bit();
// the one [dec | status | post] hand-off region;
// and the device prefix of the OSD-0 order,
// run after the last iteration of either body
// on the shots that did not converge.
//
// Device side: one kernel runs a whole shot (all max_iter iterations) on the device; between the two
// half-iterations the grid synchronises, so no kernel is launched per iteration and the host is
// involved once per shot.  The launch is cudaLaunchCooperativeKernel, which is what guarantees the
// co-residency the grid barrier needs; the grid is bounded by the occupancy check at create.  (The
// (A cluster variant -- one 16-block non-portable cluster per shot with the messages in
// distributed shared memory -- was built and measured separately and is 6.6x slower per
// iteration than this kernel on bb144.)
//
// Message flow, two measured formulations, chosen once at create:
//
//  * In-place mode (max row degree <= 64, the problems whose rows fit one warp's register path):
//    the check pass reads the row's bit-to-check values, reduces its two smallest magnitudes and its
//    sign parity, and writes each edge's check-to-bit message back into the *same* slot (b2c).  The
//    column pass then reads the messages sequentially, walks the column in the reference's addition
//    order, and overwrites the slot with the new bit-to-check value.  No c2b array: peak message
//    memory is 8 B per edge, and neither pass reads an edge's message twice.
//  * Row-summary mode (long rows whose in-place grid would not fit the device, sum_mode): the check
//    pass writes the row's (m1, m2, total_sgn) into three arrays and scatters nothing; the column
//    pass recomputes each edge's message from its row's summary and its own bit-to-check value.
//    Same arithmetic, same values, same column order.  Measured faster where the in-place check
//    pass cannot place its split (bb144 xyz 646 against 952 us, an 819,200-edge synthetic 2030
//    against 2445) and slower where it can (a 14,652-edge long-row problem 535 against 350), hence
//    the create-time rule in setup().
//
// Long rows in in-place mode are split over d.wpr warps of the same block (the partial minima,
// second minima, sign counts and decision parities are order-free, so the split is exact); that
// multiplies the number of warps a row pass has in flight, which is what its latency is bounded by.
//
// Exactness: the check update reduces the row's minimum, second minimum and sign parity -- all three
// independent of traversal order -- and multiplies once by the signed alpha; the variable update is
// one sequential walk per column (one thread per column) in the reference's addition order.  The
// minimum, second minimum, `|v| == m1`, `v <= 0` and every merge live in the integer domain (see
// mag_bits / nonpositive below); the message multiply is
// __dmul_<mode>(magnitude, +-alpha) and every addition __dadd_<mode>, with <mode> from fegetround()
// read at dbench_create, so the host's rounding mode is reproduced exactly.
#include <algorithm>
#include <cfenv>
#include <mutex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#if defined(DBENCH_STAGE_PROBES) && DBENCH_STAGE_PROBES
#include <climits>   // (measurement build only): LLONG_MIN / LLONG_MAX
#endif
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "bp_engine.hpp"
#include "osd_caps.hpp"  // fastbp::kOsdPrefixCap: the device prefix cap
#include "dbench_stages.h"  // the measurement build's stage record (nothing in normal builds)
// The host-side stage stamps of the mapped path (Engine::run), measurement build, host pass only.
#if defined(DBENCH_STAGE_PROBES) && DBENCH_STAGE_PROBES && !defined(__CUDA_ARCH__)
#define M6C_HOST_STAGES 1
#else
#define M6C_HOST_STAGES 0
#endif

// CMakeLists.txt defines these on every source of the CUDA core.
// A hand-compiled test build includes this file, so it gets the same defaults
// (0 = the device's own SM count and L2 size) when the definitions are absent.
#ifndef DBENCH_GPU_SM_BUDGET
#define DBENCH_GPU_SM_BUDGET 0
#endif
#ifndef DBENCH_GPU_L2_BUDGET
#define DBENCH_GPU_L2_BUDGET 0
#endif
#ifndef DBENCH_INEXACT_MEASURE
#define DBENCH_INEXACT_MEASURE 0
#endif


// The degree-aware warp-group layout.  Default 1: the interleaved
// layout assigns each block's 32-column groups to the block's warp groups by degree, so the highest-
// degree groups are not left to the warps that already carry two groups (see balance_block_groups).
// 0 is the base's degree-sorted order, kept for the probe build that measures the change against it.
// It is a create-time permutation of whole groups: every column keeps its own addition order, the
// block's column set, its row footprint and its edge slots are unchanged, so no per-shot or
// per-iteration work is added; only which warp walks which column moves.
#ifndef DBENCH_LAYOUT_BALANCE
#define DBENCH_LAYOUT_BALANCE 1
#endif



namespace cg = cooperative_groups;

namespace {



// ---------------------------------------------------------------------------
// Rounding modes: the four instantiations of the BP kernel.  The host picks the one matching
// fegetround() at create.
// ---------------------------------------------------------------------------
enum { kRN = 0, kRD = 1, kRU = 2, kRZ = 3 };

template <int RM>
__device__ __forceinline__ double dadd(double a, double b) {
    if constexpr (RM == kRN) return __dadd_rn(a, b);
    if constexpr (RM == kRD) return __dadd_rd(a, b);
    if constexpr (RM == kRU) return __dadd_ru(a, b);
    return __dadd_rz(a, b);
}
template <int RM>
__device__ __forceinline__ double dmul(double a, double b) {
    if constexpr (RM == kRN) return __dmul_rn(a, b);
    if constexpr (RM == kRD) return __dmul_rd(a, b);
    if constexpr (RM == kRU) return __dmul_ru(a, b);
    return __dmul_rz(a, b);
}

// ---------------------------------------------------------------------------
// The integer domain of the check pass.  Every comparison the check pass makes is
// made on the IEEE bit pattern with integer instructions:
//
//   * |v| is the pattern with the sign bit cleared (kMagMask).  The unsigned order of those 63 bits is
//     the order of |v| -- monotone in the exponent, then in the mantissa, for every finite value, for
//     +-inf and for subnormals alike -- and every NaN maps above +inf.  The minimum and second
//     minimum start at the bits of DBL_MAX (kDblMaxBits) with a strict <, so no NaN and no inf ever
//     enters them, exactly as `|NaN| < x` is false and `fmin(DBL_MAX, inf) == DBL_MAX` in FP64.
//   * |v| == m1 is equality of those 63 bits (the map v -> |v| bits is injective on magnitudes).
//   * v <= 0 is `nonpositive()` below: false for every NaN whatever its sign bit, true for +0 and -0.
//   * fmin/fmax of two magnitudes that are never NaN (a lane's m1/m2 are DBL_MAX or magnitudes taken
//     from the row) are the integer min/max of their magnitude bits.
//
// The one non-integer step left per message is the reference's single multiply, `dmul<RM>(base, +-alpha)`,
// whose operand is the magnitude bits reinterpreted as a double (__longlong_as_double of a magnitude is
// exactly that magnitude) and whose sign is in the alpha operand, exactly bp.hpp:262.
// ---------------------------------------------------------------------------
constexpr unsigned long long kMagMask = 0x7fffffffffffffffULL;   // |v| = the bits with the sign cleared
constexpr unsigned long long kDblMaxBits = 0x7fefffffffffffffULL;  // |DBL_MAX|
constexpr unsigned long long kInfBits = 0x7ff0000000000000ULL;     // |+-inf|

// |v| as the 64-bit pattern with the sign bit cleared, emitted as an opaque `and.b64` rather than as
// a C++ `&` of the pattern: nvcc recognizes the C++ form -- even on a value that came straight out
// of a 64-bit integer load -- and rewrites it as the double `fabs` of the same bits
// (`DADD R, -RZ, |R|`), one FP64 instruction on a part whose FP64 rate is 1/64 of FP32 (measured
// before this change: 14 such DADDs in the in-place kernel and 5 in the row-summary kernel).  The
// mask and fabs agree on every bit pattern that can reach them, NaN included: the mask keeps the
// payload and clears the sign, and fabs is exact.  The unsigned order of the 63 bits is the order of
// |v| (see above), so every comparison on them is unchanged.
__device__ __forceinline__ unsigned long long mag_of(unsigned long long b) {
    unsigned long long r;
    asm("and.b64 %0, %1, 0x7fffffffffffffff;" : "=l"(r) : "l"(b));
    return r;
}
static_assert(kMagMask == 0x7fffffffffffffffULL, "the asm above clears the sign with this mask");
__device__ __forceinline__ unsigned long long mag_bits(double v) {
    return mag_of(__double_as_longlong(v));
}

// v <= 0.0 for every bit pattern: true for +0 and -0, for every negative finite value and for -inf;
// false for +inf, for positive values and for every NaN whatever its sign bit (|NaN| > |inf|).
__device__ __forceinline__ unsigned nonpositive(double v) {
    const unsigned long long b = __double_as_longlong(v);
    const unsigned long long m = mag_of(b);
    return (unsigned)(m <= kInfBits) & ((unsigned)(b >> 63) | (unsigned)(m == 0ull));
}

// The hard decision of the column pass, `t <= 0.0`, in the same integer domain: one DSETP per
// *column* (per iteration) is the only FP64 comparison that would be left outside the check
// pass, so it is moved to the integer domain as well.  BP_FP_DECISION=1
// builds the FP64 form instead, which is how the two were A/B'd in the probe; the shipped build (and
// the shipped library) uses the integer form.  The two are value-equal on every bit pattern t can
// take, by the same case analysis: nonpositive(NaN) = 0 and NaN <= 0.0 is false,
// nonpositive(+-0) = 1 and +-0 <= 0.0 is true.
#ifndef BP_FP_DECISION
#define BP_FP_DECISION 0
#endif
__device__ __forceinline__ unsigned dec_bit(double t) {
#if BP_FP_DECISION
    return (t <= 0.0) ? 1u : 0u;
#else
    return nonpositive(t);
#endif
}
__device__ __forceinline__ int lane_of() { return (int)(threadIdx.x & 31u); }
__device__ __forceinline__ int warp_of() { return (int)(threadIdx.x >> 5); }

constexpr int kThreads = 256;      // in-place formulation: 128 and 256 measured equal on bb144, 512
                                   // is 25% slower; 256 with the 3-blocks/SM launch bounds
constexpr int kThreadsSum = 512;   // row-summary formulation: 512 threads at one block per SM is 7%
                                   // faster on bb144 xyz (644.3 against 693.8 us) than 256 at the cap

// Columns of degree <= kRegDeg keep their messages in registers across the two halves of the column
// pass; longer columns fall back to the c2b scratch array in global memory.  The target problems
// have column degree <= 9 (bb144 xyz); higher degrees stay exact through the fallback (acceptance
// tests up to 64), they only cost the extra array traffic.
constexpr uint32_t kRegDeg = 9;

// Grid barrier.  The library ships cooperative groups' grid.sync() only: the
// hand-written single-counter barrier this library started from is gone from the shipped kernels, and
// the run-time branch that selected it is gone with it, so every iteration's barrier is the
// cooperative-groups spin loop (a load and a backward branch).  A copy of the old
// barrier is kept in a standalone probe, so the SASS of both can still be compared.

// ---------------------------------------------------------------------------
// The shot
// ---------------------------------------------------------------------------
// The syndrome of the shot travels in the kernel parameters,
// bit-packed, one bit per detector (kSynWords 64-bit words = 2,048 detectors), instead of a device
// array filled by an in-stream H2D copy.  m <= kSynWords * 64 takes the parameter path; a larger
// problem keeps the device array and the copy (nothing else changes -- syn_bit() reads whichever the
// handle set up).
constexpr uint32_t kSynWords = 32;

struct Dev {
    const uint32_t *cptr;   // n + 1, CSC column pointers of H
    const uint32_t *rptr;   // m + 1, row pointers
    const uint32_t *redge;  // nnz, edge slot per row slot (interleaved layout: the new slot)
    const uint32_t *rcol;   // nnz, column of each row slot
    const uint32_t *rind;   // edge slot -> row (interleaved layout) / nnz, row in CSC order
    unsigned long long *sm1;  // m: row-summary minimum, second minimum, as magnitude bits (see above)
    unsigned long long *sm2;  // m
    const unsigned long long *sm1i;   // m: the same two, from the priors (iteration 1)
    const unsigned long long *sm2i;   // m
    const int *spar0;       // m: #{prior <= 0} per row (spar = spar0 + syn)
    int *spar;              // m: total_sgn = syndrome bit + #{b2c <= 0}
    uint32_t sum_mode;      // 1: the row-summary formulation (long rows, see below)
    const double *prior;    // n, prior LLRs
    // prior_pos: the priors in the interleaved layout's position order,
    // prior_pos[p] = prior[cperm[p]], written at create and in set_priors (never in the decode path).
    // The bytes are a pure copy of d.prior, so every value is bit-identical; the column pass reads
    // it coalesced at p and without the dependent d.cperm[p] load in front of the gather.  Null when
    // the layout is not interleaved (those bodies read d.prior in the caller's column order).
    const double *prior_pos;
    double *b2c;            // nnz: bit-to-check values after a column pass, messages between the two
    double *post;           // n
    uint8_t *dec;           // n
    const uint8_t *syn;     // m: device copy of the syndrome (values 0/1), only when m > kSynWords * 64;
                            // every read goes through syn_bit(), which picks it or syn_bits
    uint32_t *mis;          // 3 (ping-pong mismatch totals; two in the two-barrier body)
    uint32_t *status;       // 2: iterations, converged
    double *c2b;            // nnz, scratch for columns of degree > kRegDeg
    // ---- create-time layout (interleaved, degree-grouped columns; only when every column
    // degree <= kRegDeg -- the register path -- so no fallback array is ever reached).  Positions
    // p are the new column order: the original columns of each block's chunk sorted by degree
    // (stable, so equal degrees keep the reference order), 32 of similar degree per warp.  The
    // column's edge i lives at slot ilv_slot0(p) + 32*i, so a warp's step i is one 256 B run.
    const uint32_t *cperm;  // n: original column of each position
    const uint8_t *cdeg;    // n: degree of the column at each position
    const uint8_t *rslot;   // nslots: create-time slot -> row slot (unsigned char; RSLOT kernels)
    const uint32_t *colbase; // nblocks + 1: the first column position of each block's chunk
                             // (the balanced table, or the uniform chunks)
    uint32_t ilv;           // 1: interleaved layout above
    // ---- fused single-barrier schedule (sum formulation only; see bp_body_fused).  The
    // block/row incidence of the column chunking the kernel launches with: rowlist/rowbase are the
    // rows each block's columns touch, ascending; owner* is, for each row, the (block, partial slot)
    // list of every block that touches it; pm1/pm2/pcnt/psign hold the per-(row, block) partial (the
    // two smallest magnitudes, the count of nonpositive values, the decision parity).  ns_max is the
    // largest number of rows any block touches (= the shared table size).
    const uint32_t *rowbase, *rowlist, *owner_ptr, *owner_slot;
    // The block's merge metadata, precomputed at create in
    // the exact form the merge walks it -- per block [rows (ns), slot_base, owner slots],
    // each record padded to a multiple of four u32 so the per-shot stage is a uint4 copy.  Pure layout
    // metadata (a function of H and the create-time chunking).  Null when the fused schedule is not used.
    const uint32_t *meta_off, *merge_meta;   // nblocks+1 offsets into merge_meta (u32); the records
    uint32_t rslot_bytes;                    // the rslot stage's size in bytes (0 unless the kernel is RSLOT)
    // The per-position row-slot table is staged per block, each block's
    // region rslot_bytes wide (16-B aligned), so the kernel finds its own region at blockIdx.x * stride.
    // The table holds kRegDeg row-slot bytes per position (two u32 words and one byte for the 9th
    // round) instead of one byte per ILV slot; same byte count per block, same dynamic shared size.
    uint32_t rslot_stride;
    // The b2c window's byte offset in the RSLOT fused
    // kernel's dynamic shared memory (after the rslot and merge-metadata stages) and its edge-round count:
    // kRegDeg when the create-time gate reserved it, 0 for the global d.b2c path.
    uint32_t win_off, win_rounds;
    unsigned long long *pm1, *pm2;
    unsigned *pcnt, *psign;
    uint32_t pm_stride;     // partial slots per buffer: pm1/pm2/psign hold two buffers, (it & 1) * pm_stride
    // The merge as a push: the triple-buffered L2 row
    // accumulators, indexed [buffer * am_stride + row]: the two smallest magnitudes (magnitude bits, as
    // everywhere) and the sign / count-parity word.  Every block that owns an edge in a row pushes its own
    // two-smallest pair into them at publish; every block that merges the row reads the three words after
    // the iteration's grid barrier.  The fused body only (the two-barrier and in-place bodies keep pm*).
    unsigned long long *am1, *am2;
    unsigned *asign;
    uint32_t am_stride;     // m: rows per buffer (three buffers)
    // Post / dec double-buffered by iteration parity.  The
    // fused body's column pass of iteration it writes buffer it & 1 (post / dec are buffer 0, post2 / dec2
    // buffer 1), so the state of it - 2 survives in the other buffer and the deferred exit test's restore
    // copy disappears.  status2 is buffer 1's copy of the status word, written by the kernel epilogue after
    // the body, so the host's one-transfer [dec | status] hand-off reads the same shape from either buffer.
    // Null in a kernel without the double buffer (sum_mode != 1).
    double *post2;          // n: iteration-parity buffer 1's posterior
    uint8_t *dec2;          // n: iteration-parity buffer 1's decision
    uint32_t *status2;      // 2: buffer 1's [iterations, converged]
    uint32_t ns_max;
    uint32_t m, n, max_iter, wpr;
    uint32_t use_synparam;  // 1: read the syndrome out of syn_bits, 0: out of syn
    double alpha;
    unsigned long long syn_bits[kSynWords];  // the shot's syndrome, bit-packed (row r -> bit r)
    // The handle's mapped status block (see StatusCtl, in mapped pinned host memory).  The
    // kernel publishes the shot's status and generation here at the end of its body; the host polls
    // the generation and reads the status.  Never null.
    unsigned long long *pub;
    // The front of the OSD-0 order.  Only touched when use_osd is set
    // (decoder=bposd with osd_method=osd0); a bp handle keeps them null and never enters the pass.
    unsigned long long *okey;      // n: the order key of every column (osd_key of the posterior)
    unsigned long long *osamp;     // kOsdS: the sorted stride sample of those keys (no longer used by
                                   // the pass; kept allocated so the Dev layout stays comparable)
    // The pushed coarse histogram.  kOsdHBins bins over a value window [osd_lo, osd_lo +
    // kOsdHBins * osd_step] fixed by the priors (set_priors).  ohist is this shot's histogram (the
    // blocks push their shared counts into it in P1); ohist_next is the other half of the ping-pong,
    // zeroed once at the top of every shot's kernel so the next shot starts from zeros.
    unsigned *ohist;               // kOsdHBins: this shot's global histogram
    unsigned *ohist_next;          // kOsdHBins: the other buffer (zeroed for the next shot)
    double osd_lo;                 // the window's low edge
    double osd_step;               // (hi - lo) / kOsdHBins
    double osd_iscale;             // kOsdHBins / (hi - lo)
    unsigned long long *otau;      // 1: the chosen threshold
    unsigned *ocnt;                // 1: collected candidates; the host copies this (see above)
    unsigned char *ocand;          // kOsdCap x 16 B: the collected (key, column) pairs
    unsigned char *ocand_sorted;   // kOsdCap x 16 B: the same records in the exact OSD-0 order (P6)
    // The device-side view of the handle's mapped pinned
    // hand-over buffer (h_pref), or null when this launch leaves the hand-over to the host's D2H copy.
    // When set, the prefix pass publishes [header | records] here before the kernel's status release,
    // so an OSD shot needs no device-to-host copy and no stream synchronisation.
    unsigned char *hand_host;
    uint32_t osd_r;                // the last rank bucket the pass reads (<= kOsdS)
    unsigned ohist_stride;         // osd_r + 1, the buckets zeroed and scanned
    unsigned use_osd;              // 1: this handle computes the OSD-0 order prefix on the device
    // The threshold selection of one rung (osd_prefix_pass and osd_repick_kernel share the
    // rule; the host sets these before each launch: rung one kOsdK/kOsdCapA, the repick kOsdK2/kOsdCap).
    unsigned osd_k;                // the first osd_k entries of the order must be inside tau's set
    unsigned osd_cap;              // the most records this rung may collect
    // Rung two, collected speculatively in the BP kernel's
    // tail on every non-converged OSD shot.  ocand2 / ocnt2 are the device block the tail collects
    // into, in the hand-over block's layout ([u32 count | u32 sorted | 8 B pad][kOsdCap x 16 B
    // records]); hand2 is the device view of the handle's mapped rung-two block (the same layout) and
    // pub2 the device view of StatusCtl::gen2, which the tail releases at system scope after the block
    // is visible.  All three are null for a bp handle, and hand2 / pub2 are null on a handle whose
    // block could not be mapped (the tail then collects only and the host copies the device block).
    unsigned char *ocand2;
    unsigned *ocnt2;
    unsigned char *hand2;
    unsigned long long *pub2;
    // 1 when this shot's max_iter is inside the tail's cap gate (see
    // kOsdTailMaxIter); the kernel's tail and the host's tail path both read it.
    unsigned tail_on;
};

// Slot of edge 0 of the column at position p (see GpuState::ilv).  Warp groups are global (a
// position's warp is p >> 5), so this needs no per-warp metadata; each warp group owns exactly
// 32 * kRegDeg slots.
__device__ __forceinline__ uint32_t ilv_slot0(uint32_t p) {
    return (p >> 5) * (32u * (uint32_t)kRegDeg) + (p & 31u);
}

// A block's column-position range, from the create-time table
// (GpuState::d_colbase, Dev::colbase).  The table holds the uniform chunks when the balanced
// partition's gate did not fire, so every reader takes one path; the boundaries are multiples of 32
// for the interleaved layout, so ilv_slot0's warp groups do not straddle two blocks.
__device__ __forceinline__ uint32_t col_begin(const Dev &d) { return d.colbase[blockIdx.x]; }
__device__ __forceinline__ uint32_t col_end(const Dev &d) { return d.colbase[blockIdx.x + 1u]; }

__device__ __forceinline__ unsigned syn_bit(const Dev &d, uint32_t row) {
    if (d.use_synparam == 1u) return (unsigned)((d.syn_bits[row >> 6] >> (row & 63u)) & 1ull);
    return (unsigned)d.syn[row];
}


__device__ __forceinline__ void grid_barrier(const Dev &d) {
    (void)d;
    cg::this_grid().sync();
}

// The front of the OSD-0 order, computed on the device.
//
// The order the reference (and osd0_engine.hpp on the host) walks is: ascending signed posterior,
// equal values by ascending column index, where equal is `==` on doubles (-0.0 equals +0.0, +inf
// equals +inf) and every NaN sorts after every number whatever its sign bit or payload.
// osd_key_bits below is that order as one unsigned key per column: -0.0 and +0.0 to one key, every
// NaN to one key above +inf, and the unsigned order of the keys is the value order for every pair
// that is not a NaN.  It is the same map the host engine's osd_key applies (osd0_engine.hpp), so a
// prefix chosen here is a prefix of exactly the host's order.
//
// What leaves the device is the candidate set (key, column) and its count -- never the n posteriors.
// The columns are found by an exact rank selection rather than by sorting the n keys (n is 76,032 on
// bb144 xyz; only its first ~10^3 entries are ever walked):
//
//   P1  every column's key; a kOsdS-entry stride sample of the keys; the kept buckets of the
//       global rank histogram zeroed.
//   P2  the sample sorted by a grid-wide rank (every block gathers it, every warp ranks one element by
//       (key, index)).
//   P3  every column's rank in the first kOsdR + 1 sample values (binary search in shared memory),
//       counted into the one global histogram -- and only the columns that can have such a rank: a
//       column is filtered against samp[kOsdR] first, and only the ones at or below it are searched
//       and counted (see kOsdR below).
//   P4  every block scans the global histogram itself (a few dozen integer adds) and takes the smallest r with its cumulative count >= kOsdK.  That count is exact,
//       so the threshold is exact: what it does NOT bound is the number of columns whose key equals
//       samp[r] (a tie bucket), which is why the collection is capped.
//   P5  every column with key <= tau is collected (key, index) with one atomic per warp's run of
//       collected columns (~10^3 of them, not n).
// A bitonic sort of the candidate set in one block is not used here: it costs
// 37 us on bb144 xyz (measured against a build with it cut out),
// while the host's own lazy chunk order does the same job in ~10 us on the walk the shots take.  (P6 below is
// a grid-wide rank sort of the small prefix.)  Otherwise the host receives the candidates in the order the grid collected them.
//
// What leaves the device is rung one of a two-rung chain -- the *small*
// prefix (d.osd_k = kOsdK, d.osd_cap = kOsdCapA) -- because the walk only ever consumes the first
// few records (measured on a development corpus: median 38, p99 180, max 588 over 2000 shots) while the
// rung-two set carries 1024-2048 of them, and the host's lazy chunk selection, the copy back and the
// collect are all proportional to the set.  A shot whose walk runs past rung one gets rung two (the
// larger kOsdK2 / kOsdCap) from osd_repick_kernel, which re-uses this pass's sample and histogram;
// a shot that runs past *that* falls back to the full posterior and the unchanged host engine -- a
// fallback, not a different result: every rung is a prefix of the same order, so the output is
// bit-identical on all of them.
// Test-only builds: a chosen half of the blocks (the parity of
// blockIdx.x + salt) spins FASTBP_TEST_DELAY_PROLOGUE ns on %globaltimer before each grid-stride
// initialisation or reset of the shot (the calls below).  The delayed blocks' writes then land after the
// other blocks' accesses to the same global state, so a missing barrier / fence between an initialisation
// and another block's first use of it shows up as a wrong result under the ordinary exactness checks.  The
// spin is bounded by %globaltimer; the macro is never defined in a shipped or timed build (no code there).
// With FASTBP_TEST_DELAY_RANDOM as well, every block draws its own pseudo-random
// delay in [0, FASTBP_TEST_DELAY_PROLOGUE) ns at every site -- a hash of blockIdx.x, the site and %globaltimer at ~1 ms
// granularity, so the pattern changes per site and per launch -- which widens a race between any two blocks at the sites
// after a grid barrier too (the parity form never separates two blocks of the same parity).  Test builds only.
#if defined(FASTBP_TEST_DELAY_PROLOGUE) && FASTBP_TEST_DELAY_PROLOGUE && defined(FASTBP_TEST_DELAY_RANDOM) && FASTBP_TEST_DELAY_RANDOM
__device__ __forceinline__ void test_delay_prologue(unsigned salt) {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    unsigned h = ((unsigned)blockIdx.x * 2654435761u) ^ (salt * 2246822519u) ^ (unsigned)(t0 >> 20);
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    const unsigned long long ns = (unsigned long long)(h % (unsigned)(FASTBP_TEST_DELAY_PROLOGUE));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}
#elif defined(FASTBP_TEST_DELAY_PROLOGUE) && FASTBP_TEST_DELAY_PROLOGUE
__device__ __forceinline__ void test_delay_prologue(unsigned salt) {
    if (((blockIdx.x + salt) & 1u) != 0u) return;
    const unsigned long long ns = (unsigned long long)(FASTBP_TEST_DELAY_PROLOGUE);
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}
#else
__device__ __forceinline__ void test_delay_prologue(unsigned) {}
#endif

// Directed test only (never defined in a shipped or timed build): blocks 0..3 --
// the ones whose threads cover rows 0..2047 of any grid-stride per-row loop -- start ~20 us late, as a
// cooperative grid's first blocks can when other kernels hold SMs.  Called at the entry of both shot bodies.
// DBENCH_TEST_LATE_INIT == 2 delays every block by its own pseudo-random 0..20 us (a hash of
// blockIdx.x and the launch's %globaltimer at ~1 ms granularity, so the pattern changes from launch to launch and is
// uniform inside a block): it widens races between any two blocks, where the half-the-blocks form leaves same-parity
// pairs unwidened and the blocks-0..3 form only widens those.  Bounded by %globaltimer; never in a shipped or timed build.
__device__ __forceinline__ void test_late_init() {
#if defined(DBENCH_TEST_LATE_INIT) && DBENCH_TEST_LATE_INIT
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
#if DBENCH_TEST_LATE_INIT == 2
    unsigned h = (unsigned)blockIdx.x * 2654435761u ^ (unsigned)(t0 >> 20);
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    const unsigned long long dl = (unsigned long long)(h % 20000u);
#else
    const unsigned long long dl = (blockIdx.x < 4u) ? 20000ull : 0ull;
#endif
    if (dl != 0ull) {
        do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < dl);
    }
#endif
}

// ---------------------------------------------------------------------------
constexpr uint32_t kOsdS = 512;      // keys in the stride sample (the P6 rank sort's staging is sized by it)
// The pushed coarse histogram.  P1 bins every column's posterior
// into kOsdHBins equal-width bins of a value window fixed at set_priors from the priors, each block
// pushes its shared counts into the one global histogram with atomics, and after the pass's first grid
// barrier every block derives the threshold itself from that histogram (redundantly, integer and
// deterministic, so all blocks agree bit for bit).  The window: |prior| <= P, and the posterior of a
// minimum-sum iteration is the prior plus at most max_iter check messages of magnitude
// alpha * min|b2c|, so the interesting part of the order (the first K entries, the smallest
// posteriors) sits inside a few times P; +-2.5 P was measured on the development shots of every preset below
// (the set sizes and repick counts).
// The bins are uniform in the *value*, which is what gives the boundary the resolution the prefix
// needs: a key-space window's bins are exponential in the value and were measured too coarse there
// (measured on the development shots).
constexpr uint32_t kOsdHBins = 256;    // histogram bins: one per thread at the pass's 256-thread block
constexpr double kOsdHWinMul = 2.5;    // the window is [-kOsdHWinMul * P, +kOsdHWinMul * P]
// Rung one (the small prefix, appended to the BP kernel) and rung two (the repick kernel the host
// launches when a walk runs past rung one).  Both values and their coupling with the host engine's
// candidate buffer (Osd0Order::kCand) live in osd_caps.hpp, which is also where the asserts are.
static constexpr uint32_t kOsdK = fastbp::kOsdPrefixKA;
static constexpr uint32_t kOsdCapA = fastbp::kOsdPrefixCapA;
static constexpr uint32_t kOsdK2 = fastbp::kOsdPrefixK;
// Ranks above the last kept bucket are never needed: the threshold is the FIRST bucket whose
// cumulative count reaches kOsdK, so no bucket above it is read (the exactness argument above holds
// for any bucket order statistic).  Truncating the rank there is what makes the rank pass local to
// the columns that can matter: every column is filtered against that bucket's upper key with one
// shared-memory compare, and only the ones at or below it run the binary search and the histogram
// increment.  How many buckets are kept is a create-time value (GpuState::osd_r, Dev::osd_r), not a
// constant: the threshold's bucket index is about kOsdK * kOsdS / n (one sample element per
// n / kOsdS columns), i.e. ~7 on bb144 xyz (n = 76,032) but ~50 on bb144 z (n = 10,512) and ~101 on
// bb72 (n = 5,184), so setup() keeps 8x that estimate with a floor of kOsdRMin buckets and a cap of
// the whole sample -- the truncation is switched off where it would save nothing (a small problem)
// and cuts the rank pass hardest where it pays (a large one).  A shot whose cumulative count at the
// last kept bucket is still below kOsdK collects nothing and takes the host's full-order fallback,
// which is exact as well; a shot whose threshold bucket is within the kept range gets exactly the
// threshold, count and collected set of the full kOsdS sample.
constexpr uint32_t kOsdRMin = 64;   // the fewest buckets the pass keeps (the truncation's floor)
// The kOsdK above is the larger K = 1024, which is rung two's kOsdK2: setup()
// sizes the kept buckets with kOsdK2, the K the pass was validated with and the one the repick
// needs (rung one's K = 384 crosses further inside the kept range).  The threshold rule
// (osd_pick_threshold) and the repick scan only the kept buckets 0..osd_r: the buckets above it are
// neither zeroed nor counted.
static constexpr uint32_t kOsdCap = fastbp::kOsdPrefixCap;
// The hand-over block's header -- [u32 count | u32 sorted | 8 B pad] -- in front of the
// records (GpuState::d_hand); 16 bytes so the records stay 16-byte aligned for P6's ulonglong2 stores.
static constexpr size_t kHandHdr = 16;
// The bound on the tail's rendezvous spin, in ns of %globaltimer
// (generous: the collect it waits for is ~2 us).  On expiry nothing is published, the host takes the
// stream-ordered copy, and the kernel ends -- a cancelled or broken run cannot leave a kernel spinning.
constexpr unsigned long long kTailSpinNs = 2000000000ull;
// The bound on the host's poll of the mapped rung-two generation, in pause iterations (no clock read on
// the decode path, as the status poll).  Generous next to the tail's ~4 us; on expiry the stream-ordered
// copy does the wait, so the bound can never change an answer.
constexpr unsigned long long kTailPollBudget = 1ull << 22;
// The bound on the event-query spin that observes rung two's stream-ordered D2H copy, in pause
// iterations (no clock read on the decode path).  On expiry the stream synchronisation does the wait.
constexpr unsigned long long kTailCopySpinBudget = 1ull << 22;
// The tail's cap gate (measured on the development machine): the
// tail runs only at max_iter <= this.  With the tail on, the handle's next launch waits for the kernel
// to end, so on a shot whose host work after the status is shorter than the tail the tail adds to the
// next call: bb288 z a05ms31 (1.3 us of host work after the status) +6.5 us pooled p50 (+3.3 %), xyz
// a05c11 +1.8, with the device body unchanged (the stage A/B).  At the primary cap 3 the publish ->
// next-entry gap's minimum is 5.06 us against the tail's 3.46 us maximum, the single-handle rows are
// flat (+0.11 us at the p50), and the tail's lever -- the repick round trip -- is what the
// measurement targets there.  The rung-two set size and its constants are unchanged; this gate only
// decides when the tail runs.
constexpr uint32_t kOsdTailMaxIter = 3u;
static_assert(kOsdCap >= 1u && kOsdCapA >= 1u,
              "a device prefix cap must leave room for at least one candidate");
static_assert(kOsdCapA <= kOsdCap, "rung one is the smaller prefix");
static_assert(kOsdK <= kOsdCapA && kOsdK2 <= kOsdCap, "each rung's K fits its own cap");

// ---------------------------------------------------------------------------
// The mapped status block, one per handle, in mapped pinned host memory.
//
// The plain hand-off is one cudaMemcpyAsync of [decisions (n) | status (8)] before the stream
// synchronisation, issued for every shot of a use_osd handle.  Which of the two the shot needs is
// known before that copy starts: a shot that converged has the BP decision as its correction and no
// OSD-0 prefix at all, a shot that did not converge has a prefix and never reads the decisions.  The
// kernel therefore publishes its two status words here, at the end of its body, with the shot's
// generation (a system-scope release store), and the host polls for that generation before it issues
// the copies, in the plain pipeline: only the ones this shot needs.  The poll is bounded by
// iterations -- no clock read on the decode path -- and a shot whose status the poll does not see in
// its budget takes the unconditional hand-off below, so the block can never stall a decode.
// ---------------------------------------------------------------------------
struct StatusCtl {
    volatile unsigned long long gen;         // device -> host: this shot's generation;          pub[0]
    volatile unsigned long long st;          // device -> host: iterations lo 32 | converged hi 32; pub[1]
    // The generation of the handle's mapped rung-two block.  The BP
    // kernel's tail releases it at system scope after the block's records are visible, so a host that
    // observes it (with an acquire load) may read this shot's rung-two set in place.  A shot that did
    // not run the tail leaves it at an earlier shot's generation: no other shot's value can equal this
    // shot's `gen`, and the caller falls back to the stream-ordered copy (or the repick kernel).
    volatile unsigned long long gen2;        // pub[2]
#if defined(DBENCH_STAGE_PROBES) && DBENCH_STAGE_PROBES
    // Stage-probe build only: %globaltimer at the kernel's
    // entry and just before the status publish (block 0 thread 0), so the mapped path -- which has no
    // stream synchronisation for the stage tool to split the BP interval at -- still yields the device
    // body time.  Normal builds have neither the fields nor the stores.
    volatile unsigned long long t_dev0;      // pub[3]
    volatile unsigned long long t_dev1;      // pub[4]
    // Stage-probe build only: the tail's end on the device clock (block 0, after the
    // mapped block is visible and just before gen2 is released), so the stage tool measures the
    // tail's real cost inside the BP kernel as pub[5] - t_dev1.
    volatile unsigned long long t_tail;      // pub[5]
#endif
};

// The generation publish: a system-scope release store, so a host that observes it (with an acquire
// load) sees the status word the kernel wrote before it.
__device__ __forceinline__ void st_release_sys_u64(volatile unsigned long long *p,
                                                   unsigned long long v) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"((void *)p), "l"(v) : "memory");
}

#if defined(DBENCH_STAGE_PROBES) && DBENCH_STAGE_PROBES
// (the measurement build only -- normal builds have neither the block, the kernel
// nor any call): the %globaltimer <-> host CLOCK_MONOTONIC calibration of a handle, so the mapped path's launch
// return -> first observed generation interval splits into the launch latency (launch return -> kernel entry) and
// the observation (status publish -> the host's first acquire that sees it) on one clock.
// Ping-pong over mapped memory, kCalibRounds rounds: the host reads its clock (t0), stores round r into `ping`;
// one device thread, spinning on `ping`, reads %globaltimer (g) when it sees r and releases `pong` = r; the host
// reads its clock when it sees pong (t1).  The device saw the store between t0 and t1, so the offset
// theta = globaltimer - host clock lies in [g - t1, g - t0]; the rounds' intersection is the bound kept.
// Before the rounds the thread reads %globaltimer back to back kCalibGranReads times: the smallest non-zero step
// is the timer's granularity on this GPU (the split is only as fine as that).
struct CalibBlk {
    volatile unsigned long long ping;    // host -> device: the round
    volatile unsigned long long pong;    // device -> host: the round seen
    volatile unsigned long long g;       // device -> host: %globaltimer when it was seen
    volatile unsigned long long gran;    // device -> host: smallest non-zero step of back-to-back reads (ns)
    volatile unsigned long long steps;   // device -> host: distinct values in those reads
};
constexpr unsigned kCalibRounds = 512;
constexpr unsigned kCalibGranReads = 4096;
constexpr unsigned long long kCalibSpin = 1ull << 20;   // device-side bound per round (~1 s of PCIe reads)

__global__ void calib_kernel(CalibBlk *c) {
    unsigned long long prev, t, minstep = ~0ull, distinct = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(prev));
    for (unsigned i = 0; i < kCalibGranReads; ++i) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        if (t != prev) {
            ++distinct;
            if (t - prev < minstep) minstep = t - prev;
            prev = t;
        }
    }
    c->gran = minstep;
    c->steps = distinct;
    __threadfence_system();
    for (unsigned long long r = 1; r <= kCalibRounds; ++r) {
        unsigned long long spin = 0;
        while (c->ping != r) {
            if (++spin > kCalibSpin) return;   // the host gave up: never hang the stream
        }
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        c->g = t;
        __threadfence_system();
        st_release_sys_u64(&c->pong, r);
    }
}
#endif

// The raw IEEE-754 pattern of a posterior as its order key.  Value-equal to the host engine's
// osd_key(): +-0.0 -> one key (0x8000000000000000, the smallest), every NaN (any sign, any payload)
// -> 0xffffffffffffffff, every other value -> its bit pattern with the sign handled (negatives
// complemented, non-negatives with the sign bit set), so the unsigned order of the keys is the
// ascending signed order of the values.  Written on the 32-bit halves so nvcc does not recognise a
// fabs() on a live double.
__device__ __forceinline__ unsigned long long osd_key_bits(unsigned long long u) {
    const unsigned hi = (unsigned)(u >> 32), lo = (unsigned)u;
    const unsigned mhi = hi & 0x7fffffffu;
    if ((unsigned)(mhi > 0x7ff00000u) | ((unsigned)(mhi == 0x7ff00000u) & (unsigned)(lo != 0u))) {
        return ~0ull;  // NaN (any sign, any payload) sorts after every number
    }
    if ((mhi | lo) == 0u) return 0x8000000000000000ull;  // +-0.0: one key, below every other
    if (hi >> 31) return ~u;
    return u | 0x8000000000000000ull;
}

// The bin of one posterior under the pushed histogram: (x - osd_lo) * osd_iscale, clamped to
// [0, kOsdHBins - 1].  Written with the kernel's rounding mode, like every other FP64 operation of a
// directed-mode kernel.  The clamps are ordered fmax(fmin(y, B-1), 0) on purpose: fmin maps a NaN to
// its other operand, so NaN lands in the top bin -- after every number, where the order key puts it --
// while +inf maps to the top bin and -inf to bin 0.  The bin is monotone in x (the rounding functions
// are), so every bin is an interval of the order.
template <int RM>
__device__ __forceinline__ unsigned osd_bin_of(const Dev &d, double x) {
    const double y = dmul<RM>(dadd<RM>(x, -d.osd_lo), d.osd_iscale);
    const double z = fmax(fmin(y, (double)(kOsdHBins - 1u)), 0.0);
    return (unsigned)z;
}

// The order key of the upper edge of bin b, minus one: the largest key a column of bin b can have.
// Every key <= it belongs to a bin at or below b (the bin map is monotone and bin b is
// [edge(b), edge(b+1))), so the set {key <= tau_b} is exactly the bins 0..b -- a prefix of the
// reference order, with the whole tie run at the boundary inside it.
template <int RM>
__device__ __forceinline__ unsigned long long osd_bin_tau(const Dev &d, unsigned b) {
    const double e = dadd<RM>(d.osd_lo, dmul<RM>((double)(b + 1u), d.osd_step));
    const unsigned long long k = osd_key_bits(__double_as_longlong(e));
    return k == 0ull ? ~0ull : k - 1ull;
}

// P4: the threshold of one rung from the global coarse histogram.  Every thread takes one bin (one
// coalesced read of the whole histogram), the block's inclusive prefix sum gives each bin's cumulative
// count, and the two bins the rule needs are reduced with shared atomics:
//   sh_r    = the first bin whose cumulative count reaches this rung's d.osd_k   (kOsdHBins: none)
//   sh_last = the last bin whose cumulative count fits d.osd_cap, plus one      (0: none)
// The rule itself then runs on thread 0: rung one's small K and cap (rung two's kOsdK2 / kOsdCap are
// the same code with the other Dev fields).  The cumulative counts are exact counts of columns at or
// below the bins' edges (the histogram is the whole grid's), so the chosen set is exactly a prefix of
// the reference order and its size is the cumulative count printed here.
// The rule's two bins over an explicit (k, cap), so the BP kernel's
// tail runs the same block-wide scan as the prefix pass and the repick kernel (d.osd_k / d.osd_cap are
// rung one's in the pass, rung two's here).
__device__ __forceinline__ void osd_hist_scan(const Dev &d, unsigned k, unsigned cap,
                                              unsigned *sh_r, unsigned *sh_last) {
    __shared__ unsigned wsum[16];
    const unsigned tid = (unsigned)threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned nw = (unsigned)blockDim.x >> 5u;
    const unsigned h = (tid < kOsdHBins) ? d.ohist[tid] : 0u;
    unsigned s = h;
#pragma unroll
    for (unsigned o = 1u; o < 32u; o <<= 1u) {
        const unsigned t = __shfl_up_sync(0xffffffffu, s, o);
        if (lane >= o) s += t;
    }
    if (lane == 31u) wsum[tid >> 5] = s;
    __syncthreads();
    if (tid < 32u) {
        unsigned t = (tid < nw) ? wsum[tid] : 0u;
#pragma unroll
        for (unsigned o = 1u; o < 32u; o <<= 1u) {
            const unsigned u = __shfl_up_sync(0xffffffffu, t, o);
            if (lane >= o) t += u;
        }
        if (tid < nw) wsum[tid] = t;   // inclusive prefix of the warp sums; lanes >= nw are never read
    }
    __syncthreads();
    const unsigned base = (tid >= 32u) ? wsum[(tid >> 5) - 1u] : 0u;
    const unsigned cum = base + s;   // inclusive cumulative count through bin tid
    if (tid < kOsdHBins) {
        if (cum >= k) atomicMin(sh_r, tid);
        if (cum <= cap) atomicMax(sh_last, tid + 1u);
    }
}

// Test-only builds, never defined in a shipped or timed build: the chosen bin is moved
// up by FASTBP_TEST_TAU_SHIFT bins (clamped to the top bin), so the collected set {key <= tau} is larger than the
// cumulative count the rule checked against the cap.  On the shots where it exceeds the rung's cap the collect
// drops the records past it and the host must reject the set (count > cap) and take the next rung or the
// full-order fallback; where it fits, the walk gets a longer prefix of the same order.  Both must stay exact.
#if defined(FASTBP_TEST_TAU_SHIFT) && FASTBP_TEST_TAU_SHIFT
#define OSD_TAU_BIN(b) min((unsigned)(b) + (unsigned)(FASTBP_TEST_TAU_SHIFT), kOsdHBins - 1u)
#else
#define OSD_TAU_BIN(b) (b)
#endif

// The threshold of one rung from the two scan results (thread 0, after osd_hist_scan): the first bin
// reaching K when its count fits the cap, else the largest bin that fits the cap (a shorter prefix of
// the same order), else none -- the rung collects nothing and the host moves on, exactly as before.
template <int RM>
__device__ __forceinline__ bool osd_pick_threshold(const Dev &d, unsigned r, unsigned lastp,
                                                   unsigned long long *tau_out) {
    if (r < kOsdHBins && (lastp == 0u || r <= lastp - 1u)) { *tau_out = osd_bin_tau<RM>(d, OSD_TAU_BIN(r)); return true; }
    if (lastp != 0u) { *tau_out = osd_bin_tau<RM>(d, OSD_TAU_BIN(lastp - 1u)); return true; }
    return false;
}

// The histogram ping-pong's one write.  Every shot of a use_osd handle zeroes the half
// the *next* shot will push into, so the shot that uses it starts from zeros without a grid-wide
// zeroing and a barrier inside P1 (the blocks' pushes cannot be ordered behind a grid-wide zeroing
// without a second rendezvous).  The halves alternate with the shot
// generation; the pair was zeroed at create.  Called at the top of both shot bodies, before any early
// return, so a converged shot -- which skips the pass -- still prepares the next shot's buffer.
__device__ __forceinline__ void osd_hist_prepare(const Dev &d) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t gstride = gridDim.x * blockDim.x;
    test_delay_prologue(12u);   // test-only: the next shot's histogram half
    for (uint32_t i = gid; i < kOsdHBins; i += gstride) d.ohist_next[i] = 0u;
}

// Collect every column with key <= tau, one atomicAdd per *warp* instead of per candidate.  A warp's
// qualifying lanes take one contiguous run of slots: the lowest qualifying lane gets the base with a
// single atomicAdd of the warp's popcount and broadcasts it, and every other lane takes base + its
// rank inside the ballot.  The set of records is exactly {key <= tau} (only the slots differ, and
// the host sorts the records by (key, column)); the counter d.ocnt[0] sums to its size.
// The threshold rule guarantees the count is <= d.osd_cap, so no record is dropped; the bound is
// kept as a safety net (a record beyond the cap would make the host reject the set).
// The records go to `dst` -- the pass's working buffer d.ocand when the
// rank sort will run, the hand-over buffer d.ocand_sorted when it will not; the repick's d.ocand.
// The collect over an explicit (key array, counter, cap), so the BP
// kernel's tail can collect rung two into its own buffer; osd_collect below is the rung-one / repick
// form over the handle's Dev fields.
__device__ __forceinline__ void osd_collect_to(const unsigned long long *okey, uint32_t n,
                                               unsigned *cnt, unsigned cap,
                                               unsigned long long tau, unsigned char *dst) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t gstride = gridDim.x * blockDim.x;
    const uint32_t iters = (n + gstride - 1u) / gstride;   // uniform over the whole grid
    for (uint32_t it = 0; it < iters; ++it) {
        const uint32_t j = gid + it * gstride;
        const bool in = (j < n);
        const unsigned long long kk = in ? okey[j] : ~0ull;
        const bool hit = in && (kk <= tau);
        const unsigned mask = __ballot_sync(0xffffffffu, hit);
        if (hit) {
            const unsigned lane = (unsigned)threadIdx.x & 31u;
            const unsigned first = (unsigned)__ffs(mask) - 1u;
            unsigned base = 0u;
            if (lane == first) base = atomicAdd(&cnt[0], (unsigned)__popc(mask));
            base = __shfl_sync(mask, base, (int)first);
            const unsigned slot = base + (unsigned)__popc(mask & ((1u << lane) - 1u));
            if (slot < cap) {
                unsigned char *rec = dst + (size_t)slot * 16u;
                *reinterpret_cast<unsigned long long *>(rec) = kk;
                *reinterpret_cast<unsigned *>(rec + 8) = j;
                *reinterpret_cast<unsigned *>(rec + 12) = 0u;
            }
        }
    }
}

__device__ __forceinline__ void osd_collect(const Dev &d, unsigned long long tau, unsigned char *dst) {
    osd_collect_to(d.okey, d.n, d.ocnt, d.osd_cap, tau, dst);
}

// Rung two, collected
// speculatively in the BP kernel's tail.
//
// The shot's rung one has already been published -- its records into the mapped hand-over block,
// its status and generation into the mapped status block -- and the host is walking rung one
// while this runs, so the work is off the host's path.  Every block re-derives rung two's threshold
// from the pushed histogram the prefix pass built (the same block-wide scan and the same rule as
// osd_repick_kernel, with rung two's constants; the selection does not depend on the rung's K and cap)
// and collects {key <= tau2} at (kOsdK2, kOsdCap) into the handle's own rung-two device block.  The
// blocks then rendezvous on that block's second header word (a device-scope fence, one
// atomic increment per block, block 0 spins for the grid with a %globaltimer bound); block 0 writes the
// header [count | sorted = 0 | pad] and copies the records into the handle's mapped rung-two block,
// makes them visible at system scope and only then releases gen2 = this shot's generation.  The host,
// on a repick shot, polls gen2 (bounded) and reads the set in place: no launch, no 32 KB D2H copy and
// no stream synchronisation.  A handle whose block could not be mapped (hand2 / pub2 null) collects
// only, and the host reads the device block with one stream-ordered D2H.
//
// Why the tail is inline: an out-of-line form (__noinline__) grows the
// SHSUM instantiation's STACK to 24 B; any stack growth is to be avoided, and the inline form's
// cost is +16 B of static shared memory (the two shared words and the scan's staging), inside the
// b2c window's 1,824 B of headroom at 84 blocks.
//
// Exactness.  The threshold is osd_hist_scan's rule over the same pushed histogram with rung two's
// constants, and the collect is osd_collect_to, the same code as osd_repick_kernel's: so the set is
// the repick kernel's {key <= tau2} (the whole tie run at tau2 included), record for record.  The host
// walks it with the unsorted walk (the chunk walk, or resume_candidates with after_key =
// rung one's largest key), whose fold, stop and x are the plain walk's.  The hand-off contract: the writer
// (block 0) makes every covered record visible at system scope (__threadfence_system()) before it
// writes the flag (st.release.sys.global.u64), and the reader (the host) acquire-loads gen2 and
// compares it with the shot's unique generation before reading anything; a shot that did not run the
// tail leaves the generation old, and the copy (or the repick kernel) produces the same set.  The
// block's count and arrival word are zeroed by the shot's own prologue, before the prefix pass's grid
// barrier that orders them before the tail's atomics, and the histogram / key array are this shot's.
// A shot that did not run the tail -- converged, a bp handle, a failed launch -- never takes this path.
template <int RM>
__device__ __forceinline__ void osd_tail_collect(const Dev &d, unsigned long long shot_gen) {
    __shared__ unsigned long long sh_tau2;
    __shared__ unsigned sh_ok2, sh_r2, sh_last2;
    // The threshold: the same block-wide scan and the same rule as osd_repick_kernel, with rung two's
    // constants.  (A serial scan on thread 0 was measured at 10.7 us -- 256 dependent global
    // loads at the histogram; the block-wide scan is one coalesced read of all bins.)
    if (threadIdx.x == 0) { sh_r2 = kOsdHBins; sh_last2 = 0u; }
    __syncthreads();
    osd_hist_scan(d, kOsdK2, kOsdCap, &sh_r2, &sh_last2);
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long tau;
        const bool ok = osd_pick_threshold<RM>(d, sh_r2, sh_last2, &tau);
        sh_tau2 = ok ? tau : 0ull;
        sh_ok2 = ok ? 1u : 0u;
    }
    __syncthreads();
    if (sh_ok2 != 0u) {
        osd_collect_to(d.okey, d.n, d.ocnt2, kOsdCap, sh_tau2, d.ocand2);
    }
    if (d.hand2 != nullptr && d.pub2 != nullptr) {
        // The mapped publication: every block reports its collect done with a device-scope fence and
        // one atomic increment of the block's second header word (zeroed by the shot prologue), block
        // 0 waits for the whole grid, writes the header and copies the records into the handle's
        // mapped block, makes them visible at system scope and only then releases gen2 = this shot's
        // generation with a system-scope release store.
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(&d.ocnt2[1], 1u);
        if (blockIdx.x == 0) {
            __shared__ unsigned sh_timeout;
            if (threadIdx.x == 0) {
                unsigned long long t0;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
                unsigned long long spins = 0, t = t0;
                sh_timeout = 0u;
                while (*reinterpret_cast<volatile const unsigned *>(&d.ocnt2[1]) < gridDim.x) {
                    if (((++spins) & 1023ull) == 0ull) {
                        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
                        if (t - t0 > kTailSpinNs) { sh_timeout = 1u; break; }
                    }
                }
            }
            __syncthreads();
            if (sh_timeout == 0u) {
                __threadfence();             // acquire: every block's records are now visible
                const uint32_t pn = d.ocnt2[0];
                // A count outside the cap is "no usable set" to the host, which reads it from the
                // header; then only the header is copied (the records past the cap were never written).
                const uint32_t nrec = (pn >= 1u && pn <= kOsdCap) ? pn : 0u;
                uint64_t *dst = reinterpret_cast<uint64_t *>(d.hand2);
                if (threadIdx.x == 0) {
                    // [count | sorted | pad]: the tail's set is unsorted (the rank sort is rung one's
                    // only), so the host takes its usual chunk walk (or the unsorted resume).
                    dst[0] = (uint64_t)pn;
                    dst[1] = 0ull;
                }
                const uint32_t nw64 = (uint32_t)((size_t)nrec * 16u / 8u);   // 2 words per record
                const uint64_t *src = reinterpret_cast<const uint64_t *>(d.ocand2);
                for (uint32_t i = threadIdx.x; i < nw64; i += blockDim.x) dst[2u + i] = src[i];
                __threadfence_system();      // the host sees the records before the generation below
                __syncthreads();             // every copy is fenced before thread 0 releases
                if (threadIdx.x == 0) {
#if defined(DBENCH_STAGE_PROBES) && DBENCH_STAGE_PROBES
                    unsigned long long t;
                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
                    d.pub[5] = t;            // the tail's end on the device clock (measurement build)
#endif
                    st_release_sys_u64(d.pub2, shot_gen);
                }
            }
        }
    } else {
#if defined(DBENCH_STAGE_PROBES) && DBENCH_STAGE_PROBES
        // No mapped block: the collect-only path; stamp block 0's end (measurement build only).
        __threadfence();
        __syncthreads();
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            unsigned long long t;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            d.pub[5] = t;
        }
#endif
    }
}

// P6: the collected records ordered by the exact OSD-0 order, on the device, so the
// host walk consumes them with no selection at all.
//
// The sort is a *rank sort*: at most kOsdCap (2048) records are collected and the grid has thousands
// of warps, so one warp can compare its record against all of them.  Warp w takes record i = w,
// w + nwarp, ...; lane l counts the records j = l, l + 32, ... that are strictly smaller in the
// OSD-0 order -- (key, then column index), the same comparator the host engine's CandLess uses, so
// the key map above decides the tie rules (-0.0 == +0.0, every NaN after every number, NaNs by
// index).  The 32 partial counts are reduced with shuffles and lane 0 writes the record to slot
// rank.  Column indices are unique, so the order is total, the ranks are exactly 0..cnt-1 and the
// scatter is a permutation: the output is sorted whatever order the collection left behind.
//
// Why a rank sort rather than a bitonic or a radix sort in one block (both measured on the development machine):
// sorting the 1.5K collected records with a bitonic sort costs 44.3 us and one
// radix-style pass (histogram + scan + scatter) 8.7 us, against the 10.1 us p50 / 20.0 us p99 the
// host's chunk selection costs -- a one-block sort cannot pay.  The whole grid spreads cnt^2/2 =
// 1.1M comparisons over the blocks that hold the collected set (about 1 us), at one extra grid
// barrier.
__device__ __forceinline__ bool osd_rec_less(unsigned long long ka, unsigned ia,
                                             unsigned long long kb, unsigned ib) {
    return (ka < kb) | ((ka == kb) & (ia < ib));
}

// Records staged into shared memory per pass.  512 records = 6 KiB (keys + indices), small enough
// that the kernel's static shared footprint stays close to the base's, so the L1 the BP iterations
// use is not carved away (a 24 KiB staging cost 3.3 % per iteration and turned the (0.5, 31)
// preset into a regression, as measured in the device-sorted-prefix ablation).
// (The staging area is the pass's own sample array, reused: see osd_rank_sort.)

// The sort's gate.  The rank
// sort costs cnt^2/2 comparisons plus the staging, while what it removes is the host's chunk selection over
// the same records, whose cost grows with the walk the shot takes; both sides were measured.
//
// The two sides were measured at the short caps (AVX2), which gave
// `n >= kOsdSortMinN && max_iter <= kOsdSortMaxIter`:
//   * bb144 xyz (n = 76032): the host stage is 10.1 us p50 / 20.0 us p99 and the sort pays at every
//     cap up to 11 (a03125c3 -9 us p50, a0375c7 -2.3, a05c11 +0.1).
//   * bb144 z (n = 10512): the host's walk is short there, its chunk selection costs ~1 us, and the
//     sort does not pay at all (a05c15 +0.8 p50 / +4.6 p99, a05ms31 +8.2 osd-path p50).
//   * the cap term existed only because *that* form of the sort added 6 KiB of static shared memory
//     to every kernel instantiation, which cost the BP iterations ~0.45 us each (3.3 % at cap 15);
//     at cap 31 that outweighed the host stage.  The kept form (staging in the pass's own sh_samp,
//     osd_rank_sort below) adds no static shared memory and leaves the BP kernels' resources at the
//     base's, so the per-iteration cost the cap term priced is no longer there.
//
// The gate was re-derived on an RTX 5080 development machine (AVX-512 build, preset a05ms31,
// 4000 shots each): at the caps the old gate excluded, the host stage is still worth removing --
// bb144 xyz a05ms31 3.64 us p50 / 10.11 us p99 on 94 % of the shots, bb288 z a05ms31 3.46 / 6.46 us
// on 84 % -- while the sort's device cost does not depend on the cap (the same cnt^2/2 comparison set
// over a rung-one set of 384..640 records).  So the cap term is dropped and only the shape term
// remains; bb144 z (n = 10512) keeps the host's cheap walk.  A gated-out shot hands the records over
// unsorted, exactly as the plain pipeline does, and the host takes its usual chunk walk -- which is why
// the flag below exists.
constexpr uint32_t kOsdSortMinN = 16384u;      // between bb144 z (10512) and bb288 z (31392)

// __noinline__ is load-bearing, not style: osd_prefix_pass is inlined
// into every bp_kernel instantiation, and with the sort's loops inlined next to the BP loop nvcc's
// register allocator re-coloured the fused kernels (SHARED 5280: STACK 24 against the base's 16; the
// 5408 kernels: REG 124/STACK 0/8 against 126/0); the BP loop then ran 1.0-2.7 % slower per iteration,
// which at cap 31 was the (0.5, 31) preset's regression.  With the call the kernels' REG/STACK/
// SHARED are the base's (cuobjdump -res-usage) and the per-iteration cost ratio is 1.000.
__device__ __noinline__ void osd_rank_sort(const Dev &d, uint32_t cnt, unsigned char *stage,
                                           uint32_t stage_bytes) {
    // One warp per element in each round, the records staged in chunks of kOsdSortChunk.  Every warp
    // counts, in lane-parallel, the records strictly smaller than its element on the OSD-0 order,
    // reduces the 32 partial counts with shuffles, and lane 0 writes the record to its rank's slot.
    // The staged chunk is read once per element-warp instead of the whole array being read from
    // memory (L1 holds none of it: the pass just wrote the 608 KiB key array), and the staging itself
    // is one coalesced read of kOsdSortChunk records per pass.  The barrier inside the chunk loop is
    // reached by every thread of the block (the round and chunk loops are block-uniform); the
    // per-element work is guarded by `active`.
    // `stage` is the pass's *own* shared-memory sample area (sh_samp, 4 KiB), which the selection
    // above has finished with: reusing it keeps the kernel's static shared footprint at the base's,
    // so the L1 the BP iterations work out of is not carved away (an extra 6 KiB of staging cost
    // about 0.45 us per iteration and turned the (0.5, 31) preset into a regression,
    // as measured in the device-sorted-prefix ablations).
    const uint32_t chunk = stage_bytes / 16u;   // records staged per pass
    const uint32_t tid = (uint32_t)threadIdx.x;
    const uint32_t nt = (uint32_t)blockDim.x;
    const uint32_t lane = tid & 31u;
    const uint32_t nw = nt >> 5u;
    const uint32_t gw = blockIdx.x * nw + (tid >> 5u);   // this warp's global index
    const uint32_t nwarp = gridDim.x * nw;               // warps in the grid
    const ulonglong2 *rp = reinterpret_cast<const ulonglong2 *>(d.ocand);
    const uint32_t rounds = (cnt + nwarp - 1u) / nwarp;
    for (uint32_t r = 0; r < rounds; ++r) {
        const uint32_t my_i = r * nwarp + gw;
        const bool active = (my_i < cnt);
        unsigned long long ki = 0ull;
        unsigned ii = 0u;
        unsigned c = 0u;
        if (active) {
            const ulonglong2 ri = rp[my_i];
            ki = ri.x;
            ii = (unsigned)ri.y;
        }
        for (uint32_t base = 0; base < cnt; base += chunk) {
            const uint32_t nch = ((cnt - base) < chunk) ? (cnt - base) : chunk;
            for (uint32_t i = tid; i < nch; i += nt) {
                *reinterpret_cast<ulonglong2 *>(stage + (size_t)i * 16u) = rp[base + i];
            }
            __syncthreads();
            if (active) {
                for (uint32_t j = lane; j < nch; j += 32u) {
                    const ulonglong2 rj =
                        *reinterpret_cast<const ulonglong2 *>(stage + (size_t)j * 16u);
                    if (rj.x < ki) ++c;
                    else if (rj.x == ki) c += (unsigned)((unsigned)rj.y < ii);
                }
            }
            __syncthreads();   // the next chunk overwrites the staging
        }
        if (active) {
#pragma unroll
            for (unsigned o = 16u; o; o >>= 1u) c += __shfl_down_sync(0xffffffffu, c, o);
            if (lane == 0u) {
                unsigned char *rec = d.ocand_sorted + (size_t)c * 16u;
                *reinterpret_cast<unsigned long long *>(rec) = ki;
                *reinterpret_cast<unsigned *>(rec + 8u) = ii;
                *reinterpret_cast<unsigned *>(rec + 12u) = 0u;
            }
        }
    }
}

// The pass, with the pushed coarse histogram.
// FOLD (the fused body): this pass's first grid barrier also orders the merge of iteration max_iter,
// whose mismatch the caller's status word needs.  The pass reads nothing of that merge but the mis
// word below, and its P1 (the keys, the bins and the blocks' histogram push) reads only post, which
// the last column pass wrote before the iteration's barrier -- so P1 runs first, the pass's own first
// barrier publishes it AND orders the merge, and the verdict is read after it.  A shot that converged
// at max_iter returns 0 here, before P4..P6: the host never takes the OSD-0 path for it (its status
// says BP_CONVERGED), and the per-shot scratch P1 wrote (okey, ohist) is rewritten by the next shot's
// P1.  The decision is the same in every block (the same mis word, read after a grid barrier), so the
// early return and the pass's remaining barriers stay grid-uniform.  Returns the mis word (0 =
// converged); bp_body discards it (FOLD = false).
//
// The earlier sample machinery -- P1's global histogram zeroing, P2's stride sample and
// grid-wide rank sort, P3's binary-search rank sweep, the second grid barrier and P4's scan of the
// sample-rank histogram -- is replaced by one pushed coarse histogram.  P1 bins every posterior into
// kOsdHBins value-uniform bins in a per-block shared histogram, then adds the block's counts to the
// one global histogram with atomics.  After the pass's first (now only) grid barrier, every block
// scans the whole histogram with a block-wide prefix sum and derives the threshold itself.  The
// threshold is a bin edge, so tau changes only the set's size, never its membership rule: the set is
// still exactly {key <= tau}, a prefix of the reference order with the whole tie run inside it.
template <bool FOLD, int RM>
__device__ __forceinline__ uint32_t osd_prefix_pass(const Dev &d, const double *post) {
    // `post` is the parity buffer that holds the shot's final state (max_iter & 1 in
    // the fused body; d.post in bp_body).
    // osd_rank_sort stages records in this array with 16-byte ulonglong2 accesses
    // (P6), so its base must be 16-byte aligned -- the tail's added shared words moved the
    // static layout and exposed the old 8-byte-aligned declaration as a misaligned 16 B write.
    __shared__ __align__(16) unsigned long long sh_samp[kOsdS];
    __shared__ unsigned long long sh_tau;
    __shared__ unsigned sh_ok;
    __shared__ unsigned sh_r, sh_last;
    // The per-block histogram lives in the sample array's storage (kOsdS u64 = 4 KiB), which the base
    // already pays as static shared memory in every instantiation, so the histogram adds none, and the
    // array is free from the block's push until P6's rank sort stages records in it again.
    unsigned *sh_hist = reinterpret_cast<unsigned *>(sh_samp);
    static_assert(kOsdHBins * sizeof(unsigned) <= sizeof(sh_samp), "the histogram fits the sample array");
    const unsigned tid = (unsigned)threadIdx.x;
    const unsigned nt = (unsigned)blockDim.x;
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t gstride = gridDim.x * blockDim.x;
    const uint32_t n = d.n;
    // The sort's gate, read before the collection.
    // The shape gate alone (see kOsdSortMinN).  n is uniform over the
    // grid, so the gate is too -- and so is the grid barrier of P6 it guards.
    const bool sortable = (n >= kOsdSortMinN);

    // P1: every column's key, and its bin in this block's shared histogram.  The global histogram this
    // shot pushes into is already zero: the previous shot's kernel zeroed the other half of the
    // two-buffer ping-pong (osd_hist_prepare), and a kernel's writes are ordered before the next
    // launch on the stream.
    for (unsigned i = tid; i < kOsdHBins; i += nt) sh_hist[i] = 0u;
    __syncthreads();
    test_delay_prologue(13u);   // test-only: the key array and the push
    for (uint32_t j = gid; j < n; j += gstride) {
        const double x = post[j];   // the final state's parity buffer
        d.okey[j] = osd_key_bits(__double_as_longlong(x));
        atomicAdd(&sh_hist[osd_bin_of<RM>(d, x)], 1u);
    }
    __syncthreads();
    // The push: one atomic per non-empty bin of this block (a block's n / nblocks columns land in at
    // most kOsdHBins of them, and on the development shapes a few hundred do).  The grid barrier below
    // publishes the whole grid's counts to every block.
    for (unsigned i = tid; i < kOsdHBins; i += nt) {
        const unsigned c = sh_hist[i];
        if (c != 0u) atomicAdd(&d.ohist[i], c);
    }
    grid_barrier(d);
    uint32_t conv = 1u;
    if constexpr (FOLD) {
        __shared__ unsigned sh_conv;
        if (threadIdx.x == 0) sh_conv = __ldcg((const unsigned *)&d.mis[d.max_iter % 3u]);
        __syncthreads();
        conv = sh_conv;
        if (conv == 0u) return 0u;   // converged after max_iter: no prefix for this shot
    }

    // P4: every block scans the complete global histogram itself and derives the same threshold (the
    // counts are integers and the sums deterministic, so all blocks agree bit for bit).  The scan is
    // block-wide: one bin per thread, a prefix sum over the block, and the two bins the rule needs.
    // Because every block has the threshold locally, the collect below needs no grid barrier after it.
    if (tid == 0) { sh_r = kOsdHBins; sh_last = 0u; }
    __syncthreads();
    osd_hist_scan(d, d.osd_k, d.osd_cap, &sh_r, &sh_last);
    __syncthreads();
    if (tid == 0) {
        unsigned long long tau;
        const bool ok = osd_pick_threshold<RM>(d, sh_r, sh_last, &tau);
        sh_tau = ok ? tau : 0ull;
        sh_ok = ok ? 1u : 0u;
    }
    __syncthreads();
    // The branch stays *uniform* across the grid (sh_ok is the same in every block: the same
    // histogram, the same integer arithmetic) and the collection is skipped rather than returned from,
    // so on a gated-in shot every block still reaches the grid barrier of P6 below.
    if (sh_ok != 0u) {
        // P5 (osd_collect): every column with key <= tau, at most this rung's cap.  The
        // destination is the gate's: the working buffer when the sort will run (it moves
        // the records into the hand-over buffer), the hand-over buffer itself when it will not -- that
        // shot's pass is then the plain one, record for record: no barrier, no copy, no flag.
        osd_collect(d, sh_tau, sortable ? d.ocand : d.ocand_sorted);
    }
    // P6: order the hand-over -- only on a gated-in shot.  The barrier publishes the collection
    // (every record written, the count final, the atomics' values visible in system and device scope)
    // to every block, which the rank sort reads.  The bound is this rung's own cap
    // d.osd_cap (rung one, kOsdCapA), not kOsdCap: a count above the rung's cap means the collect
    // dropped records (the host rejects such a set), and sorting it would read record slots this shot
    // never wrote.
    if (sortable) {
        grid_barrier(d);
        const uint32_t ocnt = d.ocnt[0];
        if (ocnt >= 1u && ocnt <= d.osd_cap) {
            // d.ocnt[1] is the hand-over's "the records are sorted" flag; the prologue clears it, so 0
            // means "unsorted, take the host's usual order" (an overflowed cap, or no prefix at all).
            // The kernel's end orders the store and the record copies before the host's D2H.  The
            // rank sort stages the records in sh_samp, which P1-P4 have finished with (its 512 x 8 B
            // carry 256 records per pass).
            osd_rank_sort(d, ocnt, reinterpret_cast<unsigned char *>(sh_samp), (uint32_t)sizeof(sh_samp));
            if (blockIdx.x == 0 && threadIdx.x == 0) d.ocnt[1] = 1u;
        }
    }
    // When the handle's hand-over buffer is mapped pinned
    // memory, the shot's whole hand-over block -- the header [count | sorted | pad] and rung one's records
    // (the rank sort's output, or the collect's when the sort is gated out) -- is published there before
    // the kernel's status release, so an OSD shot needs no device-to-host copy and no stream
    // synchronisation.  Every block reports its own collect / sort writes with a device-scope fence and
    // one atomic increment of the header's third word (d.ocnt[2], zeroed by the shot prologue); block 0
    // waits for every block, copies the block to the mapped buffer with its own threads, makes it visible
    // at system scope and returns; the kernel's epilogue then publishes [status | generation] with its
    // system-scope release store, which is the flag the host polls.  (FOLD: a shot that
    // converged at max_iter returned above, grid-uniformly, and publishes nothing here; the host takes no
    // hand-over for it.)
    if (d.hand_host != nullptr) {
        __threadfence();
        __syncthreads();
        if (tid == 0) atomicAdd(&d.ocnt[2], 1u);
        if (blockIdx.x == 0) {
            if (tid == 0) {
                // A volatile read: the spin must survive compilation and re-read the counter.
                while (*reinterpret_cast<volatile const unsigned *>(&d.ocnt[2]) < gridDim.x) {
                }
            }
            __syncthreads();
            __threadfence();                   // acquire: every block's records are now visible
            const uint32_t pn = d.ocnt[0];
            // A count outside the rung's cap is "no usable set" to the host, which reads it from the
            // header; then only the header is copied (the records past the cap were never written).
            const uint32_t nrec = (pn >= 1u && pn <= d.osd_cap) ? pn : 0u;
            const uint32_t nw64 = (uint32_t)((kHandHdr + (size_t)nrec * 16u) / 8u);
            const uint64_t *src = reinterpret_cast<const uint64_t *>(d.ocnt);
            uint64_t *dst = reinterpret_cast<uint64_t *>(d.hand_host);
            for (uint32_t i = tid; i < nw64; i += nt) dst[i] = src[i];
            __threadfence_system();            // the host sees the records before the status release
            __syncthreads();                   // every copy is fenced before thread 0 returns
        }
    }
    return conv;
}

// Rung two of the OSD-0 prefix: launched as its own kernel only on the
// shots whose walk ran past rung one.  It re-derives the threshold from the sample and the histogram
// rung one published -- neither depends on K -- and re-collects at the larger (kOsdK2, kOsdCap);
// the host zeroes d.ocnt[0] on the same stream first, so the count this pass leaves is its own.  It
// reads only this shot's arrays (nothing an earlier shot left), and it is a plain launch: no grid
// barrier is needed because the collect's counter is read by the host only after the stream sync.
// It collects into the working buffer d.ocand, unsorted (the rank sort is rung one's only);
// Engine::osd_repick copies that buffer and tells the host the records are unsorted.
// The rung-two selection reads the same pushed histogram rung one pushed (it is left
// untouched until this shot's hand-off), with the host's rung-two (kOsdK2, kOsdCap) in the Dev; its
// scan is the pass's own block-wide one.  Templated on the rounding mode like the pass, so the
// threshold's edge is computed with the mode's own DADD / DMUL (the kernel is instantiated once per
// mode and the handle launches its own).
template <int RM>
__global__ void osd_repick_kernel(Dev d) {
    __shared__ unsigned long long sh_tau;
    __shared__ unsigned sh_ok, sh_r, sh_last;
    if (threadIdx.x == 0) { sh_r = kOsdHBins; sh_last = 0u; }
    __syncthreads();
    osd_hist_scan(d, d.osd_k, d.osd_cap, &sh_r, &sh_last);
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long tau;
        const bool ok = osd_pick_threshold<RM>(d, sh_r, sh_last, &tau);
        sh_tau = ok ? tau : 0ull;
        sh_ok = ok ? 1u : 0u;
    }
    __syncthreads();
    if (sh_ok == 0u) return;
    osd_collect(d, sh_tau, d.ocand);
}

// The keep-warm kernel (fastbp_keep_warm's GPU flag): no work, no memory access.  Launched with
// one block of 32 threads on the handle's stream between decodes, it keeps the GPU's clocks and the
// stream's launch path from going idle (at a 10 ms call cycle the clocks sag inside P1).
__global__ void keep_warm_kernel() {}

// Host-side launch of the repick kernel in the handle's rounding mode (one instantiation per mode,
// as the BP kernels have; the launch is the same cooperative-free plain launch it was).
// The launch's own error is returned and checked (CK) by both callers.  An earlier form
// dropped it and relied on the cudaGetLastError() that follows; a real failure sets the last error, but an
// interposed one (the failure-injection shim returns an error without entering the runtime) does not, so the
// decode would have gone on with a hand-over the repick never wrote.
static cudaError_t osd_repick_launch(int rm, int nblocks, int nthreads, cudaStream_t stream, Dev &d) {
    void *args[] = {&d};
    const void *fn = (rm == kRU) ? (const void *)osd_repick_kernel<kRU>
                   : (rm == kRD) ? (const void *)osd_repick_kernel<kRD>
                   : (rm == kRZ) ? (const void *)osd_repick_kernel<kRZ>
                                 : (const void *)osd_repick_kernel<kRN>;
    return cudaLaunchKernel(fn, dim3((unsigned)nblocks), dim3((unsigned)nthreads), args, 0, stream);
}


// ---------------------------------------------------------------------------
// Check pass.  One warp per row (wpr == 1) or wpr warps per row for long rows (wpr > 1, the work
// units are group-aligned so the wpr partials of a row are always in one block).
//
// Each lane holds its slice's values in registers (two per lane, i.e. a slice of degree <= 64) so
// the message loop never gathers the messages a second time; rows whose slice is longer take the
// generic path, which gathers twice (that costs latency, not exactness).  The gather always covers
// the whole slice: a row of degree > 2*stride (i.e. > 512 at the wpr cap of 8) makes the lane loop
// over its slice so that m1/m2/parity are the row's, not the first 512 edges'.
//
// The sign parity is total_sgn = syndrome[row] + #{v <= 0}, exactly bp.hpp's total_sgn, and the
// message is (|v| == m1 ? m2 : m1) * (+-alpha), one multiply with the final sign in the operand.
// |NaN| == m1 is false, so a NaN message receives m1: the reference's running minimum never takes a
// NaN either.
// ---------------------------------------------------------------------------
template <bool FIRST>
__device__ __forceinline__ void row_gather(const Dev &d, uint32_t wpr, uint32_t row, uint32_t sub,
                                           int lane, double &v0, double &v1, bool &h0, bool &h1,
                                           unsigned long long &m1, unsigned long long &m2, int &cnt,
                                           unsigned &xorac) {
    const uint32_t b = d.rptr[row], e = d.rptr[row + 1];
    const uint32_t stride = wpr * 32u;
    const uint32_t base = b + sub * 32u + (uint32_t)lane;
    v0 = 0.0; v1 = 0.0;
    h0 = false; h1 = false;
    xorac = 0;
    m1 = kDblMaxBits;
    m2 = kDblMaxBits;
    cnt = 0;
    if (e - b <= 2u * stride) {
        // register path: at most two values per lane, kept for row_write
        h0 = base < e;
        h1 = base + stride < e;
        if (h0) {
            const uint32_t k = base;
            v0 = FIRST ? d.prior[d.rcol[k]] : d.b2c[d.redge[k]];
            if constexpr (!FIRST) xorac ^= (unsigned)d.dec[d.rcol[k]];
        }
        if (h1) {
            const uint32_t k = base + stride;
            v1 = FIRST ? d.prior[d.rcol[k]] : d.b2c[d.redge[k]];
            if constexpr (!FIRST) xorac ^= (unsigned)d.dec[d.rcol[k]];
        }
        if (h0) { cnt += (int)nonpositive(v0); const unsigned long long a0 = mag_bits(v0);
                  if (a0 < m1) { m2 = m1; m1 = a0; } else if (a0 < m2) m2 = a0; }
        if (h1) { cnt += (int)nonpositive(v1); const unsigned long long a1 = mag_bits(v1);
                  if (a1 < m1) { m2 = m1; m1 = a1; } else if (a1 < m2) m2 = a1; }
    } else {
        // long slice: loop over the whole slice; row_write's generic path re-gathers (h0 = h1 = false)
        for (uint32_t k = base; k < e; k += stride) {
            const double v = FIRST ? d.prior[d.rcol[k]] : d.b2c[d.redge[k]];
            if constexpr (!FIRST) xorac ^= (unsigned)d.dec[d.rcol[k]];
            cnt += (int)nonpositive(v);
            const unsigned long long a = mag_bits(v);
            if (a < m1) { m2 = m1; m1 = a; } else if (a < m2) m2 = a;
        }
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) {
        const unsigned long long o1 = __shfl_xor_sync(0xffffffffu, m1, off);
        const unsigned long long o2 = __shfl_xor_sync(0xffffffffu, m2, off);
        const unsigned long long mx = (m1 > o1) ? m1 : o1;   // fmax(m1, o1), never NaN
        const unsigned long long i2 = (m2 < o2) ? m2 : o2;   // fmin(m2, o2)
        m1 = (m1 < o1) ? m1 : o1;                            // fmin(m1, o1)
        m2 = (mx < i2) ? mx : i2;                            // fmin(fmax(m1, o1), fmin(m2, o2))
        cnt += __shfl_xor_sync(0xffffffffu, cnt, off);
        xorac ^= __shfl_xor_sync(0xffffffffu, xorac, off);
    }
}

// Writes this warp's slice's messages from the registers row_gather filled, with the row-wide
// (m1, m2, cntsy).  Slices of a row are disjoint, so no lane reads a slot another lane writes.
template <int RM, bool FIRST>
__device__ __forceinline__ void row_write(const Dev &d, uint32_t wpr, uint32_t row, uint32_t sub,
                                          int lane, double v0, double v1, bool h0, bool h1,
                                          unsigned long long m1, unsigned long long m2, int cntsy) {
    const uint32_t b = d.rptr[row], e = d.rptr[row + 1];
    const uint32_t stride = wpr * 32u;
    const uint32_t base = b + sub * 32u + (uint32_t)lane;
    if (e - b <= 2u * stride) {
        // register path: the slice's values are already in v0/v1
        if (h0) {
            const uint32_t k = base, ed = d.redge[k];
            const unsigned long long base_bits = (mag_bits(v0) == m1) ? m2 : m1;
            const double salpha = ((cntsy + (int)nonpositive(v0)) & 1) ? -d.alpha : d.alpha;
            d.b2c[ed] = dmul<RM>(__longlong_as_double(base_bits), salpha);
        }
        if (h1) {
            const uint32_t k = base + stride, ed = d.redge[k];
            const unsigned long long base_bits = (mag_bits(v1) == m1) ? m2 : m1;
            const double salpha = ((cntsy + (int)nonpositive(v1)) & 1) ? -d.alpha : d.alpha;
            d.b2c[ed] = dmul<RM>(__longlong_as_double(base_bits), salpha);
        }
    } else {
        // generic path (a slice longer than 64): a second gather over the slice.  Iteration 1 reads
        // the priors here too (b2c holds nothing yet on the first check pass).
        for (uint32_t k = base; k < e; k += stride) {
            const uint32_t ed = d.redge[k];
            const double v = FIRST ? d.prior[d.rcol[k]] : d.b2c[ed];
            const unsigned long long base_bits = (mag_bits(v) == m1) ? m2 : m1;
            const double salpha = ((cntsy + (int)nonpositive(v)) & 1) ? -d.alpha : d.alpha;
            d.b2c[ed] = dmul<RM>(__longlong_as_double(base_bits), salpha);
        }
    }
}

template <int RM, bool FIRST>
__device__ __forceinline__ void row_pass(const Dev &d, int *row_mis) {
    const int lane = threadIdx.x & 31;
    const int wid = (int)(threadIdx.x >> 5);
    const uint32_t nwarp = (gridDim.x * blockDim.x) >> 5;
    const uint32_t wpb = (uint32_t)(blockDim.x >> 5);
    const uint32_t wpr = (d.wpr < wpb) ? d.wpr : wpb;   // a row can only be split over its own block's warps
    int mis = 0;
    if (wpr == 1u) {
        const uint32_t gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        for (uint32_t row = gwarp; row < d.m; row += nwarp) {
            double v0, v1;
            unsigned long long m1, m2;
            bool h0, h1;
            int cnt;
            unsigned xorac;
            row_gather<FIRST>(d, 1u, row, 0, lane, v0, v1, h0, h1, m1, m2, cnt, xorac);
            row_write<RM, FIRST>(d, 1u, row, 0, lane, v0, v1, h0, h1, m1, m2, cnt + (int)syn_bit(d, row));
            if constexpr (!FIRST) {
                if (lane == 0) mis += (int)((xorac & 1u) != (unsigned)syn_bit(d, row));
            }
        }
    } else {
        // wpr > 1: block-uniform rounds over group-aligned windows of `wpg` work units -- the largest
        // multiple of wpr that fits the block's warps.  The window start is a multiple of wpr, so the
        // wpr slices of a row always land in the same block's window and the merge can use
        // __syncthreads(); warps past wpg sit the round out (wpr = 3, 5, 6, 7 do not divide the 8 warps
        // of a block).  Every warp of a row's group merges the wpr partials itself, then writes its own
        // slice's messages.
        __shared__ unsigned long long sh_m1[32], sh_m2[32];
        __shared__ int sh_c[32], sh_x[32];
        const uint32_t wpg = (wpb / wpr) * wpr;
        const uint32_t total = d.m * wpr;
        for (uint32_t base = blockIdx.x * wpg; base < total; base += (uint32_t)gridDim.x * wpg) {
            const uint32_t w = base + (uint32_t)wid;
            double v0 = 0.0, v1 = 0.0;
            unsigned long long m1 = kDblMaxBits, m2 = kDblMaxBits;
            bool h0 = false, h1 = false;
            int cnt = 0;
            unsigned xorac = 0;
            uint32_t row = 0, sub = 0;
            if (wid < (int)wpg && w < total) {
                row = w / wpr;
                sub = w - row * wpr;
                row_gather<FIRST>(d, wpr, row, sub, lane, v0, v1, h0, h1, m1, m2, cnt, xorac);
            }
            if (lane == 0) {
                sh_m1[wid] = m1; sh_m2[wid] = m2; sh_c[wid] = cnt; sh_x[wid] = (int)xorac;
            }
            __syncthreads();
            if (wid < (int)wpg && w < total) {
                // Start from the identity and merge all wpr partials: merging the warp's own partial
                // twice would count its edges twice, and the two-smallest merge is only idempotent in
                // the multiset sense -- (m1,m2) merged with itself is (m1,m1), not (m1,m2).
                const uint32_t g = wid - (uint32_t)(wid % (int)wpr);
                unsigned long long gm1 = kDblMaxBits, gm2 = kDblMaxBits;
                int gcnt = 0;
                unsigned gxor = 0;
                for (uint32_t t = 0; t < wpr; ++t) {
                    const unsigned long long o1 = sh_m1[g + t], o2 = sh_m2[g + t];
                    const unsigned long long mx = (gm1 > o1) ? gm1 : o1;
                    const unsigned long long i2 = (gm2 < o2) ? gm2 : o2;
                    gm1 = (gm1 < o1) ? gm1 : o1;
                    gm2 = (mx < i2) ? mx : i2;
                    gcnt += sh_c[g + t];
                    gxor ^= (unsigned)sh_x[g + t];
                }
                row_write<RM, FIRST>(d, wpr, row, sub, lane, v0, v1, h0, h1, gm1, gm2, gcnt + (int)syn_bit(d, row));
                if constexpr (!FIRST) {
                    if (lane == 0 && (wid % (int)wpr) == 0) mis += (int)((gxor & 1u) != (unsigned)syn_bit(d, row));
                }
            }
            __syncthreads();
        }
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) mis += __shfl_xor_sync(0xffffffffu, mis, off);
    if (lane == 0) row_mis[threadIdx.x >> 5] = mis;
}

// The trailing pass: only the mismatch of the state the last column pass left (nothing reads the
// messages afterwards, so the check update is skipped).  Reads dec[] only.
__device__ __forceinline__ void mismatch_pass(const Dev &d, int *row_mis) {
    const int lane = threadIdx.x & 31;
    const int wid = (int)(threadIdx.x >> 5);
    const uint32_t nwarp = (gridDim.x * blockDim.x) >> 5;
    const uint32_t wpb = (uint32_t)(blockDim.x >> 5);
    const uint32_t wpr = (d.wpr < wpb) ? d.wpr : wpb;
    if (wpr == 1u) {
        const uint32_t gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        int tot = 0;   // a warp may own several rows when m exceeds the warp count: accumulate them
        for (uint32_t row = gwarp; row < d.m; row += nwarp) {
            const uint32_t b = d.rptr[row], e = d.rptr[row + 1];
            unsigned xorac = 0;
            for (uint32_t k = b + (uint32_t)lane; k < e; k += 32u) xorac ^= (unsigned)d.dec[d.rcol[k]];
            #pragma unroll
            for (int off = 16; off; off >>= 1) xorac ^= __shfl_xor_sync(0xffffffffu, xorac, off);
            if (lane == 0) tot += (int)((xorac & 1u) != (unsigned)syn_bit(d, row));
        }
        if (lane == 0) row_mis[threadIdx.x >> 5] = tot;
        return;
    }
    // wpr > 1: the same group-aligned block-uniform windows as the check pass.  A warp handles one row
    // slice per round, so the mismatch bits have to accumulate over the rounds in a register: assigning
    // row_mis[wid] every round would keep only the last round's rows.
    __shared__ int shx[32];
    const uint32_t wpg = (wpb / wpr) * wpr;
    const uint32_t total = d.m * wpr;
    const bool leader = (wid % (int)wpr) == 0;
    int tot_mis = 0;
    for (uint32_t base = blockIdx.x * wpg; base < total; base += (uint32_t)gridDim.x * wpg) {
        const uint32_t w = base + (uint32_t)wid;
        unsigned v = 0;
        uint32_t row = 0;
        if (wid < (int)wpg && w < total) {
            row = w / wpr;
            const uint32_t sub = w - row * wpr;
            const uint32_t b = d.rptr[row], e = d.rptr[row + 1];
            for (uint32_t k = b + sub * 32u + (uint32_t)lane; k < e; k += wpr * 32u) {
                v ^= (unsigned)d.dec[d.rcol[k]];
            }
            #pragma unroll
            for (int off = 16; off; off >>= 1) v ^= __shfl_xor_sync(0xffffffffu, v, off);
        }
        if (lane == 0) shx[wid] = (int)v;
        __syncthreads();
        // A leader must own a real work unit of this round's window: with wpg < warps per block the
        // surplus warps (which may satisfy wid % wpr == 0) would otherwise merge stale shx slots of
        // a group that is not in this window and count mismatches that do not exist (found with
        // an equivalence test on a problem with max row degree 132: wpr = 3, wpg = 6, warp 6 read shx[6..8]).
        if (lane == 0 && leader && wid < (int)wpg && w < total) {
            // the row's parity is the XOR of its slices' parities; the mismatch compares it with the
            // row's syndrome bit (summing the slices' mismatch bits would not be the same thing)
            unsigned par = 0;
            const uint32_t g = wid - (uint32_t)(wid % (int)wpr);
            for (uint32_t t = 0; t < wpr; ++t) par ^= (unsigned)shx[g + t];
            if (w < total) tot_mis += (int)((par & 1u) != (unsigned)syn_bit(d, row));
        }
        __syncthreads();
    }
    if (lane == 0) row_mis[threadIdx.x >> 5] = leader ? tot_mis : 0;
}

// ---------------------------------------------------------------------------
// Row-summary mode (sum_mode): for rows too long for one warp's register path (max row degree > 64)
// the check pass writes the row's (m1, m2, total_sgn) into three arrays instead of scattering every
// edge's message; the column pass recomputes each edge's message from its row's summary and its own
// bit-to-check value, so no message array is written by the check pass at all.  Same arithmetic, same
// (m1, m2, total_sgn) values, same column order -- only where the values live changes.  Measured
// (standalone variant measurements): slower on the short-row problems
// (bb144 whole body 242.2 against 207.9 us, bb288 332.6 against 312.0) and faster where rows are long
// (bb144 xyz 693.9 against 944.4 us), because the check pass drops 438,912 scattered message stores
// per iteration and the column pass pays two cached summary loads per edge.  It therefore ships only
// where the rows are long (wpr > 1), decided at create.
// ---------------------------------------------------------------------------
template <int RM, bool FIRST>
__device__ __forceinline__ void row_sum_pass(const Dev &d, int *row_mis) {
    const int lane = threadIdx.x & 31;
    const uint32_t nwarp = (gridDim.x * blockDim.x) >> 5;
    const uint32_t gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int mis = 0;
    for (uint32_t row = gwarp; row < d.m; row += nwarp) {
        const uint32_t b = d.rptr[row], e = d.rptr[row + 1];
        unsigned long long m1 = kDblMaxBits, m2 = kDblMaxBits;
        int cnt = 0;
        unsigned xorac = 0;
        for (uint32_t k = b + (uint32_t)lane; k < e; k += 32u) {
            const double v = FIRST ? d.prior[d.rcol[k]] : d.b2c[d.redge[k]];
            if constexpr (!FIRST) xorac ^= (unsigned)d.dec[d.rcol[k]];
            cnt += (int)nonpositive(v);
            const unsigned long long a = mag_bits(v);
            if (a < m1) { m2 = m1; m1 = a; } else if (a < m2) m2 = a;
        }
        #pragma unroll
        for (int off = 16; off; off >>= 1) {
            const unsigned long long o1 = __shfl_xor_sync(0xffffffffu, m1, off);
            const unsigned long long o2 = __shfl_xor_sync(0xffffffffu, m2, off);
            const unsigned long long mx = (m1 > o1) ? m1 : o1;
            const unsigned long long i2 = (m2 < o2) ? m2 : o2;
            m1 = (m1 < o1) ? m1 : o1;
            m2 = (mx < i2) ? mx : i2;
            cnt += __shfl_xor_sync(0xffffffffu, cnt, off);
            xorac ^= __shfl_xor_sync(0xffffffffu, xorac, off);
        }
        if (lane == 0) {
            d.sm1[row] = m1;
            d.sm2[row] = m2;
            d.spar[row] = cnt + (int)syn_bit(d, row);
            if constexpr (!FIRST) mis += (int)((xorac & 1u) != (unsigned)syn_bit(d, row));
        }
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) mis += __shfl_xor_sync(0xffffffffu, mis, off);
    if (lane == 0) row_mis[threadIdx.x >> 5] = mis;
}

// Iteration 1 of the row-summary formulation.  Its (m1, m2) are functions of the priors
// alone, so they are computed once per handle by row_init_kernel() (dbench_create /
// dbench_set_priors, never in the decode path) and iteration 1 only has to publish them and add this
// shot's syndrome bit to the count:
//     spar[row] = spar0[row] + syn_bit(row).
// The whole row_sum_pass<FIRST> -- one gather of a prior per edge, plus two shuffles per row -- is
// replaced by three loads and three stores per row.  The mismatch bookkeeping is the same as
// FIRST's: row_mis stays zero (the reference tests convergence after the check pass, and iteration 1
// has no messages to test), so the caller's block reduction and the barrier after this pass are
// unchanged.
__device__ __forceinline__ void spar_pass(const Dev &d, int *row_mis) {
    const uint32_t nwarp = (gridDim.x * blockDim.x) >> 5;
    const uint32_t gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    for (uint32_t row = gwarp; row < d.m; row += nwarp) {
        if (lane == 0) {
            d.sm1[row] = d.sm1i[row];
            d.sm2[row] = d.sm2i[row];
            d.spar[row] = d.spar0[row] + (int)syn_bit(d, row);
        }
    }
    if (lane == 0) row_mis[threadIdx.x >> 5] = 0;
}

// One thread per row, run once per handle after the priors are uploaded (dbench_create and
// dbench_set_priors, never in the decode path): the row minimum, the second minimum (as magnitude
// bits) and the count of non-positive priors, in whichever sequential order the thread walks the
// row.  All three are order-free reductions -- integer min, second min and a count -- so the values
// are bit-identical to row_sum_pass<FIRST>'s warp reduction of the same multiset.
__global__ void row_init_kernel(const double *prior, const uint32_t *rptr, const uint32_t *rcol,
                                unsigned long long *sm1i, unsigned long long *sm2i, int *spar0,
                                uint32_t m) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    const uint32_t b = rptr[row], e = rptr[row + 1];
    unsigned long long m1 = kDblMaxBits, m2 = kDblMaxBits;
    int cnt = 0;
    for (uint32_t k = b; k < e; ++k) {
        const double v = prior[rcol[k]];
        cnt += (int)nonpositive(v);
        const unsigned long long a = mag_bits(v);
        if (a < m1) { m2 = m1; m1 = a; } else if (a < m2) m2 = a;
    }
    sm1i[row] = m1;
    sm2i[row] = m2;
    spar0[row] = cnt;
}

// Fill the position-ordered prior array at create and in set_priors.
// One 8-byte copy per column, on the handle's stream, ordered after the prior upload and after the
// layout's cperm upload.  Not on the decode path: the priors change only in these two calls.
__global__ void prior_pos_kernel(const double *prior, const uint32_t *cperm, double *prior_pos, uint32_t n) {
    const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n) prior_pos[p] = prior[cperm[p]];
}

// Interleaved variant (create-time layout, see GpuState::ilv and ilv_slot0): the column at
// position p holds its edges at slots ilv_slot0(p) + 32*i, i = 0..deg-1, in the reference's row
// order (rind and redge are permuted with the layout).  Same arithmetic, same per-column order,
// same per-edge values; only the thread->column map and the addresses move.  Only reached when
// every column fits the register path (create-time rule), so there is no generic branch.
template <int RM, bool FIRST>
__device__ __forceinline__ void col_sum_pass_ilv(const Dev &d) {
    const uint32_t c0 = col_begin(d), c1 = col_end(d);
    for (uint32_t p = c0 + threadIdx.x; p < c1; p += blockDim.x) {
        const uint32_t s0 = ilv_slot0(p);
        const uint32_t deg = d.cdeg[p];
        const uint32_t col = d.cperm[p];
        double t = d.prior_pos[p];   // the prior in position order (same bits)
        double cb[kRegDeg];
        double pre[kRegDeg];
        uint32_t s = s0;
        #pragma unroll
        for (uint32_t i = 0; i < kRegDeg; ++i) {
            if (i >= deg) break;
            const uint32_t row = d.rind[s];
            const double v = FIRST ? d.prior_pos[p] : d.b2c[s];
            const unsigned long long base_bits = (mag_bits(v) == d.sm1[row]) ? d.sm2[row] : d.sm1[row];
            const double salpha = ((d.spar[row] + (int)nonpositive(v)) & 1) ? -d.alpha : d.alpha;
            cb[i] = dmul<RM>(__longlong_as_double(base_bits), salpha);
            pre[i] = t;
            t = dadd<RM>(t, cb[i]);
            s += 32u;
        }
        d.post[col] = t;
        d.dec[col] = (uint8_t)dec_bit(t);
        double t2 = 0.0;  // +0.0, exactly bp.hpp's part 2
        #pragma unroll
        for (uint32_t i = kRegDeg; i-- > 0;) {
            if (i >= deg) continue;
            const uint32_t si = s0 + 32u * i;
            d.b2c[si] = dadd<RM>(pre[i], t2);
            t2 = dadd<RM>(t2, cb[i]);
        }
    }
}

template <int RM, bool FIRST, bool ILV>
__device__ __forceinline__ void col_sum_pass(const Dev &d) {
    if constexpr (ILV) { col_sum_pass_ilv<RM, FIRST>(d); return; }
    const uint32_t per_block = (d.n + gridDim.x - 1u) / gridDim.x;
    const uint32_t c0 = blockIdx.x * per_block;
    const uint32_t c1 = (c0 + per_block < d.n) ? (c0 + per_block) : d.n;
    for (uint32_t col = c0 + threadIdx.x; col < c1; col += blockDim.x) {
        const uint32_t b = d.cptr[col], e = d.cptr[col + 1];
        const uint32_t deg = e - b;
        double t = d.prior[col];
        if (deg <= kRegDeg) {
            double cb[kRegDeg];
            double pre[kRegDeg];
            #pragma unroll
            for (uint32_t i = 0; i < kRegDeg; ++i) {
                if (i >= deg) break;
                const uint32_t k = b + i, row = d.rind[k];
                const double v = FIRST ? d.prior[col] : d.b2c[k];
                const unsigned long long base_bits = (mag_bits(v) == d.sm1[row]) ? d.sm2[row] : d.sm1[row];
                const double salpha = ((d.spar[row] + (int)nonpositive(v)) & 1) ? -d.alpha : d.alpha;
                cb[i] = dmul<RM>(__longlong_as_double(base_bits), salpha);
                pre[i] = t;
                t = dadd<RM>(t, cb[i]);
            }
            d.post[col] = t;
            d.dec[col] = (uint8_t)dec_bit(t);
            double t2 = 0.0;  // +0.0, exactly bp.hpp's part 2
            #pragma unroll
            for (uint32_t i = kRegDeg; i-- > 0;) {
                if (i >= deg) continue;
                d.b2c[b + i] = dadd<RM>(pre[i], t2);
                t2 = dadd<RM>(t2, cb[i]);
            }
        } else {
            for (uint32_t k = b; k < e; ++k) {
                const uint32_t row = d.rind[k];
                const double v = FIRST ? d.prior[col] : d.b2c[k];
                const unsigned long long base_bits =
                    (mag_bits(v) == d.sm1[row]) ? d.sm2[row] : d.sm1[row];
                const double salpha = ((d.spar[row] + (int)nonpositive(v)) & 1) ? -d.alpha : d.alpha;
                const double m = dmul<RM>(__longlong_as_double(base_bits), salpha);
                d.c2b[k] = m;
                d.b2c[k] = t;
                t = dadd<RM>(t, m);
            }
            d.post[col] = t;
            d.dec[col] = (uint8_t)dec_bit(t);
            double t2 = 0.0;
            for (uint32_t k = e; k-- > b;) {
                d.b2c[k] = dadd<RM>(d.b2c[k], t2);
                t2 = dadd<RM>(t2, d.c2b[k]);
            }
        }
    }
}

// Variable update: one thread per column, the reference's addition order.  Part 1 reads the edge's
// message out of b2c and keeps it in registers (degree <= kRegDeg); part 2 adds the running suffix
// and writes the new bit-to-check value.  With kDualReg the part-1 prefix values are held in
// registers too, so an edge costs one load and one store in total (part 2 writes dadd(prefix, t2)
// directly instead of loading the value part 1 stored and adding to it); the association is the
// reference's in both cases.
constexpr bool kDualReg = true;

// The columns are spread over every block (a contiguous chunk per block, threads striding inside
// it): with the flat grid-stride mapping only ceil(n / blockDim.x) blocks had any column work at all
// (41 of 117 on bb144), and the pass is latency-bound, so the blocks that did have work had too few
// loads in flight.
// Interleaved variant of col_pass (same layout as col_sum_pass_ilv): part 1 reads the edge's
// message out of the interleaved slot, part 2 writes it back; one load and one store per edge.
template <int RM>
__device__ __forceinline__ void col_pass_ilv(const Dev &d) {
    const uint32_t c0 = col_begin(d), c1 = col_end(d);
    for (uint32_t p = c0 + threadIdx.x; p < c1; p += blockDim.x) {
        const uint32_t s0 = ilv_slot0(p);
        const uint32_t deg = d.cdeg[p];
        const uint32_t col = d.cperm[p];
        double t = d.prior_pos[p];   // the prior in position order (same bits)
        double msg[kRegDeg];
        double pre[kRegDeg];
        uint32_t s = s0;
        #pragma unroll
        for (uint32_t i = 0; i < kRegDeg; ++i) {
            if (i >= deg) break;
            const double m = d.b2c[s];
            msg[i] = m;
            pre[i] = t;
            t = dadd<RM>(t, m);
            s += 32u;
        }
        d.post[col] = t;
        d.dec[col] = (uint8_t)dec_bit(t);
        double t2 = 0.0;  // +0.0, exactly bp.hpp's part 2
        #pragma unroll
        for (uint32_t i = kRegDeg; i-- > 0;) {
            if (i >= deg) continue;
            d.b2c[s0 + 32u * i] = dadd<RM>(pre[i], t2);
            t2 = dadd<RM>(t2, msg[i]);
        }
    }
}

template <int RM, bool ILV>
__device__ __forceinline__ void col_pass(const Dev &d) {
    if constexpr (ILV) { col_pass_ilv<RM>(d); return; }
    const uint32_t per_block = (d.n + gridDim.x - 1u) / gridDim.x;
    const uint32_t c0 = blockIdx.x * per_block;
    const uint32_t c1 = (c0 + per_block < d.n) ? (c0 + per_block) : d.n;
    for (uint32_t col = c0 + threadIdx.x; col < c1; col += blockDim.x) {
        const uint32_t b = d.cptr[col], e = d.cptr[col + 1];
        const uint32_t deg = e - b;
        double t = d.prior[col];
        if (deg <= kRegDeg) {
            double msg[kRegDeg];
            double pre[kRegDeg];
            #pragma unroll
            for (uint32_t i = 0; i < kRegDeg; ++i) {
                if (i >= deg) break;
                const double m = d.b2c[b + i];
                msg[i] = m;
                pre[i] = t;
                t = dadd<RM>(t, m);
            }
            d.post[col] = t;
            d.dec[col] = (uint8_t)dec_bit(t);
            double t2 = 0.0;  // +0.0, exactly bp.hpp's part 2
            #pragma unroll
            for (uint32_t i = kRegDeg; i-- > 0;) {
                if (i >= deg) continue;
                if constexpr (kDualReg) d.b2c[b + i] = dadd<RM>(pre[i], t2);
                else d.b2c[b + i] = dadd<RM>(d.b2c[b + i], t2);
                t2 = dadd<RM>(t2, msg[i]);
            }
        } else {
            for (uint32_t k = b; k < e; ++k) {
                const double msg = d.b2c[k];
                d.c2b[k] = msg;
                d.b2c[k] = t;
                t = dadd<RM>(t, msg);
            }
            d.post[col] = t;
            d.dec[col] = (uint8_t)dec_bit(t);
            double t2 = 0.0;
            for (uint32_t k = e; k-- > b;) {
                d.b2c[k] = dadd<RM>(d.b2c[k], t2);
                t2 = dadd<RM>(t2, d.c2b[k]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The fused single-barrier iteration (ported into the interleaved-layout kernels): the check pass
// is gone.  The column pass of a block accumulates, for every edge, the value it has just written to
// b2c as a partial of that edge's row (the two smallest magnitudes by two atomicMin, the count of
// nonpositive values, the decision parity); the partials are published, ONE grid barrier follows, and
// the merge turns them into the shared summary the next iteration's column pass reads (each block
// merges the rows its own columns touch, from every block that owns one of them).  The values are the
// ones the shipped check pass would have read: m1/m2 of the bit-to-check value, its nonpositiveness
// and the decision parity are all order-free, and the 2-smallest merge of the partials is exact.
// ---------------------------------------------------------------------------
struct FusedSh {
    unsigned long long *pa_m1, *pa_m2, *sm_m1, *sm_m2;
    unsigned *pa_sign, *sm_cnt, *sm_sign, *row2slot;   // pa_cnt is gone (never read)
    // Under RN / RZ the interleaved RSLOT kernel's row
    // summary is one 32 B record per row -- {m1 (raw bits), dmul<RM>(m1, +alpha) ^ par << 63,
    // dmul<RM>(m2, +alpha) ^ par << 63, 0} with par the row's count parity -- written by the merge (and by
    // SHSUM's iteration-1 staging) as two 16 B stores and read by the column pass as two 16 B loads issued
    // together; the pass tests the raw m1 and selects the product in registers, then flips its sign bit
    // for a nonpositive edge value.  Exact: RN and RZ round sign-symmetrically, so dmul(m, -alpha) ==
    // -dmul(m, alpha) bit for bit, and the two flips are exactly the base's salpha choice
    // ((spar + nonpositive(v)) & 1).  sm_m1 / sm_m2 / sm_cnt / sm_sign are not allocated in that kernel
    // (sm_sign was written and never read).  RU / RD and every other kernel keep this layout and the
    // per-edge DMUL.
    unsigned long long *sm_rec;
};

// Whether an instantiation uses the 32 B record (the interleaved, shared-summary, RSLOT
// kernel under RN / RZ: the shipped preset kernel).
template <bool ILV, bool SHSUM, bool RSLOT, int RM>
__device__ __forceinline__ constexpr bool fused_rec() {
    return ILV && SHSUM && RSLOT && (RM == kRN || RM == kRZ);
}

template <bool ILV, bool SHSUM, bool RSLOT, int RM>
__device__ __forceinline__ FusedSh fused_shared(const Dev &d, unsigned long long *sh) {
    constexpr bool REC = fused_rec<ILV, SHSUM, RSLOT, RM>();
    FusedSh f{};
    const uint32_t ns = d.ns_max;
    unsigned long long *p = sh;
    f.pa_m1 = p; p += ns;
    f.pa_m2 = p; p += ns;
    if constexpr (REC) {
        f.sm_rec = p; p += 4u * ns;   // 32 B per row; 16 B aligned (2 x ns x 8 B precede it)
    } else {
        f.sm_m1 = p; p += ns;
        f.sm_m2 = p; p += ns;
    }
    unsigned *q = (unsigned *)p;
    f.pa_sign = q; q += ns;
    if constexpr (!REC) {
        f.sm_cnt = q; q += ns;
        f.sm_sign = q; q += ns;
    }
    // A RSLOT kernel neither stages nor reads row2slot (both parts of its column pass take
    // the row slot from the staged rslot table), so no space is reserved for it there.
    if constexpr (RSLOT) f.row2slot = nullptr;
    else f.row2slot = q;
    return f;
}

// Where the block stages its slice of the create-time slot -> row-slot
// table (unsigned char, one byte per ILV slot of the block's padded range).  It sits after the fused
// tables; the offset matches the host's fused_shbytes computation byte for byte (fused_rslot_offset).
// The tables are 4 x 8 B + 3 x 4 B per row slot (pa_cnt gone) + the m-entry
// row2slot table, rounded up to 16 B so the stages behind it are uint4 copies.  52 B per
// row slot in the record kernel (2 x 8 B partials + the 32 B record + pa_sign); the host sizes the RSLOT
// variant's dynamic shared memory for 52 B (the RU / RD instantiations of that variant use 44 of it).
// The m-entry row2slot table is reserved only in kernels that use it (not RSLOT ones).
template <bool ILV, bool SHSUM, bool RSLOT, int RM>
__device__ __forceinline__ size_t fused_rslot_offset(const Dev &d) {
    constexpr size_t per_row = fused_rec<ILV, SHSUM, RSLOT, RM>() ? 52u : 44u;
    const size_t r2s = RSLOT ? 0u : (size_t)d.m * 4u;
    return ((size_t)d.ns_max * per_row + r2s + 15u) & ~(size_t)15u;
}
template <bool ILV, bool SHSUM, bool RSLOT, int RM>
__device__ __forceinline__ uint8_t *fused_rslot(const Dev &d, unsigned long long *sh) {
    return (uint8_t *)((char *)sh + fused_rslot_offset<ILV, SHSUM, RSLOT, RM>(d));
}

// Where the block stages its merge metadata (the rows
// of its slots only), right after the rslot stage (d.rslot_bytes = 0 in a kernel without one).  Both offsets
// are 16-byte aligned (fused_rslot_offset rounds up; rslot_bytes is a multiple of 288 = 18 x 16), and
// every record is padded to four u32, so the stage is a uint4 copy.  Matches the host's fused_shbytes.
template <bool ILV, bool SHSUM, bool RSLOT, int RM>
__device__ __forceinline__ uint32_t *fused_meta(const Dev &d, unsigned long long *sh) {
    return (uint32_t *)((char *)sh + fused_rslot_offset<ILV, SHSUM, RSLOT, RM>(d) + (size_t)d.rslot_bytes);
}

// The column pass of the fused schedule: the shipped arithmetic, the summary from shared memory
// (the global arrays on the FIRST iteration, whose summary comes from the priors' check pass) and the
// accumulation of the value the next iteration's summary must hold.
template <int RM, bool FIRST>
__device__ __forceinline__ void col_fused_pass(const Dev &d, const FusedSh &f,
                                               const uint32_t *rlist, uint32_t ns, uint32_t base,
                                               uint32_t pbuf, bool last, const uint32_t *rows_sh,
                                               uint32_t am_off, double *post, uint8_t *dec) {
    (void)rlist; (void)ns; (void)base; (void)pbuf;
    const uint32_t per_block = (d.n + gridDim.x - 1u) / gridDim.x;
    const uint32_t c0 = blockIdx.x * per_block;
    const uint32_t c1 = (c0 + per_block < d.n) ? (c0 + per_block) : d.n;
    for (uint32_t col = c0 + threadIdx.x; col < c1; col += blockDim.x) {
        const uint32_t b = d.cptr[col], e = d.cptr[col + 1];
        const uint32_t deg = e - b;
        double t = d.prior[col];
        if (deg <= kRegDeg) {
            double cb[kRegDeg], pre[kRegDeg];
            uint32_t rowid[kRegDeg];
            #pragma unroll
            for (uint32_t i = 0; i < kRegDeg; ++i) {
                if (i >= deg) break;
                const uint32_t k = b + i, row = d.rind[k];
                rowid[i] = row;
                const uint32_t s = FIRST ? 0u : f.row2slot[row];
                const double v = FIRST ? d.prior[col] : d.b2c[k];
                // FIRST reads the priors' own summary (row_init_kernel's arrays,
                // one value per row for the whole handle) instead of the copy spar_pass used to make:
                // `spar` is the same expression spar_pass wrote, evaluated here per edge.  No cross-block
                // write sits between the prologue and this pass any more.
                const unsigned long long m1 = FIRST ? d.sm1i[row] : f.sm_m1[s];
                const unsigned long long m2 = FIRST ? d.sm2i[row] : f.sm_m2[s];
                const int spar = FIRST ? (d.spar0[row] + (int)syn_bit(d, row)) : (int)f.sm_cnt[s];
                const unsigned long long base_bits = (mag_bits(v) == m1) ? m2 : m1;
                const double salpha = ((spar + (int)nonpositive(v)) & 1) ? -d.alpha : d.alpha;
                cb[i] = dmul<RM>(__longlong_as_double(base_bits), salpha);
                pre[i] = t;
                t = dadd<RM>(t, cb[i]);
            }
            post[col] = t;
            const uint8_t dc = (uint8_t)dec_bit(t);
            dec[col] = dc;
            double t2 = 0.0;  // +0.0, exactly bp.hpp's part 2
            #pragma unroll
            for (uint32_t i = kRegDeg; i-- > 0;) {
                if (i >= deg) continue;
                const uint32_t s = f.row2slot[rowid[i]];   // the summary being built for the next iteration
                if (!last) {
                    const double nv = dadd<RM>(pre[i], t2);
                    d.b2c[b + i] = nv;
                    t2 = dadd<RM>(t2, cb[i]);
                    const unsigned long long a = mag_bits(nv);
                    const unsigned long long old1 = atomicMin(&f.pa_m1[s], a);
                    atomicMin(&f.pa_m2[s], (a <= old1) ? old1 : a);
                    // the count is kept as its parity in bit 1 of the sign word: the column pass reads
                    // only (spar + nonpositive(v)) & 1, and the XOR of the parities is that sum's parity
                    atomicXor(&f.pa_sign[s], (unsigned)dc | ((unsigned)nonpositive(nv) << 1u));
                } else {
                    // The cap's last column
                    // pass -- the whole reverse pass is dead.  Its b2c values are read only by a later
                    // column pass, and of its partials only the decision parity (bit 0) is read, by the
                    // mismatch-only merge of this iteration.  The two FP64 additions, the b2c store, the
                    // minima and the count bit all go; the decision XOR is the same XOR over the same
                    // edges (order-free).
                    atomicXor(&f.pa_sign[s], (unsigned)dc);
                }
            }
        } else {
            for (uint32_t k = b; k < e; ++k) {
                const uint32_t row = d.rind[k];
                const uint32_t s = FIRST ? 0u : f.row2slot[row];
                const double v = FIRST ? d.prior[col] : d.b2c[k];
                // FIRST reads the priors' own summary (row_init_kernel's arrays,
                // one value per row for the whole handle) instead of the copy spar_pass used to make:
                // `spar` is the same expression spar_pass wrote, evaluated here per edge.  No cross-block
                // write sits between the prologue and this pass any more.
                const unsigned long long m1 = FIRST ? d.sm1i[row] : f.sm_m1[s];
                const unsigned long long m2 = FIRST ? d.sm2i[row] : f.sm_m2[s];
                const int spar = FIRST ? (d.spar0[row] + (int)syn_bit(d, row)) : (int)f.sm_cnt[s];
                const unsigned long long base_bits = (mag_bits(v) == m1) ? m2 : m1;
                const double salpha = ((spar + (int)nonpositive(v)) & 1) ? -d.alpha : d.alpha;
                const double m = dmul<RM>(__longlong_as_double(base_bits), salpha);
                d.c2b[k] = m;
                d.b2c[k] = t;
                t = dadd<RM>(t, m);
            }
            post[col] = t;
            const uint8_t dc = (uint8_t)dec_bit(t);
            dec[col] = dc;
            double t2 = 0.0;
            for (uint32_t k = e; k-- > b;) {
                const uint32_t row = d.rind[k];
                const uint32_t s = f.row2slot[row];
                if (!last) {
                    const double nv = dadd<RM>(d.b2c[k], t2);
                    d.b2c[k] = nv;
                    t2 = dadd<RM>(t2, d.c2b[k]);
                    const unsigned long long a = mag_bits(nv);
                    const unsigned long long old1 = atomicMin(&f.pa_m1[s], a);
                    atomicMin(&f.pa_m2[s], (a <= old1) ? old1 : a);
                    // the count is kept as its parity in bit 1 of the sign word: the column pass reads
                    // only (spar + nonpositive(v)) & 1, and the XOR of the parities is that sum's parity
                    atomicXor(&f.pa_sign[s], (unsigned)dc | ((unsigned)nonpositive(nv) << 1u));
                } else {
                    atomicXor(&f.pa_sign[s], (unsigned)dc);   // see the register path
                }
            }
        }
    }
    __syncthreads();
    // Push the block's partial pairs into the iteration's L2 row
    // accumulators instead of publishing them for every merging block to pull.  Per row slot of the
    // block:  old = atomicMin(am1[row], m1);  atomicMin(am2[row], min(m2, max(old, m1)));
    // atomicXor(asign[row], sign).  fused_merge carries the argument that the three words end at the
    // row's two smallest magnitudes (with multiplicity) and its sign / count parity.
    unsigned long long *a1 = d.am1 + am_off;
    unsigned long long *a2 = d.am2 + am_off;
    unsigned *asg = d.asign + am_off;
    if (!last) {
        for (uint32_t j = threadIdx.x; j < ns; j += blockDim.x) {
            const uint32_t r = rows_sh[j];
            const unsigned long long m1 = f.pa_m1[j], m2 = f.pa_m2[j];
            const unsigned long long old = atomicMin(&a1[r], m1);
            const unsigned long long mx = (old > m1) ? old : m1;
            atomicMin(&a2[r], (m2 < mx) ? m2 : mx);
            atomicXor(&asg[r], f.pa_sign[j]);
        }
    } else {
        // The last iteration's merge reads bit 0 of the sign word only, so the
        // last push is the sign XOR alone (no minima).
        for (uint32_t j = threadIdx.x; j < ns; j += blockDim.x) atomicXor(&asg[rows_sh[j]], f.pa_sign[j]);
    }
    __syncthreads();
}

// Interleaved variant of col_fused_pass (interleaved layout + fused schedule): identical arithmetic and the same
// per-column chain order as col_sum_pass_ilv (the column's edges at ilv_slot0(p) + 32*i, in increasing
// row order, one thread per column), plus the per-edge accumulation above.  The layout is taken only
// when every column degree fits the register path (create-time rule), so there is no general branch.
// SHSUM: iteration 1 takes its row summary out of the block's own
// shared tables (f.sm_*[js], the same slots the merge writes for it >= 2, filled before the loop from
// the row-init arrays sm1i/sm2i/spar0 + the shot's syndrome bit) instead of the three scattered global
// arrays d.sm1/d.sm2/d.spar.  Same values, same parity, same arithmetic -- only the copy the pass
// reads changes: the merge writes f.sm_cnt as ((gs >> 1) ^ syn) & 1 and spar_pass writes
// d.spar = spar0 + syn, and both are read as `spar + nonpositive(v)` & 1, whose value depends on the
// parity alone.
template <int RM, bool FIRST, bool SHSUM, bool RSLOT = false>
__device__ __forceinline__ void col_fused_pass_ilv(const Dev &d, const FusedSh &f,
                                                   const uint32_t *rlist, uint32_t ns, uint32_t base,
                                                   uint32_t pbuf, bool last, const uint32_t *rows_sh,
                                                   uint32_t am_off, double *post, uint8_t *dec,
                                                   const uint8_t *rslot_sh = nullptr,
                                                   uint32_t lbase = 0, uint32_t rslot_n = 0,
                                                   double *win = nullptr) {
    (void)rlist; (void)ns; (void)base; (void)rslot_n; (void)pbuf; (void)win;
    constexpr bool REC = fused_rec<true, SHSUM, RSLOT, RM>();   // the 32 B row record
    const uint32_t c0 = col_begin(d), c1 = col_end(d);
    // The message array's origin, chosen once for the
    // whole pass.  With the window (d.win_rounds != 0: all kRegDeg edge rounds of every column of the block,
    // in the ILV slot order) it is shifted by the block's first slot, so the ILV slot index s is the same in
    // both cases: mb[s] is the window's double s - lbase or d.b2c[s].  Per block:
    // the block's slots are [lbase, lbase + rslot_n) with lbase = (col_begin >> 5) * 288 and rslot_n its own
    // ((col_end - col_begin + 31) >> 5) * 288 -- the RSLOT slice's extent, which the host sizes by the
    // largest block (rslot_bytes) -- so every mb[s] the pass touches lies inside the block's window.
    // The block range variables of the balanced layout are used.  The choice is grid-uniform (a
    // create-time value); the arithmetic, the per-column addition order and the values are the base's: only
    // the memory space of the message changes.  The window belongs to the RSLOT kernel only.
    double *mb = d.b2c;
    if constexpr (RSLOT) {
        if (d.win_rounds != 0u) mb = win - (size_t)lbase;
    }
    // The block's per-position row-slot bytes.  For the block's position
    // lp (= p - c0) the row slot of round i is byte i of the packed triple: bytes 0..3 in word lp of
    // w0, 4..7 in word lp of w1, and the ninth byte in b8[lp].  Three independent word loads give a
    // thread all kRegDeg row slots of its position at once, instead of kRegDeg byte loads 32 B apart
    // that each sit in front of their own record load.
    const uint32_t npos = c1 - c0;
    const uint32_t *const rslot_w0 = (const uint32_t *)rslot_sh;            // rounds 0..3
    const uint32_t *const rslot_w1 = rslot_w0 + npos;                       // rounds 4..7
    const uint8_t *const rslot_b8 = (const uint8_t *)(rslot_w0 + 2u * npos); // round 8
    for (uint32_t p = c0 + threadIdx.x; p < c1; p += blockDim.x) {
        const uint32_t s0 = ilv_slot0(p);
        const uint32_t deg = d.cdeg[p];
        const uint32_t col = d.cperm[p];
        const uint32_t lp = p - c0;
        const uint32_t jsw0 = RSLOT ? rslot_w0[lp] : 0u;
        const uint32_t jsw1 = RSLOT ? rslot_w1[lp] : 0u;
        const uint32_t jsb8 = RSLOT ? (uint32_t)rslot_b8[lp] : 0u;
        double t = d.prior_pos[p];   // the prior in position order (same bits)
        double cb[kRegDeg], pre[kRegDeg];
        uint32_t rowid[kRegDeg];
        uint32_t s = s0;
        #pragma unroll
        for (uint32_t i = 0; i < kRegDeg; ++i) {
            if (i >= deg) break;
            uint32_t row = 0, js = 0;
            if constexpr (RSLOT) {
                // The row slot comes straight out of the block's
                // staged table, one byte per slot, so the pass neither loads rind from L2 nor hops
                // through row2slot.  The byte is the same js the two-load path produced.
                // Round i's byte out of the packed triple of the position.
                js = (i < 4u) ? ((jsw0 >> (8u * i)) & 0xffu)
                   : (i < 8u) ? ((jsw1 >> (8u * (i - 4u))) & 0xffu) : jsb8;
                rowid[i] = js;
            } else {
                row = d.rind[s];
                rowid[i] = row;
                js = (SHSUM || !FIRST) ? f.row2slot[row] : 0u;
            }
            const double v = FIRST ? d.prior_pos[p] : mb[s];   // the window or d.b2c
            if constexpr (REC) {
                // Both 16 B halves of the row's record in flight at once;
                // the test is on the raw m1, the product (count parity already in its sign) is selected in
                // registers, and a nonpositive edge value flips its sign bit -- exactly the base's
                // dmul<RM>(base_bits, salpha) under RN / RZ (see FusedSh).
                const ulonglong2 ra = *reinterpret_cast<const ulonglong2 *>(f.sm_rec + 4u * js);
                const ulonglong2 rb = *reinterpret_cast<const ulonglong2 *>(f.sm_rec + 4u * js + 2u);
                const unsigned long long pbits = (mag_bits(v) == ra.x) ? rb.x : ra.y;
                cb[i] = __longlong_as_double(pbits ^ ((unsigned long long)nonpositive(v) << 63));
            } else {
            unsigned long long m1, m2; int spar;
            if constexpr (SHSUM || RSLOT || !FIRST) {
                m1 = f.sm_m1[js]; m2 = f.sm_m2[js]; spar = (int)f.sm_cnt[js];
            } else {
                // the row-init arrays, as col_fused_pass (no spar_pass in the fused body)
                m1 = d.sm1i[row]; m2 = d.sm2i[row]; spar = d.spar0[row] + (int)syn_bit(d, row);
            }
            const unsigned long long base_bits = (mag_bits(v) == m1) ? m2 : m1;
            const double salpha = ((spar + (int)nonpositive(v)) & 1) ? -d.alpha : d.alpha;
            cb[i] = dmul<RM>(__longlong_as_double(base_bits), salpha);
            }
            pre[i] = t;
            t = dadd<RM>(t, cb[i]);
            s += 32u;
        }
        post[col] = t;
        const uint8_t dc = (uint8_t)dec_bit(t);
        dec[col] = dc;
        double t2 = 0.0;  // +0.0, exactly bp.hpp's part 2
        #pragma unroll
        for (uint32_t i = kRegDeg; i-- > 0;) {
            if (i >= deg) continue;
            // the summary being built for the next iteration
            // The same packed triple, round i (part 2 re-reads it only if
            // the compiler does not keep the three words live; either way it is 3 word loads, not 9 bytes).
            const uint32_t js = RSLOT ? ((i < 4u) ? ((jsw0 >> (8u * i)) & 0xffu)
                                         : (i < 8u) ? ((jsw1 >> (8u * (i - 4u))) & 0xffu) : jsb8)
                                      : f.row2slot[rowid[i]];
            if (!last) {
                const double nv = dadd<RM>(pre[i], t2);
                mb[s0 + 32u * i] = nv;   // the same message, in the window or in d.b2c
                t2 = dadd<RM>(t2, cb[i]);
                const unsigned long long a = mag_bits(nv);
                const unsigned long long old1 = atomicMin(&f.pa_m1[js], a);
                atomicMin(&f.pa_m2[js], (a <= old1) ? old1 : a);
                atomicXor(&f.pa_sign[js], (unsigned)dc | ((unsigned)nonpositive(nv) << 1u));
            } else {
                atomicXor(&f.pa_sign[js], (unsigned)dc);   // see col_fused_pass
            }
        }
    }
    __syncthreads();
    // Push the block's partial pairs into the iteration's L2 row
    // accumulators instead of publishing them for every merging block to pull (see col_fused_pass and
    // fused_merge for the argument).  The pushes of the iteration sit behind the grid barrier, so a
    // block whose push runs long is waited for once, at the merge, like the publish it replaces.
    unsigned long long *a1 = d.am1 + am_off;
    unsigned long long *a2 = d.am2 + am_off;
    unsigned *asg = d.asign + am_off;
    if (!last) {
        for (uint32_t j = threadIdx.x; j < ns; j += blockDim.x) {
            const uint32_t r = rows_sh[j];
            const unsigned long long m1 = f.pa_m1[j], m2 = f.pa_m2[j];
            const unsigned long long old = atomicMin(&a1[r], m1);
            const unsigned long long mx = (old > m1) ? old : m1;
            atomicMin(&a2[r], (m2 < mx) ? m2 : mx);
            atomicXor(&asg[r], f.pa_sign[j]);
        }
    } else {
        // The last iteration's merge reads bit 0 of the sign word only, so the
        // last push is the sign XOR alone (no minima).
        for (uint32_t j = threadIdx.x; j < ns; j += blockDim.x) atomicXor(&asg[rows_sh[j]], f.pa_sign[j]);
    }
    __syncthreads();
}

// The deferred exit test needs no restore copy.  The column pass of iteration
// it writes post / dec into buffer it & 1, so when the test at iteration it or in the tail judges the
// state of it - 2, that state is still in the other buffer, untouched; the status word names the
// iteration whose state is final and the host (and the prefix pass) read that buffer's parity.

// Reset one accumulator buffer to the identity, grid-strided over the rows.  The merge of
// iteration it resets the buffer iteration it-1 used ((R + it + 2) % 3 = (R + it - 1) % 3): every block
// finished its merge of it-1 (the buffer's only reader) before it reached the barrier of iteration it,
// which precedes this merge, so no read of it is outstanding; and its next pushes are iteration it+2's
// (this shot) or the next shot's iterations 1 / 2, both ordered after this store (the barrier of it+1,
// resp. the kernel boundary).  Done in every merge, the last included (an earlier form skipped the resets
// whose buffer this shot does not use again and initialised buffers 1 / 2 in the prologue instead, which raced).
__device__ __forceinline__ void accum_reset(const Dev &d, uint32_t am_next) {
    unsigned long long *n1 = d.am1 + am_next;
    unsigned long long *n2 = d.am2 + am_next;
    unsigned *nsg = d.asign + am_next;
    for (uint32_t r = blockIdx.x * blockDim.x + threadIdx.x; r < d.m; r += gridDim.x * blockDim.x) {
        n1[r] = kDblMaxBits; n2[r] = kDblMaxBits; nsg[r] = 0u;
    }
}

// The merge: each block turns its own rows' summaries into the shared table the next column pass
// reads, and counts the rows whose decision parity does not match the syndrome (the mismatch; a row
// is merged by every block that touches it, which is a multiple of the true count -- 0 iff 0; the
// count is read only as a zero test, and the push leaves it bit-identical: the same blocks merge the
// row and read the same parity).
//
// The row's summary is not pulled from every owner
// block's published partial pair (an owner loop over the staged metadata) -- it is read as three
// words from the iteration's L2 accumulator, into which every owner block pushed its own pair at publish
// (col_fused_pass / col_fused_pass_ilv).  Exactness: let the owner blocks hold pairs (a_i, b_i), a_i <=
// b_i (kDblMaxBits when fewer than two edges).  The atomicMin on am1 serialises the owners: owner i
// gets old_i = the minimum of the a's pushed before it and contributes c_i = min(b_i, max(old_i, a_i))
// to am2.  am1 = min a_i = v1, and min c_i = v2, the second smallest of the multiset {a_i} u {b_i}
// (multiplicity included): no c_i is below v2 (max(old_i, a_i) < v2 forces it to equal v1 = a_i, so v1
// has multiplicity 2 and v2 = v1), and some owner reaches v2 in either push order, whether v2 is an a_k
// or the b_k of v1's own block.  The comparisons are the unsigned order of the 63 magnitude bits
// (merge_pair's): DBL_MAX bits are the identity, +inf / NaN never enter.  asign is the XOR of the
// owners' pa_sign words (bit 0 the decision parity, bit 1 the count parity): order-free.
template <int RM, bool REC>
__device__ __forceinline__ void fused_merge(const Dev &d, FusedSh &f, uint32_t ns, const uint32_t *meta_sh,
                                            int *row_mis, uint32_t am_off, uint32_t am_next, bool last) {
    int mis = 0;
    const uint32_t *rows = meta_sh;               // ns: the row of each block-local slot
    if (last) {
        // The last iteration's merge.  Nothing
        // reads the row summary it would build (no later column pass exists; the prefix pass reads only
        // post / dec), and the convergence test reads only the decision parity: read the row's pushed
        // sign word (bit 0, the only bit the last column pass pushes) and count the mismatch; no minima,
        // no sm_* stores, no partial reset (the next shot's prologue re-initialises pa_* for every slot).
        for (uint32_t j = threadIdx.x; j < ns; j += blockDim.x) {
            const uint32_t row = rows[j];
            mis += (int)((d.asign[am_off + row] & 1u) != syn_bit(d, row));
        }
        test_delay_prologue(11u);  // test-only: the accumulator reset
        accum_reset(d, am_next);   // the rotation needs the last merge's reset too
    } else {
    for (uint32_t j = threadIdx.x; j < ns; j += blockDim.x) {
        const uint32_t row = rows[j];
        const unsigned long long gm1 = d.am1[am_off + row], gm2 = d.am2[am_off + row];
        const unsigned gs = d.asign[am_off + row];
        const unsigned syn = syn_bit(d, row);
        if constexpr (REC) {
            // The next column pass's operands as
            // one 32 B record, two 16 B stores; the count parity pre-applied to both products' sign bits.
            const unsigned long long par = (unsigned long long)(((gs >> 1) ^ syn) & 1u) << 63;
            ulonglong2 *r = reinterpret_cast<ulonglong2 *>(f.sm_rec + 4u * j);
            r[0] = make_ulonglong2(gm1, (unsigned long long)__double_as_longlong(
                                            dmul<RM>(__longlong_as_double(gm1), d.alpha)) ^ par);
            r[1] = make_ulonglong2((unsigned long long)__double_as_longlong(
                                       dmul<RM>(__longlong_as_double(gm2), d.alpha)) ^ par, 0ull);
        } else {
            f.sm_m1[j] = gm1; f.sm_m2[j] = gm2;
            f.sm_cnt[j] = ((gs >> 1) ^ syn) & 1u;
            f.sm_sign[j] = gs;
        }
        mis += (int)((gs & 1u) != syn);
    }
    __syncthreads();
    for (uint32_t j = threadIdx.x; j < d.ns_max; j += blockDim.x) {   // reset for the next iteration
        f.pa_m1[j] = kDblMaxBits; f.pa_m2[j] = kDblMaxBits; f.pa_sign[j] = 0;
    }
    test_delay_prologue(11u);   // test-only: the accumulator reset
    accum_reset(d, am_next);
    __syncthreads();
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) mis += __shfl_xor_sync(0xffffffffu, mis, off);
    if (lane_of() == 0) row_mis[warp_of()] = mis;
    // Every warp's count is in row_mis before the caller's thread 0 sums it: without this barrier
    // thread 0 could read the zero a slower warp had not overwritten yet, undercount the mismatch to
    // 0 and report a convergence that does not exist.
    __syncthreads();
}

// The fused body.  The convergence test of iteration it judges the state the column pass of it-2
// left, and the tail judges the states max_iter-1 and max_iter; every state 1..max_iter the shipped
// body can stop at is judged, at the earliest point bp.hpp would stop, so the status word, the
// correction and the posterior are the shipped body's.
template <int RM, bool ILV, bool SHSUM = false, bool RSLOT = false>
// The body returns whether the BP kernel's tail must run: 1 on a
// non-converged shot of an OSD handle (the same condition under which the prefix pass ran), 0 otherwise.
// The value is the same in every block (the convergence verdict is grid-uniform), so the tail's block
// barriers are reached by every block.
__device__ uint32_t bp_body_fused(const Dev &d) {
    __shared__ int row_mis[32];
    __shared__ uint32_t tot_sh;
    // The 32 B record is read and written with 16 B (ulonglong2) accesses at
    // 16 B-multiple offsets of this array, so the array's base must be 16 B-aligned; the declaration says
    // so instead of relying on the toolchain's placement of the dynamic shared base.
    extern __shared__ __align__(16) unsigned long long sh[];
    constexpr bool REC = fused_rec<ILV, SHSUM, RSLOT, RM>();   // the 32 B row record
    FusedSh f = fused_shared<ILV, SHSUM, RSLOT, RM>(d, sh);
    test_late_init();   // (directed test only; see test_late_init)
    // The block stages its slice of the create-time slot -> row-slot
    // table, one unsigned char per ILV slot (see fused_rslot).  Copied as 32-bit words: lbase is a
    // multiple of 4 (288 slots per warp group) and so is the slice length.
    uint8_t *rslot_sh = nullptr;
    uint32_t lbase = 0, rslot_n = 0;
    double *win = nullptr;   // the b2c window (RSLOT kernels; used only when d.win_rounds != 0)
    if constexpr (RSLOT) {
        rslot_sh = fused_rslot<ILV, SHSUM, RSLOT, RM>(d, sh);
        lbase = (col_begin(d) >> 5) * (32u * (uint32_t)kRegDeg);
        rslot_n = ((col_end(d) - col_begin(d) + 31u) >> 5) * (32u * (uint32_t)kRegDeg);
        win = (double *)((char *)sh + d.win_off);
    }
    const uint32_t base = d.rowbase[blockIdx.x];
    const uint32_t ns = d.rowbase[blockIdx.x + 1] - base;
    const uint32_t *rlist = d.rowlist + base;
    test_delay_prologue(4u);    // test-only: the block-0 reset below (block 0 is delayed here)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        d.mis[0] = 0u; d.mis[1] = 0u; d.mis[2] = 0u;
        d.status[0] = 0u; d.status[1] = 0u;
        // The clear of the OSD-0 prefix counter, as in bp_body -- a shot that
        // converges (and returns from the loop or the tail below) leaves "no prefix" behind.
        // The header's third word is the mapped hand-over's per-shot
        // completion counter (see osd_prefix_pass); cleared with the count and the sorted flag.
        if (d.use_osd != 0u) {
            d.ocnt[0] = 0u; d.ocnt[1] = 0u; d.ocnt[2] = 0u;
            // Rung two's collected set -- its count and its arrival
            // word, cleared before the tail's collect (the prefix pass's grid barrier orders this store
            // before the tail's atomicAdds).
            d.ocnt2[0] = 0u; d.ocnt2[1] = 0u;
        }
    }
    // Zero the histogram half the next shot pushes into (see osd_hist_prepare).
    if (d.use_osd != 0u) osd_hist_prepare(d);
    test_delay_prologue(5u);    // test-only: the partial tables
    for (uint32_t j = threadIdx.x; j < d.ns_max; j += blockDim.x) {
        f.pa_m1[j] = kDblMaxBits; f.pa_m2[j] = kDblMaxBits; f.pa_sign[j] = 0;
    }
    // The rotation (it replaces a prologue init of buffers 1 and 2, which raced with
    // other blocks' iteration-1 pushes: no grid barrier separates the prologue from iteration 1's publish
    // in the fused schedule).  Iteration it pushes into buffer (R + it) % 3, R = the handle's rotation word
    // d.mis[3].  Invariant at kernel entry: buffers (R + 1) % 3 and (R + 2) % 3 hold the identity
    // (create sets all three; every shot leaves its next two clean, below), ordered by the stream.  Every
    // thread reads R right before the iteration loop (below), before the barrier of
    // iteration 1; block 0 writes the next shot's R only after that barrier, so no block of this shot can
    // read the new value.
    // The row2slot table is read only by the
    // non-RSLOT fused passes (the RSLOT column pass takes the row slot from its own staged table, in both
    // parts), so a RSLOT kernel does not stage it.
    if constexpr (!RSLOT) {
        test_delay_prologue(6u);   // test-only: the row2slot staging
        for (uint32_t j = threadIdx.x; j < ns; j += blockDim.x) f.row2slot[rlist[j]] = j;
    }
    if constexpr (SHSUM) {
        test_delay_prologue(7u);   // test-only: the iteration-1 summary staging
        // The block's own row summaries of the priors, in the same
        // shared slots the merge later writes and in the same form (f.sm_cnt holds only the parity,
        // which is all the column pass reads).  This is spar_pass's content, staged where the
        // column pass wants it: iteration 1 then reads one shared table per edge instead of the three
        // scattered global arrays d.sm1/d.sm2/d.spar.
        for (uint32_t j = threadIdx.x; j < ns; j += blockDim.x) {
            const uint32_t row = rlist[j];
            if constexpr (REC) {
                // Iteration 1's record from the priors' summaries, the
                // count parity pre-applied to both products (see the merge).
                const unsigned long long m1 = d.sm1i[row], m2 = d.sm2i[row];
                const unsigned long long par =
                    (unsigned long long)(((unsigned)d.spar0[row] + syn_bit(d, row)) & 1u) << 63;
                ulonglong2 *r = reinterpret_cast<ulonglong2 *>(f.sm_rec + 4u * j);
                r[0] = make_ulonglong2(m1, (unsigned long long)__double_as_longlong(
                                               dmul<RM>(__longlong_as_double(m1), d.alpha)) ^ par);
                r[1] = make_ulonglong2((unsigned long long)__double_as_longlong(
                                           dmul<RM>(__longlong_as_double(m2), d.alpha)) ^ par, 0ull);
            } else {
                f.sm_m1[j] = d.sm1i[row];
                f.sm_m2[j] = d.sm2i[row];
                f.sm_cnt[j] = ((unsigned)d.spar0[row] + syn_bit(d, row)) & 1u;
            }
        }
    }
    test_delay_prologue(8u);    // test-only: the row-slot staging
    if (threadIdx.x < blockDim.x / 32) row_mis[threadIdx.x] = 0;
    if constexpr (RSLOT) {
        // The block stages its own per-position region -- kRegDeg bytes per
        // position of its column range, packed as [w0: npos u32 | w1: npos u32 | b8: npos bytes] -- so the
        // copy is `((c1 - c0) * kRegDeg)` bytes rounded up to 16 (the region is 16-B aligned and at least
        // that long; the padding bytes are never read).  A uint4 copy when both sides are 16-byte aligned
        // (the shared stage is aligned by fused_rslot_offset), with the u32 form as the fallback.
        const uint32_t npos = col_end(d) - col_begin(d);
        const uint32_t words = ((npos * (uint32_t)kRegDeg) + 15u) >> 4 << 2;
        const uint32_t *src = (const uint32_t *)(d.rslot + (size_t)blockIdx.x * (size_t)d.rslot_stride);
        uint32_t *dst = (uint32_t *)rslot_sh;
        if ((((uintptr_t)src | (uintptr_t)dst) & 15u) == 0u) {
            const uint4 *s4 = (const uint4 *)src;
            uint4 *d4 = (uint4 *)dst;
            const uint32_t n4 = words >> 2;
            #pragma unroll 2
            for (uint32_t w = threadIdx.x; w < n4; w += blockDim.x) d4[w] = s4[w];
            for (uint32_t w = (n4 << 2) + threadIdx.x; w < words; w += blockDim.x) dst[w] = src[w];
        } else {
            for (uint32_t j = threadIdx.x; j < words; j += blockDim.x) dst[j] = src[j];
        }
    }
    // Stage the block's merge metadata (create-time layout data)
    // before the first column pass, as a uint4 copy (the u32 form is the fallback).  A per-shot cost, not
    // a per-iteration one.  The record is the block's rows only (the push needs no owner list).
    uint32_t *meta_sh = fused_meta<ILV, SHSUM, RSLOT, RM>(d, sh);
    {
        test_delay_prologue(9u);   // test-only: the merge-metadata staging
        const uint32_t meta_o = d.meta_off[blockIdx.x];
        const uint32_t meta_cnt = d.meta_off[blockIdx.x + 1u] - meta_o;
        const uint32_t *msrc = d.merge_meta + meta_o;
        if ((((uintptr_t)msrc | (uintptr_t)meta_sh) & 15u) == 0u) {
            const uint4 *s4 = (const uint4 *)msrc;
            uint4 *d4 = (uint4 *)meta_sh;
            const uint32_t n4 = meta_cnt >> 2;
            #pragma unroll 2
            for (uint32_t w = threadIdx.x; w < n4; w += blockDim.x) d4[w] = s4[w];
            for (uint32_t w = (n4 << 2) + threadIdx.x; w < meta_cnt; w += blockDim.x) meta_sh[w] = msrc[w];
        } else {
            for (uint32_t w = threadIdx.x; w < meta_cnt; w += blockDim.x) meta_sh[w] = msrc[w];
        }
    }
    __syncthreads();
    // Every thread loads
    // this shot's rotation R itself, here, right before the loop.  Its first use is iteration 1's push, a
    // column pass later, so the L2 round trip is hidden; an earlier form had thread 0 load it at entry and store it
    // to shared at once, which stalled warp 0 an L2 round trip before its share of the prologue staging.  The
    // load is still program-ordered before this block's arrival at the barrier of iteration 1 (block 0 writes
    // R' only after leaving that barrier), so it reads the previous shot's value.
    const uint32_t am_rot = __ldcg((const unsigned *)&d.mis[3]) % 3u;

    // The exit test sits two iterations after the column pass whose state it judges (the merge of
    // iteration it-2 is ordered by iteration it-1's publish barrier), so there is one grid barrier per
    // iteration.  A converged shot restores post/dec from the scratch the column passes fill, so the
    // correction, the posterior and the status word stay the shipped body's; the only price is the two
    // extra column passes on the 5% of shots that converge.
    for (uint32_t it = 1; it <= d.max_iter; ++it) {
        const uint32_t mslot = it % 3u;
        const uint32_t pbuf = (it & 1u) * d.pm_stride;   // this iteration's partial buffer (unused by the push)
        // This iteration's accumulator buffer ((R + it) % 3), the buffer its merge resets
        // ((R + it + 2) % 3, the one iteration it-1 used: always, the last merge included, so a shot whose
        // last merge is L leaves (R + L + 1) % 3 and (R + L + 2) % 3 clean for the next shot, R' = R + L),
        // and the rows of the block's slots (meta_sh[0..ns), staged with the merge metadata -- the push
        // needs the row of every slot exactly as the merge does).
        const uint32_t am_off = ((am_rot + it) % 3u) * d.am_stride;
        const uint32_t am_next = ((am_rot + it + 2u) % 3u) * d.am_stride;
        // This iteration's post / dec buffer -- iteration it writes buffer it & 1.
        double *const post = (it & 1u) ? d.post2 : d.post;
        uint8_t *const dec = (it & 1u) ? d.dec2 : d.dec;
        if (threadIdx.x == 0) {
            tot_sh = (it >= 3u) ? __ldcg((const unsigned *)&d.mis[(it - 2u) % 3u]) : 1u;
            test_delay_prologue(10u);   // test-only: the per-iteration mismatch-counter reset
            if (blockIdx.x == 0) d.mis[mslot] = 0u;   // the slot this iteration's merge fills
        }
        __syncthreads();
        if (it >= 3u && tot_sh == 0u) {
            // converged after iteration it-2: its state is in this iteration's parity buffer, intact --
            // the pass of it-1 wrote the other one; no restore
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                d.status[0] = it - 2u;
                d.status[1] = 1u;
                d.mis[3] = (am_rot + it - 1u) % 3u;   // the last merge was it-1's (R' = R + L)
            }
            return 0u;   // converged: no prefix pass ran, no rung-two tail
        }
        // The cap is a run-time parameter (d.max_iter),
        // so the last iteration -- whose column pass and merge do only the mismatch -- is selected at run
        // time, with a grid-uniform branch, inside the instantiations the base already carries.
        const bool last = (it == d.max_iter);
        if (it == 1u) {
            // No prologue pass and no prologue grid barrier.
            // SHSUM staged iteration 1's summaries in the block's own shared tables before the loop;
            // the non-SHSUM passes read row_init_kernel's arrays (create / set_priors) directly.
            // Either way no cross-block write sits between the prologue and this column pass, and the
            // prologue's block barrier covers the dynamic shared tables the pass reads.  That
            // is why the push's buffers are not initialised in the prologue but left clean by the
            // previous shot -- the rotation.
            if constexpr (ILV) col_fused_pass_ilv<RM, true, SHSUM, RSLOT>(d, f, rlist, ns, base, pbuf, last,
                                                                          meta_sh, am_off, post, dec,
                                                                          rslot_sh, lbase, rslot_n, win);
            else               col_fused_pass<RM, true>(d, f, rlist, ns, base, pbuf, last, meta_sh, am_off, post, dec);
        } else {
            if constexpr (ILV) col_fused_pass_ilv<RM, false, SHSUM, RSLOT>(d, f, rlist, ns, base, pbuf, last,
                                                                           meta_sh, am_off, post, dec,
                                                                           rslot_sh, lbase, rslot_n, win);
            else               col_fused_pass<RM, false>(d, f, rlist, ns, base, pbuf, last, meta_sh, am_off, post, dec);
        }
        grid_barrier(d);                                        // the single barrier of the iteration
        if (threadIdx.x < blockDim.x / 32) row_mis[threadIdx.x] = 0;
        __syncthreads();
        fused_merge<RM, REC>(d, f, ns, meta_sh, row_mis, am_off, am_next, last);
        if (threadIdx.x == 0) {
            int tot = 0;
            const int nw = ((int)blockDim.x) >> 5;
            for (int w = 0; w < nw; ++w) tot += row_mis[w];
            if (tot != 0) atomicAdd(&d.mis[mslot], (unsigned)tot);
        }
    }
    // Every path from here on has merged iteration max_iter last (L = max_iter), so the
    // next shot's rotation is R + max_iter.  Written after the barrier of iteration 1 (every block has
    // read R), read only at the next kernel's entry (stream order).
    if (blockIdx.x == 0 && threadIdx.x == 0) d.mis[3] = (am_rot + d.max_iter) % 3u;

    // The two states the loop's exit test cannot judge: max_iter-1 and max_iter.  The loop's test at
    // iteration it reads the merge of it-2, so inside the loop the latest state judged is max_iter-2;
    // the shipped body's check pass at its iteration max_iter judges the state after the column pass
    // of max_iter-1 and returns (max_iter-1, converged) there, without ever running the last column
    // pass.  This body has run it, so if that state converged the correction and the posterior have to
    // be rolled back to it (the scratch still holds it: the column pass of max_iter saved it on entry),
    // and its answer takes precedence over the later state (bp.hpp returns at the earlier test).
    // prev_sh_s (the merge of max_iter-1) is already ordered by the loop's
    // last grid barrier: every block passed it after its merge of max_iter-1 (which precedes its column
    // pass of max_iter), so the max_iter-1 test needs no barrier of its own.
    __shared__ uint32_t prev_sh_s;
    if (threadIdx.x == 0) {
        prev_sh_s = (d.max_iter >= 2u) ? __ldcg((const unsigned *)&d.mis[(d.max_iter + 2u) % 3u]) : 1u;
    }
    __syncthreads();
    if (d.max_iter >= 2u && prev_sh_s == 0u) {
        // converged after iteration max_iter-1: its state is in the other parity buffer (the column
        // pass of max_iter wrote this one) and is intact; no restore
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            d.status[0] = d.max_iter - 1u;
            d.status[1] = 1u;
        }
        return 0u;   // converged: no prefix pass ran, no rung-two tail
    }
    // The merge of max_iter (the state bp.hpp tests after its last check pass)
    // fills d.mis[max_iter % 3], and its atomicAdds must be ordered by a grid barrier before the status
    // word is read.  On an OSD-0 handle that barrier is the prefix pass's own first barrier: the pass's
    // P1/P2 read only post (written before the iteration's barrier) and its verdict is read after the
    // barrier (osd_prefix_pass<true>).  Every block reaches it, and its early return on a converged shot
    // is grid-uniform (the same mis word in every block).  A bp-only handle keeps a trailing barrier of
    // its own.  The pass touches only its own arrays (okey, osamp, ohist, ocnt, ocand) and its own static
    // shared memory: none of the fused schedule's partials, dynamic shared tables or scratch.
    // Block 0 writes d.status after the pass; bp_kernel's publish reads it after the body, in block 0's
    // program order.
    uint32_t last_s;
    if (d.use_osd != 0u) {
        // The prefix pass reads the final state, which the last column pass left in buffer
        // max_iter & 1.
        last_s = osd_prefix_pass<true, RM>(d, (d.max_iter & 1u) ? d.post2 : d.post);
    } else {
        grid_barrier(d);
        __shared__ uint32_t last_sh_s;
        if (threadIdx.x == 0) last_sh_s = __ldcg((const unsigned *)&d.mis[d.max_iter % 3u]);
        __syncthreads();
        last_s = last_sh_s;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        d.status[0] = d.max_iter;
        d.status[1] = (last_s == 0u) ? 1u : 0u;
    }
    // Rung two's tail (see osd_tail_collect) runs on the shots the
    // prefix pass ran: every iteration done, not converged, an OSD handle.
    return (d.use_osd != 0u && last_s != 0u) ? 1u : 0u;
}

// ---------------------------------------------------------------------------
// The shot body.  Iteration 1 has no state to measure (the reference's convergence test runs after
// each check pass, never before the first one) and no bit-to-check values yet: the first check pass
// reads the priors directly and writes the first messages.  The last check update is followed by a
// mismatch-only pass, because bp.hpp tests convergence after the last part 1 as well.
// ---------------------------------------------------------------------------
template <int RM, bool SUM, bool ILV>
// The tail flag, exactly as bp_body_fused's (the pass's verdict is
// grid-uniform here too).
__device__ uint32_t bp_body(const Dev &d) {
    __shared__ int row_mis[32];
    __shared__ uint32_t tot_sh;
    // The late-start test hook applies here too.  Block 0's clear of mis[0..1] below has only a block
    // barrier before the other blocks' iteration-1 atomicAdd(mis[1]); benign as written (iteration 1's
    // row pass counts no mismatch, so no atomic is issued, and slot 1 is re-zeroed at the top of iteration
    // 2, ordered by iteration 1's barriers), and the delayed-start builds keep it that way.
    test_late_init();
    test_delay_prologue(16u);   // test-only: the block-0 reset below (block 0 is delayed here)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        d.mis[0] = 0u; d.mis[1] = 0u; d.status[0] = 0u; d.status[1] = 0u;
        // Cleared here, before anything runs, so a shot whose kernel returns early (BP converged) or
        // whose pass is skipped leaves no order prefix behind: the host reads the collect's own
        // counter, and 0 means "no prefix" exactly as the old oprefix_n = 0 did.
        // The header's third word is the mapped hand-over's per-shot
        // completion counter (see osd_prefix_pass); cleared with the count and the sorted flag.
        if (d.use_osd != 0u) {
            d.ocnt[0] = 0u; d.ocnt[1] = 0u; d.ocnt[2] = 0u;
            // Rung two's collected set -- its count and its arrival
            // word, cleared before the tail's collect (the prefix pass's grid barrier orders this store
            // before the tail's atomicAdds).
            d.ocnt2[0] = 0u; d.ocnt2[1] = 0u;
        }
    }
    // Zero the histogram half the next shot pushes into (see osd_hist_prepare).  This
    // body can return from inside the loop on a converged shot (the pass then never runs), so the
    // preparation cannot live in the pass.
    if (d.use_osd != 0u) osd_hist_prepare(d);
    __syncthreads();

    for (uint32_t it = 1; it <= d.max_iter; ++it) {
        const uint32_t slot = it & 1u;
        test_delay_prologue(17u);   // test-only: the per-iteration mismatch-counter reset
        if (blockIdx.x == 0 && threadIdx.x == 0) d.mis[slot ^ 1u] = 0u;
        test_delay_prologue(18u);   // test-only: the per-warp mismatch slot init
        if (threadIdx.x < blockDim.x / 32) row_mis[threadIdx.x] = 0;
        __syncthreads();
        if constexpr (SUM) {
            if (it == 1) spar_pass(d, row_mis);   // from the priors, at create / set_priors
            else         row_sum_pass<RM, false>(d, row_mis);
        } else {
            if (it == 1) row_pass<RM, true>(d, row_mis);
            else         row_pass<RM, false>(d, row_mis);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int tot = 0;
            const int nw = ((int)blockDim.x) >> 5;
            for (int w = 0; w < nw; ++w) tot += row_mis[w];
            if (tot != 0) atomicAdd(&d.mis[slot], (unsigned)tot);
        }
        grid_barrier(d);
        if (threadIdx.x == 0) tot_sh = __ldcg((const unsigned *)&d.mis[slot]);
        __syncthreads();
        if (tot_sh == 0u && it >= 2u) {
            // the state after iteration it-1 reproduces the syndrome: bp.hpp returns there
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                d.status[0] = it - 1u;
                d.status[1] = 1u;
            }
            return 0u;   // converged: no prefix pass ran, no rung-two tail
        }
        if constexpr (SUM) {
            if (it == 1) col_sum_pass<RM, true, ILV>(d);    // iteration 1's input is the priors
            else         col_sum_pass<RM, false, ILV>(d);
        } else {
            col_pass<RM, ILV>(d);
        }
        grid_barrier(d);
    }

    // The last state (after part 1 of max_iter) is what bp.hpp tests after the loop's last iteration.
    // The accumulation target is the slot the loop's last iteration zeroed, so the zeroing is already
    // ordered before the accumulation by that iteration's barriers.
    const uint32_t fslot = (d.max_iter + 1u) & 1u;
    test_delay_prologue(19u);   // test-only: the trailing mismatch pass's slot init
    if (threadIdx.x < blockDim.x / 32) row_mis[threadIdx.x] = 0;
    __syncthreads();
    mismatch_pass(d, row_mis);
    __syncthreads();
    if (threadIdx.x == 0) {
        int tot = 0;
        const int nw = ((int)blockDim.x) >> 5;
        for (int w = 0; w < nw; ++w) tot += row_mis[w];
        if (tot != 0) atomicAdd(&d.mis[fslot], (unsigned)tot);
    }
    grid_barrier(d);
    __shared__ uint32_t last_sh;
    if (threadIdx.x == 0) last_sh = __ldcg((const unsigned *)&d.mis[fslot]);
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        d.status[0] = d.max_iter;
        d.status[1] = (last_sh == 0u) ? 1u : 0u;
    }
    // On the shots that ran every iteration and did NOT converge, the device
    // also computes the front of the OSD-0 order.  The posteriors of the last column pass are already
    // visible (the loop's last grid barrier orders them before this point).  A bp-only handle
    // (use_osd == 0) skips it and pays nothing, and so does a shot that converged -- whether it
    // returned from the loop or ran every iteration and tested clean at the end (last_sh): the host
    // never takes the OSD-0 path for those, so the pass would be pure cost.  last_sh is the same in
    // every block (all of them load the same d.mis[fslot] after the same barrier), so the branch is
    // grid-uniform and the barriers inside the pass are still reached by every block.
    if (d.use_osd != 0u && last_sh != 0u) (void)osd_prefix_pass<false, RM>(d, d.post);
    // Rung two's tail (see osd_tail_collect), on the shots the pass ran.
    return (d.use_osd != 0u && last_sh != 0u) ? 1u : 0u;
}

// One instantiation per (rounding mode, message formulation): the formulation cannot change over a
// handle's life, and keeping it a compile-time parameter keeps each kernel's register budget and
// occupancy what they were measured at (the two paths in one kernel cost 166 registers against 80,
// which would halve the grid).  __launch_bounds__ pins the measured occupancy of each formulation:
// 3 blocks/SM at 256 threads in-place, 1 block of 512 per SM for the summary.
template <int RM, int SUM, bool ILV, int NT, bool SHSUM = false, bool RSLOT = false>
__global__ void __launch_bounds__(NT, SUM ? 1 : 3) bp_kernel(Dev d, unsigned long long shot_gen) {
#if defined(DBENCH_STAGE_PROBES) && DBENCH_STAGE_PROBES
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        d.pub[3] = t;   // t_dev0 (pub[2] is StatusCtl::gen2, shipped)
    }
#endif
    // The body's tail flag (grid-uniform, see the bodies).
    uint32_t tail;
    if constexpr (SUM == 1) tail = bp_body_fused<RM, ILV, SHSUM, RSLOT>(d);
    else tail = bp_body<RM, SUM != 0, ILV>(d);
    // The shot's status, straight into the handle's mapped status block (see StatusCtl).  Block 0
    // thread 0 wrote d.status itself after the body's last grid barrier (both formulations), so this
    // reads no other block's value, and the release store of the generation publishes both words at
    // system scope for the host's poll.  The prefix count stays in device memory: its collecting
    // blocks write it concurrently with block 0's body tail, and its 4 bytes come back with the
    // prefix buffer.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
#if defined(DBENCH_STAGE_PROBES) && DBENCH_STAGE_PROBES
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        d.pub[4] = t;   // t_dev1 (pub[3] is t_dev0; pub[2] is gen2)
#endif
        // Buffer 1 carries a copy of the status word, so the host's one-transfer
        // [dec | status] hand-off has the same shape whichever parity the shot's final state has.
        // The D2H copies are stream-ordered behind this kernel, so this store is visible to them;
        // nothing else reads it.  Null in a kernel without the parity double buffer.  (it sits outside
        // the publish-off guard, so the forced-missed-poll build keeps the hand-off's shape too.)
        if (d.status2 != nullptr) {
            d.status2[0] = d.status[0];
            d.status2[1] = d.status[1];
        }
#if !(defined(FASTBP_TEST_STATUS_PUBLISH_OFF) && FASTBP_TEST_STATUS_PUBLISH_OFF)
        // (the forced-missed-poll test build turns the publication off; never in a shipped build)
        d.pub[1] = (unsigned long long)d.status[0] | ((unsigned long long)d.status[1] << 32);
        st_release_sys_u64(&d.pub[0], shot_gen);
#endif
    }
    // The speculative rung two, in the kernel's tail.  It runs after
    // the status release above (block 0's program order), so the host's poll and its rung-one walk do
    // not wait for it; the tail's device time sits in the gap between the status publish and the
    // handle's next launch.
    if (tail != 0u && d.ocnt2 != nullptr && d.tail_on != 0u) osd_tail_collect<RM>(d, shot_gen);
}

// The kernel to launch for one (rounding mode, formulation, layout); the formulation and the layout
// are compile-time parameters so no variant carries another's code (the two paths in one kernel
// spilled 192 B of stack and cost the plain problems 11% per iteration).
// SUM: 0 = in-place (base), 1 = the fused single-barrier summary, 2 = the two-barrier summary.
// The fused interleaved kernel stages the prior summaries in its own shared tables (SHSUM): that is
// exact for every problem (same values, same parity) and only changes where iteration 1 reads them.
// RSLOT additionally stages the slot -> row-slot map (create-time, unsigned char; only where the map
// fits a byte, i.e. ns_max <= 255).
template <int RM>
const void *pick_kernel(int sum, bool ilv, bool rslot = false) {
    if (sum == 1) {
        if (ilv && rslot) return (const void *)bp_kernel<RM, 1, true, kThreadsSum, true, true>;
        return ilv ? (const void *)bp_kernel<RM, 1, true, kThreadsSum, true>
                   : (const void *)bp_kernel<RM, 1, false, kThreadsSum>;
    }
    if (sum == 2) {
        return ilv ? (const void *)bp_kernel<RM, 2, true, kThreadsSum>
                   : (const void *)bp_kernel<RM, 2, false, kThreadsSum>;
    }
    return ilv ? (const void *)bp_kernel<RM, 0, true, kThreads>
               : (const void *)bp_kernel<RM, 0, false, kThreads>;
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
#define CK(x)                                                                          \
    do {                                                                               \
        cudaError_t e_ = (x);                                                          \
        if (e_ != cudaSuccess) {                                                       \
            cudaGetLastError(); /* consume it, do not leave it for the next call */    \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_));  \
        }                                                                              \
    } while (0)

void *dev_alloc(size_t bytes) {
    void *p = nullptr;
    CK(cudaMalloc(&p, bytes ? bytes : 1));
    return p;
}

constexpr int kMaxBlocks = 1024;  // the occupancy check is the real bound; this is a request cap

int pick_rounding_mode() {
    switch (fegetround()) {
        case FE_UPWARD: return kRU;
        case FE_DOWNWARD: return kRD;
        case FE_TOWARDZERO: return kRZ;
        default: return kRN;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// The handle's GPU resources
// ---------------------------------------------------------------------------
namespace fastbp {

// The bounded poll budget of the mapped status block.  Iterations, not time: the decode path
// reads no clock.  At ~20-30 ns an iteration this is a few tenths of a second of body time, far more
// than any shot of the target workload, and a shot that exceeds it takes the plain hand-off.
// The forced-missed-poll test build sets FASTBP_TEST_POLL_BUDGET=1 (with
// the status publication off), so the host takes the plain unconditional hand-off on every shot.
#ifndef FASTBP_TEST_POLL_BUDGET
#define FASTBP_TEST_POLL_BUDGET (1ull << 25)
#endif
constexpr unsigned long long kStatusPollBudget = FASTBP_TEST_POLL_BUDGET;

struct GpuState {
    uint32_t m = 0, n = 0, nnz = 0;
    int rm = kRN;
    int nblocks = 1;
    int nthreads = kThreads;   // 256 in-place, 512 for the row-summary formulation
    uint32_t wpr = 1;
    int sum_mode = 0;                // 0 in-place, 1 the fused single-barrier summary, 2 summary
    int sms = 0;                     // SMs the shapes were sized for (device, capped by the budget)
    unsigned long long l2 = 0;       // bytes of L2 the capacities were sized for (device, capped)
    int ilv = 0;                     // 1: the interleaved degree-grouped column layout
    uint32_t nslots = 0;             // nnz in the natural layout, padded slots in the interleaved one
    // the fused schedule's incidence (sum_mode == 1): the rows of each block, and for each row the
    // (block, partial slot) list of every block whose columns touch it, plus the partial arrays
    uint32_t *d_rowbase = nullptr, *d_rowlist = nullptr, *d_owner_ptr = nullptr, *d_owner_slot = nullptr;
    // The block's merge metadata (the rows of its
    // slots only), one padded record per block; the kernel stages its record.
    uint32_t *d_meta_off = nullptr, *d_merge_meta = nullptr;
    uint32_t meta_max = 0;           // the largest block record (padded), in u32
    uint32_t rslot_bytes = 0;        // the rslot stage's size in bytes (0 unless the RSLOT kernel runs)
    uint32_t win_off = 0, win_rounds = 0;   // the b2c window (see Dev::win_off)
    unsigned long long *d_pm1 = nullptr, *d_pm2 = nullptr;
    unsigned *d_pcnt = nullptr, *d_psign = nullptr;
    // Three buffers of m rows each, [buffer * m + row].
    unsigned long long *d_am1 = nullptr, *d_am2 = nullptr;
    unsigned *d_asign = nullptr;
    // The iteration-parity double buffer of the fused body.  d_out (the hand-off region)
    // is buffer 0; d_out2 is buffer 1 with the same layout [dec (n) | pad | status (8) | post (8n)], so the
    // host's [dec | status] copy and its n-byte decision copy read the same offsets from either.  Allocated
    // only when the fused schedule runs (sum_mode == 1); null otherwise.  No scratch is kept for the
    // deferred exit test's restore: the other buffer *is* the saved state.
    uint8_t *d_out2 = nullptr;
    double *d_post2 = nullptr;     // d_out2 + post_off
    uint8_t *d_dec2 = nullptr;     // d_out2 + 0
    uint32_t *d_status2 = nullptr; // d_out2 + status_off
    uint32_t ns_max = 0;
    uint32_t pm_stride = 0;          // partial slots per buffer (pm1/pm2/psign hold two, by iteration parity)
    size_t fused_shbytes = 0;

    uint32_t *d_cptr = nullptr, *d_rptr = nullptr, *d_redge = nullptr;
    uint32_t *d_rcol = nullptr;
    uint32_t *d_rind = nullptr;   // row of each edge in CSC order (sum_mode's column pass)
    uint32_t *d_cperm = nullptr;  // ilv: n, original column of each position
    uint8_t *d_cdeg = nullptr;    // ilv: n, degree of the column at each position
    uint8_t *d_rslot = nullptr;   // nslots, create-time slot -> row slot (unsigned char)
    int rslot = 0;                // 1: the fused kernel stages d_rslot in shared memory (RSLOT)
    uint32_t *d_colbase = nullptr;   // nblocks + 1, the per-block column ranges
    int balanced = 0;                // 1: the balanced partition was taken (0: the uniform chunks)
    LayoutStats lay{};               // the create-time layout record (Engine::layout)
#if M6C_HOST_STAGES
    // (measurement build only): this handle's clock calibration (calibrate_clocks): theta =
    // %globaltimer - CLOCK_MONOTONIC lies in [calib_lo, calib_hi] ns; 0 / 0 when it could not be taken.
    long long calib_lo = 0, calib_hi = 0;
#endif
    unsigned long long *d_sm1 = nullptr, *d_sm2 = nullptr;
    unsigned long long *d_sm1i = nullptr, *d_sm2i = nullptr;   // from the priors, at create
    int *d_spar = nullptr, *d_spar0 = nullptr;
    double *d_prior = nullptr, *d_b2c = nullptr, *d_c2b = nullptr;
    // prior_pos: n doubles in the interleaved layout's position order (see
    // Dev::prior_pos).  Allocated with the layout's tables (never when the layout is not interleaved),
    // filled by prior_pos_kernel at create and in set_priors.
    double *d_prior_pos = nullptr;
    // One device region holding [dec (n) | status (8) | post (8n)], so that one D2H covers
    // the whole hand-off (the prefix [dec | status] alone when the shot's posteriors are not needed).
    uint8_t *d_out = nullptr;
    double *d_post = nullptr;    // d_out + post_off
    uint8_t *d_dec = nullptr;    // d_out + 0
    uint8_t *d_syn = nullptr;    // m: only for problems too wide for the parameter syndrome
    uint32_t *d_mis = nullptr;   // 4: mismatch totals (the fused body's three slots, two in the others) +
                                 // the accumulator rotation word [3] (fused body only)
    uint32_t *d_status = nullptr;  // d_out + status_off
    size_t status_off = 0, post_off = 0, pre_bytes = 0, out_bytes = 0;

    // The OSD-0 order prefix (only a decoder=bposd osd_method=osd0 handle
    // allocates these; a bp handle leaves them null and the kernel branches over the pass).
    unsigned long long *d_okey = nullptr, *d_osamp = nullptr, *d_otau = nullptr;
    // The hand-over block d_hand is
    // [u32 count | u32 sorted | 8 B pad][kOsdCap x 16 B records], one allocation, and h_pref has the
    // same layout, so each hand-off is ONE transfer of the header plus the rung's records.  d_ocnt
    // (the collect's counter and the sorted flag) is the header, d_ocand_sort (the records the host reads:
    // the sorted output, a gated-out collect, the repick's collect) is d_hand + kHandHdr -- both are
    // aliases, never freed on their own.  d_ocand is the rank sort's working buffer.
    unsigned char *d_hand = nullptr;
    unsigned char *d_ocand = nullptr;
    unsigned char *d_ocand_sort = nullptr;   // alias: d_hand + kHandHdr
    // Rung two's device block (the tail's collect, the hand-over
    // block's layout) and the pinned staging the host copies it into on a repick shot.  Allocated for a
    // use_osd handle only; h_pref2_dev is the mapped device view (null when the block could not be
    // mapped, and then the tail collects only).
    unsigned char *d_hand2 = nullptr;
    unsigned char *h_pref2 = nullptr;
    unsigned char *h_pref2_dev = nullptr;
    int tail_armed = 0;              // this shot's kernel ran (or may have run) the rung-two tail
    unsigned *d_ohist = nullptr, *d_ocnt = nullptr;   // d_ocnt alias: (unsigned *)d_hand
    // d_ohist holds the two halves of the pushed histogram's ping-pong (each kOsdHBins
    // u32); hist_sel is the half the most recent run pushed into (the repick reads the same half).
    // The window [osd_lo, osd_lo + kOsdHBins * osd_step] is a function of the priors only, computed in
    // set_priors (create or a caller's re-set), never on the decode path.
    unsigned hist_sel = 0;
    double osd_lo = -(double)kOsdHWinMul, osd_step = (2.0 * kOsdHWinMul) / (double)kOsdHBins;
    double osd_iscale = (double)kOsdHBins / (2.0 * kOsdHWinMul);
    int use_osd = 0;
    uint32_t osd_r = kOsdS;          // set in setup(): the last rank bucket the pass reads (retired by
                                     // the pushed histogram; kept as the warm-up Dev's value)
    unsigned ohist_stride = kOsdS + 1u;
    unsigned char *h_pref = nullptr;  // pinned staging, d_hand's layout: [count | sorted | pad][records]
    unsigned char *h_pref_dev = nullptr;  // the mapped device view of h_pref (Dev::hand_host)

    uint8_t *h_syn = nullptr;    // pinned staging: the shot's syndrome (H2D, wide problems only)
    // The pinned hand-off, laid out exactly like d_out.  fastbpgpu6g.cpp's correction and
    // posterior pointers alias this buffer, so no host-side memcpy of either remains.
    uint8_t *h_res = nullptr;

    cudaStream_t stream = nullptr;
    cudaEvent_t ev = nullptr;

    // ---- the mapped status block (see StatusCtl) ----
    int dev = -1;
    StatusCtl *h_status = nullptr;   // the host's view (mapped pinned memory)
    StatusCtl *d_status_pub = nullptr;   // the device's view of the same memory
    unsigned long long shot_gen = 0;   // the last generation this handle launched (0 is never a shot)

    ~GpuState() {
        // Every kernel of this handle is a per-shot launch on its stream, so the stream
        // synchronisation below is the only thing destroy needs to wait for.  Nothing here throws
        // (the ABI's destroy does not).
        // Destroy waits on the handle's own stream before it frees anything: no
        // device-wide synchronisation, and no cudaFree while a kernel of this handle is running.
        if (stream) cudaStreamSynchronize(stream);
        if (ev) cudaEventDestroy(ev);
        if (stream) cudaStreamDestroy(stream);
        // Free only what was allocated: the layout's tables (cperm/cdeg), the fused schedule's
        // incidence and partials and the OSD-0 prefix arrays are created only for the problems and
        // handles that use them, so the rest of the pointers are null here.  A cudaFree(nullptr) is a
        // no-op for the driver, but the allocation and failure-injection shims of the test tools count
        // the call, and a handle must free exactly as many times as it allocated (the visible check's
        // create/destroy cycles).  d_post, d_dec and d_status point into d_out and are not
        // allocations of their own.
        auto fre = [](void *p) { if (p) cudaFree(p); };
        fre(d_cptr); fre(d_rptr); fre(d_redge); fre(d_rcol);
        fre(d_rind); fre(d_cperm); fre(d_cdeg); fre(d_rslot); fre(d_colbase); fre(d_sm1);
        fre(d_sm2); fre(d_sm1i); fre(d_sm2i); fre(d_spar);
        fre(d_spar0); fre(d_rowbase); fre(d_rowlist); fre(d_owner_ptr);
        fre(d_owner_slot); fre(d_meta_off); fre(d_merge_meta); fre(d_pm1); fre(d_pm2); fre(d_pcnt);
        fre(d_psign); fre(d_am1); fre(d_am2); fre(d_asign);
        fre(d_out2); fre(d_prior); fre(d_prior_pos);
        fre(d_b2c); fre(d_c2b); fre(d_out);
        fre(d_syn); fre(d_mis);
        fre(d_okey); fre(d_osamp); fre(d_ohist); fre(d_otau);
        fre(d_hand); fre(d_ocand);   // d_ocnt and d_ocand_sort point into d_hand
        fre(d_hand2);                // rung two's collect block
        if (h_syn) cudaFreeHost(h_syn);
        if (h_res) cudaFreeHost(h_res);
        if (h_pref) cudaFreeHost(h_pref);
        if (h_pref2) cudaFreeHost(h_pref2);   // rung two's pinned staging
        if (h_status) cudaFreeHost(h_status);
    }
};

#if M6C_HOST_STAGES
// (measurement build only): run the ping-pong of calib_kernel on the handle's stream (see CalibBlk) and
// keep the intersection of the rounds' offset bounds.  Create time only; a failure leaves the bound at 0 / 0 (the
// split tool then reports the raw per-shot sequences only).  The bound reaches the tools through the stage record
// only (marks 14 / 15 of every polled shot); the library writes no side file.
static void calibrate_clocks(GpuState *g) {
    g->calib_lo = g->calib_hi = 0;
    CalibBlk *h = nullptr, *dp = nullptr;
    if (cudaHostAlloc((void **)&h, sizeof(CalibBlk), cudaHostAllocMapped) != cudaSuccess) { cudaGetLastError(); return; }
    std::memset((void *)h, 0, sizeof(CalibBlk));
    if (cudaHostGetDevicePointer((void **)&dp, h, 0) != cudaSuccess) {
        cudaGetLastError();
        cudaFreeHost(h);
        return;
    }
    calib_kernel<<<1, 1, 0, g->stream>>>(dp);
    if (cudaGetLastError() != cudaSuccess) { cudaFreeHost(h); return; }
    long long lo = LLONG_MIN, hi = LLONG_MAX;
    unsigned kept = 0;
    for (unsigned long long r = 1; r <= kCalibRounds; ++r) {
        const uint64_t t0 = dbench_stage_now();
        __atomic_store_n(&h->ping, r, __ATOMIC_RELEASE);
        bool seen = false;
        for (unsigned long long s = 0; s < (1ull << 26); ++s) {
            if (__atomic_load_n(&h->pong, __ATOMIC_ACQUIRE) == r) { seen = true; break; }
            __builtin_ia32_pause();
        }
        const uint64_t t1 = dbench_stage_now();
        if (!seen) break;
        const unsigned long long gg = h->g;
        const long long a = (long long)(gg - t1), b = (long long)(gg - t0);
        if (a > lo) lo = a;
        if (b < hi) hi = b;
        ++kept;
    }
    cudaStreamSynchronize(g->stream);   // the kernel returns after its last round (or its spin bound)
    cudaGetLastError();
    if (kept == kCalibRounds && lo <= hi) { g->calib_lo = lo; g->calib_hi = hi; }
    cudaFreeHost(h);
}
#endif

Engine::Engine() = default;
Engine::~Engine() { delete gpu_; }


// ---------------------------------------------------------------------------
// The cross-block balance of work at create time.
//
// Every block's column chunk used to be the same size (per_block = ceil(n / blocks), rounded up to a
// warp group), so the DEM's degree mix along its column order decided how long each block's column
// pass took, and the grid barrier waited for the slowest one.  Here the chunk boundaries are chosen
// from H alone: the columns stay contiguous (a block's row footprint, and so the merge and the shared
// row-slot table, keep the locality the base layout was built for), only c0/c1 move, and each
// boundary is still a multiple of 32 positions so the interleaved slot formula (ilv_slot0) keeps
// every warp group on one 256 B run.
//
// The model is the block's column-pass work: sum over its degree-sorted 32-column groups of
// 32 x the group's largest degree, plus a per-column term.  A phase probe fitted a block's
// time to its total step count (t = 1.483 + 0.000801 steps + 0.001611 cols us,
// R^2 0.937), so only the RATIO of the two weights is used
// here (the constant and the absolute scale cancel).  The greedy linear partition below walks the
// 32-column groups and starts a new block when that lands closer to total/blocks.
//
// The gate: the balanced partition replaces the uniform one only when it predicts a lower slowest
// block by more than kBalMarginPct -- both predictions are create-time functions of H.  Otherwise the
// base's uniform chunking is used, to the column.
// ---------------------------------------------------------------------------
constexpr double kBalStepW = 0.000801;   // us per warp step (32 lanes x a group's largest degree)
constexpr double kBalColW = 0.001611;    // us per column
constexpr uint32_t kBalGroup = 32;       // a warp group: the interleaved layout's packing
constexpr unsigned kBalMarginPct = 0;    // a balanced layout must predict a strictly lower slowest block

// The predicted column-pass cost of one block: the block's own degree-sorted 32-column groups.
static double bal_block_cost(uint32_t c0, uint32_t c1, const uint32_t *deg,
                             std::vector<uint32_t> &scratch) {
    scratch.assign(deg + c0, deg + c1);
    std::sort(scratch.begin(), scratch.end());
    unsigned long long steps = 0;
    for (size_t i = 0; i < scratch.size(); i += kBalGroup) {
        const size_t e = std::min(i + (size_t)kBalGroup, scratch.size());
        steps += (unsigned long long)(e - i) * scratch[e - 1];
    }
    return kBalColW * (double)(c1 - c0) + kBalStepW * (double)steps;
}

// The greedy linear partition: boundaries at multiples of kBalGroup, each block's accumulated weight
// (kBalColW + kBalStepW x the group's largest degree, the same model summed per group) as close to
// total/blocks as the group granularity allows.  Every block takes at least one group and the last
// block takes the remainder.
static void bal_partition(uint32_t n, uint32_t blocks, const uint32_t *deg, uint32_t max_groups,
                          std::vector<uint32_t> &colbase) {
    colbase.assign((size_t)blocks + 1, 0);
    if (blocks == 0) { return; }
    const uint32_t G = (n + kBalGroup - 1u) / kBalGroup;
    if (G < blocks) {   // fewer warp groups than blocks: no room to balance, the caller keeps uniform
        for (uint32_t b = 0; b <= blocks; ++b) colbase[b] = (b < blocks) ? std::min(b * kBalGroup, n) : n;
        return;
    }
    // The weight of a warp group: its columns' per-column term plus kBalStepW per edge.  A block's
    // degree-sorted groups are near-homogeneous, so 32 x the group's largest degree (the model's
    // step count) is close to the group's sum of degrees -- and the sum is the additive form the
    // partition needs (a group's cost must not depend on which block it lands in).
    std::vector<double> gw((size_t)G, 0.0);
    double remaining = 0.0;
    for (uint32_t gi = 0; gi < G; ++gi) {
        const uint32_t c0 = gi * kBalGroup;
        const uint32_t c1 = std::min(c0 + kBalGroup, n);
        double edges = 0.0;
        for (uint32_t j = c0; j < c1; ++j) edges += (double)deg[j];
        gw[gi] = kBalColW * (double)(c1 - c0) + kBalStepW * edges;
        remaining += gw[gi];
    }
    // Greedy against the running target: what is left, over the blocks still to place.  A fixed
    // target of total/blocks would take the group granularity's rounding out of the LAST blocks (the
    // first measurement of this partition starved block 83 to 352 columns that way).
    uint32_t gi = 0;
    for (uint32_t b = 1; b < blocks; ++b) {
        const double target = remaining / (double)(blocks - b + 1);
        const uint32_t start = gi;
        const uint32_t cap = start + std::max(max_groups, 1u);
        double acc = 0.0;
        while (gi < G && gi < cap && (acc + gw[gi] < target || gi == start) && (G - gi) > (blocks - b)) {
            acc += gw[gi];
            remaining -= gw[gi];
            ++gi;
        }
        if (gi < G && gi < cap && (G - gi) > (blocks - b) &&
            std::abs(acc + gw[gi] - target) < std::abs(acc - target)) {
            acc += gw[gi];
            remaining -= gw[gi];
            ++gi;
        }
        colbase[b] = std::min(gi * kBalGroup, n);
    }
    colbase[blocks] = n;
}

// ---------------------------------------------------------------------------
// The degree-aware warp-group assignment.  A block's kernel walks the
// column positions c0 + threadIdx.x, c0 + blockDim.x + threadIdx.x, ... so warp w of the block takes
// the 32-column groups whose index is w mod warps, i.e. one group per round.  The base's order is
// degree-sorted, so with G groups over W warps the last rounds carry the highest degrees and the
// warps that carry two groups also carry the heaviest ones: the block's column pass ends with its
// slowest warp (seen in per-warp timing marks).
//
// Here the block's groups (the same 32-column groups, each a run of the degree-sorted list) are
// re-assigned to the warp slots: the heaviest groups go to the warps that have only one group, and
// the rest are paired heaviest-with-lightest (LPT by the group's largest degree, then a bounded
// pairwise swap search).  Every column keeps its own row order and addition sequence; only which warp
// walks which column moves, so the arithmetic, post / dec, the block's row footprint and its edge
// slots are unchanged.  Create-time only (a function of H and the create-time chunking).
static void balance_block_groups(const std::vector<uint32_t> &idx,
                                 const std::vector<uint8_t> &cdeg_host, uint32_t warps,
                                 std::vector<uint32_t> &out) {
    const uint32_t C = (uint32_t)idx.size();
    const uint32_t G = (C + 31u) / 32u;
    const uint32_t W = warps ? warps : 1u;
    if (G <= 1u || W <= 1u) { out = idx; return; }
    std::vector<uint32_t> gmax((size_t)G, 0u);
    for (uint32_t g = 0; g < G; ++g) {
        const uint32_t e = std::min((g + 1u) * 32u, C);
        uint32_t mx = 0;
        for (uint32_t k = g * 32u; k < e; ++k) mx = std::max(mx, (uint32_t)cdeg_host[idx[k]]);
        gmax[g] = mx;
    }
    // heaviest first (stable: equal degrees keep their group order)
    std::vector<uint32_t> ord((size_t)G);
    for (uint32_t g = 0; g < G; ++g) ord[g] = g;
    std::stable_sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return gmax[a] > gmax[b]; });
    // the warp slots' capacities: bucket w holds the group indices congruent to w (mod W)
    std::vector<uint32_t> cap((size_t)W, 0u), load((size_t)W, 0u);
    for (uint32_t g = 0; g < G; ++g) cap[g % W]++;
    std::vector<std::vector<uint32_t>> bucket((size_t)W);
    for (uint32_t oi = 0; oi < G; ++oi) {
        const uint32_t g = ord[oi];
        uint32_t best = W;
        unsigned long long bestv = ~0ull;
        for (uint32_t w = 0; w < W; ++w) {
            if (bucket[w].size() >= cap[w]) continue;
            const unsigned long long v = ((unsigned long long)(load[w] + gmax[g]) << 32) |
                                         (unsigned long long)load[w];
            if (v < bestv) { bestv = v; best = w; }
        }
        bucket[best].push_back(g);
        load[best] += gmax[g];
    }
    // bounded local search on swaps with the heaviest bucket
    for (uint32_t iter = 0; iter < 512u; ++iter) {
        uint32_t wmax = 0;
        for (uint32_t w = 1; w < W; ++w) {
            if (load[w] > load[wmax]) wmax = w;
        }
        uint32_t bw = W, bi = 0, bj = 0, bestmax = load[wmax];
        for (uint32_t w = 0; w < W; ++w) {
            if (w == wmax) continue;
            for (size_t i = 0; i < bucket[wmax].size(); ++i) {
                for (size_t j = 0; j < bucket[w].size(); ++j) {
                    const uint32_t gi = bucket[wmax][i], gj = bucket[w][j];
                    if (gmax[gi] == gmax[gj]) continue;
                    const uint32_t nw = load[wmax] - gmax[gi] + gmax[gj];
                    const uint32_t nv = load[w] - gmax[gj] + gmax[gi];
                    const uint32_t mx = std::max(nw, nv);
                    if (mx < bestmax) { bestmax = mx; bw = w; bi = (uint32_t)i; bj = (uint32_t)j; }
                }
            }
        }
        if (bw == W) break;
        const uint32_t gi = bucket[wmax][bi], gj = bucket[bw][bj];
        bucket[wmax][bi] = gj;
        bucket[bw][bj] = gi;
        load[wmax] = load[wmax] - gmax[gi] + gmax[gj];
        load[bw] = load[bw] - gmax[gj] + gmax[gi];
    }
    // The degree-sorted list's last group is the only one that can be partial (C % 32 != 0) and its
    // columns must occupy the block's *last* position index G-1 (the positions before it are full
    // 32-column groups).  The emission below walks each bucket in reverse, so the group at the
    // bucket's front lands on the bucket's last round; put the partial group there.  (Without this a
    // block whose column count is not a multiple of 32 would write a full 32-column group over the
    // partial group's tail, out of the layout buffer: this crashed an equivalence test.)
    if ((C & 31u) != 0u) {
        auto &bk = bucket[(G - 1u) % W];
        for (size_t i = 0; i < bk.size(); ++i) {
            if (bk[i] == G - 1u) {
                std::swap(bk[i], bk[0]);
                break;
            }
        }
    }
    // emit: bucket w's k-th group at local position index w + k * W.  The bucket is emitted in
    // reverse LPT order (lightest first), matching the base layout's round structure: a warp's first
    // round is its lightest group, so the cold first touches of the iteration's column pass (the L1
    // is invalidated by the barrier's acquire) fall on the short groups.
    std::vector<uint32_t> tmp((size_t)C, 0u);
    std::vector<uint8_t> seen((size_t)C, 0u);
    bool ok = true;
    for (uint32_t w = 0; w < W && ok; ++w) {
        for (size_t k = 0; k < bucket[w].size() && ok; ++k) {
            const uint32_t g = bucket[w][bucket[w].size() - 1u - k];
            const uint32_t e = std::min((g + 1u) * 32u, C);
            uint32_t d = ((uint32_t)k * W + w) * 32u;
            for (uint32_t q = g * 32u; q < e; ++q, ++d) {
                if (d >= C || seen[d] != 0u) { ok = false; break; }
                seen[d] = 1u;
                tmp[d] = idx[q];
            }
        }
    }
    if (ok) out.swap(tmp);
    else    out = idx;   // structural failure (never on the tested shapes): the base's own order
}

bool Engine::setup(uint32_t m_, uint32_t n_, const uint32_t *colptr, const uint32_t *rowind,
                   bool osd0, bool wide_ok) {
    int ndev = 0;
    cudaError_t e = cudaGetDeviceCount(&ndev);
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("no usable GPU: cudaGetDeviceCount: ") +
                                 cudaGetErrorString(e));
    }
    if (ndev <= 0) {
        throw std::runtime_error("no usable GPU: no CUDA device visible (CUDA_VISIBLE_DEVICES)");
    }
    int dev = 0;
    e = cudaSetDevice(dev);
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("no usable GPU: cudaSetDevice: ") +
                                 cudaGetErrorString(e));
    }
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, dev));
    if (prop.cooperativeLaunch == 0) {
        throw std::runtime_error("no usable GPU: device does not support cooperative launch");
    }

    GpuState *g = new GpuState();
    try {
        g->dev = dev;
        g->m = m_;
        g->n = n_;
        g->nnz = n_ ? colptr[n_] : 0;
        g->rm = pick_rounding_mode();
        g->use_osd = osd0 ? 1 : 0;

        // CSR of H: ascending columns within a row, edge ids in CSC order, plus the column of each
        // row slot (the check pass needs it for the candidate syndrome).
        std::vector<uint32_t> rptr((size_t)m_ + 1, 0), redge(g->nnz), rcol(g->nnz);
        std::vector<uint32_t> cptr(colptr, colptr + n_ + 1);
        uint32_t maxrowdeg = 0;
        for (uint32_t j = 0; j < n_; ++j) {
            for (uint32_t k = colptr[j]; k < colptr[j + 1]; ++k) rptr[rowind[k] + 1]++;
        }
        for (uint32_t i = 0; i < m_; ++i) {
            rptr[i + 1] += rptr[i];
            const uint32_t deg = rptr[i + 1] - rptr[i];
            if (deg > maxrowdeg) maxrowdeg = deg;
        }
        {
            std::vector<uint32_t> fill(rptr.begin(), rptr.end() - 1);
            for (uint32_t j = 0; j < n_; ++j) {
                for (uint32_t k = colptr[j]; k < colptr[j + 1]; ++k) {
                    const uint32_t slot = fill[rowind[k]]++;
                    redge[slot] = k;
                    rcol[slot] = j;
                }
            }
        }

        // Device sizing.  Every capacity and launch shape below is derived from the
        // device's SM count and L2 size (cudaDeviceProp::multiProcessorCount / l2CacheSize), each
        // capped by the build's budget: DBENCH_GPU_SM_BUDGET and DBENCH_GPU_L2_BUDGET from
        // CMakeLists.txt (0 = the device's own value), so a build for the RTX 5080
        // (84 SMs, 64 MiB of L2) sizes its grids, its formulation and its capacities for that part
        // while it runs on a larger GPU (170 SMs).  Nothing here is hard-coded to that GPU's SM count or L2.
        g->sms = prop.multiProcessorCount;
        if (DBENCH_GPU_SM_BUDGET > 0 && DBENCH_GPU_SM_BUDGET < g->sms) g->sms = DBENCH_GPU_SM_BUDGET;
        g->l2 = (unsigned long long)prop.l2CacheSize;
        if (DBENCH_GPU_L2_BUDGET > 0 && (unsigned long long)DBENCH_GPU_L2_BUDGET < g->l2) {
            g->l2 = (unsigned long long)DBENCH_GPU_L2_BUDGET;
        }

        // Shape (measured): one warp per row covers a row of up to 64 edges on
        // the register path; longer rows are split over several warps.  The column pass wants one
        // thread per column.  The occupancy check below caps the grid so the cooperative launch can
        // place it.
        g->wpr = 1;
        if (maxrowdeg > 64u) {
            g->wpr = (maxrowdeg + 63u) / 64u;
            if (g->wpr > 8u) g->wpr = 8u;
        }
        uint32_t need_rows = (m_ * g->wpr + 7u) / 8u;   // warps of 8 per block
        uint32_t need_cols = (n_ + kThreads - 1u) / kThreads;
        uint32_t blocks = need_rows > need_cols ? need_rows : need_cols;
        if (blocks < 1) blocks = 1;
        if (blocks > (uint32_t)kMaxBlocks) blocks = kMaxBlocks;
        g->nblocks = (int)blocks;
        // Formulation (measured): the row-summary path is used when the
        // in-place check pass cannot place the `wpr` warps per row it wants -- its grid request is
        // larger than the occupancy cap.  Then the split is pointless anyway and the summary's
        // scatter-free check pass wins (bb144 xyz: 645.8 against 951.9 us; the 819,200-edge
        // synthetic long-row problem: 2030 against 2445 us).  Where the in-place grid does fit, its
        // register path keeps the column pass at one load per edge and wins (bb72 loss: 350.4
        // against the summary's 535.4 us), and the 200,000-edge synthetic pays 4% for the rule
        // (929.3 against 892.9 us).  Both paths are exact and audited (by a long-row test).
        // The second half of the rule is the L2 budget: the in-place formulation writes one message
        // per edge in the check pass and reads it back in the column pass, so its message array
        // (8 B/edge) must stay in the L2 the handle may use; when it does not, the summary
        // formulation is preferred -- its check pass writes 3 words per row instead and the column
        // pass streams the same array once.  Evaluating the rule
        // on every corpus problem gives a margin of >= 8x: no corpus problem
        // comes near the threshold.

        CK(cudaStreamCreateWithFlags(&g->stream, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&g->ev, cudaEventDisableTiming));

        g->d_cptr = (uint32_t *)dev_alloc((size_t)(n_ + 1) * 4);
        g->d_rptr = (uint32_t *)dev_alloc((size_t)(m_ + 1) * 4);
        g->d_redge = (uint32_t *)dev_alloc((size_t)g->nnz * 4);
        g->d_rcol = (uint32_t *)dev_alloc((size_t)g->nnz * 4);
        g->d_sm1 = (unsigned long long *)dev_alloc((size_t)(m_ ? m_ : 1) * 8);
        g->d_sm2 = (unsigned long long *)dev_alloc((size_t)(m_ ? m_ : 1) * 8);
        g->d_sm1i = (unsigned long long *)dev_alloc((size_t)(m_ ? m_ : 1) * 8);
        g->d_sm2i = (unsigned long long *)dev_alloc((size_t)(m_ ? m_ : 1) * 8);
        g->d_spar = (int *)dev_alloc((size_t)(m_ ? m_ : 1) * 4);
        g->d_spar0 = (int *)dev_alloc((size_t)(m_ ? m_ : 1) * 4);
        g->d_prior = (double *)dev_alloc((size_t)n_ * 8);
        // [dec (n) | status (8) | post (8n)] in one device region and one pinned region of
        // the same shape.  The status word sits on an 8-byte boundary so the prefix copy of a
        // BP-converged shot carries dec and status in one cudaMemcpyAsync, and the posterior is
        // 8-byte aligned for the double load the OSD-0 engine and the soft output do.  (b2c and c2b
        // are allocated below, once the layout has fixed their slot count.)
        {
            const size_t dec_bytes = (size_t)n_;
            const size_t status_off_ = (dec_bytes + 7u) & ~(size_t)7u;
            g->status_off = status_off_;
            g->post_off = status_off_ + 8u;
            g->pre_bytes = g->post_off;
            g->out_bytes = g->post_off + (size_t)n_ * 8u;
            g->d_out = (uint8_t *)dev_alloc(g->out_bytes);
            // initcheck reported uninitialised reads on two problems (n = 2450 and n = 14966):
            // the alignment pad between the decisions and the status word, [n, status_off), is never written by
            // any kernel, and a decoder=bp handle's [dec | status] copy (pre_bytes) reads it -- the host never uses
            // those bytes.  Zeroed once here; only shapes with n % 8 != 0 have a pad.  Create path only.
            if (g->status_off > dec_bytes) {
                CK(cudaMemsetAsync(g->d_out + dec_bytes, 0, g->status_off - dec_bytes, g->stream));
            }
            g->d_dec = g->d_out;
            g->d_status = (uint32_t *)(g->d_out + g->status_off);
            g->d_post = (double *)(g->d_out + g->post_off);
            g->d_syn = (uint8_t *)dev_alloc((size_t)(m_ ? m_ : 1));
            g->d_mis = (uint32_t *)dev_alloc(4 * 4);   // + the rotation word
            if (cudaHostAlloc((void **)&g->h_res, g->out_bytes, cudaHostAllocDefault) != cudaSuccess ||
                cudaHostAlloc((void **)&g->h_syn, (size_t)m_, cudaHostAllocDefault) != cudaSuccess) {
                throw std::runtime_error("pinned host allocation failed");
            }
        }
        cudaGetLastError();

        // Every create-time upload goes on the handle's own stream: a pageable cudaMemcpy on the
        // legacy default stream is not ordered against a cudaStreamNonBlocking stream, and the
        // kernels read all of this.  create ends with a synchronisation of that
        // stream -- never a device-wide one: the handle's own state is complete
        // before it is returned, and a co-tenant's kernels are neither waited for nor disturbed.
        // Load every rounding-mode instantiation now (module loading is lazy; a later launch of a
        // kernel that is not loaded yet waits for running kernels, which a persistent kernel would
        // never allow), and check that the grid fits co-resident.
        cudaFuncAttributes attr{};
        // Load every (rounding mode, formulation, layout) instantiation now (see pick_kernel):
        // module loading is lazy, and a launch of a kernel that is not loaded yet waits for running
        // kernels, which a persistent kernel would never allow.
        // The keep-warm kernel is loaded at create too (lazy module loading would make its
        // first launch wait for running kernels).
        CK(cudaFuncGetAttributes(&attr, (const void *)keep_warm_kernel));
        for (int sm_ = 0; sm_ <= 2; ++sm_) {
            for (int lv = 0; lv <= 1; ++lv) {
                CK(cudaFuncGetAttributes(&attr, pick_kernel<kRN>(sm_, lv != 0)));
                CK(cudaFuncGetAttributes(&attr, pick_kernel<kRD>(sm_, lv != 0)));
                CK(cudaFuncGetAttributes(&attr, pick_kernel<kRU>(sm_, lv != 0)));
                CK(cudaFuncGetAttributes(&attr, pick_kernel<kRZ>(sm_, lv != 0)));
            }
        }
        int per_sm = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, pick_kernel<kRN>(0, false), kThreads, 0));
        const long long cap_inplace = (long long)per_sm * (long long)g->sms;
        const bool too_wide = (g->wpr > 1u && (long long)need_rows > cap_inplace);
        const bool msg_out_of_l2 = ((unsigned long long)g->nnz * 8ull > g->l2);
        // Prefer the one-block-per-SM fused
        // summary schedule (and the b2c window where its gate fits) whenever the in-place grid needs
        // more blocks than the device has SMs (at 84 SMs: bb144 z 117, hgp98 xyz 245, bb288 z 252
        // blocks), not only when it exceeds the occupancy cap or the messages exceed L2.  Both
        // schedules are exact; this term only chooses between them at create.
        const bool wide_grid = wide_ok && (long long)g->nblocks > (long long)g->sms;
        const bool sum_needed = (too_wide || msg_out_of_l2 || wide_grid);
        const bool wide_only = wide_grid && !too_wide && !msg_out_of_l2;   // see the end of the fused gate
        g->sum_mode = sum_needed ? 2 : 0;   // 2 = two-barrier summary; the fused one is chosen below
        int per_sm_ilv = 0;
        if (sum_needed) {
            // The grid is capped by the two-barrier summary kernel (both layouts): that is the
            // formulation the fused schedule falls back to, and the fused kernel carries the same
            // __launch_bounds__ and is checked against the final grid below.
            CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, pick_kernel<kRN>(2, false), kThreadsSum, 0));
            CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm_ilv, pick_kernel<kRN>(2, true), kThreadsSum, 0));
            g->nthreads = (int)kThreadsSum;
        } else {
            CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm_ilv, pick_kernel<kRN>(0, true), kThreads, 0));
            g->nthreads = (int)kThreads;
        }
        // The grid has to fit either layout (the layout is chosen below, from the final grid), and
        // the two instantiations are pinned by the same __launch_bounds__, so this is normally a
        // no-op.
        if (per_sm_ilv > 0 && per_sm_ilv < per_sm) per_sm = per_sm_ilv;
        const long long max_blocks = (long long)per_sm * (long long)g->sms;
        if (max_blocks <= 0) {
            throw std::runtime_error("no usable GPU: the BP kernel does not fit any SM");
        }
        if ((long long)g->nblocks > max_blocks) g->nblocks = (int)max_blocks;
        if (g->nblocks < 1) g->nblocks = 1;

        // The last rank bucket the device prefix pass keeps.  The
        // threshold is the first bucket whose cumulative count reaches kOsdK, and that bucket's index
        // is about kOsdK * kOsdS / n: the stride sample has one element per n / kOsdS columns, so the
        // kOsdK-th column of the order lies that many sample elements in.  Eight times the estimate,
        // floored at kOsdRMin and capped at the whole sample, is what the pass keeps -- see the block
        // above osd_prefix_pass.  This is a property of the problem's shape (n), computed once here;
        // nothing about a shot enters it.
        {
            const unsigned long long nz = (n_ > 0u) ? (unsigned long long)n_ : 1ull;
            // kOsdK2 (= 1024, rung two's K, the one the truncation was validated with), not the rung-one
            // kOsdK (384): the repick must find its K = 1024 crossing inside the kept buckets.
            const unsigned long long exp_r = ((unsigned long long)kOsdK2 * (unsigned long long)kOsdS) / nz;
            unsigned long long rr = 8ull * exp_r;
            if (rr < (unsigned long long)kOsdRMin) rr = kOsdRMin;
            if (rr > (unsigned long long)kOsdS) rr = kOsdS;
            g->osd_r = (uint32_t)rr;
            g->ohist_stride = g->osd_r + 1u;
        }

        // ---------------------------------------------------------------------------
        // Create-time layout.  The column order and the edge-slot layout are decided here, once,
        // after the grid is final (the partition is per block, so the number of blocks fixes it).
        //
        // The block's contiguous chunk of columns is unchanged: it is what keeps the block's row
        // footprint small (on bb144 xyz a block touches ~194 of the 1872 rows, 4.4 KB of the row
        // summary record, because the corpus lays the columns out in circuit order and a block gets
        // one time region).  Inside the chunk the columns are sorted by degree (stable, so equal
        // degrees keep the reference order) and packed 32 to a warp, and the edge slots are
        // interleaved 32 wide: the message of edge i of the column at position p lives at
        // ilv_slot0(p) + 32*i.  A warp's step i is then one 256 B run instead of 32 scattered
        // elements, and the unrolled `i < deg` loop no longer runs to the largest degree of a
        // random 32-column window: on bb144 xyz the shipped layout executes 664,512 warp-steps for
        // the 438,912 edges (51.4% wasted) against 447,456 (1.9%) for this one
        // (from the create-time layout statistics).
        //
        // Nothing about the messages changes: a column's edges are still visited in the reference's
        // row order (rind is permuted with the slots) and each column keeps its own addition
        // sequence; post and dec stay in the original column order (the caller's).  Only reached
        // when every column degree fits the register path, so the layout never needs the general
        // (c2b) fallback; the shipped CSC layout stays for every other problem.
        // ---------------------------------------------------------------------------
        std::vector<uint32_t> rind_host;   // ILV: the padded slot -> row array (also used by RSLOT below)
        // The per-block column ranges the kernel launches with (built below;
        // the fused schedule's row incidence and the row-slot slices read the same table).
        std::vector<uint32_t> colbase;
        {
            uint32_t maxcoldeg = 0;
            std::vector<uint8_t> cdeg_host((size_t)n_, 0);
            for (uint32_t j = 0; j < n_; ++j) {
                const uint32_t dg = colptr[j + 1] - colptr[j];
                cdeg_host[j] = (uint8_t)(dg > 255u ? 255u : dg);
                if (dg > maxcoldeg) maxcoldeg = dg;
            }
            g->nslots = g->nnz;
            std::vector<uint32_t> cperm, cpos, redge_ilv;
            std::vector<uint8_t> cdeg_new;
            // Built for every layout: the base's uniform chunks (the plain per-block count) unless
            // the interleaved layout below replaces them with the balanced partition.
            colbase.assign((size_t)g->nblocks + 1, 0);
            {
                const uint32_t pbp = (n_ + (uint32_t)g->nblocks - 1u) / (uint32_t)g->nblocks;
                for (int b = 0; b <= g->nblocks; ++b)
                    colbase[(size_t)b] = std::min((uint32_t)b * pbp, n_);
            }
            if (n_ > 0 && maxcoldeg <= kRegDeg) {
                const uint32_t blocks = (uint32_t)g->nblocks;
                uint32_t pb = (n_ + blocks - 1u) / blocks;
                pb = (pb + 31u) & ~31u;   // a block's chunk starts on a warp-group boundary
                // The base's uniform chunking (32-aligned: the interleaved layout's rule) and the
                // balanced candidate; the gate takes the balanced one only when it predicts a lower
                // slowest block.  Both are create-time functions of H (see bal_partition above).
                {
                    std::vector<uint32_t> degv((size_t)n_);
                    for (uint32_t j = 0; j < n_; ++j) degv[j] = colptr[j + 1] - colptr[j];
                    std::vector<uint32_t> uni((size_t)blocks + 1, 0);
                    for (uint32_t b = 0; b < blocks; ++b) uni[b] = std::min(b * pb, n_);
                    uni[blocks] = n_;
                    // A block of more than two groups per warp puts a third group on some warp and
                    // that warp sets the block's time (measured: the 35-group blocks of an
                    // edges-only partition ran 8.93 us against 6.75 for the 26-group ones).
                    // The cap is two groups per warp, but never less
                    // than the base's own chunk, so a grid where the base already exceeds it (70 SMs:
                    // 34 groups) is not made worse.
                    const uint32_t warps = std::max(1u, (uint32_t)g->nthreads >> 5u);
                    const uint32_t cap_groups = std::max(2u * warps, pb / kBalGroup);
                    std::vector<uint32_t> bal;
                    bal_partition(n_, blocks, degv.data(), cap_groups, bal);
                    std::vector<uint32_t> scratch;
                    double umax = 0.0, bmax = 0.0;
                    for (uint32_t b = 0; b < blocks; ++b) {
                        umax = std::max(umax, bal_block_cost(uni[b], uni[b + 1], degv.data(), scratch));
                        bmax = std::max(bmax, bal_block_cost(bal[b], bal[b + 1], degv.data(), scratch));
                    }
                    // The row-footprint gate: the balanced ranges must not cost the block-local
                    // row-slot table (ns_max <= 255, the RSLOT kernel) or grow the fused kernel's shared
                    // tables into a fallback to the two-barrier path.  Both footprints are computed
                    // here from H and the two partitions; the balanced one is taken only if it keeps
                    // the same side of the 255 gate as the base and does not grow more than a quarter.
                    uint32_t ns_uni = 0, ns_bal = 0;
                    {
                        // The footprint of each partition is the real build's
                        // ns_max below -- a fresh `seen` per partition (an earlier form shared one vector between
                        // the two calls with no reset, so a row the uniform block b touched last was skipped
                        // for the balanced block b: ns_bal was undercounted and the gate reduced to
                        // bmax < umax), and block 0 also carries the degree-0 rows (the real build adds them).
                        uint32_t n_empty = 0;
                        for (uint32_t r = 0; r < m_; ++r) n_empty += (rptr[r + 1] == rptr[r]) ? 1u : 0u;
                        std::vector<uint32_t> seen((size_t)m_, UINT32_MAX), rows;
                        auto ns_of = [&](const std::vector<uint32_t> &cb) {
                            std::fill(seen.begin(), seen.end(), UINT32_MAX);
                            uint32_t mx = 0;
                            for (uint32_t b = 0; b < blocks; ++b) {
                                rows.clear();
                                for (uint32_t c = cb[b]; c < cb[b + 1]; ++c)
                                    for (uint32_t e = colptr[c]; e < colptr[c + 1]; ++e) {
                                        const uint32_t r = rowind[e];
                                        if (seen[r] != b) { seen[r] = b; rows.push_back(r); }
                                    }
                                const uint32_t nb = (uint32_t)rows.size() + (b == 0 ? n_empty : 0u);
                                if (nb > mx) mx = nb;
                            }
                            return mx;
                        };
                        ns_uni = ns_of(uni);
                        ns_bal = ns_of(bal);
#if defined(FASTBP_TEST_LAYOUT_PROBE) && FASTBP_TEST_LAYOUT_PROBE
                        // test-only (the host table): the earlier footprints, as the earlier gate computed them
                        // (one shared `seen`, no degree-0 rows), and the gate they gave.
                        {
                            std::vector<uint32_t> seen0((size_t)m_, UINT32_MAX), rows0;
                            auto ns_old = [&](const std::vector<uint32_t> &cb) {
                                uint32_t mx = 0;
                                for (uint32_t b = 0; b < blocks; ++b) {
                                    rows0.clear();
                                    for (uint32_t c = cb[b]; c < cb[b + 1]; ++c)
                                        for (uint32_t e = colptr[c]; e < colptr[c + 1]; ++e) {
                                            const uint32_t r = rowind[e];
                                            if (seen0[r] != b) { seen0[r] = b; rows0.push_back(r); }
                                        }
                                    if (rows0.size() > mx) mx = (uint32_t)rows0.size();
                                }
                                return mx;
                            };
                            const uint32_t u0 = ns_old(uni), b0 = ns_old(bal);
                            const bool ok0 = (u0 > 255u || b0 <= 255u) &&
                                             (uint64_t)b0 * 4ull <= (uint64_t)u0 * 5ull + 8ull;
                            g->lay.ns_uni_6e = u0; g->lay.ns_bal_6e = b0;
                            g->lay.gate_6e = (bmax < umax * (1.0 - (double)kBalMarginPct / 100.0) && ok0) ? 1u : 0u;
                        }
#endif
                    }
                    const bool row_ok = (ns_uni > 255u || ns_bal <= 255u) &&
                                        (uint64_t)ns_bal * 4ull <= (uint64_t)ns_uni * 5ull + 8ull;
                    bool take_bal = (bmax < umax * (1.0 - (double)kBalMarginPct / 100.0) && row_ok);
                    g->lay.ns_uni = ns_uni; g->lay.ns_bal = ns_bal;
                    g->lay.gate = take_bal ? 1u : 0u;
                    g->lay.bal_cost_ratio_ppm = umax > 0.0 ? (uint32_t)(bmax / umax * 1e6) : 0u;
#if defined(FASTBP_TEST_OLD_GATE) && FASTBP_TEST_OLD_GATE
                    take_bal = (g->lay.gate_6e != 0u);   // test-only: the earlier gate's decision, to tabulate its layout
#endif
                    if (take_bal) {
                        colbase = bal;
                        g->balanced = 1;
                    } else {
                        colbase = uni;
                        g->balanced = 0;
                    }
                }
                cperm.assign((size_t)n_, 0);
                cpos.assign((size_t)n_, 0);
                redge_ilv.assign(g->nnz, 0);
                cdeg_new.assign((size_t)n_, 0);
                uint32_t p = 0;
                for (uint32_t b = 0; b < blocks; ++b) {
                    const uint32_t c0 = colbase[b], c1 = colbase[b + 1];
                    if (c0 >= c1) continue;
                    std::vector<uint32_t> idx(c1 - c0);
                    for (uint32_t k = 0; k < idx.size(); ++k) idx[k] = c0 + k;
                    std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t c) {
                        return cdeg_host[a] < cdeg_host[c];
                    });
#if DBENCH_LAYOUT_BALANCE
                    // Re-assign the block's whole 32-column groups to its
                    // warp slots by degree (see balance_block_groups).  The groups are the same runs
                    // of the degree-sorted list, so the step count and the ILV waste gate below are
                    // unchanged; only the warp that walks each group moves.
                    {
                        std::vector<uint32_t> bal_idx;
                        balance_block_groups(idx, cdeg_host, (uint32_t)g->nthreads >> 5u, bal_idx);
                        idx.swap(bal_idx);
                    }
#endif
                    for (uint32_t k = 0; k < idx.size(); ++k, ++p) {
                        cperm[p] = idx[k];
                        cpos[idx[k]] = p;
                        cdeg_new[p] = cdeg_host[idx[k]];
                    }
                }
                // Gate: a warp executes the unrolled loop to the largest degree among its columns, so
                // each warp group costs 32 * max(deg) steps whatever the columns' own degrees are.
                // Predict both layouts' step counts at create and take the interleaved one only when
                // the sort has done its job, i.e. when the interleaved order's own residual waste
                // (steps - sum(deg)) / sum(deg) is at most kIlvWasteLimit.  A block chunk that holds
                // only a warp group or two cannot make its warp groups homogeneous (the sorted order
                // is still mixed), and there the per-column indirection and the 9-slot padding cost
                // more than the steps saved.  The four measured points bracket the limit: bb144 xyz
                // 1.9% waste (1.44x per iteration faster), bb288 z 18.9% (-10% shot time), bb144 z
                // 23.6% (+6%) and hgp98 z 47.1% (+10% per iteration) -- the latter two keep the
                // base's CSC layout.
                constexpr unsigned kIlvWasteLimitPct = 12;
                auto steps_for = [&](const std::vector<uint32_t> &ord, const std::vector<uint32_t> &cb) {
                    unsigned long long tot = 0;
                    for (uint32_t b = 0; b < blocks; ++b) {
                        const uint32_t c0 = cb[b], c1 = cb[b + 1];
                        if (c0 >= c1) continue;
                        for (uint32_t w = c0; w < c1; w += 32u) {
                            const uint32_t we = (w + 32u < c1) ? (w + 32u) : c1;
                            uint32_t mx = 0;
                            for (uint32_t k = w; k < we; ++k) {
                                const uint32_t dg = cdeg_host[ord[k]];
                                if (dg > mx) mx = dg;
                            }
                            tot += 32u * mx;
                        }
                    }
                    return tot;
                };
                std::vector<uint32_t> ident((size_t)n_);
                for (uint32_t j = 0; j < n_; ++j) ident[j] = j;
                std::vector<uint32_t> uni_plain((size_t)blocks + 1, 0);
                for (uint32_t b = 0; b <= blocks; ++b)
                    uni_plain[b] = std::min(b * ((n_ + blocks - 1u) / blocks), n_);
                const unsigned long long shipped_steps = steps_for(ident, uni_plain);
                const unsigned long long ilv_steps = steps_for(cperm, colbase);
                const unsigned long long sum_deg = g->nnz;
                (void)shipped_steps;
                if (ilv_steps * 100ull > sum_deg * (100ull + kIlvWasteLimitPct)) {
                    cperm.clear();   // keep the base's CSC layout
                    colbase = uni_plain;   // and the base's chunking (the table holds it)
                    g->balanced = 0;
                } else {
                    g->ilv = 1;
                }
            }
            if (g->ilv) {
                g->nslots = ((n_ + 31u) / 32u) * (32u * (uint32_t)kRegDeg);
                rind_host.assign((size_t)g->nslots, 0u);
                for (uint32_t j = 0; j < n_; ++j) {
                    const uint32_t s0 = (cpos[j] >> 5) * (32u * (uint32_t)kRegDeg) + (cpos[j] & 31u);
                    for (uint32_t e = colptr[j]; e < colptr[j + 1]; ++e) {
                        rind_host[s0 + 32u * (e - colptr[j])] = rowind[e];
                    }
                }
                for (uint32_t k = 0; k < g->nnz; ++k) {
                    const uint32_t j = rcol[k];          // original column of this row slot
                    const uint32_t s0 = (cpos[j] >> 5) * (32u * (uint32_t)kRegDeg) + (cpos[j] & 31u);
                    redge_ilv[k] = s0 + 32u * (redge[k] - colptr[j]);
                }
                g->d_cperm = (uint32_t *)dev_alloc((size_t)n_ * 4);
                g->d_cdeg = (uint8_t *)dev_alloc((size_t)n_);
                // prior_pos: the priors in position order; filled below, after the
                // create-time prior upload (prior_pos_kernel), and again in set_priors.
                g->d_prior_pos = (double *)dev_alloc((size_t)n_ * 8);
                CK(cudaMemcpyAsync(g->d_cperm, cperm.data(), (size_t)n_ * 4, cudaMemcpyHostToDevice, g->stream));
                CK(cudaMemcpyAsync(g->d_cdeg, cdeg_new.data(), (size_t)n_, cudaMemcpyHostToDevice, g->stream));
                std::memcpy(redge.data(), redge_ilv.data(), (size_t)g->nnz * 4);
            } else {
                rind_host.assign(rowind, rowind + g->nnz);
            }
            g->d_rind = (uint32_t *)dev_alloc((size_t)g->nslots * 4);
            g->d_b2c = (double *)dev_alloc((size_t)g->nslots * 8);
            g->d_c2b = (double *)dev_alloc((size_t)g->nslots * 8);
            CK(cudaMemcpyAsync(g->d_rind, rind_host.data(), (size_t)g->nslots * 4, cudaMemcpyHostToDevice,
                               g->stream));
            // The block -> column-range table (the kernels' col_begin/col_end).
            g->d_colbase = (uint32_t *)dev_alloc((size_t)(g->nblocks + 1) * 4);
            CK(cudaMemcpyAsync(g->d_colbase, colbase.data(), (size_t)(g->nblocks + 1) * 4,
                               cudaMemcpyHostToDevice, g->stream));
        }

        // ---------------------------------------------------------------------------
        // The fused single-barrier schedule, built on the column chunking the layout above fixed.
        // The whole incidence is create-time: the rows each block's chunk of columns touches, and for
        // every row the list of (block, partial slot) owners.  The schedule is used only when its
        // per-block shared footprint fits the device and the fused kernel still co-resides with the
        // grid; otherwise the two-barrier summary kernel of the chosen layout runs (g->sum_mode 2).
        // ---------------------------------------------------------------------------
        if (sum_needed && n_ > 0 && m_ > 0) {
            // The block's column range is the create-time table's, not a
            // uniform per-block count (colbase holds the base's uniform chunks when the balanced
            // partition's gate did not fire, so this is the base's schedule then, to the column).
            std::vector<uint32_t> rowbase((size_t)g->nblocks + 1, 0), rowlist;
            std::vector<uint32_t> seen(m_, UINT32_MAX), rows;
            uint32_t ns_max = 0;
            for (int b = 0; b < g->nblocks; ++b) {
                const uint32_t c0 = colbase[(size_t)b];
                const uint32_t c1 = colbase[(size_t)b + 1];
                rows.clear();
                for (uint32_t c = c0; c < c1; ++c) {
                    for (uint32_t e = colptr[c]; e < colptr[c + 1]; ++e) {
                        const uint32_t r = rowind[e];
                        if (seen[r] != (uint32_t)b) { seen[r] = (uint32_t)b; rows.push_back(r); }
                    }
                }
                // Rows without edges belong to no column, so no block would merge them and their
                // syndrome bit would never reach the mismatch: a shot with a 1 on an empty row would
                // converge here while bp.hpp (and the shipped check pass, which walks every row)
                // never converges.  Block 0 carries them: their partials stay the identity (parity 0),
                // so the merge counts exactly the empty rows whose syndrome bit is 1.
                if (b == 0) {
                    for (uint32_t r = 0; r < m_; ++r) {
                        if (rptr[r + 1] == rptr[r]) rows.push_back(r);
                    }
                }
                std::sort(rows.begin(), rows.end());
                rowbase[b + 1] = rowbase[b] + (uint32_t)rows.size();
                rowlist.insert(rowlist.end(), rows.begin(), rows.end());
                if (rowbase[b + 1] - rowbase[b] > ns_max) ns_max = rowbase[b + 1] - rowbase[b];
            }
            // row -> (block, partial slot): the slot is the block's local index in global coordinates
            std::vector<uint32_t> owner_ptr(m_ + 1, 0), owner_slot, cnt(m_, 0), slot_of(m_, UINT32_MAX);
            for (uint32_t r : rowlist) cnt[r]++;
            for (uint32_t r = 0; r < m_; ++r) owner_ptr[r + 1] = owner_ptr[r] + cnt[r];
            owner_slot.resize(owner_ptr[m_]);
            {
                std::vector<uint32_t> fill(owner_ptr.begin(), owner_ptr.end() - 1);
                for (int b = 0; b < g->nblocks; ++b) {
                    for (uint32_t j = 0; j < rowbase[b + 1] - rowbase[b]; ++j)
                        slot_of[rowlist[rowbase[b] + j]] = rowbase[b] + j;
                    for (uint32_t j = 0; j < rowbase[b + 1] - rowbase[b]; ++j) {
                        const uint32_t r = rowlist[rowbase[b] + j];
                        owner_slot[fill[r]++] = slot_of[r];
                    }
                }
            }
            g->ns_max = ns_max;
            g->pm_stride = owner_ptr[m_];
            // The block's merge metadata, staged once per shot into shared
            // memory.  The merge reads no owner list (the
            // push accumulates per row in L2), so the record is the block's rows alone -- [rows (ns)] =
            // rowlist[rowbase[b] .. rowbase[b + 1]), the row of each block-local slot, which the push and
            // the merge both index by -- padded to a multiple of four u32 (a uint4 staging copy).  An earlier
            // record form also carried [slot_base (ns + 1) | owner slots (tot)].
            std::vector<uint32_t> meta_off((size_t)g->nblocks + 1, 0), merge_meta;
            uint32_t meta_max = 0;
            for (int b = 0; b < g->nblocks; ++b) {
                const uint32_t j0 = rowbase[b], j1 = rowbase[b + 1];
                meta_off[b] = (uint32_t)merge_meta.size();
                for (uint32_t j = j0; j < j1; ++j) merge_meta.push_back(rowlist[j]);
                while ((merge_meta.size() & 3u) != 0u) merge_meta.push_back(0u);   // uint4 padding
                const uint32_t cnt = (uint32_t)merge_meta.size() - meta_off[b];
                if (cnt > meta_max) meta_max = cnt;
            }
            meta_off[g->nblocks] = (uint32_t)merge_meta.size();
            g->meta_max = meta_max;
            // 4 x 8 B + 3 x 4 B per row slot (pa_cnt gone) + row2slot, 16-B aligned (the
            // device's fused_rslot_offset), then [rslot stage] then the merge metadata stage.
            const size_t tables = ((size_t)ns_max * 8 * 4 + (size_t)ns_max * 4 * 3 + (size_t)m_ * 4 + 15u) &
                                  ~(size_t)15u;
            // The RSLOT variant's RN / RZ kernels keep a 32 B record per row
            // slot instead of sm_m1 / sm_m2 / sm_cnt / sm_sign: 52 B per row slot (the device's
            // fused_rslot_offset<..., REC>); one dynamic size is passed to all four rounding modes, so the
            // variant is sized for the record (RU / RD use 44 B of it).
            // and without the m x 4 B row2slot table, which no RSLOT kernel stages or reads
            // (the device's fused_rslot_offset<..., RSLOT>); RU / RD's 44 B rows fit inside the same size.
            const size_t tables_rec = ((size_t)ns_max * 52u + 15u) & ~(size_t)15u;
            const size_t meta_bytes = (size_t)meta_max * 4u;
            g->fused_shbytes = tables + meta_bytes;
            // The create-time slot -> row-slot table, one unsigned char
            // per ILV slot, built only where the fused ILV kernel is a candidate and a block's row count
            // fits a byte (ns_max <= 255).  rslot[s] is the row slot the column pass would get from
            // f.row2slot[rind[s]]: a function of H and this create-time layout alone, so it is computed
            // once here and uploaded, never per shot.
            std::vector<uint8_t> rslot_host;
            // The stage holds kRegDeg row-slot bytes per *position* of the
            // block's column range, packed as [w0: npos u32 | w1: npos u32 | b8: npos bytes].  Its size is
            // kept exactly the base's (the 32-rounded ILV slice, groups * 32 * kRegDeg): the b2c window is
            // sized by it (win_bytes = rslot_bytes * 8) and the window must cover every ILV slot the
            // pass addresses, so the stage is not shrunk even where the packed table would need less.  The
            // engine's dynamic shared memory, the window gate and the create-time numbers are unchanged.
            uint32_t rslot_per_block = 0u;
            if (g->ilv) {
                for (int b = 0; b < g->nblocks; ++b) {
                    const uint32_t nb_ = (((colbase[(size_t)b + 1] - colbase[(size_t)b]) + 31u) >> 5) *
                                         (32u * (uint32_t)kRegDeg);
                    if (nb_ > rslot_per_block) rslot_per_block = nb_;
                }
            }
            const uint32_t rslot_bytes = rslot_per_block;
            {   // the widest block, for the layout record (the window is sized by it)
                uint32_t mx = 0u;
                for (int b = 0; b < g->nblocks; ++b) mx = std::max(mx, colbase[(size_t)b + 1] - colbase[(size_t)b]);
                g->lay.max_block_cols = mx;
            }
            if (g->ilv && ns_max <= 255u) {
                // One 16-B-aligned region per block, kRegDeg bytes per
                // position of its range.  The values are the old per-slot table's (rslot[s] =
                // f.row2slot[rind[s]]), repacked so the column pass reads three words per position
                // instead of kRegDeg bytes 32 B apart.  Padding rounds (i >= the column's degree) carry
                // whatever the old table carried there; the pass never reads them.
                rslot_host.assign((size_t)g->nblocks * (size_t)rslot_bytes, 0u);
                std::vector<uint32_t> pos(m_, UINT32_MAX);
                for (int b = 0; b < g->nblocks; ++b) {
                    // A chunk past the last column (the 32-rounded chunking leaves the last
                    // blocks empty on some grids; the balanced table has no empty blocks when there
                    // are as many groups as blocks) has no slots; its staged slice is never read.
                    if (colbase[(size_t)b] >= colbase[(size_t)b + 1]) continue;
                    const uint32_t nb = rowbase[b + 1] - rowbase[b];
                    // The row slot the column pass wants is the *block-local* one (the index into its
                    // own f.sm_* tables, exactly what f.row2slot holds), not the global position in
                    // rowlist.
                    for (uint32_t j = 0; j < nb; ++j) pos[rowlist[rowbase[b] + j]] = j;
                    const uint32_t c0 = colbase[(size_t)b];
                    const uint32_t c1 = colbase[(size_t)b + 1];
                    const uint32_t npos = c1 - c0;
                    uint8_t *const dst = rslot_host.data() + (size_t)b * (size_t)rslot_bytes;
                    uint32_t *const w0 = (uint32_t *)dst;
                    uint32_t *const w1 = w0 + npos;
                    uint8_t *const b8 = (uint8_t *)(w0 + 2u * npos);
                    for (uint32_t p = c0; p < c1; ++p) {
                        const uint32_t lp = p - c0;
                        const uint32_t s0 = (p >> 5) * (32u * (uint32_t)kRegDeg) + (p & 31u);
                        uint32_t lo = 0u, hi = 0u, last = 0u;
                        for (uint32_t i = 0; i < (uint32_t)kRegDeg; ++i) {
                            const uint32_t js = (uint32_t)(uint8_t)pos[rind_host[s0 + 32u * i]];
                            if (i < 4u) lo |= (js & 0xffu) << (8u * i);
                            else if (i < 8u) hi |= (js & 0xffu) << (8u * (i - 4u));
                            else last = js;
                        }
                        w0[lp] = lo; w1[lp] = hi; b8[lp] = (uint8_t)last;
                    }
                    for (uint32_t j = 0; j < nb; ++j) pos[rowlist[rowbase[b] + j]] = UINT32_MAX;
                }
            }
            g->d_rowbase = (uint32_t *)dev_alloc((size_t)(g->nblocks + 1) * 4);
            g->d_rowlist = (uint32_t *)dev_alloc((size_t)rowlist.size() * 4);
            g->d_owner_ptr = (uint32_t *)dev_alloc((size_t)(m_ + 1) * 4);
            g->d_owner_slot = (uint32_t *)dev_alloc((size_t)owner_ptr[m_] * 4);
            g->d_meta_off = (uint32_t *)dev_alloc((size_t)(g->nblocks + 1) * 4);
            g->d_merge_meta = (uint32_t *)dev_alloc((size_t)merge_meta.size() * 4);
            g->d_pm1 = (unsigned long long *)dev_alloc((size_t)owner_ptr[m_] * 2 * 8);   // two buffers
            g->d_pm2 = (unsigned long long *)dev_alloc((size_t)owner_ptr[m_] * 2 * 8);
            g->d_pcnt = (unsigned *)dev_alloc((size_t)owner_ptr[m_] * 4);
            g->d_psign = (unsigned *)dev_alloc((size_t)owner_ptr[m_] * 2 * 4);
            // The row accumulators, three buffers of m rows (m1 / m2 as
            // magnitude bits, 8 B each; the sign and count parity, 4 B).  They stay in L2 (3 * m * 20 B:
            // 112 KiB on bb144 xyz) and are indexed by row, so every block that owns an edge in a row
            // pushes into the same three words.
            g->d_am1 = (unsigned long long *)dev_alloc((size_t)m_ * 3 * 8);
            g->d_am2 = (unsigned long long *)dev_alloc((size_t)m_ * 3 * 8);
            g->d_asign = (unsigned *)dev_alloc((size_t)m_ * 3 * 4);
            // The rotation: all three buffers start at the identity; from then on every shot
            // leaves the two buffers the next shot pushes into first at the identity (fused_merge).
            {
                const std::vector<unsigned long long> ident((size_t)m_ * 3, kDblMaxBits);
                CK(cudaMemcpyAsync(g->d_am1, ident.data(), (size_t)m_ * 3 * 8, cudaMemcpyHostToDevice, g->stream));
                CK(cudaMemcpyAsync(g->d_am2, ident.data(), (size_t)m_ * 3 * 8, cudaMemcpyHostToDevice, g->stream));
                CK(cudaMemsetAsync(g->d_asign, 0, (size_t)m_ * 3 * 4, g->stream));
                CK(cudaStreamSynchronize(g->stream));   // ident is a local: the async copies must finish
            }
            // Iteration-parity buffer 1, laid out exactly like d_out (dec | pad | status |
            // post) so the host's hand-off copies keep their shape and offsets.  The alignment pad is zeroed once
            // here, as for d_out (a decoder=bp handle's [dec | status] copy reads it and the host never
            // uses those bytes).  Create path only; nothing allocates per shot.
            g->d_out2 = (uint8_t *)dev_alloc(g->out_bytes);
            g->d_dec2 = g->d_out2;
            g->d_status2 = (uint32_t *)(g->d_out2 + g->status_off);
            g->d_post2 = (double *)(g->d_out2 + g->post_off);
            if (g->status_off > (size_t)n_) {
                CK(cudaMemsetAsync(g->d_out2 + n_, 0, g->status_off - (size_t)n_, g->stream));
            }
            CK(cudaMemcpyAsync(g->d_rowbase, rowbase.data(), (size_t)(g->nblocks + 1) * 4,
                               cudaMemcpyHostToDevice, g->stream));
            CK(cudaMemcpyAsync(g->d_rowlist, rowlist.data(), (size_t)rowlist.size() * 4,
                               cudaMemcpyHostToDevice, g->stream));
            CK(cudaMemcpyAsync(g->d_owner_ptr, owner_ptr.data(), (size_t)(m_ + 1) * 4,
                               cudaMemcpyHostToDevice, g->stream));
            CK(cudaMemcpyAsync(g->d_owner_slot, owner_slot.data(), (size_t)owner_ptr[m_] * 4,
                               cudaMemcpyHostToDevice, g->stream));
            CK(cudaMemcpyAsync(g->d_meta_off, meta_off.data(), (size_t)(g->nblocks + 1) * 4,
                               cudaMemcpyHostToDevice, g->stream));
            CK(cudaMemcpyAsync(g->d_merge_meta, merge_meta.data(), (size_t)merge_meta.size() * 4,
                               cudaMemcpyHostToDevice, g->stream));
            // The fused kernel's dynamic shared memory is ns_max * 32 B + m * 4 B; the schedule is
            // used only if the device accepts that opt-in and the kernel then still places the grid.
            // The opt-in is set to the device's own maximum, not to this problem's size: the setting
            // is per kernel and process-global, so a second handle (another problem, another thread)
            // must not be able to shrink it below what this handle's launches pass.
            int max_optin = 0;
            const bool have_optin =
                cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) == cudaSuccess &&
                max_optin > 0;
            cudaGetLastError();
            g->lay.max_optin = have_optin ? (uint32_t)max_optin : 0u;   // recorded in the layout record
            const size_t cap = 128u * 1024u;
            const bool ilv_ = (g->ilv != 0);
            // The opt-in attribute is the largest dynamic size the kernel accepts: the device's
            // maximum minus the kernel's own static shared memory (setting the bare device maximum
            // is cudaErrorInvalidValue).  Static and dynamic together are what the limit covers, so
            // this is the same value for every problem and the setting never shrinks.
            auto set_max_smem = [&](const void *fn) {
                cudaFuncAttributes finfo{};
                if (cudaFuncGetAttributes(&finfo, fn) != cudaSuccess) return false;
                const int md = (int)((size_t)max_optin - finfo.sharedSizeBytes);
                return md >= 0 && cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                       md) == cudaSuccess;
            };
            // Try one variant's dynamic size: opt in, then co-residency.  The caller tries the
            // row-slot-staged variant first and falls back to the base's fused kernel (never to the
            // two-barrier path for a problem the base ran fused).
            auto try_fused = [&](bool rslot, size_t shb) -> bool {
                if (!have_optin || shb > (size_t)max_optin || shb > cap) return false;
                bool ok = true;
                ok &= set_max_smem(pick_kernel<kRN>(1, ilv_, rslot));
                ok &= set_max_smem(pick_kernel<kRD>(1, ilv_, rslot));
                ok &= set_max_smem(pick_kernel<kRU>(1, ilv_, rslot));
                ok &= set_max_smem(pick_kernel<kRZ>(1, ilv_, rslot));
                int fper_sm = 0;
                if (ok) {
                    ok &= cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fper_sm, pick_kernel<kRN>(1, ilv_, rslot),
                                                                        kThreadsSum, shb) == cudaSuccess;
                    ok &= fper_sm > 0 && (long long)fper_sm * (long long)g->sms >= (long long)g->nblocks;
                }
                if (ok) {
                    g->sum_mode = 1;
                    g->rslot = rslot ? 1 : 0;
                    g->fused_shbytes = shb;
                    g->rslot_bytes = rslot ? rslot_bytes : 0u;   // where fused_meta lands
                }
                cudaGetLastError();   // the attribute probe above is allowed to fail
                return ok;
            };
            // The b2c window -- every edge round of the
            // block's columns, one double per ILV slot of the block's padded range -- after the rslot and merge-
            // metadata stages, sized by the largest block (rslot_bytes slots x 8 B; the balanced blocks
            // differ in width).  The gate tries the window first (the opt-in limit + the co-residency probe in
            // try_fused); when it does not fit, the same RSLOT kernel runs with the global d.b2c (win_rounds 0).
            const size_t win_bytes = ilv_ ? (size_t)rslot_bytes * 8u : 0u;
            g->win_off = 0u; g->win_rounds = 0u;
            bool rslot_ok = false;
            if (ilv_ && !rslot_host.empty()) {
                const size_t wb = tables_rec + rslot_bytes + meta_bytes;
                if (win_bytes != 0u && try_fused(true, wb + win_bytes)) {
                    g->win_off = (uint32_t)wb;
                    g->win_rounds = kRegDeg;
                    rslot_ok = true;
                } else {
                    rslot_ok = try_fused(true, wb);
                }
                g->lay.win_bytes = g->win_rounds ? (uint32_t)win_bytes : 0u;
                g->lay.win_rounds = g->win_rounds;
#if defined(FASTBP_TEST_LAYOUT_PROBE) && FASTBP_TEST_LAYOUT_PROBE
                g->lay.win_request = win_bytes != 0u ? (uint32_t)(wb + win_bytes) : 0u;   // layout record
#endif
            }
            if (rslot_ok) {
                {
                    // One region per block, rslot_bytes wide; the kernel
                    // stages its own region (blockIdx.x * rslot_stride) rounded up to 16 B, so the
                    // allocation is exactly nblocks * rslot_bytes (every byte inside a region is
                    // defined: the host table covers each block's kRegDeg * npos bytes and zeroes the
                    // rest of the region).
                    const size_t rslot_alloc = (size_t)g->nblocks * (size_t)rslot_bytes;
                    g->d_rslot = (uint8_t *)dev_alloc(rslot_alloc);
                    CK(cudaMemsetAsync(g->d_rslot, 0, rslot_alloc, g->stream));
                    CK(cudaMemcpyAsync(g->d_rslot, rslot_host.data(), rslot_host.size(),
                                       cudaMemcpyHostToDevice, g->stream));
                }
            } else {
                try_fused(false, tables + meta_bytes);
            }
            if (g->sum_mode != 1) {   // no fused schedule for this problem: the two-barrier summary path
                g->fused_shbytes = 0;
                g->win_off = 0u; g->win_rounds = 0u;   // no b2c window
                g->lay.win_rounds = 0u; g->lay.win_bytes = 0u;
                g->rslot = 0;
                g->rslot_bytes = 0;
                g->meta_max = 0;
                cudaFree(g->d_meta_off); g->d_meta_off = nullptr;
                cudaFree(g->d_merge_meta); g->d_merge_meta = nullptr;
                cudaFree(g->d_rowbase); g->d_rowbase = nullptr;
                cudaFree(g->d_rowlist); g->d_rowlist = nullptr;
                cudaFree(g->d_owner_ptr); g->d_owner_ptr = nullptr;
                cudaFree(g->d_owner_slot); g->d_owner_slot = nullptr;
                cudaFree(g->d_pm1); g->d_pm1 = nullptr;
                cudaFree(g->d_pm2); g->d_pm2 = nullptr;
                cudaFree(g->d_pcnt); g->d_pcnt = nullptr;
                cudaFree(g->d_psign); g->d_psign = nullptr;
                cudaFree(g->d_am1); g->d_am1 = nullptr;
                cudaFree(g->d_am2); g->d_am2 = nullptr;
                cudaFree(g->d_asign); g->d_asign = nullptr;
                cudaFree(g->d_out2);
                g->d_out2 = nullptr; g->d_post2 = nullptr; g->d_dec2 = nullptr; g->d_status2 = nullptr;
            }
        }
        // Caveat on the wide-grid preference (measured: bb144 z without the window at 82 SMs, a05c15 +4.7 us;
        // bb72 z loss1e-2 at 84 SMs, the converged class +11.9 us p50 on the fused non-ILV kernel): the
        // wide-grid preference holds only where the fused schedule it chose runs with the b2c window.  When the
        // term alone moved the shape and the window did not fit (no fused schedule, the fused non-ILV kernel,
        // or the ILV RSLOT kernel with the global b2c), create starts over without the term, so the handle is
        // the one the in-place rule builds.  Create path only; at most one retry.
        if (wide_only && !(g->sum_mode == 1 && g->win_rounds != 0u)) {
            delete g;
            g = nullptr;
            return setup(m_, n_, colptr, rowind, osd0, false);
        }

        // The mapped control block and the device-side pieces of the persistent kernel.  All of
        // it is created here (nothing on the decode path allocates); a handle that never decodes
        // never launches the kernel, and one that is not alone on the device never uses it.
        // The mapped status block every handle publishes into (see StatusCtl).  Pinned and
        // mapped: the kernel writes it, the host reads it after the shot's stream synchronisation.
        CK(cudaHostAlloc((void **)&g->h_status, sizeof(StatusCtl), cudaHostAllocMapped));
        std::memset((void *)g->h_status, 0, sizeof(StatusCtl));
        CK(cudaHostGetDevicePointer((void **)&g->d_status_pub, g->h_status, 0));
        CK(cudaMemcpyAsync(g->d_cptr, cptr.data(), (size_t)(n_ + 1) * 4, cudaMemcpyHostToDevice, g->stream));
        CK(cudaMemcpyAsync(g->d_rptr, rptr.data(), (size_t)(m_ + 1) * 4, cudaMemcpyHostToDevice, g->stream));
        CK(cudaMemcpyAsync(g->d_redge, redge.data(), (size_t)g->nnz * 4, cudaMemcpyHostToDevice, g->stream));
        CK(cudaMemcpyAsync(g->d_rcol, rcol.data(), (size_t)g->nnz * 4, cudaMemcpyHostToDevice, g->stream));
        // The default prior LLRs live in the caller's channel probabilities: set_priors() uploads them.
        std::vector<double> zeros(n_ > 0 ? n_ : 1, 0.0);
        CK(cudaMemcpyAsync(g->d_prior, zeros.data(), (size_t)n_ * 8, cudaMemcpyHostToDevice, g->stream));
        // prior_pos: the position-ordered copy of the priors just uploaded, for the
        // interleaved layout's column pass.  Ordered on the stream after both the prior upload and the
        // layout's cperm upload above; set_priors refills it whenever the caller replaces the priors.
        if (g->ilv != 0) {
            prior_pos_kernel<<<(n_ + 255u) / 256u, 256, 0, g->stream>>>(g->d_prior, g->d_cperm,
                                                                        g->d_prior_pos, n_);
            CK(cudaGetLastError());
        }
        // All four words, the accumulator rotation R = 0 included (the accumulators start
        // at the identity, set where they are allocated).
        CK(cudaMemsetAsync(g->d_mis, 0, 16, g->stream));

        // The first check pass's values for the (zero) priors uploaded above.  set_priors
        // recomputes them whenever the caller replaces the priors.
        if (m_ > 0 && n_ > 0 && sum_needed) {
            row_init_kernel<<<(m_ + 127u) / 128u, 128, 0, g->stream>>>(g->d_prior, g->d_rptr, g->d_rcol,
                                                                      g->d_sm1i, g->d_sm2i, g->d_spar0, m_);
            CK(cudaGetLastError());
        }

        // The OSD-0 order prefix.  A decoder=bposd osd_method=osd0 handle sizes
        // its key array by n and its per-block histogram slices by the grid it will launch; a bp
        // handle (osd0 == false) allocates none of it and never touches the pass.
        if (g->use_osd) {
            g->d_okey = (unsigned long long *)dev_alloc((size_t)n_ * 8);
            g->d_osamp = (unsigned long long *)dev_alloc((size_t)kOsdS * 8);
            // The one global histogram, twice over -- the shot-parity ping-pong (see
            // osd_hist_prepare): this shot pushes into one half while the shot's kernel zeroes the
            // other for the next.  Both halves are zeroed below before the warm-up kernel.
            g->d_ohist = (unsigned *)dev_alloc((size_t)(2u * kOsdHBins) * 4);
            g->d_otau = (unsigned long long *)dev_alloc(8);
            g->d_ocand = (unsigned char *)dev_alloc((size_t)kOsdCap * 16);   // P6's working buffer
            // The hand-over block (see GpuState): header [count | sorted | pad], then the
            // records (16-byte aligned: the header is 16 B).  One transfer carries both.
            g->d_hand = (unsigned char *)dev_alloc(kHandHdr + (size_t)kOsdCap * 16);
            // Zeroed once at create.  Every hand-off copies the block at a
            // fixed size (rung one kHandHdr + kOsdCapA x 16 B, the repick kHandHdr + kOsdCap x 16 B)
            // while a shot writes only its `count` records; the host reads only those, so the value of
            // the slots past the count never matters -- this only makes them defined (initcheck clean).
            CK(cudaMemsetAsync(g->d_hand, 0, kHandHdr + (size_t)kOsdCap * 16, g->stream));
            g->d_ocnt = reinterpret_cast<unsigned *>(g->d_hand);
            g->d_ocand_sort = g->d_hand + kHandHdr;
            // The same block is mapped as well as pinned, so the
            // prefix pass can publish it with system-scope stores and an OSD shot needs no D2H copy and no
            // stream synchronisation.  The D2H path stays for the cases the mapped publication does not
            // cover (a missed status poll, a converged shot's decisions, rung two's repick).
            // If the block cannot be mapped (or its device view
            // cannot be had), the handle keeps the copy path -- plain pinned memory, h_pref_dev = nullptr,
            // so Dev::hand_host is null (the pass publishes nothing) and run() never takes pref_mapped:
            // exactly the plain hand-over.  Only a failed pinned allocation fails the create.
            g->h_pref_dev = nullptr;
            if (cudaHostAlloc((void **)&g->h_pref, kHandHdr + (size_t)kOsdCap * 16, cudaHostAllocMapped) !=
                cudaSuccess) {
                cudaGetLastError();
                g->h_pref = nullptr;
                if (cudaMallocHost((void **)&g->h_pref, kHandHdr + (size_t)kOsdCap * 16) != cudaSuccess) {
                    throw std::runtime_error("pinned host allocation failed");
                }
            } else if (cudaHostGetDevicePointer((void **)&g->h_pref_dev, g->h_pref, 0) != cudaSuccess) {
                g->h_pref_dev = nullptr;
            }
            cudaGetLastError();
            // No second record buffer; the walk reads the pinned
            // hand-off block in place (order_records).
            // Rung two's device block (the hand-over block's layout,
            // zeroed once here so the slots past a shot's count are defined) and the pinned staging the
            // repick-shot copy lands in.  A failed pinned allocation fails the create (the buffer is
            // needed only on a repick shot, but the path must not fall back to a pageable copy).
            g->d_hand2 = (unsigned char *)dev_alloc(kHandHdr + (size_t)kOsdCap * 16);
            CK(cudaMemsetAsync(g->d_hand2, 0, kHandHdr + (size_t)kOsdCap * 16, g->stream));
            g->h_pref2_dev = nullptr;
            if (cudaHostAlloc((void **)&g->h_pref2, kHandHdr + (size_t)kOsdCap * 16,
                              cudaHostAllocMapped) != cudaSuccess) {
                cudaGetLastError();
                g->h_pref2 = nullptr;
                if (cudaMallocHost((void **)&g->h_pref2, kHandHdr + (size_t)kOsdCap * 16) != cudaSuccess) {
                    throw std::runtime_error("pinned host allocation failed");
                }
            } else if (cudaHostGetDevicePointer((void **)&g->h_pref2_dev, g->h_pref2, 0) != cudaSuccess) {
                g->h_pref2_dev = nullptr;   // unmapped: the tail collects only, the host copies
            }
            std::memset(g->h_pref2, 0, kHandHdr + (size_t)kOsdCap * 16);

            // Warm rung two's device path once, here, at
            // create.  A CUDA module loads a kernel on its first launch; that first
            // osd_repick_kernel launch measured 25-45 us (the repick hand-off 43.8-48.9 us on the first repick
            // decode of a process against 10-12.5 us on every later one), and a benchmark warm-up never
            // reaches a repick shot (0.4 % of the development shots take one).  The sequence is exactly the one
            // Engine::osd_repick issues (counter reset, the kernel, the 32 KB hand-over copy), with a Dev
            // that computes nothing: n = 0 makes the collection loop empty, and with the histogram zeroed
            // and osd_k = ~0u no threshold is picked, so the kernel writes no record.  Shot-independent
            // work, once per handle; setup's final synchronisation waits for it.  The decode path is
            // unchanged.
            {
                Dev wd = {};
                wd.n = 0;
                wd.okey = g->d_okey;
                wd.osamp = g->d_osamp;
                wd.ohist = g->d_ohist;                 // the half shot 1 (odd) will push into
                wd.ohist_next = g->d_ohist + kOsdHBins;
                wd.osd_lo = g->osd_lo;
                wd.osd_step = g->osd_step;
                wd.osd_iscale = g->osd_iscale;
                wd.osd_r = g->osd_r;
                wd.ohist_stride = g->ohist_stride;
                wd.ocnt = g->d_ocnt;
                wd.ocand = g->d_ocand_sort;
                wd.osd_k = ~0u;   // no bin can reach it: no threshold, no collect (n = 0 as well)
                wd.osd_cap = 0u;
                CK(cudaMemsetAsync(g->d_ohist, 0, (size_t)(2u * kOsdHBins) * 4, g->stream));
                CK(cudaMemsetAsync(g->d_ocnt, 0, 8, g->stream));
                CK(osd_repick_launch(g->rm, g->nblocks, (int)g->nthreads, g->stream, wd));
                CK(cudaGetLastError());
                CK(cudaMemcpyAsync(g->h_pref, g->d_hand, kHandHdr + (size_t)kOsdCap * 16,
                                   cudaMemcpyDeviceToHost, g->stream));
            }
        }

        CK(cudaStreamSynchronize(g->stream));
#if M6C_HOST_STAGES
        calibrate_clocks(g);   // measurement build only
#endif
        // the layout record, from the final create-time decisions
        g->lay.words = (uint32_t)(sizeof(LayoutStats) / 4u);
        g->lay.sms = (uint32_t)g->sms; g->lay.nblocks = (uint32_t)g->nblocks; g->lay.nthreads = (uint32_t)g->nthreads;
        g->lay.sum_mode = (uint32_t)g->sum_mode; g->lay.ilv = (uint32_t)g->ilv;
        g->lay.balanced = (uint32_t)g->balanced; g->lay.rslot = (uint32_t)g->rslot;
        g->lay.rec = (g->sum_mode == 1 && g->ilv && g->rslot && (g->rm == kRN || g->rm == kRZ)) ? 1u : 0u;
        g->lay.ns_max = g->ns_max; g->lay.rslot_bytes = g->rslot_bytes; g->lay.meta_max = g->meta_max;
        g->lay.fused_shbytes = (uint32_t)g->fused_shbytes;
#if defined(FASTBP_TEST_LAYOUT_PROBE) && FASTBP_TEST_LAYOUT_PROBE
        {   // (layout-probe builds only): the launched kernel's static shared bytes and the dynamic
            // bytes the driver has left for it at this grid's blocks per SM (the window's margin is measured
            // against this, not against the opt-in limit).  Queries only; a failure records 0.
            const int sm_ = g->sum_mode;
            const bool ilv = (g->ilv != 0), rs = (sm_ == 1) && (g->rslot != 0);
            const void *fn = (g->rm == kRU) ? pick_kernel<kRU>(sm_, ilv, rs)
                           : (g->rm == kRD) ? pick_kernel<kRD>(sm_, ilv, rs)
                           : (g->rm == kRZ) ? pick_kernel<kRZ>(sm_, ilv, rs)
                                            : pick_kernel<kRN>(sm_, ilv, rs);
            cudaFuncAttributes fa{};
            if (cudaFuncGetAttributes(&fa, fn) == cudaSuccess) {
                g->lay.fused_static = (uint32_t)fa.sharedSizeBytes;
                g->lay.fused_max_dyn_attr = (uint32_t)fa.maxDynamicSharedSizeBytes;   // layout record
            }
            const int per_sm = g->sms > 0 ? (g->nblocks + g->sms - 1) / g->sms : 1;
            size_t avail = 0;
            if (cudaOccupancyAvailableDynamicSMemPerBlock(&avail, fn, per_sm < 1 ? 1 : per_sm, g->nthreads) == cudaSuccess)
                g->lay.fused_avail_dyn = (uint32_t)avail;
            cudaGetLastError();   // a failed query leaves no error behind
        }
#endif
    } catch (...) {
        delete g;
        throw;
    }
    gpu_ = g;
    m = m_;
    n = n_;
    // The caller's correction and posterior *are* the pinned hand-off buffer.  Nothing in
    // the decode path copies them on the host any more; the D2H lands where they are read.
    decoding = g->h_res + 0u;
    posterior = (const double *)(g->h_res + g->post_off);
    return true;
}

void Engine::layout(LayoutStats *out) const {
    if (out == nullptr) return;
    *out = gpu_ ? gpu_->lay : LayoutStats{};
    out->words = (uint32_t)(sizeof(LayoutStats) / 4u);
}

bool Engine::set_priors(const double *prior_llr_n) {
    if (!gpu_ || n == 0) return true;
    GpuState *g = gpu_;
    try {
        // Every step below can fail: the upload, the prior_pos / row_init launches, the sync.  They are
        // ordered on the handle's stream and complete before the call returns, so a successful call
        // leaves the handle consistent.  A failure after the upload was enqueued leaves the device
        // partly on the new priors (d_prior new; the position copy, the first-pass arrays or the OSD
        // window possibly old or new): the caller then re-runs this with its committed priors, which
        // rewrites all of them, and returns an error (dbench_set_priors).
        CK(cudaMemcpyAsync(g->d_prior, prior_llr_n, (size_t)n * 8, cudaMemcpyHostToDevice, g->stream));
        // prior_pos: refill the position-ordered copy on the same stream (ordered
        // after the upload above); create's fill already ran for the zero priors.
        if (g->ilv != 0) {
            prior_pos_kernel<<<(n + 255u) / 256u, 256, 0, g->stream>>>(g->d_prior, g->d_cperm,
                                                                       g->d_prior_pos, n);
            CK(cudaGetLastError());
        }
        // The first check pass's (m1, m2, count) depend only on the priors, so they are
        // recomputed here, on the device, once per set_priors -- never in the decode path.
        if (g->m > 0 && n > 0 && g->sum_mode != 0) {
            row_init_kernel<<<(g->m + 127u) / 128u, 128, 0, g->stream>>>(g->d_prior, g->d_rptr, g->d_rcol,
                                                                        g->d_sm1i, g->d_sm2i, g->d_spar0,
                                                                        g->m);
            CK(cudaGetLastError());
        }
        // The pushed histogram's window, a function of the priors only: P = the largest
        // finite |prior|, the window [-2.5 P, +2.5 P] split into kOsdHBins equal-width value bins.
        // Host work at create / set_priors, never on the decode path.  P = 0 (all priors zero) and
        // non-finite priors fall back to P = 1: the window only decides how the threshold is picked --
        // any bin edge is a valid tau, so a badly placed window costs a shorter set (a repick or the
        // exact full fallback), never exactness.
        if (g->use_osd) {
            double p = 0.0;
            for (uint32_t j = 0; j < n; ++j) {
                const double a = std::fabs(prior_llr_n[j]);
                if (std::isfinite(a) && a > p) p = a;
            }
            if (!(p > 0.0)) p = 1.0;
            g->osd_lo = -kOsdHWinMul * p;
            const double hi = kOsdHWinMul * p;
            g->osd_step = (hi - g->osd_lo) / (double)kOsdHBins;
            g->osd_iscale = (double)kOsdHBins / (hi - g->osd_lo);
        }
        CK(cudaStreamSynchronize(g->stream));
    } catch (const std::exception &) {
        return false;
    }
    return true;
}

uint32_t Engine::run(const uint8_t *syn01, uint32_t max_iter, double alpha, bool need_post) {
    GpuState *g = gpu_;
    posterior_valid = false;

    Dev d{};
    d.cptr = g->d_cptr; d.rptr = g->d_rptr; d.redge = g->d_redge;
    d.rcol = g->d_rcol; d.rind = g->d_rind;
    d.cperm = g->d_cperm; d.cdeg = g->d_cdeg; d.rslot = g->d_rslot; d.ilv = (uint32_t)g->ilv;
    d.colbase = g->d_colbase;
    d.sm1 = g->d_sm1; d.sm2 = g->d_sm2; d.spar = g->d_spar;
    d.sm1i = g->d_sm1i; d.sm2i = g->d_sm2i; d.spar0 = g->d_spar0;
    d.sum_mode = (uint32_t)g->sum_mode;
    d.prior = g->d_prior; d.prior_pos = g->d_prior_pos;   // prior_pos
    d.b2c = g->d_b2c; d.c2b = g->d_c2b; d.post = g->d_post;
    d.dec = g->d_dec; d.syn = g->d_syn; d.mis = g->d_mis;
    d.status = g->d_status;
    d.rowbase = g->d_rowbase; d.rowlist = g->d_rowlist;
    d.owner_ptr = g->d_owner_ptr; d.owner_slot = g->d_owner_slot;
    d.meta_off = g->d_meta_off; d.merge_meta = g->d_merge_meta;   // the merge metadata
    d.rslot_bytes = g->rslot_bytes;
    d.rslot_stride = g->rslot_bytes;   // the per-block region stride
    d.win_off = g->win_off; d.win_rounds = g->rslot ? g->win_rounds : 0u;   // the b2c window
    d.pm1 = g->d_pm1; d.pm2 = g->d_pm2; d.pcnt = g->d_pcnt; d.psign = g->d_psign;
    d.am1 = g->d_am1; d.am2 = g->d_am2; d.asign = g->d_asign; d.am_stride = g->m;   // the L2 accumulators
    d.post2 = g->d_post2; d.dec2 = g->d_dec2; d.status2 = g->d_status2;   // the parity buffers
    d.ns_max = g->ns_max;
    d.pm_stride = g->pm_stride;
    d.m = g->m; d.n = g->n; d.max_iter = max_iter; d.alpha = alpha; d.wpr = g->wpr;
    d.okey = g->d_okey; d.osamp = g->d_osamp; d.otau = g->d_otau;
    d.ocnt = g->d_ocnt; d.ocand = g->d_ocand; d.ocand_sorted = g->d_ocand_sort;
    // The prefix pass publishes the hand-over into this mapped block (and
    // the host then skips the D2H copy and the synchronisation of an OSD shot); a handle without the
    // OSD arrays keeps the null pointer and the pass never runs.
    d.hand_host = g->use_osd ? g->h_pref_dev : nullptr;
    // Rung two's device block, the tail collects into it; the
    // mapped view of the handle's rung-two block and of its generation word (null for a bp handle and
    // for a handle whose block could not be mapped, and then the tail collects only).
    d.ocand2 = g->d_hand2 ? (g->d_hand2 + kHandHdr) : nullptr;
    d.ocnt2 = g->d_hand2 ? reinterpret_cast<unsigned *>(g->d_hand2) : nullptr;
    d.hand2 = (g->use_osd != 0) ? g->h_pref2_dev : nullptr;
    d.pub2 = reinterpret_cast<unsigned long long *>(g->d_status_pub) + 2;   // StatusCtl::gen2
    d.tail_on = (max_iter <= kOsdTailMaxIter) ? 1u : 0u;   // the tail's cap gate
    d.ohist_stride = g->ohist_stride; d.use_osd = (unsigned)g->use_osd;
    // The kernel's pass is rung one of the OSD-0 prefix chain (the small K and cap); the
    // repick kernel the host may launch afterwards sets rung two's values in its own Dev.
    d.osd_k = kOsdK; d.osd_cap = kOsdCapA;
    d.osd_r = g->osd_r;
    // Where the kernel publishes the shot's status (see StatusCtl) and this shot's generation.
    d.pub = reinterpret_cast<unsigned long long *>(g->d_status_pub);
    const unsigned long long shot_gen = ++g->shot_gen;
    // This shot's half of the histogram ping-pong (osd_hist_prepare zeroes the other
    // one inside the kernel, for the next shot), and the window set_priors fixed.
    g->hist_sel = (unsigned)(shot_gen & 1ull);
    d.ohist = g->d_ohist + (size_t)g->hist_sel * kOsdHBins;
    d.ohist_next = g->d_ohist + (size_t)(1u - g->hist_sel) * kOsdHBins;
    d.osd_lo = g->osd_lo; d.osd_step = g->osd_step; d.osd_iscale = g->osd_iscale;

    // The syndrome of a problem that fits the parameter budget (m <= kSynWords * 64) is
    // bit-packed into the launch parameters here, on the host, and is read by the kernel out of the
    // constant bank: no H2D, no device syndrome array, no in-stream copy to order the launch behind.
    // The mask is the same one the device array got (syn01[i] & 1); the kernel measures the initial
    // mismatch itself.  Wider problems keep the H2D (both are on the handle's stream, so the launch
    // stays ordered behind the copy).
    if (g->m <= kSynWords * 64u) {
        d.use_synparam = 1u;
        for (uint32_t w = 0; w < kSynWords; ++w) d.syn_bits[w] = 0ull;
        // Detectors are packed 64 at a time into one parameter word: the
        // eight bytes of a group are taken as one 64-bit load and their low bits compressed with the
        // byte-pack multiply -- ((v & 0x0101..) * 0x0102040810204080) >> 56 puts byte j's bit 0 at
        // bit j -- instead of eight byte loads, eight tests and seven shifts/ors (the straight-line
        // form measured in a standalone probe: 0.631 us at m = 1872 against 2.995 byte by byte);
        // and the whole 64-detector word is assembled in a register (one store per word instead of
        // eight read-modify-write ORs).  The masked bit is exactly `syn01[i] & 1` (the multiply works
        // on any byte values because the mask takes bit 0), and the tail (m not a multiple of 64, and
        // the last m % 8 detectors) is the same two loops as before.
        uint32_t i = 0;
        uint32_t w = 0;
        for (; i + 64u <= g->m; i += 64u, ++w) {
            unsigned long long word = 0ull;
            for (uint32_t g8 = 0; g8 < 64u; g8 += 8u) {
                unsigned long long v;
                std::memcpy(&v, syn01 + i + g8, 8);
                const unsigned long long packed =
                    ((v & 0x0101010101010101ull) * 0x0102040810204080ull) >> 56;
                word |= packed << g8;
            }
            d.syn_bits[w] = word;
        }
        for (; i + 8u <= g->m; i += 8u) {
            unsigned long long v;
            std::memcpy(&v, syn01 + i, 8);
            const unsigned long long packed =
                ((v & 0x0101010101010101ull) * 0x0102040810204080ull) >> 56;
            d.syn_bits[i >> 6] |= packed << (i & 63u);
        }
        for (; i < g->m; ++i) {
            d.syn_bits[i >> 6] |= (unsigned long long)(syn01[i] & 1u) << (i & 63u);
        }
    } else {
        d.use_synparam = 0u;
        for (uint32_t i = 0; i < g->m; ++i) g->h_syn[i] = (uint8_t)(syn01[i] & 1u);
        CK(cudaMemcpyAsync(g->d_syn, g->h_syn, g->m, cudaMemcpyHostToDevice, g->stream));
    }
    unsigned long long gen_arg = shot_gen;
    void *args[] = {&d, &gen_arg};
    dim3 grid(g->nblocks), block((unsigned)g->nthreads);
    const bool ilv = (g->ilv != 0);
    const int sm_ = g->sum_mode;
    const bool rslot = (sm_ == 1) && (g->rslot != 0);
    const size_t shb = (sm_ == 1) ? g->fused_shbytes : 0u;
    const void *fn = (g->rm == kRU) ? pick_kernel<kRU>(sm_, ilv, rslot)
                   : (g->rm == kRD) ? pick_kernel<kRD>(sm_, ilv, rslot)
                   : (g->rm == kRZ) ? pick_kernel<kRZ>(sm_, ilv, rslot)
                                    : pick_kernel<kRN>(sm_, ilv, rslot);
    CK(cudaLaunchCooperativeKernel(fn, grid, block, args, shb, g->stream));
#if M6C_HOST_STAGES
    const uint64_t stg_launched = dbench_stage_now();   // host clock at the launch's return
#endif
    // The plain hand-off is one D2H of [dec (n) | status (8)] plus the prefix and
    // its count, all issued before the synchronisation.  The kernel publishes this shot's status and
    // generation into the mapped block at the end of its body; the bounded poll below (no clock read
    // on the decode path) waits for that generation, so the host knows which copies this shot needs
    // *before* it issues them, in the plain pipeline: a shot that converged has the BP decision
    // as its correction and no OSD-0 prefix, a shot that did not converge with osd0 never reads the
    // decisions.  A shot the poll does not catch in its budget (a very long body, or a kernel that
    // died) takes the plain unconditional hand-off, which is also what a decoder=bp handle takes.
    bool have_status = false, converged = false;
    uint32_t polled_iters = 0;
    if (g->use_osd) {
        for (unsigned long long i = 0; i < kStatusPollBudget; ++i) {
            if (__atomic_load_n(&g->h_status->gen, __ATOMIC_ACQUIRE) == shot_gen) {
                have_status = true;
#if M6C_HOST_STAGES
                // The stage tool ignores accumulators 1-3, so they carry the
                // mapped path's split for the split tool: [1] launch return -> first observed
                // generation (host clock), [2] the kernel's body on the device clock (globaltimer).
                const uint64_t stg_seen = dbench_stage_now();
                dbench_stage_tls.acc[1] = stg_seen - stg_launched;
                dbench_stage_tls.acc[2] = g->h_status->t_dev1 - g->h_status->t_dev0;
                // marks 11-15, which the stage tool records but never reads (its marks are
                // 0-10), carry the per-shot split for the calibration split tool: [11] launch return (host
                // CLOCK_MONOTONIC), [12] kernel entry (%globaltimer) - launch return, [13] first observed
                // generation - status publish (%globaltimer), both raw (each holds theta: L = [12] - theta,
                // O = [13] + theta, L + O exact), [14] / [15] the handle's theta bound (calibrate_clocks).
                dbench_stage_tls.mark[11] = stg_launched;
                dbench_stage_tls.mark[12] = (uint64_t)(g->h_status->t_dev0 - stg_launched);
                dbench_stage_tls.mark[13] = (uint64_t)(stg_seen - g->h_status->t_dev1);
                dbench_stage_tls.mark[14] = (uint64_t)g->calib_lo;
                dbench_stage_tls.mark[15] = (uint64_t)g->calib_hi;
#endif
                break;
            }
            __builtin_ia32_pause();
        }
        if (have_status) {
            const unsigned long long pack = __atomic_load_n(&g->h_status->st, __ATOMIC_ACQUIRE);
            polled_iters = (uint32_t)pack;
            converged = (pack >> 32) != 0ull;
        }
    }
    // Which iteration-parity buffer holds this shot's final state.  The fused body's
    // column pass of iteration it writes buffer it & 1, and every exit path reports the iteration
    // whose state it leaves in the status word, so the buffer is (iteration & 1).  The poll's status
    // names the iteration before the copies are issued; when there was no poll (a decoder=bp handle,
    // or a shot whose body outran the poll budget) the parity is predicted from max_iter -- exact for
    // every shot that runs to the cap, which is every non-converged shot -- and corrected after the
    // status read below when the shot converged early in the other parity.  A kernel without the
    // double buffer (sum_mode != 1) writes buffer 0 only.
    const bool fused_parity = (g->sum_mode == 1);
    const uint32_t predicted_parity = fused_parity ? (max_iter & 1u) : 0u;
    const uint32_t read_parity = have_status ? (polled_iters & 1u) : predicted_parity;
    const uint8_t *const src_out = (fused_parity && read_parity) ? g->d_out2 : g->d_out;
    const double *const src_post = (fused_parity && read_parity) ? g->d_post2 : g->d_post;
    bool copied = false;   // any D2H issued below (then the stream must be synchronised)
    if (g->use_osd == 0 || !have_status) {
        CK(cudaMemcpyAsync(g->h_res, src_out, g->pre_bytes, cudaMemcpyDeviceToHost, g->stream));
        copied = true;
    } else if (converged) {
        // The correction is the BP decision (n bytes; the plain copy carries the status with it,
        // which the mapped block has already given the host).
        CK(cudaMemcpyAsync(g->h_res, src_out, (size_t)g->n, cudaMemcpyDeviceToHost, g->stream));
        copied = true;
    }
    if (need_post) {
        CK(cudaMemcpyAsync(g->h_res + g->post_off, src_post, (size_t)g->n * 8,
                           cudaMemcpyDeviceToHost, g->stream));
        copied = true;
    }
    // Rung one: the shot's order prefix, in the same round trip as the
    // decision and the status.  Nothing is copied on a handle that never takes the OSD-0 path.  A
    // converged shot (or one whose kernel returned early) left the counter at 0, so the copy carries
    // "no set at this rung" and the host moves on (rung two, or the full posterior).  Rung one's cap
    // is the small one, so the copy enqueued on every shot of a use_osd handle is kOsdCapA x 16 B
    // instead of kOsdCap x 16 B.  A single transfer on the header-first
    // block: the header [count | sorted] and rung one's records are ONE copy of
    // kHandHdr + kOsdCapA x 16 bytes.
    // A shot the poll saw converge needs no prefix at all.
    // On a non-converged OSD shot whose status the poll saw, the
    // whole hand-over block is already in the mapped pinned buffer -- the prefix pass published it and made
    // it visible at system scope before the status release the poll observed -- so this shot needs neither
    // a device-to-host copy nor a stream synchronisation.  Every other case (a missed poll, a converged
    // shot's decisions, a bp handle, a posterior copy) keeps the plain transfers and syncs.  The
    // host then walks the records where the device put them.
    const bool pref_mapped = (g->use_osd != 0) && (g->h_pref_dev != nullptr) && have_status && !converged;
    // Whether this shot's kernel ran the rung-two tail (and so may
    // have written this shot's rung-two set).  The device runs it on every non-converged OSD shot; a
    // shot the poll saw converge published no rung two.  A missed poll leaves the flag set (the
    // kernel's own verdict is unknown here); such a shot's set, if any, is this shot's -- the block is
    // written only under this shot's kernel and read only after the stream-ordered copy below -- and a
    // converged shot never takes the rung-two path at all.
    g->tail_armed = (g->use_osd != 0 && g->d_hand2 != nullptr && max_iter <= kOsdTailMaxIter &&
                     !(have_status && converged)) ? 1 : 0;
#if M6C_HOST_STAGES
    dbench_stage_tls.acc[3] = pref_mapped ? 1u : 0u;   // [3] 1 = this shot took the mapped hand-over
#endif
    if (g->use_osd && (!have_status || !converged) && !pref_mapped) {
        CK(cudaMemcpyAsync(g->h_pref, g->d_hand, kHandHdr + (size_t)kOsdCapA * 16, cudaMemcpyDeviceToHost,
                           g->stream));
        copied = true;
    }
    if (copied) CK(cudaStreamSynchronize(g->stream));

    // The status of this shot.  When the poll above caught the kernel's publication, the two
    // words are that shot's and are already known; otherwise they come from the plain copy in h_res.
    uint32_t it = 0;
    if (g->use_osd && have_status) {
        it = polled_iters;
        last_converged = converged;
    } else {
        const uint32_t *st = (const uint32_t *)(g->h_res + g->status_off);
        it = st[0];
        last_converged = (st[1] != 0);
    }
    // A missed poll's copies were issued for the predicted parity.  A shot that converged
    // early leaves its final state in the other buffer; the status word just read says which, and only
    // then is a second copy needed (the copies are stream-ordered after the kernel, so every write is
    // complete).  Non-converged shots run to max_iter and never come here.
    const uint32_t final_parity = fused_parity ? (it & 1u) : 0u;
    if (!have_status && final_parity != predicted_parity) {
        const uint8_t *const fix = final_parity ? g->d_out2 : g->d_out;
        CK(cudaMemcpyAsync(g->h_res, fix, g->pre_bytes, cudaMemcpyDeviceToHost, g->stream));
        if (need_post) {
            CK(cudaMemcpyAsync(g->h_res + g->post_off, fix + g->post_off, (size_t)g->n * 8,
                               cudaMemcpyDeviceToHost, g->stream));
        }
        CK(cudaStreamSynchronize(g->stream));
    }
    last_post_parity = final_parity;
    if (need_post) posterior_valid = true;
    if (g->use_osd && (!have_status || !converged)) {
        // A usable rung-one set is 1..kOsdCapA records (its threshold rule guarantees the count is
        // inside the cap); 0 means "this rung has no set" -- a converged shot, a shot whose kernel
        // returned early, or a threshold neither the small K nor the cap could serve.
        const unsigned pn = *reinterpret_cast<const unsigned *>(g->h_pref);
        order_cand_n = (pn >= 1u && pn <= kOsdCapA) ? pn : 0u;
        // The device sorted the records (P6's gate) or it did not: the host must walk them the same
        // way the device left them.
        order_sorted = (order_cand_n > 0) && (*reinterpret_cast<const unsigned *>(g->h_pref + 4u) == 1u);
        // the walk reads the records in the pinned staging block
        // itself; no host copy.  The block stays untouched until this handle's next run()/osd_repick().
        order_records = order_cand_n ? static_cast<const void *>(g->h_pref + kHandHdr) : nullptr;
    } else {
        // A converged shot has no usable prefix (the count is 0 and no records were collected): the
        // host takes the full posterior / the decision, exactly as the plain path does for such a shot.
        order_cand_n = 0;
        order_sorted = false;
        order_records = nullptr;
    }
    return it;
}

// Rung two: the larger prefix, from the sample and the histogram rung one published (their
// selection does not depend on K, so this pass only re-derives the threshold and re-collects).  The
// host calls it on a non-converged OSD-0 shot whose walk ran past rung one (or whose rung one had no
// set at all).  Everything here is this shot's own data -- no reuse of an earlier shot's order,
// pivots, posterior or flags -- and the walk over the result is the same reference order, so the
// result is bit-identical to a prefix collected in one rung and to the full-order fallback.
bool Engine::keep_warm_gpu() {
    GpuState *g = gpu_;
    if (g == nullptr) return false;
    try {
        // The launch's own error is checked (an interposed failure sets no last error), then the last error.
        CK(cudaLaunchKernel((const void *)keep_warm_kernel, dim3(1u), dim3(32u), nullptr, 0, g->stream));
        CK(cudaGetLastError());
    } catch (const std::exception &) {
        return false;
    }
    return true;
}

bool Engine::osd_repick() {
    GpuState *g = gpu_;
    repick_failed = false;   // set by the two failure paths below
    if (!g || !g->use_osd) return false;
    // On a shot whose kernel ran the tail (a non-converged OSD
    // shot), rung two is already collected in the handle's device block.  The host reads it in place
    // (mapped block, generation-checked) or with one stream-ordered D2H -- no launch, no memset and no
    // second collect.  Anything else (a shot that did not run the tail, a handle without the buffer)
    // takes the repick kernel below, which produces the same set.
    if (g->tail_armed && g->d_hand2 != nullptr) {
        order_sorted = false;   // rung two arrives unsorted (the rank sort is rung one's only)
        // (1) The mapped publication: the kernel's tail published this shot's set into the handle's
        // mapped block -- every covered record visible at system scope (__threadfence_system) before
        // the generation word's system-scope release store.  The host polls the generation (bounded,
        // no clock read, the status poll's pattern); once observed (an acquire load), the set is read
        // in place: no launch, no copy, no synchronisation.  Generation `gen` is unique to the shot, so
        // no other shot's set can pass the check.
        bool have_set = false;
        if (g->h_pref2_dev != nullptr) {
            const unsigned long long want = g->shot_gen;
            for (unsigned long long i = 0; i < kTailPollBudget; ++i) {
                if (__atomic_load_n(&g->h_status->gen2, __ATOMIC_ACQUIRE) == want) {
                    have_set = true;
                    break;
                }
                __builtin_ia32_pause();
            }
        }
        if (!have_set) {
            // (2) The stream-ordered copy: a publication the poll did not observe, or a block that
            // could not be mapped.  The copy is enqueued on this handle's stream (after this shot's
            // kernel, so the set is the shot's own) and its completion is observed by a bounded
            // event-query spin; expiry falls back to the stream synchronisation, so a stuck copy
            // cannot hang the decode.
            try {
                CK(cudaMemcpyAsync(g->h_pref2, g->d_hand2, kHandHdr + (size_t)kOsdCap * 16,
                                   cudaMemcpyDeviceToHost, g->stream));
                CK(cudaEventRecord(g->ev, g->stream));
                bool copy_done = false;
                for (unsigned long long i = 0; i < kTailCopySpinBudget; ++i) {
                    const cudaError_t q = cudaEventQuery(g->ev);
                    if (q == cudaSuccess) { copy_done = true; break; }
                    if (q != cudaErrorNotReady) CK(q);
                    __builtin_ia32_pause();
                }
                if (!copy_done) CK(cudaStreamSynchronize(g->stream));
                have_set = true;
            } catch (const std::exception &) {
                order_cand_n = 0;
                order_records = nullptr;
                repick_failed = true;   // an error return of the decode, not the posterior fallback
                return false;
            }
        }
        const unsigned pn = *reinterpret_cast<const unsigned *>(g->h_pref2);
        order_cand_n = (pn >= 1u && pn <= kOsdCap) ? pn : 0u;
        order_records = order_cand_n ? static_cast<const void *>(g->h_pref2 + kHandHdr) : nullptr;
#if M6C_HOST_STAGES
        // The measurement build only (never shipped): mark the shot as one that took the kernel
        // tail's rung two (mark[15] = 2; the stage tool's marks are 0-10, so it ignores this), and pack
        // rung two's set size and the tail's device time (pub[5] - t_dev1) into acc[3] -- which holds
        // the mapped-hand-over flag on every other shot -- for the stage tool.
        dbench_stage_tls.mark[15] = 2;
        const int64_t ttail = (g->h_status->t_tail >= g->h_status->t_dev1)
                                  ? (int64_t)(g->h_status->t_tail - g->h_status->t_dev1) : 0;
        dbench_stage_tls.acc[3] = ((int64_t)order_cand_n << 32) | ttail;
#endif
        return true;
    }
    try {
        Dev d = {};
        d.n = g->n;
        d.okey = g->d_okey;
        d.osamp = g->d_osamp;
        // Rung two reads the histogram half the most recent run pushed into (nothing
        // zeroed it: the ping-pong halves alternate per shot), and its window is the handle's.
        d.ohist = g->d_ohist + (size_t)g->hist_sel * kOsdHBins;
        d.ohist_next = g->d_ohist + (size_t)(1u - g->hist_sel) * kOsdHBins;
        d.osd_lo = g->osd_lo;
        d.osd_step = g->osd_step;
        d.osd_iscale = g->osd_iscale;
        d.osd_r = g->osd_r;
        d.ohist_stride = g->ohist_stride;
        d.ocnt = g->d_ocnt;
        // Rung two is collected unsorted straight into the hand-over block's records, so
        // its header and records are one transfer too (kHandHdr + kOsdCap x 16 B).  The 8-byte reset
        // clears the count and the "sorted" flag word; the host walks this set with the chunk walk.
        d.ocand = g->d_ocand_sort;
        d.osd_k = kOsdK2;
        d.osd_cap = kOsdCap;
        order_sorted = false;
        CK(cudaMemsetAsync(g->d_ocnt, 0, 8, g->stream));
        CK(osd_repick_launch(g->rm, g->nblocks, (int)g->nthreads, g->stream, d));
        CK(cudaGetLastError());
        CK(cudaMemcpyAsync(g->h_pref, g->d_hand, kHandHdr + (size_t)kOsdCap * 16, cudaMemcpyDeviceToHost,
                           g->stream));
        CK(cudaStreamSynchronize(g->stream));
        const unsigned pn = *reinterpret_cast<const unsigned *>(g->h_pref);
        order_cand_n = (pn >= 1u && pn <= kOsdCap) ? pn : 0u;
        order_records = order_cand_n ? static_cast<const void *>(g->h_pref + kHandHdr) : nullptr;
        return true;
    } catch (const std::exception &) {
        order_cand_n = 0;
        order_sorted = false;
        order_records = nullptr;
        repick_failed = true;   // an error return of the decode, not the posterior fallback
        return false;
    }
}

bool Engine::fetch_posterior() {
    if (posterior_valid) return true;
    GpuState *g = gpu_;
    if (!g) return false;
    try {
        // The device buffer still holds the last shot's posteriors (no other decode ran in between,
        // and a failed decode invalidates this handle's outputs instead of leaving them stale).  The
        // copy lands directly in the pinned region posterior[] aliases.
        // The copy queues on the handle's stream behind the shot's kernel.  With the mapped
        // hand-over, on a polled non-converged OSD shot run() returned without synchronising the stream
        // (the host saw the kernel's status release, not its completion), so the stream order is what
        // makes this copy read the finished posteriors; the synchronisation below waits for both.
        cudaStream_t ds = g->stream;
        // The posterior lives in the parity buffer the last shot's final state was written
        // to (run() records it); a kernel without the double buffer always wrote buffer 0.
        const double *src = (g->sum_mode == 1 && last_post_parity) ? g->d_post2 : g->d_post;
        CK(cudaMemcpyAsync(g->h_res + g->post_off, src, (size_t)g->n * 8,
                           cudaMemcpyDeviceToHost, ds));
        CK(cudaStreamSynchronize(ds));
    } catch (const std::exception &) {
        return false;
    }
    posterior_valid = true;
    return true;
}

}  // namespace fastbp
