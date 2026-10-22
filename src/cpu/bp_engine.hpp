// fastbpmt: exact min-sum BP engine, one in-place message array, chunks padded
// only locally (SELL-C-sigma).
//
// Bit-identical to bp_decode_parallel() MINIMUM_SUM in bp.hpp of the vendored ldpc sources (ldpc 2.4.1) as called
// by ldpc-ref, in every rounding mode of the caller.  The exactness argument, the layout bound
// and the fixed capacities are
// given where they are used below.
//
// Layout.  Both sides are cut into chunks: the side is sorted by degree (descending, stable by index)
// and a chunk takes members while
//     (members) * (first member's degree) <= floor(1.3 * (sum of the members' degrees)) + 16,
// up to L members.  A chunk with L members is DENSE (vector path); a chunk that could not be filled
// keeps its fewer members and is THIN (scalar path); thin chunks are emitted after the dense ones.
// Every chunk is a (position, lane) rectangle padded to *its own* longest member, so no array is ever
// sized by a global maximum degree and the padding is at most 0.3 * nnz + 16 * (number of chunks) per
// side.
//
// There is ONE message array: the column layout.  Slot (c, p, lane) = cbase[c] + p*clanes[c] + lane
// holds the bit-to-check message b2c before the check update of an iteration and the check-to-bit
// message c2b of that edge afterwards.  The check update gathers b2c through the row-side index map
// cslot_of_rslot (row slot -> column slot) into vloc, and scatters c2b back to the same slots; the
// variable update then reads and rewrites those slots in place.  A row slot that no edge owns (a
// padding position of a row rectangle) maps to one of the T padding slots msg[cslots + t], which
// hold NaN and are therefore a no-op for every comparison the check update makes (|NaN| < x is
// false, NaN <= 0 is false, and a NaN magnitude never becomes m1 or m2).  The dense production pass
// writes every lane of the row rectangle, and the lanes without an edge at that position store NaN
// into a padding slot (the only value such a lane can write anywhere); the thin path writes only
// positions below a lane's degree, and the variable update only writes inside its own column
// rectangles.  The padding slot a padding position points at is the one of the thread that owns its
// row chunk (setup builds cslot_of_rslot that way), so no two threads ever write the same location -
// with a single shared slot every dense pass of every thread stores NaN into it, which is a data
// race even though all of them store the same value.  The invariant "every padding
// slot is NaN after every pass" therefore holds by construction: the only writes to one are NaN, by
// its owner.  (The pre-fix form masked those lanes out instead, scatter_masked; the unconditional
// store with a NaN padding value is the form used by the earlier engine.)  The development
// NaN-padding test shows the slot is read, and the
// FASTBPOSD_DEBUG_POISON_PAD build shows that a finite value there corrupts the decode.
//
// Check update.  Per row keep m1 <= m2, the two smallest |b2c| of the row (m2 == m1 when the minimum
// repeats, both DBL_MAX when the row has none) and cnt = syndrome bit + #{edges with b2c <= 0}; every
// outgoing message is then
//     (|b2c_e| == m1 ? m2 : m1) * (cnt_e odd ? -alpha : +alpha),  cnt_e = cnt + (b2c_e <= 0),
// one multiply by the signed alpha: exactly bp.hpp's
//     sgn = syndrome + #{b2c <= 0} + (b2c_e <= 0); e.check_to_bit_msg = min_{j != e} |b2c_j| * ((sgn % 2) ? -alpha : +alpha),
// whose min and count are both independent of traversal order.  The sign is an operand of the single
// multiply, never a negation after it: -(m * alpha) and m * (-alpha) are different doubles under
// FE_UPWARD / FE_DOWNWARD.
//
// Variable update, fused.  Per column chunk: the reverse pass accumulates the suffix from the last
// row back (t = 0 upward, exactly bp.hpp's part 2), the forward pass computes the running prefix from
// the prior LLR (b2c := t; t += c2b), the posterior LLR (the final t) and the hard decision (t <= 0)
// in the same visit, and the new b2c is prefix + suffix.  The suffix of the last iteration is dead
// work (bp.hpp leaves the loop before part 2 there) and is skipped by var_last.  Every column keeps
// the reference's double addition order; no reassociation, no FMA, no float32.
//
// Convergence is tracked incrementally: cand is the XOR of the columns whose hard decision is 1, and
// mismatch counts the rows where cand differs from the syndrome.  A flip XORs the column into cand
// and adjusts the count, so no per-iteration syndrome pass exists.  XOR is a commutative, associative
// GF(2) sum, so this is bit-exact and equivalent to bp.hpp's per-iteration candidate_syndrome rebuild.
#ifndef FASTBPOSD_BP_ENGINE_HPP
#define FASTBPOSD_BP_ENGINE_HPP

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
#define FASTBP_LANES 8
#elif defined(__AVX2__)
#include <immintrin.h>
#define FASTBP_LANES 4
#else
#define FASTBP_LANES 1
#endif

// The test-only switches -- FASTBPOSD_DEBUG_POISON_PAD (below), FASTBPOSD_DEBUG_TRUNCATE_ORDER
// (osd0_engine.hpp), FASTBPOSD_TEST_INT32_LIMIT (below), the two observable-path variants
// FASTBPOSD_OBS_MASK_CONVERGED / FASTBPOSD_OBS_ROW_PARITY (fastbpcpu6g.cpp) and the store variant
// FASTBPOSD_NATIVE_SCATTER (below, bit-identical but a different instruction) -- change the decode
// or the setup result.  A stray -D of one of them in a library build would silently change the
// shipped behaviour, so each is rejected unless the build also defines FASTBPOSD_TEST_BUILD, which
// only the development test command lines that need them set (the NaN-padding, order-regression,
// int32-guard, observable-mask and store-variant checks).
#if defined(FASTBPOSD_DEBUG_POISON_PAD) || defined(FASTBPOSD_DEBUG_TRUNCATE_ORDER) || \
    defined(FASTBPOSD_TEST_INT32_LIMIT) || defined(FASTBPOSD_OBS_MASK_CONVERGED) || \
    defined(FASTBPOSD_OBS_ROW_PARITY) || defined(FASTBPOSD_NATIVE_SCATTER)
#if !defined(FASTBPOSD_TEST_BUILD)
#error "a FASTBPOSD_* test-only switch is defined without -DFASTBPOSD_TEST_BUILD (these change the decode or refuse problems; they are not part of the library build)"
#endif
#endif

// The padding value the dense check pass stores for a lane without an edge at this position: NaN,
// the no-op the gathers of the later chunks need.  Test-only switch (NaN-padding test): it
// builds the same engine with +0.0 there and shows the shot changes, i.e. that the padding gathers
// do read the shared slot and that a finite value there corrupts the decode.  Never defined for the
// library build.
#ifdef FASTBPOSD_DEBUG_POISON_PAD
#define FASTBP_PAD_VALUE(v, zero_v) (zero_v)
#else
#define FASTBP_PAD_VALUE(v, zero_v) (v)
#endif

namespace fastbp {

// The int32 index limit of the slot tables: slots, lane ids and column ids must stay below 2^31
// (setup() fails above it, so dbench_create refuses the problem instead of truncating).  A problem
// that large is impractical, so a test builds the header with -DFASTBPOSD_TEST_INT32_LIMIT=N
// (the int32 guard test) and reaches the same comparisons with a small problem.
#ifdef FASTBPOSD_TEST_INT32_LIMIT
constexpr uint64_t kIndexLimit = FASTBPOSD_TEST_INT32_LIMIT;
#else
constexpr uint64_t kIndexLimit = (uint64_t)std::numeric_limits<int32_t>::max();
#endif

// ---------------------------------------------------------------------------
// Placement of the hot arrays.
//
// One arena per BP engine: every engine array -- the hot ones and the per-thread ones -- is carved
// out of ONE 2 MB-aligned allocation (posix_memalign, hinted MADV_HUGEPAGE once), at a fixed offset
// from the arena base: the running offset plus FASTBPMT_ARENA_GAP, rounded up to FASTBPMT_ARENA_GRAN,
// plus the array's own skew (a distinct multiple of 64 B per array).  The layout an engine sees is
// therefore a function of (problem, thread count) alone, never of the process's allocation history
// (glibc's dynamic mmap threshold would otherwise put the same array in a fresh mapping or in the
// main heap depending on what the process freed before).
//
// The arena is sized EXACTLY, by measurement: Engine::setup() runs the same setup twice.  The first
// pass (a scratch Engine, ArenaRange::measuring) takes its arrays from the heap and only adds up the
// offsets the carves would take; the second pass carves the real arena, whose capacity is that sum
// rounded up to a whole 2 MiB.  Both passes perform the same allocations in the same order (setup is
// deterministic and every engine array is allocated exactly once, with its final size: no vector of
// the arena ever grows), so the second pass ends exactly at the measured offset; anything else is a
// bug and setup() fails (dbench_create refuses) instead of falling back to the heap.  This replaces
// the first-fit chain of 2 MB chunks used before (one exactly sized
// allocation per engine; each chunk of the chain cost a page plus glibc's 2 MiB alignment lead).
//
// posix_memalign and madvise are the two calls allowed here (mmap is not).  Only setup()
// (dbench_create) allocates; the decode path never does.  release() of a pointer inside a registered
// arena is a no-op; the arena is freed with the engine.
// ---------------------------------------------------------------------------
struct ArenaRange {
    char *base = nullptr;
    size_t cap = 0;
    size_t off = 0;
    bool measuring = false;   // first pass of setup(): heap-backed arrays, offsets only
    bool overflow = false;    // a carve did not fit (never happens when the two passes agree)
};
inline thread_local ArenaRange *tl_arena = nullptr;
inline std::mutex &arena_mu() { static std::mutex mu; return mu; }
inline std::vector<std::pair<char *, size_t>> &arena_registry() {
    static std::vector<std::pair<char *, size_t>> r;
    return r;
}
inline bool in_arena(const void *p) {
    if (p == nullptr) return false;
    std::lock_guard<std::mutex> g(arena_mu());
    for (auto &a : arena_registry())
        if (static_cast<const char *>(p) >= a.first && static_cast<const char *>(p) < a.first + a.second) return true;
    return false;
}

// Granularity of an array's start offset and the gap inserted before each array (measurement knobs;
// they change only the placement, never a value).  The defaults keep the earlier placement rule
// inside one arena: 4 KB-rounded starts plus the array's skew, no gap (a sweep
// of the gap at T = 1 found no systematic sensitivity; a sweep of an earlier engine
// preferred 64 B-packed starts).  Not re-swept for this engine at T = 8.
#ifndef FASTBPMT_ARENA_GRAN
#define FASTBPMT_ARENA_GRAN 4096u
#endif
#ifndef FASTBPMT_ARENA_GAP
#define FASTBPMT_ARENA_GAP 0u
#endif
static_assert((FASTBPMT_ARENA_GRAN & (FASTBPMT_ARENA_GRAN - 1u)) == 0 && FASTBPMT_ARENA_GRAN >= 64u,
              "FASTBPMT_ARENA_GRAN must be a power of two >= 64");

struct AlignedAlloc {
    static constexpr size_t kHuge = 2u << 20;   // 2 MB: the huge-page granularity we hint

    static size_t start_of(size_t off) {
        return (off + FASTBPMT_ARENA_GAP + (FASTBPMT_ARENA_GRAN - 1u)) & ~(size_t)(FASTBPMT_ARENA_GRAN - 1u);
    }
    // Heap placement (the measuring pass and the tests that build an Engine without setup()):
    // 4 KB base alignment (2 MB for arrays >= 2 MB) plus the array's own skew.
    static void *heap(size_t bytes, size_t skew) {
        const size_t align = bytes >= kHuge ? kHuge : 4096u;
        void *raw = nullptr;
        if (::posix_memalign(&raw, align, bytes + skew) != 0) return nullptr;
        return static_cast<char *>(raw) + skew;
    }
    // The array's skew: every engine array starts at a *distinct offset modulo 4096*, which keeps two
    // arrays walked together from sharing a cache-set pattern; the static_assert in Engine forbids
    // two coupled arrays with the same skew index.
    static void *allocate(size_t bytes, size_t skew_index) {
        const size_t skew = 64u * (skew_index + 1u);
        if (ArenaRange *a = tl_arena) {
            const size_t start = start_of(a->off);
            const size_t need = start + skew + bytes;
            if (a->measuring) {
                a->off = need;
                return heap(bytes, skew);
            }
            if (need > a->cap) {   // never happens: the capacity is the measured offset
                a->overflow = true;
                return nullptr;
            }
            a->off = need;
            return a->base + start + skew;
        }
        return heap(bytes, skew);
    }
    static void release(void *p, size_t skew_index) {
        if (p == nullptr || in_arena(p)) return;
        std::free(static_cast<char *>(p) - 64u * (skew_index + 1u));
    }
};

// Owns one engine's arena; declared as the Engine's first member, so it is destroyed after every array.
struct EngineArena {
    ArenaRange r;
    EngineArena() = default;
    EngineArena(const EngineArena &) = delete;
    EngineArena &operator=(const EngineArena &) = delete;
    // One 2 MB-aligned allocation of `bytes` rounded up to a whole 2 MiB.
    bool reserve(size_t bytes) {
        const size_t kHuge = AlignedAlloc::kHuge;
        const size_t cap = (bytes + kHuge - 1) / kHuge * kHuge;
        void *p = nullptr;
        if (cap == 0 || ::posix_memalign(&p, kHuge, cap) != 0) return false;
        (void)::madvise(p, cap, MADV_HUGEPAGE);   // 2 MB-aligned, so this returns 0
        r.base = static_cast<char *>(p);
        r.cap = cap;
        r.off = 0;
        r.measuring = false;
        r.overflow = false;
        std::lock_guard<std::mutex> g(arena_mu());
        arena_registry().emplace_back(r.base, r.cap);
        return true;
    }
    // True when p lies in this engine's arena (the coverage check).
    bool covers(const void *p) const {
        return r.base != nullptr && static_cast<const char *>(p) >= r.base && static_cast<const char *>(p) < r.base + r.cap;
    }
    ~EngineArena() {
        if (!r.base) return;
        {
            std::lock_guard<std::mutex> g(arena_mu());
            auto &reg = arena_registry();
            for (size_t i = 0; i < reg.size(); ++i) {
                if (reg[i].first == r.base) { reg.erase(reg.begin() + (long)i); break; }
            }
        }
        std::free(r.base);
        r.base = nullptr;
    }
};
struct ArenaScope {
    explicit ArenaScope(ArenaRange *a) { tl_arena = a; }
    ~ArenaScope() { tl_arena = nullptr; }
};

// Every engine array has its own skew index (offset = 64 * (index + 1) bytes), and the arrays that
// are walked together in check_pass()/var_pass() are asserted distinct below.  Two arrays that share
// a 64 B offset can land at a power-of-two stride from each other when both are 2 MB-aligned.
namespace skewidx {
constexpr size_t kCbase = 0, kCwidth = 1, kClanes = 2, kClaneBase = 3, kCcol = 4, kCidxOfCol = 5;
constexpr size_t kCdegLane = 6, kCuni = 7, kRbase = 8, kRwidth = 9, kRlanes = 10, kRlaneBase = 11;
constexpr size_t kRrow = 12, kRdegLane = 13, kRuni = 14, kCslotOfRslot = 15, kClaneOfRslot = 16;
constexpr size_t kMsg = 17, kPriorLane = 18, kLlrLane = 19, kVlocT = 20, kSuffT = 21, kSynLane = 22;
constexpr size_t kC2bRow = 23, kPrevMaskT = 24, kCandT = 25, kSynBits = 26, kCscPtr = 27, kCscRow = 28;
constexpr size_t kDecoding = 29, kPosterior = 30, kRslotOfCslot = 31, kRSplit = 32, kCSplit = 33;
constexpr size_t kOSplit = 34, kBarSlots = 35;
}  // namespace skewidx

// The message layout.  Default: the single message array, as in the earlier engine -- b2c
// before and c2b after the check update live in the column-layout msg, the check pass gathers b2c
// through cslot_of_rslot and scatters c2b back through it, and the variable pass streams its own
// column chunks in place.  Measured faster than the split layout at every thread count (bb144
// bp.ms30, AVX2: 0.99 x the earlier engine at T = 1 and 0.32 at T = 4, against 1.51 and 0.52 for the split
// layout): the variable pass's extra gather through a second index map costs more than the check
// pass's scatter into lines other threads will read.  FASTBPMT_SINGLE_MSG=0 selects that split
// layout for measurement -- b2c in msg and c2b in the row-layout c2b_row, each written
// contiguously by its own pass and read by the other through rslot_of_cslot, so no pass writes a
// line another thread owns.  Both layouts are bit-identical.
#ifndef FASTBPMT_SINGLE_MSG
#define FASTBPMT_SINGLE_MSG 1
#endif
constexpr bool kSingleMsg = FASTBPMT_SINGLE_MSG != 0;

template <size_t... I> constexpr bool distinct_skews() {
    const size_t v[] = {I...};
    for (size_t a = 0; a < sizeof...(I); ++a) {
        for (size_t b = a + 1; b < sizeof...(I); ++b) {
            if (v[a] == v[b]) return false;
        }
    }
    return true;
}

// A std::vector over AlignedAlloc with its own skew index (skewidx::k*; the engine never reuses an
// index for two arrays).
template <class T, size_t Skew>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() = default;
    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Skew> &) {}
    T *allocate(size_t n) {
        void *p = AlignedAlloc::allocate(n * sizeof(T), Skew);
        if (p == nullptr) throw std::bad_alloc();
        return static_cast<T *>(p);
    }
    void deallocate(T *p, size_t) { AlignedAlloc::release(p, Skew); }
    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Skew>;
    };
    template <class U>
    bool operator==(const AlignedAllocator<U, Skew> &) const { return true; }
    template <class U>
    bool operator!=(const AlignedAllocator<U, Skew> &) const { return false; }
};
template <class T, size_t Skew>
using AVec = std::vector<T, AlignedAllocator<T, Skew>>;

// ---------------------------------------------------------------------------
// Vector primitives: one specialisation per build ISA, LANES rows (check pass)
// or columns (variable pass) in flight.  No FMA, no reassociation.
// ---------------------------------------------------------------------------
template <int L> struct Vops;

#if FASTBP_LANES == 1
template <> struct Vops<1> {
    using V = double;
    using M = bool;
    static constexpr int L = 1;
    static V set1(double x) { return x; }
    static V load(const double *p) { return *p; }
    static void store(double *p, V v) { *p = v; }
    static V add(V a, V b) { return a + b; }
    static V mul(V a, V b) { return a * b; }
    static V absv(V a) { return std::fabs(a); }
    static M cmple(V a, V b) { return a <= b; }   // false when either is NaN
    static M cmplt(V a, V b) { return a < b; }    // false when either is NaN
    static M cmpeq(V a, V b) { return a == b; }   // false when either is NaN
    static M cmpne(V a, V b) { return a != b; }
    static M mxor(M a, M b) { return a != b; }
    static V select(M m, V a, V b) { return m ? a : b; }
    static V min_pd(V a, V b) { return (a < b) ? a : b; }  // b when a is NaN
    static V max_pd(V a, V b) { return (a > b) ? a : b; }  // b when a is NaN
    static uint32_t movemask(M m) { return m ? 1u : 0u; }
    static V gather(const double *base, const int32_t *idx) { return base[*idx]; }
    static void scatter(double *base, const int32_t *idx, V v) { base[*idx] = v; }
};
#endif

#if FASTBP_LANES == 4
template <> struct Vops<4> {
    using V = __m256d;
    using M = __m256d;
    static constexpr int L = 4;
    static V set1(double x) { return _mm256_set1_pd(x); }
    static V load(const double *p) { return _mm256_loadu_pd(p); }
    static void store(double *p, V v) { _mm256_storeu_pd(p, v); }
    static V add(V a, V b) { return _mm256_add_pd(a, b); }
    static V mul(V a, V b) { return _mm256_mul_pd(a, b); }
    static V absv(V a) {
        return _mm256_and_pd(a, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));
    }
    static M cmple(V a, V b) { return _mm256_cmp_pd(a, b, _CMP_LE_OQ); }
    static M cmplt(V a, V b) { return _mm256_cmp_pd(a, b, _CMP_LT_OQ); }
    static M cmpeq(V a, V b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OQ); }
    static M cmpne(V a, V b) { return _mm256_cmp_pd(a, b, _CMP_NEQ_UQ); }
    static M mxor(M a, M b) { return _mm256_xor_pd(a, b); }
    static V select(M m, V a, V b) { return _mm256_blendv_pd(b, a, m); }
    // vminpd/vmaxpd return their second operand when either is NaN.  Safe here because the second
    // operand (the running minimum) is never NaN: m1 and m2 start at DBL_MAX and only ever receive
    // non-NaN values (a NaN |b2c| leaves both unchanged, like bp.hpp's `if (|x| < temp)`).
    static V min_pd(V a, V b) { return _mm256_min_pd(a, b); }
    static V max_pd(V a, V b) { return _mm256_max_pd(a, b); }
    static uint32_t movemask(M m) { return (uint32_t)_mm256_movemask_pd(m); }
    // Scattered read of one vector: four scalar accesses.  Measured on the development machine: the AVX2 gather
    // instruction (vpgatherdq, microcoded) costs ~50% more than four loads here (0.93 vs 0.58 ns per
    // edge-iteration on bb72_unif_p1e-3_r6_z, 2000 shots).
    static V gather(const double *base, const int32_t *idx) {
        return _mm256_setr_pd(base[idx[0]], base[idx[1]], base[idx[2]], base[idx[3]]);
    }
    static void scatter(double *base, const int32_t *idx, V v) {
        double buf[4];
        _mm256_storeu_pd(buf, v);
        base[idx[0]] = buf[0];
        base[idx[1]] = buf[1];
        base[idx[2]] = buf[2];
        base[idx[3]] = buf[3];
    }
};
#endif

#if FASTBP_LANES == 8
template <> struct Vops<8> {
    using V = __m512d;
    using M = __mmask8;
    static constexpr int L = 8;
    static V set1(double x) { return _mm512_set1_pd(x); }
    static V load(const double *p) { return _mm512_loadu_pd(p); }
    static void store(double *p, V v) { _mm512_storeu_pd(p, v); }
    static V add(V a, V b) { return _mm512_add_pd(a, b); }
    static V mul(V a, V b) { return _mm512_mul_pd(a, b); }
    static V absv(V a) {
        return _mm512_and_pd(a, _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL)));
    }
    static M cmple(V a, V b) { return (M)_mm512_cmp_pd_mask(a, b, _CMP_LE_OQ); }
    static M cmplt(V a, V b) { return (M)_mm512_cmp_pd_mask(a, b, _CMP_LT_OQ); }
    static M cmpeq(V a, V b) { return (M)_mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ); }
    static M cmpne(V a, V b) { return (M)_mm512_cmp_pd_mask(a, b, _CMP_NEQ_UQ); }
    static M mxor(M a, M b) { return (M)((unsigned)a ^ (unsigned)b); }
    static V select(M m, V a, V b) { return _mm512_mask_blend_pd(m, b, a); }
    static V min_pd(V a, V b) { return _mm512_min_pd(a, b); }  // b when either is NaN
    static V max_pd(V a, V b) { return _mm512_max_pd(a, b); }
    static uint32_t movemask(M m) { return (uint32_t)m; }
    // Scattered read of one vector, eight scalar accesses (see Vops<4>::gather; measured on the development
    // machine: 0.61 vs 0.89 ns per edge-iteration on bb72 with the gather instruction).
    static V gather(const double *base, const int32_t *idx) {
        return _mm512_setr_pd(base[idx[0]], base[idx[1]], base[idx[2]], base[idx[3]],
                              base[idx[4]], base[idx[5]], base[idx[6]], base[idx[7]]);
    }
    // AVX-512 has a scatter instruction, but measured on the development machine the 8 scalar stores are
    // faster: 12-18% at p99 on bb72, bb144, bb144 xyz, bb72 loss and bb288 bp.ms30 with bit-identical
    // decodes, and the instruction does not pay for itself (perf counters: it saves instructions,
    // not cycles).  The scalar form is the default; FASTBPOSD_NATIVE_SCATTER turns the instruction
    // back on for measurement.  Both forms are exact.
    static void scatter(double *base, const int32_t *idx, V v) {
#if defined(FASTBPOSD_NATIVE_SCATTER)
        _mm512_i32scatter_pd(base, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(idx)), v, 8);
#else
        double buf[8];
        _mm512_storeu_pd(buf, v);
        for (int l = 0; l < 8; ++l) base[idx[l]] = buf[l];
#endif
    }
};
#endif

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------
class Engine {
public:
    static constexpr int L = FASTBP_LANES;
    using O = Vops<L>;
    using V = typename O::V;
    using M = typename O::M;

    EngineArena arena_;   // first member: every array below is carved from it, so it is destroyed last
    uint32_t m = 0, n = 0;

    // ---- column side ------------------------------------------------------
    uint32_t ncc = 0;         // column chunks (dense first, then thin)
    uint32_t ncc_dense = 0;   // column chunks with exactly L lanes
    uint32_t nclanes = 0;     // column lanes (one per column); lane nclanes is the trash lane
    uint64_t cslots = 0;      // column-layout slots (edges + padding)
    uint32_t ctrash = 0;      // == cslots: the first of this engine's T padding slots (one per thread)
    AVec<uint64_t, skewidx::kCbase> cbase;     // ncc + 1
    AVec<uint32_t, skewidx::kCwidth> cwidth;   // positions per chunk (its longest member's degree)
    AVec<uint32_t, skewidx::kClanes> clanes;   // active lanes per chunk
    AVec<uint32_t, skewidx::kClaneBase> clane_base;
    AVec<int32_t, skewidx::kCcol> ccol;        // nclanes + 1: lane -> column (-1: unused lane)
    AVec<int32_t, skewidx::kCidxOfCol> cidx_of_col;  // n: column -> lane (-1: none)
    AVec<double, skewidx::kCdegLane> cdeg_lane;  // nclanes + 1: per-lane degree (0 for the unused lanes)
    AVec<uint8_t, skewidx::kCuni> cuni;          // ncc: 1 when every lane of the chunk has the same degree

    // ---- row side ---------------------------------------------------------
    uint32_t ngc = 0;         // row chunks (dense first, then thin)
    uint32_t ngc_dense = 0;   // row chunks with exactly L lanes
    uint32_t nrlanes = 0;     // row lanes (one per row); lane nrlanes is the trash lane
    uint64_t rslots = 0;      // row-layout slots (edges + padding)
    uint32_t rtrash = 0;      // == rslots: shared destination of row-side padding
    AVec<uint64_t, skewidx::kRbase> rbase;   // ngc + 1
    AVec<uint32_t, skewidx::kRwidth> rwidth;
    AVec<uint32_t, skewidx::kRlanes> rlanes;
    AVec<uint32_t, skewidx::kRlaneBase> rlane_base;
    AVec<int32_t, skewidx::kRrow> rrow;      // nrlanes + 1: lane -> row (-1: unused lane)
    AVec<double, skewidx::kRdegLane> rdeg_lane;  // nrlanes + 1: per-lane degree (0 for the unused lanes)
    AVec<uint8_t, skewidx::kRuni> runi;      // ngc: 1 when every lane of the row chunk has the same degree

    // Row slot -> column slot of the edge that owns it (padding slots: ctrash), and row slot ->
    // column lane of that edge (padding slots: nclanes, the NaN prior lane).
    AVec<int32_t, skewidx::kCslotOfRslot> cslot_of_rslot;  // rslots + 1
    AVec<int32_t, skewidx::kClaneOfRslot> clane_of_rslot;  // rslots + 1

    // ---- the message array and the working vectors -------------------------
    AVec<double, skewidx::kMsg> msg;         // cslots + 1: b2c before / c2b after the check update; [cslots] = NaN
    AVec<double, skewidx::kPriorLane> prior_lane;  // nclanes + 1: prior LLRs of the lanes; [nclanes] = NaN
    AVec<double, skewidx::kLlrLane> llr_lane;      // nclanes + 1: posterior LLRs
    AVec<double, skewidx::kSynLane> syn_lane;  // nrlanes + 1: syndrome bit of a row lane (0.0 or 1.0)

    // ---- per-shot state (one private copy per thread) ------
    AVec<uint8_t, skewidx::kPrevMaskT> prev_mask;  // T * pm_stride: decisions of the thread's chunks
    AVec<uint64_t, skewidx::kCandT> cand_t;        // T * cand_stride bytes: partial candidate syndrome
    AVec<uint32_t, skewidx::kCscPtr> csc_ptr;
    AVec<uint32_t, skewidx::kCscRow> csc_row;      // CSC copy of H (flip bookkeeping)
    uint32_t cand_words = 0;    // uint64 words of a candidate-syndrome bitset: (m + 63) / 64

    // ---- outputs ----------------------------------------------------------
    AVec<uint8_t, skewidx::kDecoding> decoding;    // n
    AVec<double, skewidx::kPosterior> posterior;   // n
    bool last_converged = false;

    // ---- the team's partition, scratch and broadcast ----------------------------------------
    // Thread tid computes the row chunks [rsplit[tid], rsplit[tid+1]) in the check pass and the
    // column chunks [csplit[tid], csplit[tid+1]) in the variable pass, both balanced by the number
    // of inner-loop iterations of the pass that walks them; every row and every column is computed
    // by exactly one thread, in the reference order.  The first chunk of every partition starts at a
    // 128 B aligned lane / slot / column-slot offset (setup inserts padding lanes and slots in front
    // of it) so no cache line is written by two threads, and the per-thread arrays below are each
    // padded to 128 B.  T = 1 keeps the reference's single-partition layout.
    uint32_t T = 1;
    AVec<uint32_t, skewidx::kRSplit> rsplit;   // T + 1
    AVec<uint32_t, skewidx::kCSplit> csplit;   // T + 1
    AVec<uint32_t, skewidx::kOSplit> osp;      // T + 1: natural-order output ranges
    AVec<double, skewidx::kVlocT> vloc;        // T * vl_stride bytes: row-side gather scratch
    AVec<double, skewidx::kSuffT> suff;        // T * sf_stride bytes: column-side suffix scratch
    uint32_t vl_stride = 0, sf_stride = 0, cand_stride = 0, pm_stride = 0;   // bytes, 128 B rounded
    AVec<uint64_t, skewidx::kSynBits> syn_bits;  // the shot's masked syndrome, one bit per row
    AVec<int32_t, skewidx::kRslotOfCslot> rslot_of_cslot;  // cslots + 1: padding -> rtrash
    AVec<double, skewidx::kC2bRow> c2b_row;      // rslots + 1: row-layout c2b (split layout); [rslots] = +0.0

    // The job the calling thread publishes for one shot (plain fields, released by job_seq).
    std::atomic<uint64_t> job_seq{0};
    // The value job_seq has before the first shot (0, or counter_base): written by set_counter_base before any
    // thread is started (starting a thread orders it before everything the thread does), read by each thread as
    // the sequence it has seen.  A thread must not read job_seq itself for that: the calling thread may already
    // have published the first shot when a thread starts running, and that thread would then wait for the next.
    uint64_t seq_start_ = 0;
    const uint8_t *job_syn = nullptr;
    uint32_t job_max_iter = 0;
    double job_alpha = 0.0;
    static constexpr uint64_t kStopSeq = ~(uint64_t)0;

    // The barrier: one cache line for the shared counters plus this thread's local sense in its own
    // 128 B slot (bar_slots), so no line is shared between the waiters' write sets.
    alignas(128) std::atomic<uint32_t> bar_count_{0};
    alignas(128) std::atomic<uint32_t> bar_gen_{0};
    AVec<uint8_t, skewidx::kBarSlots> bar_slots;   // T * 128 bytes

    // Builds the layout for (H, threads) in this engine's single, exactly sized arena (two passes:
    // measure, then carve; see EngineArena).  False when the problem is refused (int32 index guard)
    // or the arena cannot be allocated.
    bool setup(uint32_t m_, uint32_t n_, const uint32_t *colptr, const uint32_t *rowind, uint32_t threads);
    void set_priors(const double *prior_llr_n);
    // Test hook (config key counter_base=N, 0 <= N <= 2^63): every counter this engine carries
    // across shots starts as if N shots had been decoded -- job_seq (uint64) at N, the barrier
    // generation bar_gen_ and every thread's local sense (uint32, equality-compared, so wrap-safe at
    // any value) at N mod 2^32.  Called after setup() and before the threads start.
    void set_counter_base(uint64_t base);
    // One shot: thread 0 (the calling thread) prepares the shot state, publishes the job and computes
    // its own share; every other thread runs run_mt(tid) from thread_loop().
    uint32_t decode_shot(const uint8_t *syn01, uint32_t max_iter, double alpha);
    void thread_loop(int tid);
    // The team's barrier (one episode = every thread waits for every other).  Public so that
    // a barrier test can time the library's own barrier pinned the way the library pins
    // its threads; the decode path calls it exactly as this.
    void barrier(int tid);

private:
    bool setup_impl(uint32_t m_, uint32_t n_, const uint32_t *colptr, const uint32_t *rowind, uint32_t threads);
    template <bool FIRST> void check_pass(int tid, uint32_t g0, uint32_t g1, double alpha, double *vlp);
    template <bool PART2> void var_pass(int tid, uint32_t c0, uint32_t c1, double *sf, uint64_t *ct);
    void update_flips(int32_t j, uint64_t *ct) {
        for (uint32_t e = csc_ptr[j]; e < csc_ptr[j + 1]; ++e) {
            const uint32_t i = csc_row[e];
            ct[i >> 6] ^= 1ull << (i & 63);
        }
    }
    uint32_t run_mt(int tid);
    bool scan_converged() const {
        const uint64_t *const sb = syn_bits.data();
        const uint64_t *const ctp = cand_t.data();
        uint64_t tot = 0;
        for (uint32_t w = 0; w < cand_words; ++w) {
            uint64_t x = sb[w];
            for (uint32_t t = 0; t < T; ++t) x ^= ctp[(size_t)t * (cand_stride / 8) + w];
            tot += (uint64_t)__builtin_popcountll(x);
        }
        return tot == 0;
    }

    // The arrays the decode loops touch together must not share a skew index (see AlignedAlloc).
    static_assert(distinct_skews<skewidx::kMsg, skewidx::kPriorLane, skewidx::kLlrLane,
                                 skewidx::kVlocT, skewidx::kSuffT, skewidx::kCslotOfRslot,
                                 skewidx::kClaneOfRslot, skewidx::kCdegLane, skewidx::kCcol,
                                 skewidx::kSynLane, skewidx::kCscPtr, skewidx::kCscRow,
                                 skewidx::kCandT, skewidx::kRslotOfCslot, skewidx::kC2bRow,
                                 skewidx::kPrevMaskT, skewidx::kSynBits>(),
                  "arrays walked together in one decode loop must have distinct skew indices");
};

// --- setup -----------------------------------------------------------------
// Both sides are cut into degree-similar chunks (see the file header): sort by degree (descending,
// stable), fill a chunk while members * (first degree) <= floor(1.3 * sum + 16) and fewer than L
// members are in it, then emit chunks with fewer than L members after the dense ones.  Every chunk
// is a dense (position, lane) rectangle padded to its own longest member.
//
// The chunk lists are built first, then split into T contiguous ranges balanced by the
// number of inner-loop iterations of the pass that walks them (a dense chunk costs its whole
// rectangle, a thin chunk the sum of its lanes' degrees), then emitted with 128 B alignment at the
// start of every partition (16 padding lanes / slots / column slots, never read or written).
static inline void split_by_weight(const std::vector<uint64_t> &w, uint32_t T, uint32_t *split) {
    const uint32_t nc = (uint32_t)w.size();
    uint64_t total = 0;
    for (uint64_t x : w) total += x;
    uint64_t acc = 0;
    uint32_t c = 0;
    split[0] = 0;
    for (uint32_t t = 1; t < T; ++t) {
        const uint64_t target = (total * (uint64_t)t) / (uint64_t)T;
        while (c < nc && acc < target) acc += w[c++];
        split[t] = c;
    }
    split[T] = nc;
}

// The two passes of the exact arena (see EngineArena): a scratch engine runs setup_impl() with the
// measuring arena (heap-backed arrays, the carve offsets added up), then this engine reserves exactly
// that many bytes (rounded up to 2 MiB) and runs setup_impl() again into the real arena.
inline bool Engine::setup(uint32_t m_, uint32_t n_, const uint32_t *colptr, const uint32_t *rowind,
                          uint32_t threads) {
    size_t need = 0;
    {
        Engine probe;
        ArenaRange meas;
        meas.measuring = true;
        {
            ArenaScope scope(&meas);
            if (!probe.setup_impl(m_, n_, colptr, rowind, threads)) return false;
        }
        need = meas.off;
    }   // the probe's heap arrays are freed here, outside any arena scope
    if (!arena_.reserve(need)) return false;
    {
        ArenaScope scope(&arena_.r);
        if (!setup_impl(m_, n_, colptr, rowind, threads)) return false;
    }
    // Both passes carve the same arrays in the same order: the real pass ends exactly where the
    // measuring pass did.  Anything else would be a layout that depends on more than (H, T).
    return !arena_.r.overflow && arena_.r.off == need;
}

inline void Engine::set_counter_base(uint64_t base) {
    seq_start_ = base;
    job_seq.store(base, std::memory_order_relaxed);
    bar_gen_.store((uint32_t)base, std::memory_order_relaxed);
    bar_count_.store(0, std::memory_order_relaxed);
    for (uint32_t t = 0; t < T; ++t) {
        uint32_t *const local = reinterpret_cast<uint32_t *>(bar_slots.data() + (size_t)t * 128u);
        *local = (uint32_t)base;
    }
}

inline bool Engine::setup_impl(uint32_t m_, uint32_t n_, const uint32_t *colptr, const uint32_t *rowind,
                               uint32_t threads) {
    m = m_;
    n = n_;
    T = threads < 1 ? 1u : (threads > 64u ? 64u : threads);
    // The row and column slot indices, the lane ids and the column ids are int32
    // (cslot_of_rslot, clane_of_rslot, ccol, cidx_of_col): a problem whose counts or layouts would
    // reach 2^31 is refused here (setup fails, dbench_create returns NULL with a message) rather
    // than silently truncated.  A problem that large is impractical (2^31 columns or slots); the
    // guard is what makes the arithmetic safe, and the int32 guard test reaches the same
    // comparisons through the kIndexLimit test-only build.
    if ((uint64_t)n >= kIndexLimit) return false;
    if ((uint64_t)m >= kIndexLimit) return false;
    const uint32_t nnz = n > 0 ? colptr[n] : 0;
    // Every engine array below (and the per-thread arrays of the team) is allocated once,
    // with its final size, inside the arena scope setup() opened (the measuring pass or the real
    // arena), so its layout is a function of (problem, T) alone.
    csc_ptr.assign(colptr, colptr + (size_t)n + 1);
    csc_row.assign(rowind, rowind + nnz);
    struct Chunk { uint32_t i, j, d0; };

    // ---- column chunks: member lists (degree-sorted, dense first), then the thread ranges ----
    std::vector<int32_t> corder(n);
    std::iota(corder.begin(), corder.end(), 0);
    std::stable_sort(corder.begin(), corder.end(), [&](int32_t a, int32_t b) {
        return (colptr[a + 1] - colptr[a]) > (colptr[b + 1] - colptr[b]);
    });
    std::vector<Chunk> cchunks;
    {
        std::vector<Chunk> th;   // thin chunks, emitted after the dense ones
        for (uint32_t i = 0; i < n;) {
            const uint32_t d0 = colptr[corder[i] + 1] - colptr[corder[i]];
            uint32_t j = i;
            uint64_t sum = 0;
            while (j < n && j - i < (uint32_t)L &&
                   (uint64_t)(j - i + 1) * d0 <=
                       (13ull * (sum + (uint64_t)(colptr[corder[j] + 1] - colptr[corder[j]]))) / 10ull + 16ull) {
                sum += (uint64_t)(colptr[corder[j] + 1] - colptr[corder[j]]);
                ++j;
            }
            if (j == i) j = i + 1;  // the first member always fits (1 * d0 <= 1.3 * d0 + 16)
            if (j - i < (uint32_t)L) th.push_back({i, j, d0});
            else cchunks.push_back({i, j, d0});
            i = j;
        }
        ncc_dense = (uint32_t)cchunks.size();
        for (const Chunk &t : th) cchunks.push_back(t);
        ncc = (uint32_t)cchunks.size();
    }
    {
        std::vector<uint64_t> w(ncc);
        for (uint32_t c = 0; c < ncc; ++c) {
            const Chunk &ch = cchunks[c];
            if (ch.j - ch.i == (uint32_t)L) {
                w[c] = (uint64_t)ch.d0 * (uint64_t)L;   // a dense chunk walks its whole rectangle
            } else {
                uint64_t s = 0;
                for (uint32_t u = ch.i; u < ch.j; ++u) s += colptr[corder[u] + 1] - colptr[corder[u]];
                w[c] = s ? s : 1;                        // a thin chunk walks its lanes' degrees
            }
        }
        std::vector<uint32_t> sp((size_t)T + 1, 0);
        split_by_weight(w, T, sp.data());
        csplit.assign(sp.begin(), sp.end());
    }
    std::vector<int32_t> col_of_lane;  // lane -> column (holes from the 128 B alignment: -1)
    {
        // Built in plain vectors and assigned once (an arena vector never grows: see EngineArena).
        std::vector<uint64_t> cb;
        std::vector<uint32_t> cw, cl, clb;
        cb.reserve((size_t)ncc + 1);
        cw.reserve(ncc);
        cl.reserve(ncc);
        clb.reserve(ncc);
        uint64_t base = 0;
        uint32_t lane = 0;
        uint32_t c = 0;
        for (uint32_t t = 0; t < T; ++t) {
            if (t > 0 && csplit[t] < csplit[t + 1]) {   // 128 B alignment of the partition's first chunk
                base = (base + 15u) & ~(uint64_t)15u;
                lane = (lane + 15u) & ~(uint32_t)15u;
            }
            for (; c < csplit[t + 1]; ++c) {
                const Chunk &ch = cchunks[c];
                cb.push_back(base);
                cw.push_back(ch.d0);
                cl.push_back(ch.j - ch.i);
                clb.push_back(lane);
                base += (uint64_t)(ch.j - ch.i) * (uint64_t)ch.d0;
                lane += ch.j - ch.i;
            }
        }
        nclanes = lane;
        cb.push_back(base);
        cbase.assign(cb.begin(), cb.end());
        cwidth.assign(cw.begin(), cw.end());
        clanes.assign(cl.begin(), cl.end());
        clane_base.assign(clb.begin(), clb.end());
        cslots = base;
        // Every column-layout slot index (and the T padding slots) must fit in the int32 tables
        // below; cslots + T + 1 slots are addressed (the padding slots included).
        if (cslots + (uint64_t)T + 1 > kIndexLimit) return false;
        ctrash = (uint32_t)cslots;
    }
    // Lane -> column, column -> lane and the per-lane degrees.  The 128 B alignment of a
    // partition's first chunk leaves holes between the partitions' lane ranges; a hole has no
    // column (-1) and degree 0 and is never read or written by any pass.
    col_of_lane.assign(nclanes, -1);
    for (uint32_t c = 0; c < ncc; ++c) {
        const Chunk &ch = cchunks[c];
        for (uint32_t u = ch.i; u < ch.j; ++u) col_of_lane[clane_base[c] + (u - ch.i)] = corder[u];
    }
    ccol.assign((size_t)nclanes + 1, -1);
    cidx_of_col.assign(n, -1);
    cdeg_lane.assign((size_t)nclanes + 1, 0.0);
    cuni.assign(ncc ? ncc : 1, 0);
    for (uint32_t lane = 0; lane < nclanes; ++lane) {
        const int32_t j = col_of_lane[lane];
        ccol[lane] = j;
        if (j < 0) continue;
        cidx_of_col[j] = (int32_t)lane;
        cdeg_lane[lane] = (double)(colptr[j + 1] - colptr[j]);
    }
    for (uint32_t c = 0; c < ncc; ++c) {
        bool uni = true;
        for (uint32_t l = 0; l < clanes[c]; ++l) {
            if (cdeg_lane[clane_base[c] + l] != (double)cwidth[c]) uni = false;
        }
        cuni[c] = (uint8_t)(uni ? 1 : 0);
    }

    // ---- row-major structure: CSR of H, with the column-layout slot of every edge ----
    std::vector<uint32_t> rowptr((size_t)m + 1, 0);
    for (uint32_t e = 0; e < nnz; ++e) rowptr[rowind[e] + 1]++;
    for (uint32_t i = 0; i < m; ++i) rowptr[i + 1] += rowptr[i];
    std::vector<uint32_t> rowslot(nnz);  // column-layout slot of each row-major entry
    std::vector<uint32_t> rowlane(nnz);  // column lane of each row-major entry
    {
        std::vector<uint32_t> chunk_of_lane(nclanes ? nclanes : 1, 0);
        for (uint32_t c = 0; c < ncc; ++c) {
            for (uint32_t l = 0; l < clanes[c]; ++l) chunk_of_lane[clane_base[c] + l] = c;
        }
        std::vector<uint32_t> fill(rowptr.begin(), rowptr.end() - 1);
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t lane = (uint32_t)cidx_of_col[j];
            const uint32_t c = chunk_of_lane[lane];
            const uint64_t cbase_j = cbase[c];
            const uint32_t stride = clanes[c];
            for (uint32_t e = colptr[j]; e < colptr[j + 1]; ++e) {
                const uint32_t k = e - colptr[j];  // rank of the row inside column j
                const uint32_t slot = fill[rowind[e]]++;
                rowslot[slot] = (uint32_t)(cbase_j + (uint64_t)k * stride + (lane - clane_base[c]));
                rowlane[slot] = lane;
            }
        }
    }

    // ---- row chunks: member lists (degree-sorted, dense first), then the thread ranges ----
    std::vector<int32_t> rorder(m);
    std::iota(rorder.begin(), rorder.end(), 0);
    std::stable_sort(rorder.begin(), rorder.end(), [&](int32_t a, int32_t b) {
        return (rowptr[a + 1] - rowptr[a]) > (rowptr[b + 1] - rowptr[b]);
    });
    std::vector<Chunk> rchunks;
    {
        std::vector<Chunk> th;
        for (uint32_t i = 0; i < m;) {
            const uint32_t d0 = rowptr[rorder[i] + 1] - rowptr[rorder[i]];
            uint32_t j = i;
            uint64_t sum = 0;
            while (j < m && j - i < (uint32_t)L &&
                   (uint64_t)(j - i + 1) * d0 <=
                       (13ull * (sum + (uint64_t)(rowptr[rorder[j] + 1] - rowptr[rorder[j]]))) / 10ull + 16ull) {
                sum += (uint64_t)(rowptr[rorder[j] + 1] - rowptr[rorder[j]]);
                ++j;
            }
            if (j == i) j = i + 1;
            if (j - i < (uint32_t)L) th.push_back({i, j, d0});
            else rchunks.push_back({i, j, d0});
            i = j;
        }
        ngc_dense = (uint32_t)rchunks.size();
        for (const Chunk &t : th) rchunks.push_back(t);
        ngc = (uint32_t)rchunks.size();
    }
    {
        std::vector<uint64_t> w(ngc);
        for (uint32_t g = 0; g < ngc; ++g) {
            const Chunk &ch = rchunks[g];
            if (ch.j - ch.i == (uint32_t)L) {
                w[g] = (uint64_t)ch.d0 * (uint64_t)L;
            } else {
                uint64_t s = 0;
                for (uint32_t u = ch.i; u < ch.j; ++u) s += rowptr[rorder[u] + 1] - rowptr[rorder[u]];
                w[g] = s ? s : 1;
            }
        }
        std::vector<uint32_t> sp((size_t)T + 1, 0);
        split_by_weight(w, T, sp.data());
        rsplit.assign(sp.begin(), sp.end());
    }
    {
        std::vector<uint64_t> rb;
        std::vector<uint32_t> rw, rl, rlb;
        rb.reserve((size_t)ngc + 1);
        rw.reserve(ngc);
        rl.reserve(ngc);
        rlb.reserve(ngc);
        uint64_t base = 0;
        uint32_t lane = 0;
        uint32_t g = 0;
        for (uint32_t t = 0; t < T; ++t) {
            if (t > 0 && rsplit[t] < rsplit[t + 1]) {
                base = (base + 15u) & ~(uint64_t)15u;
                lane = (lane + 15u) & ~(uint32_t)15u;
            }
            for (; g < rsplit[t + 1]; ++g) {
                const Chunk &ch = rchunks[g];
                rb.push_back(base);
                rw.push_back(ch.d0);
                rl.push_back(ch.j - ch.i);
                rlb.push_back(lane);
                base += (uint64_t)(ch.j - ch.i) * (uint64_t)ch.d0;
                lane += ch.j - ch.i;
            }
        }
        nrlanes = lane;
        rb.push_back(base);
        rbase.assign(rb.begin(), rb.end());
        rwidth.assign(rw.begin(), rw.end());
        rlanes.assign(rl.begin(), rl.end());
        rlane_base.assign(rlb.begin(), rlb.end());
        rslots = base;
        if (rslots + 1 > kIndexLimit) return false;
        rtrash = (uint32_t)rslots;
    }
    // Lane -> row (holes: -1) and the per-lane degrees, from the row chunks.
    rrow.assign((size_t)nrlanes + 1, -1);
    for (uint32_t g = 0; g < ngc; ++g) {
        for (uint32_t l = 0; l < rlanes[g]; ++l) rrow[rlane_base[g] + l] = rorder[rchunks[g].i + l];
    }
    rdeg_lane.assign((size_t)nrlanes + 1, 0.0);
    runi.assign(ngc ? ngc : 1, 0);
    for (uint32_t g = 0; g < ngc; ++g) {
        for (uint32_t l = 0; l < rlanes[g]; ++l) {
            const int32_t i = rrow[rlane_base[g] + l];
            if (i >= 0) {
                rdeg_lane[rlane_base[g] + l] = (double)(rowptr[i + 1] - rowptr[i]);
            }
        }
        bool uni = true;
        for (uint32_t l = 0; l < rlanes[g]; ++l) {
            if (rdeg_lane[rlane_base[g] + l] != (double)rwidth[g]) uni = false;
        }
        runi[g] = (uint8_t)(uni ? 1 : 0);
    }

    // ---- cross index tables (padding row slots read the shared NaN slot / trash lane) ----
    // A position of a row rectangle that no edge owns is a padding position: its gather must read a
    // NaN (so that the min/count comparisons ignore it) and, on the dense non-uniform production
    // path, the pass stores the padding value back into the same slot.  With more than one thread
    // that slot must be *per thread* -- several threads storing the same NaN into one location is a
    // data race even though every thread stores the same value -- so the padding
    // slots are ctrash + t, and a padding position points at the padding slot of the thread that
    // owns its row chunk.  Reads and writes of a padding slot are then always by its owner, and
    // every padding slot holds NaN from setup to the end of the handle.
    std::vector<uint32_t> thr_of_rchunk(ngc ? ngc : 1, 0);
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t g = rsplit[t]; g < rsplit[t + 1]; ++g) thr_of_rchunk[g] = t;
    }
    cslot_of_rslot.assign((size_t)rslots + 1, (int32_t)ctrash);
    clane_of_rslot.assign((size_t)rslots + 1, (int32_t)nclanes);
    if (!kSingleMsg) rslot_of_cslot.assign((size_t)cslots + 1, (int32_t)rtrash);
    for (uint32_t g = 0; g < ngc; ++g) {
        const uint32_t stride = rlanes[g];
        const int32_t trash = (int32_t)(ctrash + thr_of_rchunk[g]);
        for (uint32_t l = 0; l < stride; ++l) {
            const int32_t i = rrow[rlane_base[g] + l];
            if (i < 0) continue;
            const uint32_t deg = rowptr[i + 1] - rowptr[i];
            for (uint32_t p = deg; p < rwidth[g]; ++p) {
                cslot_of_rslot[(size_t)rbase[g] + (size_t)p * stride + l] = trash;
            }
            for (uint32_t p = 0; p < deg; ++p) {
                const size_t rs = (size_t)rbase[g] + (size_t)p * stride + l;
                const uint32_t cs = rowslot[rowptr[i] + p];
                cslot_of_rslot[rs] = (int32_t)cs;
                clane_of_rslot[rs] = (int32_t)rowlane[rowptr[i] + p];
                if (!kSingleMsg) rslot_of_cslot[cs] = (int32_t)rs;
            }
        }
    }

    // ---- the message array and the per-shot state ----
    // msg[cslots] is the shared destination of every padding gather and must stay NaN: the check
    // update's min/count comparisons all reject a NaN, so a padding position never influences a row.
    // On the split layout (FASTBPMT_SINGLE_MSG=0) the c2b messages live in c2b_row,
    // and rslot_of_cslot[ctrash] -- the value the variable pass reads for a column-side padding
    // position -- is c2b_row[rtrash], which stays +0.0: an exact identity for the running sums,
    // exactly what the reference reads from its never-written column-side padding slots.  On the
    // single-array layout (the default) c2b lives in msg as in the earlier engine, and the dense production
    // pass stores the padding value (NaN) into this thread's own msg[ctrash + tid] slot on every
    // pass: the slot is per thread, so the write is not a data race, and it stays NaN.
    msg.assign((size_t)cslots + (size_t)T + 1, 0.0);
    for (uint32_t t = 0; t < T; ++t) msg[cslots + t] = std::numeric_limits<double>::quiet_NaN();
    prior_lane.assign((size_t)nclanes + 1, 0.0);
    prior_lane[nclanes] = std::numeric_limits<double>::quiet_NaN();
    llr_lane.assign((size_t)nclanes + 1, 0.0);
    syn_lane.assign((size_t)nrlanes + 1, 0.0);
    if (!kSingleMsg) {
        c2b_row.assign((size_t)rslots + 1, 0.0);   // [rtrash] stays +0.0
    }
    decoding.assign(n ? n : 1, 0);
    posterior.assign(n ? n : 1, 0.0);
    // Scratch is sized by the largest chunk (its width times its lanes), never by a global maximum
    // degree: a long outlier lives in a thin chunk whose rectangle is exactly its own degree.  Every
    // thread gets its own scratch, padded to 128 B.
    uint64_t rcap = 0, ccap = 0;
    for (uint32_t g = 0; g < ngc; ++g) rcap = std::max<uint64_t>(rcap, (uint64_t)rwidth[g] * rlanes[g]);
    for (uint32_t c = 0; c < ncc; ++c) ccap = std::max<uint64_t>(ccap, (uint64_t)cwidth[c] * clanes[c]);
    vl_stride = (uint32_t)(((rcap + (uint64_t)L) * sizeof(double) + 127u) & ~(size_t)127u);
    sf_stride = (uint32_t)(((ccap + (uint64_t)L) * sizeof(double) + 127u) & ~(size_t)127u);
    cand_words = (m + 63u) / 64u;
    cand_stride = (uint32_t)(((uint64_t)cand_words * sizeof(uint64_t) + 127u) & ~(size_t)127u);
    pm_stride = (ncc + 127u) & ~(uint32_t)127u;
    if (cand_stride == 0) cand_stride = 128;
    if (pm_stride == 0) pm_stride = 128;
    vloc.assign((size_t)T * (vl_stride / sizeof(double)), 0.0);
    suff.assign((size_t)T * (sf_stride / sizeof(double)), 0.0);
    prev_mask.assign((size_t)T * pm_stride, 0);
    cand_t.assign((size_t)T * (cand_stride / sizeof(uint64_t)), 0);
    syn_bits.assign(cand_words ? cand_words : 1, 0);
    bar_slots.assign((size_t)T * 128u, 0);
    osp.assign((size_t)T + 1, 0);
    for (uint32_t t = 0; t <= T; ++t) {
        uint64_t j = ((uint64_t)n * t) / T;
        osp[t] = (uint32_t)((j + 15u) & ~(uint64_t)15u);   // 128 B aligned output ranges (T = 1: exact)
        if (osp[t] > n) osp[t] = n;
    }
    osp[T] = n;
    for (uint32_t t = 1; t < T; ++t) {
        if (osp[t] < osp[t - 1]) osp[t] = osp[t - 1];
    }
    return true;
}

inline void Engine::set_priors(const double *prior_llr_n) {
    for (uint32_t lane = 0; lane < nclanes; ++lane) {
        const int32_t j = ccol[lane];
        prior_lane[lane] = j >= 0 ? prior_llr_n[j] : 0.0;
    }
}

// --- the team barrier ------------------------------------------------------
// A sense-reversing spin barrier; release/acquire on the shared counters orders every thread's
// writes before the barrier with every thread's reads after it (so the check pass sees the whole
// c2b written by the previous barrier-to-barrier interval, the variable pass sees every c2b of this
// iteration, and the convergence scan sees every thread's candidate syndrome).  No futex, condition
// variable or yield: on dedicated CPUs a wait is a few hundred nanoseconds.  A thread that shares a
// CPU (oversubscription) waits for the scheduler, which is bounded because the barrier's other
// threads keep arriving.
static inline void spin_hint() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

inline void Engine::barrier(int tid) {
    if (T <= 1) return;
    uint32_t *const local = reinterpret_cast<uint32_t *>(bar_slots.data() + (size_t)tid * 128u);
    const uint32_t g = *local;
    if (bar_count_.fetch_add(1, std::memory_order_acq_rel) == T - 1) {
        bar_count_.store(0, std::memory_order_relaxed);
        bar_gen_.store(g + 1, std::memory_order_release);
    } else {
        while (bar_gen_.load(std::memory_order_acquire) == g) spin_hint();
    }
    *local = g + 1;
}

// --- check update ----------------------------------------------------------
// One reduction pass (the two smallest magnitudes and the count of b2c <= 0) and one production pass
// per row chunk.  FIRST is iteration 1: the incoming messages are gathered from the prior LLRs of
// the column lanes (padding row slots read prior_lane[nclanes] = NaN) and no b2c exists yet.
// Dense chunks (exactly L lanes) take the vector path; the chunks that could not be filled take a
// scalar path with their own lane count as the stride (see setup).  Thread tid walks only its own
// row chunks [g0, g1); vlp is its private gather scratch.
template <bool FIRST>
inline void Engine::check_pass(int tid, uint32_t g0, uint32_t g1, double alpha, double *const vlp) {
    (void)tid;
    const V dblmax = O::set1(std::numeric_limits<double>::max());
    const V zero = O::set1(0.0);
    const V alpha_v = O::set1(alpha);
    const V nalpha_v = O::set1(-alpha);
    // The array pointers live in locals: the vector objects are reached through `this` and GCC
    // otherwise reloads their data pointers inside the inner loops (measured: 3 extra loads per
    // position, ~15% of the BP time).
    double *const mp = msg.data();
    double *const c2b = c2b_row.data();
    const int32_t *const rs = rslot_of_cslot.data();
    const int32_t *const cs = cslot_of_rslot.data();
    const int32_t *const cl = clane_of_rslot.data();
    const double *const pl = prior_lane.data();
    const double *const rdl = rdeg_lane.data();
    for (uint32_t g = g0; g < g1 && g < ngc_dense; ++g) {
        const uint64_t rb = rbase[g];
        const uint32_t w = rwidth[g];
        const uint32_t lane0 = rlane_base[g];
        // par = parity of (syndrome bit + #{b2c <= 0}) as a per-lane mask; a NaN b2c is never <= 0,
        // so it does not count -- exactly bp.hpp's `if (b2c <= 0) total_sgn += 1`.
        M par = O::cmpne(O::load(&syn_lane[lane0]), zero);
        V m1 = dblmax, m2 = dblmax;
        for (uint32_t p = 0; p < w; ++p) {
            const size_t off = (size_t)rb + (size_t)p * L;
            const V b = FIRST ? O::gather(pl, cl + off) : O::gather(mp, cs + off);
            O::store(&vlp[(size_t)p * L], b);
            const V v = O::absv(b);
            par = O::mxor(par, O::cmple(b, zero));
            // Two smallest of {m1, m2, v}: the second smallest of the union is min(max(old m1, v),
            // old m2) and the smallest is min(v, old m1).  vminpd/vmaxpd return their second operand
            // when either is NaN, and the running minima are never NaN, so a NaN v leaves both
            // unchanged, exactly like bp.hpp's `if (|x| < temp)`.
            m2 = O::min_pd(O::max_pd(m1, v), m2);
            m1 = O::min_pd(v, m1);
        }
        // The reference's sgn for this edge is syndrome + #{b2c <= 0} + (b2c_e <= 0) (mod 2), and
        // its message is min_{j != e} |b2c_j| * ((sgn % 2) ? -alpha : +alpha): ONE multiply whose
        // second operand carries the sign.  Rounding a positive product and negating afterwards is a
        // different double under FE_UPWARD / FE_DOWNWARD.  Adding the edge's own term to the parity
        // mask (which already holds the syndrome bit and the other edges) is the same mod 2.
        if (runi[g]) {
            for (uint32_t p = 0; p < w; ++p) {
                const size_t off = (size_t)rb + (size_t)p * L;
                const V b = O::load(&vlp[(size_t)p * L]);
                const V av = O::absv(b);
                const V a = O::select(O::mxor(par, O::cmple(b, zero)), nalpha_v, alpha_v);
                const V mag = O::select(O::cmpeq(av, m1), m2, m1);
                if (kSingleMsg) O::scatter(mp, cs + off, O::mul(mag, a));
                else O::store(&c2b[off], O::mul(mag, a));
            }
        } else {
            const V dg = O::load(&rdl[lane0]);
            const V nan_v = O::set1(std::numeric_limits<double>::quiet_NaN());
            for (uint32_t p = 0; p < w; ++p) {
                const size_t off = (size_t)rb + (size_t)p * L;
                const V b = O::load(&vlp[(size_t)p * L]);
                const V av = O::absv(b);
                const V a = O::select(O::mxor(par, O::cmple(b, zero)), nalpha_v, alpha_v);
                const V mag = O::select(O::cmpeq(av, m1), m2, m1);
                // A row lane with no edge at this position holds the padding value; on the
                // single-array layout it is stored into the shared msg[ctrash] slot, which must stay
                // NaN (see the file header), on the split layout into its own never-read rectangle
                // padding.  Storing unconditionally is the form of the earlier engine.
                const V val = O::select(O::cmplt(O::set1((double)p), dg), O::mul(mag, a),
                                        FASTBP_PAD_VALUE(nan_v, zero));
                if (kSingleMsg) O::scatter(mp, cs + off, val);
                else O::store(&c2b[off], val);
            }
        }
    }
    for (uint32_t g = (g0 > ngc_dense ? g0 : ngc_dense); g < g1; ++g) {  // thin (scalar) row chunks
        const uint64_t rb = rbase[g];
        const uint32_t nl = rlanes[g];
        const uint32_t lane0 = rlane_base[g];
        for (uint32_t l = 0; l < nl; ++l) {
            const uint32_t deg = (uint32_t)rdl[lane0 + l];
            int64_t cnt = (int64_t)syn_lane[lane0 + l];
            double m1 = std::numeric_limits<double>::max();
            double m2 = m1;
            for (uint32_t p = 0; p < deg; ++p) {
                const size_t off = (size_t)rb + (size_t)p * nl + l;
                const double b = FIRST ? pl[cl[off]] : mp[cs[off]];
                vlp[(size_t)p * nl + l] = b;
                const double v = std::fabs(b);
                if (b <= 0.0) ++cnt;
                if (v < m1) {
                    m2 = m1;
                    m1 = v;
                } else if (v < m2) {
                    m2 = v;
                }
            }
            for (uint32_t p = 0; p < deg; ++p) {
                const size_t off = (size_t)rb + (size_t)p * nl + l;
                const double b = vlp[(size_t)p * nl + l];
                const double mag = (std::fabs(b) == m1) ? m2 : m1;
                const double a = (((cnt + (b <= 0.0 ? 1 : 0)) & 1) != 0) ? -alpha : alpha;
                if (kSingleMsg) mp[cs[off]] = mag * a;
                else c2b[off] = mag * a;
            }
        }
    }
}

// --- variable update -------------------------------------------------------
// Fused forward/reverse pass over one column chunk (bp.hpp's two loops of the bit-to-check update).
// The reverse pass accumulates the suffix from the last row back; the forward pass keeps the running
// prefix t (b2c := t; t += c2b), stores the new b2c = prefix + suffix and keeps the posterior and the
// hard decision (t <= 0).  PART2 is false on the last iteration, where bp.hpp would run part 2 but
// its result is never read.  Decisions that flipped update the thread's private candidate syndrome
// (no shared state, no atomics: every flip of a column belongs to the thread that owns it).  sf and
// ct are this thread's scratch and candidate bitset.
template <bool PART2>
inline void Engine::var_pass(int tid, uint32_t c0, uint32_t c1, double *const sf, uint64_t *const ct) {
    const V zero = O::set1(0.0);
    double *const mp = msg.data();
    const double *const c2b = c2b_row.data();
    const int32_t *const rs = rslot_of_cslot.data();
    const double *const pl = prior_lane.data();
    const double *const cdl = cdeg_lane.data();
    double *const ll = llr_lane.data();
    const int32_t *const cp = ccol.data();
    uint8_t *const pm = prev_mask.data() + (size_t)tid * pm_stride;
    for (uint32_t c = c0; c < c1 && c < ncc_dense; ++c) {
        const uint64_t cb = cbase[c];
        const uint32_t w = cwidth[c];
        const uint32_t lane0 = clane_base[c];
        if (PART2) {
            V u = zero;
            for (int32_t p = (int32_t)w - 1; p >= 0; --p) {
                O::store(&sf[(size_t)p * L], u);
                const size_t off = cb + (size_t)p * L;
                u = O::add(u, kSingleMsg ? O::load(&mp[off]) : O::gather(c2b, rs + off));
            }
        }
        V t = O::load(&pl[lane0]);
        if (cuni[c]) {
            for (uint32_t p = 0; p < w; ++p) {
                const size_t off = cb + (size_t)p * L;
                const V b = kSingleMsg ? O::load(&mp[off]) : O::gather(c2b, rs + off);
                if (PART2) {
                    const V v = O::add(t, O::load(&sf[(size_t)p * L]));
                    O::store(&mp[off], v);
                }
                t = O::add(t, b);
            }
        } else {
            const V dg = O::load(&cdl[lane0]);
            for (uint32_t p = 0; p < w; ++p) {
                const size_t off = cb + (size_t)p * L;
                const V b = kSingleMsg ? O::load(&mp[off]) : O::gather(c2b, rs + off);
                if (PART2) {
                    const V v = O::add(t, O::load(&sf[(size_t)p * L]));
                    // Padding slots of this rectangle (position >= the lane's degree) are never read
                    // as an edge; keep them at +0.0, an exact identity for later accumulations.
                    O::store(&mp[off], O::select(O::cmplt(O::set1((double)p), dg), v, zero));
                }
                t = O::add(t, b);
            }
        }
        O::store(&ll[lane0], t);
        const uint32_t mask = O::movemask(O::cmple(t, zero));
        const uint32_t flips = mask ^ pm[c];
        pm[c] = (uint8_t)mask;
        if (flips) {
            for (uint32_t l = 0; l < (uint32_t)L; ++l) {
                if (((flips >> l) & 1u) == 0) continue;
                const int32_t j = cp[lane0 + l];
                if (j >= 0) update_flips(j, ct);
            }
        }
    }
    for (uint32_t c = (c0 > ncc_dense ? c0 : ncc_dense); c < c1; ++c) {  // thin (scalar) column chunks
        const uint64_t cb = cbase[c];
        const uint32_t nl = clanes[c];
        const uint32_t lane0 = clane_base[c];
        uint32_t mask = 0;
        for (uint32_t l = 0; l < nl; ++l) {
            const uint32_t deg = (uint32_t)cdl[lane0 + l];
            if (PART2) {
                double s = 0.0;
                for (int32_t p = (int32_t)deg - 1; p >= 0; --p) {
                    const size_t off = (size_t)cb + (size_t)p * nl + l;
                    sf[(size_t)p * nl + l] = s;
                    s += kSingleMsg ? mp[off] : c2b[rs[off]];
                }
            }
            double t = pl[lane0 + l];
            for (uint32_t p = 0; p < deg; ++p) {
                const size_t off = (size_t)cb + (size_t)p * nl + l;
                const double b = kSingleMsg ? mp[off] : c2b[rs[off]];
                if (PART2) mp[off] = t + sf[(size_t)p * nl + l];
                t += b;
            }
            ll[lane0 + l] = t;
            mask |= (uint32_t)(t <= 0.0) << l;
        }
        const uint32_t flips = mask ^ pm[c];
        pm[c] = (uint8_t)mask;
        if (flips) {
            for (uint32_t l = 0; l < nl; ++l) {
                if (((flips >> l) & 1u) == 0) continue;
                const int32_t j = cp[lane0 + l];
                if (j >= 0) update_flips(j, ct);
            }
        }
    }
    (void)PART2;
}

// --- one decode: iterate, then materialise the natural-order outputs --------
// Every thread runs this for every shot; thread 0 is the calling thread.  The convergence test of
// iteration it is made from every thread's private candidate syndrome after the barrier that
// publishes them (barrier B), so all threads compute the same value from the same data and agree on
// where to stop without a further reduction; the test is the reference's (`cand == syndrome` over
// every row), just counted bit-parallel instead of incrementally.  Nothing here races: every array
// written per iteration is a thread-private one or a partition exclusively owned by that thread,
// and the barriers order every write before the reads that need it.
inline uint32_t Engine::run_mt(int tid) {
    const uint8_t *const syn = job_syn;
    const uint32_t max_iter = job_max_iter;
    const double alpha = job_alpha;
    const uint32_t r0 = rsplit[tid], r1 = rsplit[tid + 1];
    const uint32_t c0 = csplit[tid], c1 = csplit[tid + 1];
    double *const vlp = vloc.data() + (size_t)tid * (vl_stride / sizeof(double));
    double *const sf = suff.data() + (size_t)tid * (sf_stride / sizeof(double));
    uint64_t *const ct = cand_t.data() + (size_t)tid * (cand_stride / sizeof(uint64_t));
    uint8_t *const pm = prev_mask.data() + (size_t)tid * pm_stride;
    double *const sl = syn_lane.data();
    const uint32_t *const rlb = rlane_base.data();
    const uint32_t *const rln = rlanes.data();
    const int32_t *const rr = rrow.data();

    // ---- per-shot init: this thread's share, written before any thread reads it (barrier A/B) ----
    for (uint32_t g = r0; g < r1; ++g) {
        const uint32_t lane0 = rlb[g];
        for (uint32_t l = 0, nl = rln[g]; l < nl; ++l) {
            const int32_t i = rr[lane0 + l];
            sl[lane0 + l] = (i >= 0 && (syn[i] & 1u)) ? 1.0 : 0.0;
        }
    }
    std::memset(ct, 0, (size_t)cand_words * sizeof(uint64_t));
    std::memset(pm, 0, (size_t)ncc);

    uint32_t iters = 0;
    bool conv = false;
    for (uint32_t it = 1; it <= max_iter; ++it) {
        if (it > 1 && conv) break;   // decided after the previous iteration's variable pass
        if (it == 1) check_pass<true>(tid, r0, r1, alpha, vlp);
        else check_pass<false>(tid, r0, r1, alpha, vlp);
        barrier(tid);                // A: every c2b of this iteration is visible to its readers
        if (it < max_iter) var_pass<true>(tid, c0, c1, sf, ct);
        else var_pass<false>(tid, c0, c1, sf, ct);
        barrier(tid);                // B: every b2c and candidate syndrome bit is visible
        conv = scan_converged();
        iters = it;
    }

    // ---- outputs: this thread's natural-order range (contiguous, no line shared) ----
    {
        const double *const ll = llr_lane.data();
        const int32_t *const ic = cidx_of_col.data();
        uint8_t *const dec = decoding.data();
        double *const post = posterior.data();
        for (uint32_t j = osp[tid]; j < osp[tid + 1]; ++j) {
            const double t = ll[ic[j]];
            post[j] = t;
            dec[j] = (uint8_t)(t <= 0.0);
        }
    }
    if (tid == 0) last_converged = conv;
    barrier(tid);                    // C: the caller (thread 0) sees every thread's outputs
    return iters;
}

// One shot: thread 0 (the calling thread) masks the syndrome into syn_bits, publishes the job and
// computes its own share; the library's threads run run_mt(tid) from thread_loop().
inline uint32_t Engine::decode_shot(const uint8_t *syn01, uint32_t max_iter, double alpha) {
    uint8_t *const sb = reinterpret_cast<uint8_t *>(syn_bits.data());
    std::memset(sb, 0, (size_t)cand_words * sizeof(uint64_t));
    for (uint32_t i = 0; i < m; ++i) {
        if (syn01[i] & 1u) sb[i >> 3] |= (uint8_t)(1u << (i & 7));
    }
    job_syn = syn01;
    job_max_iter = max_iter;
    job_alpha = alpha;
    if (T > 1) job_seq.fetch_add(1, std::memory_order_release);
    return run_mt(0);
}

// The loop of a library thread.  Started and pinned in dbench_create, joined in
// dbench_destroy; between shots it spins on the job sequence -- no futex, no condition variable, no
// yield, and no allocation or I/O at any time between create and destroy.  A thread that shares its
// CPU with another (oversubscription) just waits for the scheduler; the owner's other threads keep
// arriving at the barriers, so every wait is bounded.
inline void Engine::thread_loop(int tid) {
    // The sequence the thread has seen: the engine's start value (0, or counter_base), set before the thread
    // was started.  Not job_seq itself: the first shot may already be published when this thread starts.
    uint64_t last = seq_start_;
    for (;;) {
        uint64_t s = job_seq.load(std::memory_order_acquire);
        while (s == last) {
            spin_hint();
            s = job_seq.load(std::memory_order_acquire);
        }
        if (s == kStopSeq) return;
        last = s;
        run_mt(tid);
    }
}

}  // namespace fastbp

#endif  // FASTBPOSD_BP_ENGINE_HPP
