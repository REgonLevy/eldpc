// The exact OSD-0 engine with per-column row lists and word spans.
//
// Reference semantics, the same as the earlier dense-bitset engine's:
// columns 0..n-1 sorted by the posterior LLR in ascending
// signed order -- equal values by ascending column index, NaN after every number (so -0.0 == +0.0
// and every NaN is above +inf, NaNs by index); the pivots are the first rank(H) independent columns
// of that order; x is the unique solution of H_S x_S = s supported on them.  The ordering machinery
// (class Osd0Order) is copied verbatim from the dense-bitset engine: this file changes only how a walked
// column is held and reduced.
//
// What changes against the dense-bitset engine.  It keeps every column as a dense bitset
// of words_ = ceil(m/64) words and copies all of it into the working buffer for every walked
// column, memsets all pwords parent words, and runs every XOR, the zero test and the lowest-bit
// search over the whole width.  On bb144_unif_p1e-3_r12_xyz that is 30 words (240 B) of column
// store per column, 76032 columns, 18.2 MB, against a column weight of at most 9 and a measured
// word span of 1-6 words (mean 3.66).  Here:
//
//   * Order, filter pass.  It compares the double against the threshold value with one
//     ordered vector compare instead of deriving the 64-bit key of every element: the threshold is
//     always an element of the posterior, so `key(v) <= key(t)` is `v <= t` (see collect()).  The
//     batch, the walk and the resulting column order are unchanged.
//
//   * Order.  The posterior vector is never fully sorted: a vectorised key-and-filter pass keeps the
//     smallest (key, index) pairs in a candidate buffer whose threshold is the kPrefix-th smallest of
//     what it holds (the buffer is compacted with nth_element(begin, begin + kPrefix - 1, end), which
//     puts the kPrefix smallest at the front and their maximum at position kPrefix - 1 -- the standard
//     guarantee).  Only the columns the pivot walk actually consumes are sorted, in
//     chunks of kStep with lazy doubling.  The initial threshold comes from a stride sample of the
//     posterior; when the walk needs more columns than the batch holds, the threshold widens (sample
//     rank doubling) and finally the whole order is sorted once (stable LSD radix on the key, so ties
//     keep ascending index order) and the walk continues there.  Every batch is the smallest set of
//     candidates of its (lo, hi] range, and the batches are contiguous prefixes of the order, so the
//     walk sees exactly the reference's column order.  (The device sorts the OSD-0
//     prefix with a rank sort before handing it over, so the walk over *that* set takes the
//     no-selection path of init_candidates_sorted() and none of this chunk machinery runs for it.)
//
// Exactness.  Every step computes the same bits as the dense-bitset engine: the buffers hold the same
// u, the same basis vectors, the same parent bitsets and the same t.  Only the set of words
// visited changes, and every skipped word is provably zero: the column's rows are inside [lo, hi],
// a basis vector is zero above its hi and below its lead row, and the working buffer is zero
// outside the range the previous column dirtied.  The pivot column set, the pivot rows and the
// returned x are therefore identical bit for bit.
#ifndef FASTBPOSD_B_OSD0_ENGINE_HPP
#define FASTBPOSD_B_OSD0_ENGINE_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "bp_engine.hpp"
#include "dbench_stages.h"  // stage marks of the measurement build only
#include "osd_caps.hpp"  // fastbp::kOsdPrefixCap: the device prefix cap this buffer must hold

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

// Measurement-only per-shot counters of the development replay harness.  The macro is
// never defined by the library's own build, so the shipped library carries neither the header nor a
// counter and not one instruction of the hot loops below changes.
#ifdef FASTBP6_OSD0_STATS
#include "bench/osd0_stats.hpp"
#define FASTBP6_ST(x) x
// The measurement build can switch the vector word operations off at run time (set_scalar_walk),
// so the replay harness measures the two forms paired in one process; the shipped build's
// FASTBP6_VEC_ON is a compile-time true and the branch below is gone.
#define FASTBP6_VEC_ON (!scalar_walk_)
#define FASTBP6_ST_BEGIN() uint64_t fastbp6_tn_ = fastbp6_tsc()
#define FASTBP6_ST_END() (st_.tsc_next += fastbp6_tsc() - fastbp6_tn_)
// Per-column phase attribution: one phase selected at run time (set_phase), so only
// one TSC pair per walked column is paid; the untimed run (phase < 0) is the baseline.  Phase k's own
// time is (elim_k - elim_base); phase 7 is an empty phase, i.e. the pair's overhead.
#define FASTBP6_ST_PB(k) if (fastbp6_phase_ == (k)) fastbp6_tp_ = fastbp6_tsc()
#define FASTBP6_ST_PE(k, field) \
    if (fastbp6_phase_ == (k)) st_.field += fastbp6_tsc() - fastbp6_tp_
// Measurement-only switch of the dense column table, so the replay harness can A/B the two
// materialisation forms paired in one process; the shipped build only ever runs the dense form.
#define FASTBP6_DENSE_ON (!scalar_dense_)
// Measurement-only switches of the walk's tail restructuring, one bit each, so
// the replay harness can A/B the parts paired in one process; the shipped build runs all of
// them (the branches below are compile-time constants there).
#define FASTBP6_CTZ_ON ((fast_bits_ & 1u) != 0)   // hardware bit scan instead of the de Bruijn table
#define FASTBP6_CUR_ON ((fast_bits_ & 2u) != 0)   // cursor from the dense load / the XOR's own word
#define FASTBP6_PB_ON ((fast_bits_ & 4u) != 0)    // scalar clear of the parent-bitset prefix
#define FASTBP6_PF_ON ((fast_bits_ & 8u) != 0)    // prefetch of the next two columns' dense blocks
#else
#define FASTBP6_ST(x)
#define FASTBP6_ST_BEGIN()
#define FASTBP6_ST_END()
#define FASTBP6_VEC_ON 1
#define FASTBP6_ST_PB(k)
#define FASTBP6_ST_PE(k, field)
#define FASTBP6_DENSE_ON 1
#define FASTBP6_CTZ_ON 1
#define FASTBP6_CUR_ON 1
#define FASTBP6_PB_ON 1
#define FASTBP6_PF_ON 1
#endif

namespace fastbposd {

static constexpr uint32_t kNoColumn = 0xffffffffu;

// Index of the lowest set bit of a nonzero word.  The de Bruijn multiply-and-lookup is the earlier
// engine's (it exists so the -mavx2 build emits no bit-scan encoding); it is kept as the reference
// form of the measurement build's paired A/B (replay harness).  The shipped code
// uses the hardware bit scan: on the target's Zen 5 `tzcnt` is one 1-cycle instruction, while the
// multiply (3 cycles) + shift + 4-cycle table load sits on the reduction's serial chain at every
// set bit; the two return the same index for every nonzero input (only nonzero inputs are passed).
inline uint32_t lowest_set_bit64_fast(uint64_t w) {
#if defined(__x86_64__) || defined(_M_X64)
    return (uint32_t)__builtin_ctzll(w);
#else
    return 0;
#endif
}

inline uint32_t lowest_set_bit64_bruijn(uint64_t w) {
    static const uint8_t kIndex[64] = {
        0,  1,  2,  53, 3,  7,  54, 27, 4,  38, 41, 8,  34, 55, 48, 28,
        62, 5,  39, 46, 44, 42, 22, 9,  24, 35, 59, 56, 49, 18, 29, 11,
        63, 52, 6,  26, 37, 40, 33, 47, 61, 45, 43, 21, 23, 58, 17, 10,
        51, 25, 36, 32, 60, 20, 57, 16, 50, 31, 19, 15, 30, 14, 13, 12};
    return kIndex[((w & (~w + 1)) * 0x022FDD63CC95386Dull) >> 58];
}

inline uint32_t lowest_set_bit64(uint64_t w) { return lowest_set_bit64_fast(w); }

// Ascending signed double as an unsigned key: -0.0 and +0.0 map to the same key, every NaN (any
// sign, any payload) maps above +inf.
inline uint64_t osd_key(double v) {
    uint64_t u;
    std::memcpy(&u, &v, sizeof(u));
    const uint64_t exp_mask = 0x7ff0000000000000ull;
    if ((u & exp_mask) == exp_mask && (u & 0x000fffffffffffffull) != 0) {
        return ~0ull;  // NaN (any sign, any payload) sorts after every number
    }
    if (u == 0x8000000000000000ull) u = 0;  // -0.0 == +0.0
    return (u >> 63) ? ~u : (u | 0x8000000000000000ull);
}

// The reference's column order, produced lazily.  Copied verbatim from the dense-bitset engine
// (class Osd0Order); this engine does not change the ordering.
class Osd0Order {
public:
    static constexpr uint32_t kPrefix = 256;
    // The capacity of the candidate buffer the key-and-filter pass fills, and the cap the device
    // hands over: a prefix the host cannot hold would be truncated silently by
    // init_candidates() below, so the two are asserted coupled here (fastbp::kOsdPrefixCap is the
    // device pass's kOsdCap; osd_caps.hpp).
    static constexpr uint32_t kCand = 2048;
    static_assert(kCand >= fastbp::kOsdPrefixCap,
                  "Osd0Order::kCand must hold the device's whole OSD-0 prefix (kOsdCap, bp_gpu.cu)");
    static constexpr uint32_t kSample = 128;
    static constexpr uint32_t kStep = 32;

    void setup(uint32_t n) {
        n_ = n;
        buf_.assign(kCand, Cand{0, 0, 0});
        sample_.assign(kSample, Cand{0, 0, 0});
        pk_.assign(n ? n : 1, 0);
        pi_.assign(n ? n : 1, 0);
        pi2_.assign(n ? n : 1, 0);
    }

    void init(const double *llr) {
        llr_ = llr;
        sample_n_ = 0;
        if (n_ > kPrefix) {
            const uint32_t stride = (n_ + kSample - 1) / kSample;
            for (uint32_t j = 0; j < n_ && sample_n_ < kSample; j += stride) {
                sample_[sample_n_].key = osd_key(llr[j]);
                sample_[sample_n_].idx = j;
                ++sample_n_;
            }
        }
        srank_ = 0;
        have_hi_ = false;
        hi_key_ = ~0ull;
        hi_idx_ = 0xffffffffu;
        if (sample_n_ > 1) {
            const uint32_t target = kPrefix + kPrefix / 2;
            uint32_t r = (uint32_t)(((uint64_t)target * sample_n_) / n_);
            if (r < 1) r = 1;
            if (r < sample_n_) {
                srank_ = r;
                sample_rank(r, &hi_key_, &hi_idx_);
                have_hi_ = true;
            }
        }
        lo_key_ = 0;
        lo_idx_ = 0;
        returned_ = 0;
        batches_ = 0;
        compactions_ = 0;
        sorts_ = 0;
        mode_ = kBatch;
        cand_only_ = false;
        cand_sorted_ = false;
        collect();
    }

    // Walk a *candidate set* the caller supplies instead of sampling the posteriors.
    // Every pair must be a real (key, column) of the problem and the set must contain the first
    // entries of the order the walk will consume -- the device's threshold selection guarantees
    // that.  The walk then advances exactly as it does over the sampled batches, and returning
    // kNoColumn means "the set is exhausted", which the caller turns into the fallback.
    void init_candidates(const void *records, uint32_t count) {
        if (count > kCand) count = kCand;
        std::memcpy(buf_.data(), records, (size_t)count * sizeof(Cand));
        cand_ptr_ = buf_.data();
        // The walk consumes this set through the same lazy chunk machinery the sampled batches use:
        // CandLess is a total order ((key, index), both unique), so the chunk selection is exact --
        // the n-th smallest element is unique and every chunk is a prefix of the reference's order.
        cnt_ = count;
        i_ = 0;
        w_ = 0;
        full_ = 0;
        returned_ = 0;
        have_hi_ = false;
        mode_ = kBatch;
        cand_only_ = true;
        cand_sorted_ = false;
        batches_ = 1;
        compactions_ = 0;
        sorts_ = 0;
    }

    // The records of `records` whose key is above `after_key`, walked exactly like
    // init_candidates() (the lazy chunk selection, CandLess).  The caller passes a larger prefix of
    // the order ({key <= tau2}) and the largest key of the prefix it has already walked
    // ({key <= after_key}, a whole tie run): what is left is exactly the order's next entries, in the
    // order the chunk selection returns them.
    void init_candidates_after(const void *records, uint32_t count, uint64_t after_key) {
        if (count > kCand) count = kCand;
        const Cand *src = static_cast<const Cand *>(records);
        uint32_t k = 0;
        for (uint32_t i = 0; i < count; ++i) {
            if (src[i].key > after_key) buf_[k++] = src[i];
        }
        cand_ptr_ = buf_.data();
        cnt_ = k;
        i_ = 0;
        w_ = 0;
        full_ = 0;
        returned_ = 0;
        have_hi_ = false;
        mode_ = kBatch;
        cand_only_ = true;
        cand_sorted_ = false;
        batches_ = 1;
        compactions_ = 0;
        sorts_ = 0;
    }

    // The caller's candidate set is *already* in the
    // reference's order -- the device sorted it by the same (key, index) comparator (bp_gpu.cu's
    // osd_rank_sort).  Nothing is sorted here: the walk returns the records in the order they
    // arrived, and returning kNoColumn means "the set is exhausted" as above.  The set must be a
    // prefix of the order for the result to be the reference's, which the device's exact rank
    // selection guarantees.
    //
    // The records are read *in place* -- `records`
    // must stay valid until the walk stops, which is true of the handle's pinned hand-off block (it is
    // overwritten only by that handle's next run()/osd_repick(), both after this walk; the resume's after_key is
    // read before the repick).  No copy is made: Cand is exactly the 16-byte record layout.
    void init_candidates_sorted(const void *records, uint32_t count) {
        if (count > kCand) count = kCand;
        cand_ptr_ = static_cast<const Cand *>(records);
        cnt_ = count;
        i_ = 0;
        w_ = 0;
        full_ = 0;
        returned_ = 0;
        have_hi_ = false;
        mode_ = kBatch;
        cand_only_ = true;
        cand_sorted_ = true;
        batches_ = 1;
        compactions_ = 0;
        sorts_ = 0;
    }

    uint32_t next() {
        for (;;) {
            if (cand_sorted_) {   // the device delivered the set in the reference's order: no selection
                if (i_ >= cnt_) return kNoColumn;
                ++returned_;
                return cand_ptr_[i_++].idx;
            }
            if (mode_ == kFullSort) {
                if (full_ >= n_) return kNoColumn;
                ++returned_;
                return pi_[full_++];
            }
            if (i_ < w_) {
                ++returned_;
                return buf_[i_++].idx;
            }
            if (w_ < cnt_) {
                DBENCH_STAGE_ACC_BEGIN(DBENCH_STAGE_ACC_ORDER);
                uint32_t nw = (w_ == 0) ? kStep : 2 * w_;
                if (nw > cnt_) nw = cnt_;
                if (nw < cnt_) {
                    std::nth_element(buf_.begin() + w_, buf_.begin() + nw, buf_.begin() + cnt_, CandLess{});
                }
                std::sort(buf_.begin() + w_, buf_.begin() + nw, CandLess{});
                w_ = nw;
                DBENCH_STAGE_ACC_END(DBENCH_STAGE_ACC_ORDER);
                continue;
            }
            if (cand_only_) return kNoColumn;  // the caller's candidate set is consumed
            DBENCH_STAGE_ACC_BEGIN(DBENCH_STAGE_ACC_ORDER);
            if (cnt_ > 0) {
                lo_key_ = buf_[cnt_ - 1].key;
                lo_idx_ = buf_[cnt_ - 1].idx;
            }
            if (!have_hi_) {
                ++sorts_;
                full_sort();
                mode_ = kFullSort;
                full_ = returned_;
                DBENCH_STAGE_ACC_END(DBENCH_STAGE_ACC_ORDER);
                continue;
            }
            srank_ = (srank_ >= sample_n_) ? sample_n_ + 1 : 2 * srank_;
            if (srank_ >= sample_n_) {
                have_hi_ = false;
            } else {
                sample_rank(srank_, &hi_key_, &hi_idx_);
            }
            collect();
            DBENCH_STAGE_ACC_END(DBENCH_STAGE_ACC_ORDER);
        }
    }

    // The column k positions after the last next() returned, when it is already in the sorted
    // window; kNoColumn otherwise (the next() would sort a new chunk first, or the source is the
    // full-sort array, whose next columns are known just as well but are not in buf_).  Only the
    // walk's measurement-guarded prefetch uses it: a column index c alone determines the address of
    // its dense table block, so the line can be touched before the walk consumes it.
    // A device-sorted set (cand_sorted_) is its own window, [i_, cnt_), so the prefetch also
    // runs on rung one's sorted records; a prefetch reads nothing and changes no value.
    uint32_t peek(uint32_t k) const {
        const uint32_t j = i_ + k;
        if (cand_sorted_) return (j < cnt_) ? cand_ptr_[j].idx : kNoColumn;   // in place
        if (mode_ != kBatch) return kNoColumn;
        return (j < w_) ? buf_[j].idx : kNoColumn;
    }

    uint32_t returned() const { return returned_; }
    uint32_t last_candidates() const { return cnt_; }
    uint32_t batches() const { return batches_; }
    uint64_t compactions() const { return compactions_; }
    uint64_t sorts() const { return sorts_; }

private:
    struct Cand {
        uint64_t key;
        uint32_t idx;
        uint32_t pad;
    };
    struct CandLess {
        bool operator()(const Cand &a, const Cand &b) const {
            if (a.key != b.key) return a.key < b.key;
            return a.idx < b.idx;
        }
    };
    enum Mode { kBatch = 0, kFullSort = 1 };

    void sample_rank(uint32_t r, uint64_t *key, uint32_t *idx) {
        std::nth_element(sample_.begin(), sample_.begin() + (r - 1), sample_.begin() + sample_n_, CandLess{});
        *key = sample_[r - 1].key;
        *idx = sample_[r - 1].idx;
    }

    void collect() {
        ++batches_;
        uint32_t c = 0;
        uint64_t tkey = have_hi_ ? hi_key_ : ~0ull;
        uint32_t tidx = have_hi_ ? hi_idx_ : 0xffffffffu;
        const double *x = llr_;
        const uint32_t n = n_;
        uint32_t j = 0;
        // The batch threshold is the key of an element of the posterior (hi_idx_, or the candidate
        // kept at kPrefix - 1 by the last compaction), so for a non-NaN threshold,
        //     key(v) <= key(t)   <=>   v <= t
        // on every pair of non-NaN doubles: osd_key is monotone, and it maps -0.0 and +0.0 to the
        // same key while the ordered compare makes them equal too.  A NaN v fails the ordered
        // compare, as it must (its key is above every number).  The index part of the (key, index)
        // bound is applied by push_key either way, so the vector loop only has to be a superset
        // test.  An absent upper bound (no threshold) or a NaN threshold keeps the scalar path,
        // which pushes every column.
        const bool ordered_thr = have_hi_ && tidx != 0xffffffffu && !std::isnan(x[tidx]);
        if (ordered_thr) {
            uint32_t last_tidx = tidx;
            double tval = x[tidx];
#if defined(__AVX512F__)
            {
                __m512d tv = _mm512_set1_pd(tval);
                for (; j + 8 <= n; j += 8) {
                    const __m512d v = _mm512_loadu_pd(x + j);
                    const __mmask8 hit = _mm512_cmp_pd_mask(v, tv, _CMP_LE_OQ);
                    if (hit) {
                        for (int lane = 0; lane < 8; ++lane) {
                            if (!((hit >> lane) & 1)) continue;
                            const uint32_t jj = j + (uint32_t)lane;
                            push_key(osd_key(x[jj]), jj, c, tkey, tidx);
                        }
                        if (tidx != last_tidx) {  // a compaction raised the threshold
                            last_tidx = tidx;
                            tv = _mm512_set1_pd(x[tidx]);
                        }
                    }
                }
            }
#elif defined(__AVX2__)
            {
                __m256d tv = _mm256_set1_pd(tval);
                for (; j + 4 <= n; j += 4) {
                    const __m256d v = _mm256_loadu_pd(x + j);
                    const unsigned mask = (unsigned)_mm256_movemask_pd(_mm256_cmp_pd(v, tv, _CMP_LE_OQ));
                    if (mask) {
                        for (int lane = 0; lane < 4; ++lane) {
                            if (!((mask >> lane) & 1)) continue;
                            const uint32_t jj = j + (uint32_t)lane;
                            push_key(osd_key(x[jj]), jj, c, tkey, tidx);
                        }
                        if (tidx != last_tidx) {  // a compaction raised the threshold
                            last_tidx = tidx;
                            tv = _mm256_set1_pd(x[tidx]);
                        }
                    }
                }
            }
#endif
        }
        for (; j < n; ++j) push_key(osd_key(x[j]), j, c, tkey, tidx);
        cnt_ = c;
        i_ = 0;
        w_ = 0;
    }

    inline void push_key(uint64_t k, uint32_t jj, uint32_t &c, uint64_t &tkey, uint32_t &tidx) {
        if (k < lo_key_ || (k == lo_key_ && jj <= lo_idx_)) return;
        if (k > tkey || (k == tkey && jj >= tidx)) return;
        if (c == kCand) {
            std::nth_element(buf_.begin(), buf_.begin() + (kPrefix - 1), buf_.begin() + c, CandLess{});
            ++compactions_;
            c = kPrefix;
            tkey = buf_[kPrefix - 1].key;
            tidx = buf_[kPrefix - 1].idx;
            if (k < lo_key_ || (k == lo_key_ && jj <= lo_idx_)) return;
            if (k > tkey || (k == tkey && jj >= tidx)) return;
        }
        buf_[c].key = k;
        buf_[c].idx = jj;
        ++c;
    }

    void full_sort() {
        const uint32_t n = n_;
        for (uint32_t j = 0; j < n; ++j) {
            pk_[j] = osd_key(llr_[j]);
            pi_[j] = j;
        }
        uint32_t *src = pi_.data(), *dst = pi2_.data();
        for (uint32_t pass = 0; pass < 8; ++pass) {
            const uint32_t shift = pass * 8;
            uint32_t count[256], pos[256];
            std::memset(count, 0, sizeof(count));
            for (uint32_t i = 0; i < n; ++i) ++count[(pk_[src[i]] >> shift) & 0xffu];
            uint32_t sum = 0;
            for (uint32_t b = 0; b < 256; ++b) {
                pos[b] = sum;
                sum += count[b];
            }
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t j = src[i];
                dst[pos[(pk_[j] >> shift) & 0xffu]++] = j;
            }
            uint32_t *t = src;
            src = dst;
            dst = t;
        }
        if (src != pi_.data()) {
            pi2_.swap(pi_);
        }
    }

    uint32_t n_ = 0;
    const double *llr_ = nullptr;
    std::vector<Cand> buf_;
    const Cand *cand_ptr_ = nullptr;   // the sorted set's records, read in place (else buf_)
    std::vector<Cand> sample_;
    std::vector<uint64_t> pk_;
    std::vector<uint32_t> pi_, pi2_;
    uint32_t sample_n_ = 0;
    uint32_t srank_ = 0;
    bool have_hi_ = false;
    uint64_t hi_key_ = ~0ull;
    uint32_t hi_idx_ = 0xffffffffu;
    uint64_t lo_key_ = 0;
    uint32_t lo_idx_ = 0;
    uint32_t cnt_ = 0, i_ = 0, w_ = 0, full_ = 0;
    int mode_ = kBatch;
    bool cand_only_ = false;   // init_candidates(): the batch IS the caller's candidate set
    bool cand_sorted_ = false; // ...and the device already sorted it (init_candidates_sorted)
    uint32_t returned_ = 0, batches_ = 0;
    uint64_t compactions_ = 0, sorts_ = 0;
};

// One OSD-0 decode.  setup() is called once from dbench_create; run() is called for every shot
// whose syndrome is non-zero and whose BP did not converge.
class Osd0Engine {
public:
    enum Result { kSolved = 0, kDeferred = 1 };
    static constexpr uint32_t kNone = 0xffffffffu;
    // The dense column table: one 64-byte block per column, word 0 a header
    // (lo | span << 32) and words 1..span the column's bits over its word span.  A column whose span
    // does not fit (span > kDenseSpanMax) has header 0 and is materialised from the CSC slice.
    static constexpr uint32_t kDenseWords = 8;
    static constexpr uint32_t kDenseSpanMax = kDenseWords - 1;

    // Column records of H (CSC), rank(H) and every per-shot buffer; nothing here depends on a shot.
    bool setup(uint32_t m, uint32_t n, const uint32_t *colptr, const uint32_t *rowind) {
        m_ = m;
        n_ = n;
        words_ = (m + 63u) / 64u;
        if (words_ == 0) words_ = 1;
        colptr_.assign(colptr, colptr + n + 1);
        const uint32_t nnz = n ? colptr[n] : 0;
        rows_.resize(nnz);
        lo_.assign(n ? n : 1, 0);
        hi_.assign(n ? n : 1, 0);
        for (uint32_t j = 0; j < n; ++j) {
            for (uint32_t e = colptr[j]; e < colptr[j + 1]; ++e) rows_[e] = rowind[e];
            if (colptr[j + 1] > colptr[j]) {  // CSC rows per column are increasing
                lo_[j] = rowind[colptr[j]] >> 6;
                hi_[j] = rowind[colptr[j + 1] - 1] >> 6;
            }
        }
        // The dense column table: every column whose span fits in one 64-byte block gets its bits
        // precomputed at create time, so a walked column is one cache line (header + span words)
        // instead of a colptr_ -> rows_ dependent load chain.  Columns with span > kDenseSpanMax
        // (and empty columns) have header 0 and take the CSC path at run time, bit for bit the old
        // materialisation.
        {
            dense_.assign((size_t)(n_ ? n_ : 1) * kDenseWords + kDenseWords, 0);
            uintptr_t p = (uintptr_t)dense_.data();
            p = (p + (uintptr_t)63u) & ~(uintptr_t)63u;
            dense_base_ = reinterpret_cast<uint64_t *>(p);
            for (uint32_t j = 0; j < n_; ++j) {
                if (colptr_[j + 1] == colptr_[j]) continue;
                const uint32_t lo = lo_[j], hi = hi_[j];
                const uint32_t span = hi - lo + 1;
                if (span > kDenseSpanMax) continue;
                uint64_t *b = dense_base_ + (size_t)j * kDenseWords;
                b[0] = (uint64_t)lo | ((uint64_t)span << 32);
                for (uint32_t e = colptr_[j]; e < colptr_[j + 1]; ++e) {
                    b[1 + (rows_[e] >> 6) - lo] |= 1ull << (rows_[e] & 63u);
                }
            }
        }
        // rank(H): the same greedy reduction over the natural column order.
        rcap_ = (m_ < n_ ? m_ : n_);
        if (rcap_ == 0) rcap_ = 1;
        pwords_ = (rcap_ + 63u) / 64u;
        basis_.assign((size_t)rcap_ * words_, 0);
        basis_hi_.assign(rcap_, 0);
        parent_bits_.assign((size_t)rcap_ * pwords_, 0);
        acc_bits_.assign(pwords_, 0);
        prow_.assign(m_ ? m_ : 1, 0);
        pivmask_.assign((size_t)words_ + 8, 0);  // bit r set iff prow_[r] != 0
        tvec_.assign(words_, 0);
        twords_ = (words_ + 63u) / 64u;
        ubuf_.assign(words_, 0);
        for (uint32_t w = 0; w < words_; ++w) ubuf_[w] = 0;
        pivcol_.assign(rcap_, 0);
        par_.assign(rcap_, 0);
        sup_.assign(rcap_, 0);
        ord_.setup(n_);
        // greedy rank over the natural order, with the same span reduction
        rank_h_ = 0;
        std::memset(prow_.data(), 0, (size_t)m_ * sizeof(uint32_t));
        std::memset(pivmask_.data(), 0, pivmask_.size() * 8);
        for (uint32_t j = 0; j < n_; ++j) {
            if (colptr_[j + 1] == colptr_[j]) continue;
            uint32_t hi = hi_[j];
            for (uint32_t e = colptr_[j]; e < colptr_[j + 1]; ++e) {
                ubuf_[rows_[e] >> 6] |= 1ull << (rows_[e] & 63u);
            }
            const uint32_t r = reduce_span(ubuf_.data(), lo_[j], &hi, prow_.data(), basis_.data(),
                                           basis_hi_.data(), nullptr, ubuf_[lo_[j]]);
            if (r != kNone) {
                std::memcpy(&basis_[(size_t)rank_h_ * words_], ubuf_.data(), (size_t)words_ * 8);
                basis_hi_[rank_h_] = hi;
                prow_[r] = rank_h_ + 1;
                pivmask_[r >> 6] |= 1ull << (r & 63u);
                ++rank_h_;
            }
            for (uint32_t w = lo_[j]; w <= hi; ++w) ubuf_[w] = 0;  // restore the all-zero buffer
        }
        // The rank pass filled rank_h_ basis vectors; clear them and the pivot table for the shots.
        std::memset(prow_.data(), 0, (size_t)m_ * sizeof(uint32_t));
        std::memset(pivmask_.data(), 0, pivmask_.size() * 8);
        for (uint32_t w = 0; w < words_; ++w) ubuf_[w] = 0;
        dlo_ = 0;  // the buffer is all zero: the first column clears nothing
        dhi_ = 0;
        return true;
    }

    uint32_t rank_h() const { return rank_h_; }

    // The support of the most recent kSolved run: the columns with x = 1, in the order of the
    // back-substitution walk (pivcol_ descending).  Valid only after run() returned kSolved.
    // O*x over the support is one observable-mask XOR per set column, so the caller
    // does not have to walk the correction's n bytes to find them.
    const uint32_t *support() const { return sup_.data(); }
    uint32_t support_size() const { return nsup_; }

    // Returns kSolved and writes the n-byte correction into x_out, or kDeferred and writes nothing
    // (the caller then runs the vendored post-processing for that shot).  kDeferred happens exactly
    // when the syndrome is outside the image of H.
    Result run(const double *llr, const uint8_t *syn, uint8_t *x_out) {
#ifdef FASTBP6_OSD0_STATS
        if (record_) { rec_cols_.clear(); rec_basis_.clear(); }
#endif
        ord_.init(llr);
        DBENCH_STAGE(DBENCH_STAGE_ORDER_INIT);
        return run_core([this]() { return ord_.next(); }, syn, x_out, nullptr);
    }

    // The same elimination over a *candidate set* the device delivered: `records` holds
    // count 16-byte (u64 key, u32 column, u32 pad) pairs, a superset of the first entries of the
    // order defined at the top of this file.  They are sorted lazily, exactly as the sampled batches are,
    // and the walk consumes them in the reference's order.  Returns kSolved, or kDeferred with
    // *exhausted true when the walk ran past the candidate set (the caller then continues on the full
    // posterior, which is the contract's fallback), or kDeferred with *exhausted false for a syndrome
    // outside the image of H (the caller runs the vendored solver).  Nothing is cached between calls.
    // `sorted` says the records are already in the reference's order (the device sorted them, see
    // bp_gpu.cu's osd_rank_sort): the walk then just consumes them, with no chunk selection at all.
    Result run_candidates(const void *records, uint32_t count, const uint8_t *syn, uint8_t *x_out,
                          bool *exhausted, bool sorted = false) {
#ifdef FASTBP6_OSD0_STATS
        if (record_) { rec_cols_.clear(); rec_basis_.clear(); }
#endif
        if (sorted) ord_.init_candidates_sorted(records, count);
        else        ord_.init_candidates(records, count);
        DBENCH_STAGE(DBENCH_STAGE_ORDER_INIT);
        return run_core([this]() { return ord_.next(); }, syn, x_out, exhausted);
    }

    // Continue the walk of the most recent run_candidates() that ran out of its set
    // (kDeferred with *exhausted true) over a larger prefix of the same order.  `records` is the
    // larger set ({key <= tau2}, any order), `after_key` the largest key of the set already walked
    // ({key <= after_key}).  run_core is a fold over the column sequence that keeps all of its state
    // in this object (pivot rows, reduced syndrome, basis, parents, the buffer's dirty span) and stops
    // only when the syndrome is consumed or rank(H) is reached -- neither happened, since the walk
    // ran out -- so walking the remainder from that state is the same fold as walking the larger
    // set from the first column.  Returns what run_candidates() returns.  Without a live walk, or
    // when the larger set does not reach after_key (it would then not contain the walked prefix --
    // the rung rule never produces that, the check keeps the call exact regardless), it walks the
    // set from the first column instead: run_candidates(records, count, syn, ...).
    bool resumable() const { return w_live_; }
    Result resume_candidates(const void *records, uint32_t count, uint64_t after_key, const uint8_t *syn,
                             uint8_t *x_out, bool *exhausted) {
        bool covers = false;
        {
            const unsigned char *r = static_cast<const unsigned char *>(records);
            for (uint32_t i = 0; i < count && !covers; ++i) {
                uint64_t k;
                std::memcpy(&k, r + (size_t)i * 16u, 8);
                covers = (k >= after_key);
            }
        }
        if (!w_live_ || !covers) {
            ++st_resume_fresh;
            return run_candidates(records, count, syn, x_out, exhausted, false);
        }
        ++st_resumed;
        ord_.init_candidates_after(records, count, after_key);
        DBENCH_STAGE(DBENCH_STAGE_ORDER_INIT);
        return run_core([this]() { return ord_.next(); }, syn, x_out, exhausted, true);
    }

private:
    // The elimination itself, over any source of columns in the order (`next()` returns the next
    // column or kNoColumn when the source is exhausted).  run() and run_candidates() differ only in
    // the source, so both take exactly the same path through the GF(2) work.
    template <class Next>
    Result run_core(Next next, const uint8_t *syn, uint8_t *x_out, bool *exhausted, bool resume = false) {
        uint32_t rank = 0;
        uint32_t tvec_hi = m_ ? ((m_ - 1) >> 6) : 0;  // the syndrome's own words are the first non-zero ones
        if (resume) {
            // The state the last walk left when it ran out of its set (see
            // resume_candidates): prow_, tvec_, basis_, parent_bits_, par_, pivcol_ and dlo_/dhi_ are
            // members and untouched since; rank and tvec_hi are the two locals it saved.
            rank = w_rank_;
            tvec_hi = w_tvec_hi_;
        } else {
            std::memset(prow_.data(), 0, (size_t)m_ * sizeof(uint32_t));
            std::memset(pivmask_.data(), 0, pivmask_.size() * 8);
            pack_syndrome(tvec_.data(), syn);   // zeroes tvec_, then the packed syndrome
        }
        w_live_ = false;
        bool in_image = false;
        bool ran_out = false;  // the column source was exhausted (as opposed to rank(H) reached)
        FASTBP6_ST(st_.reset(); uint64_t fastbp6_tw_ = fastbp6_tsc());
        for (;;) {
            FASTBP6_ST_BEGIN();
            const uint32_t c = next();
            FASTBP6_ST_END();
            if (c == kNoColumn) { ran_out = true; break; }
            FASTBP6_ST(++st_.columns);
#ifdef FASTBP6_OSD0_STATS
            if (record_) rec_cols_.push_back(c);  // measurement-only (attribution mode): the walked columns
#endif
            // Restructured walk (FASTBP6_PF_ON): touch the dense blocks of the next columns of the
            // order now.  Their addresses are fixed by the column indices alone, and the current
            // column's reduction runs for a few hundred cycles before the next block is read, so the
            // L2/L3 latency of that read is overlapped with work instead of sitting on the serial
            // chain.  The prefetch reads nothing the walk would not read (every column's block is
            // touched when the walk consumes it); a prefetch across a chunk boundary is skipped
            // (peek returns kNoColumn) rather than forcing the order to sort the next chunk early.
            if (FASTBP6_PF_ON) {
                const uint32_t pf0 = ord_.peek(0);
                const uint32_t pf1 = ord_.peek(1);
                if (pf0 != kNoColumn) {
                    __builtin_prefetch((const void *)(dense_base_ + (size_t)pf0 * kDenseWords), 0, 3);
                }
                if (pf1 != kNoColumn) {
                    __builtin_prefetch((const void *)(dense_base_ + (size_t)pf1 * kDenseWords), 0, 3);
                }
            }
            FASTBP6_ST(uint64_t fastbp6_tp_ = 0);
            FASTBP6_ST(FASTBP6_ST_PB(7); FASTBP6_ST_PE(7, tsc_empty));
            // The column's dense block: one 64-byte line holding the header (lo | span << 32) and the
            // column's span words.  This load is the whole CSC fetch of the common case: no colptr_ and
            // no rows_ indirection, and (unlike them) its address is known as soon as c is.
            const uint64_t *dblk = dense_base_ + (size_t)c * kDenseWords;
            const uint64_t dhdr = dblk[0];
            const uint32_t dspan = (uint32_t)(dhdr >> 32);
            FASTBP6_ST_PB(0);
            // Materialise the column in the shared buffer, clearing only what the last column dirtied.
            zero_span(ubuf_.data(), dlo_, dhi_);
            FASTBP6_ST_PE(0, tsc_zero_prev);
            FASTBP6_ST_PB(1);
            uint32_t lo, hi;
            // The word the reduction scan starts from.  The dense store hands it back from the same
            // register the column was loaded in, so the scan does not have to read a word that was
            // just written (a masked vector store immediately followed by a scalar load of the same
            // address does not forward cheaply); the CSC path loads it from the buffer as before.
            uint64_t cur0;
            if (FASTBP6_DENSE_ON && dspan != 0) {
                lo = (uint32_t)dhdr;
                hi = lo + dspan - 1;
                cur0 = dense_store_line(ubuf_.data(), dblk, lo, dspan);
            } else {
                if (colptr_[c + 1] == colptr_[c]) { FASTBP6_ST(++st_.skipped_empty); continue; }  // empty column: never a pivot
                for (uint32_t e = colptr_[c]; e < colptr_[c + 1]; ++e) {
                    ubuf_[rows_[e] >> 6] |= 1ull << (rows_[e] & 63u);
                }
                lo = lo_[c];
                hi = hi_[c];
                cur0 = FASTBP6_CUR_ON ? ubuf_[lo] : 0;
            }
            FASTBP6_ST_PE(1, tsc_mat);
            uint64_t *pbits = &parent_bits_[(size_t)rank * pwords_];
            const uint32_t live = (rank >> 6) + 1;
            FASTBP6_ST_PB(2);
            // The parent bitset's live prefix is a few words (1 below rank 64); the restructured walk
            // clears it with plain stores, whose forwarding into reduce_span's read-modify-write is
            // cheap, instead of a masked vector store followed by a scalar read-modify-write.
            if (FASTBP6_PB_ON) {
                for (uint32_t i = 0; i < live; ++i) pbits[i] = 0;
            } else {
                zero_span(pbits, 0, live - 1);
            }
            FASTBP6_ST_PE(2, tsc_pbits);
            FASTBP6_ST(st_.span_words += hi - lo + 1; st_.pbits_words += live);
            FASTBP6_ST_PB(3);
            const uint32_t r = reduce_span(ubuf_.data(), lo, &hi, prow_.data(), basis_.data(),
                                           basis_hi_.data(), pbits, cur0);
            FASTBP6_ST_PE(3, tsc_reduce);
            dlo_ = lo;
            dhi_ = hi;
            if (r == kNone) {
                FASTBP6_ST(++st_.dependent);
                continue;  // dependent column: its parent bits are dropped
            }
            FASTBP6_ST(++st_.pivots);
            FASTBP6_ST_PB(4);
            prow_[r] = rank + 1;
            pivmask_[r >> 6] |= 1ull << (r & 63u);
            // The basis vector is zero below lo_[c] and above hi; the readers start at the lead
            // row's word and stop at basis_hi_, so only [lo_[c], hi] has to be stored.
            copy_span(&basis_[(size_t)rank * words_], ubuf_.data(), lo, hi);
            basis_hi_[rank] = (uint16_t)hi;
            pivcol_[rank] = c;
            const uint8_t p = (uint8_t)((tvec_[r >> 6] >> (r & 63u)) & 1ull);
            par_[rank] = p;
            FASTBP6_ST_PE(4, tsc_pivot);
            FASTBP6_ST_PB(5);
            if (p) {
                FASTBP6_ST(st_.tvec_xor_words += hi - (r >> 6) + 1);
                xor_span(tvec_.data(), ubuf_.data(), r >> 6, hi);
                if (hi > tvec_hi) tvec_hi = hi;
            }
            ++rank;
            FASTBP6_ST_PE(5, tsc_txor);
            FASTBP6_ST_PB(6);
            const bool anynz = any_nonzero_word(tvec_.data(), tvec_hi + 1);
            FASTBP6_ST_PE(6, tsc_tzero);
            if (!anynz) {
                in_image = true;
                break;
            }
            if (rank == rank_h_) break;  // every later column is dependent: t cannot change
        }
        FASTBP6_ST(st_.tsc_walk += fastbp6_tsc() - fastbp6_tw_);
        DBENCH_STAGE(DBENCH_STAGE_WALK);
        if (!in_image) {
            if (exhausted != nullptr) *exhausted = ran_out;
            if (ran_out) {   // a walk that ran out of its set can be continued
                w_rank_ = rank;
                w_tvec_hi_ = tvec_hi;
                w_live_ = true;
            }
            ++st_deferrals;
            return kDeferred;
        }
        std::memset(acc_bits_.data(), 0, (size_t)pwords_ * 8);
        for (uint32_t k = 0; k < rank; ++k) {
            if (par_[k]) acc_bits_[k >> 6] |= 1ull << (k & 63u);
        }
        std::memset(x_out, 0, n_);
        nsup_ = 0;
        // The same reverse propagation over the same descending pivot order, but a word of acc_bits_
        // with no set bit is skipped whole instead of being tested bit by bit: the pivot ids above
        // the cursor are final (a visit only XORs parent words 0..k>>6, i.e. bits <= k), so the walk
        // can move to the next lower word as soon as the current one has no set bit left.  Every
        // write to x_out, every support entry and every XOR of a parent word happens in exactly the
        // order the plain `for (k = rank; k-- > 0;)` loop performs them.  rank(H) is 1,872 on
        // bb144 xyz, where the plain scan's bit test per pivot measured 2.76 us of the 3.64 us loop
        // (standalone tail benchmark); this shape spends it on the 30 acc_bits_ words
        // and the set bits instead.
        uint64_t *acc = acc_bits_.data();
        uint32_t w = pwords_;
        FASTBP6_ST(uint64_t fastbp6_tb_ = fastbp6_tsc());
        while (w > 0) {
            --w;
            uint64_t cur = acc[w];
            while (cur) {
                const uint32_t b = 63u - (uint32_t)__builtin_clzll(cur);
                const uint32_t k = (w << 6) + b;
                x_out[pivcol_[k]] = 1u;
                sup_[nsup_++] = pivcol_[k];
                const uint64_t *pb = &parent_bits_[(size_t)k * pwords_];
                for (uint32_t i = 0; i <= w; ++i) acc[i] ^= pb[i];
                // Bits at or above b cannot be set by the XOR (a pivot's parents are earlier pivots,
                // so pb[w] has no bit at b or above); masking anyway keeps the descent strict.
                cur = acc[w] & ((1ull << b) - 1ull);
            }
        }
        FASTBP6_ST(st_.tsc_backsub += fastbp6_tsc() - fastbp6_tb_);
        DBENCH_STAGE(DBENCH_STAGE_BACKSUB);
        st_last_walked = ord_.returned();
        st_last_pivots = rank;
        st_last_candidates = ord_.last_candidates();
        ++st_solved;
        return kSolved;
    }

    // ---- the walk's word operations, over a column's active words only -------------------------
    // Every word loop of the walk runs over a *span* [lo, hi] of the u64 row words some column can
    // touch (at most words_ = ceil(m/64): 30 on bb144 xyz, 3-4 on the average walked column).  With
    // AVX-512 one masked zmm op covers eight words, so the clear, the copy and the two XORs a walked
    // column costs are one or two instructions instead of a scalar loop with a data-dependent trip
    // count.  The mask registers (k) keep every access inside [lo, hi]: a masked-off lane is not read
    // at all, so nothing outside the buffer is ever touched, and the unmasked tail word writes of the
    // scalar form are preserved bit for bit (the masked-out words are left unchanged).  The scalar
    // form is the AVX2 build's code and, in the measurement build, the A/B reference of
    // the replay harness (set_scalar_walk); both forms compute the same words.
    static constexpr uint32_t kVecWords = 8;
    static inline __mmask8 span_tail_mask(uint32_t n) {  // n in 1..kVecWords
        return (__mmask8)((1u << n) - 1u);
    }

    // The dense table's fast path: write the block's span words (blk[0..span-1]) into u[lo..lo+span-1]
    // with one masked 8-word load and one masked store (span in 1..kDenseSpanMax).  The masked-off
    // lanes of the load are not read at all, so the block's padding is never touched.
    // Returns lane 0 of the loaded block, i.e. the word written at u[lo]: the restructured walk
    // starts its reduction scan from that register value instead of re-reading u[lo] after the store.
    inline uint64_t dense_store(uint64_t *u, const uint64_t *blk, uint32_t lo, uint32_t span) {
#if defined(__AVX512F__)
        if (FASTBP6_VEC_ON) {
            const __mmask8 m = span_tail_mask(span);
            const __m512i v = _mm512_maskz_loadu_epi64(m, (const void *)blk);
            _mm512_mask_storeu_epi64((void *)(u + lo), m, v);
            return (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(v));
        }
#endif
        for (uint32_t i = 0; i < span; ++i) u[lo + i] = blk[i];
        return blk[0];
    }

    // The same store from the block's own aligned line.  The masked 8-lane load from
    // blk + 1 that dense_store does spans 8 bytes of the next column's line, and on Zen 5 the
    // masked-off lane still fetches that line (about one extra L3/DRAM line per walked column,
    // measured in the replay harness).  Here the one aligned 64-byte load reads exactly the
    // block's line; valignq shifts the header out, so lanes 0..span-1 are blk[1..span] as before and
    // the masked store and the returned lane 0 are bit for bit those of dense_store(u, blk + 1, ...).
    inline uint64_t dense_store_line(uint64_t *u, const uint64_t *blk_line, uint32_t lo, uint32_t span) {
#if defined(__AVX512F__)
        if (FASTBP6_VEC_ON) {
            const __m512i v0 = _mm512_load_si512((const void *)blk_line);
            const __m512i v = _mm512_alignr_epi64(_mm512_setzero_si512(), v0, 1);
            _mm512_mask_storeu_epi64((void *)(u + lo), span_tail_mask(span), v);
            return (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(v));
        }
#endif
        return dense_store(u, blk_line + 1, lo, span);
    }

    // The shot's syndrome into tvec_ (words_ u64 words): the caller's syndrome is one byte per row
    // and the ABI's convention (decoder_abi.h, and the glue's masking in fastbpgpu6g.cpp) is that
    // the row's bit is bit 0 of its byte, so (v & 1) != 0 per byte is the word's bit.  AVX-512 packs
    // 64 bytes into one word per test instead of 1872 byte loads, shifts and ORs per shot.
    inline void pack_syndrome(uint64_t *t, const uint8_t *syn) {
        std::memset(t, 0, (size_t)words_ * 8);
#if defined(__AVX512F__) && defined(__AVX512BW__)
        if (FASTBP6_VEC_ON) {
            const __m512i one = _mm512_set1_epi8(1);
            uint32_t i = 0;
            for (; i + 64 <= m_; i += 64) {
                const __m512i v = _mm512_loadu_si512((const void *)(syn + i));
                t[i >> 6] |= (uint64_t)_mm512_test_epi8_mask(v, one);
            }
            for (; i < m_; ++i) {
                if (syn[i] & 1u) t[i >> 6] |= 1ull << (i & 63u);
            }
            return;
        }
#endif
        for (uint32_t i = 0; i < m_; ++i) {
            if (syn[i] & 1u) t[i >> 6] |= 1ull << (i & 63u);
        }
    }

    inline void zero_span(uint64_t *u, uint32_t lo, uint32_t hi) {
        // the span is never empty (every caller's lo <= hi); the guard keeps an empty one
        // a no-op, as the scalar loop `for (w = lo; w <= hi; ++w)` it replaces would be.
        assert(hi >= lo);
        if (hi < lo) return;
        uint32_t n = hi - lo + 1, w = lo;
#if defined(__AVX512F__)
        if (FASTBP6_VEC_ON) {
            const __m512i z = _mm512_setzero_si512();
            while (n >= kVecWords) { _mm512_storeu_si512((void *)(u + w), z); w += kVecWords; n -= kVecWords; }
            if (n) _mm512_mask_storeu_epi64((void *)(u + w), span_tail_mask(n), z);
            return;
        }
#endif
        for (; n; ++w, --n) u[w] = 0;
    }

    inline void copy_span(uint64_t *d, const uint64_t *s, uint32_t lo, uint32_t hi) {
        // the span is never empty (every caller's lo <= hi); the guard keeps an empty one
        // a no-op, as the scalar loop `for (w = lo; w <= hi; ++w)` it replaces would be.
        assert(hi >= lo);
        if (hi < lo) return;
        uint32_t n = hi - lo + 1, w = lo;
#if defined(__AVX512F__)
        if (FASTBP6_VEC_ON) {
            while (n >= kVecWords) {
                _mm512_storeu_si512((void *)(d + w), _mm512_loadu_si512((const void *)(s + w)));
                w += kVecWords; n -= kVecWords;
            }
            if (n) {
                const __mmask8 k = span_tail_mask(n);
                _mm512_mask_storeu_epi64((void *)(d + w), k, _mm512_maskz_loadu_epi64(k, (const void *)(s + w)));
            }
            return;
        }
#endif
        for (; n; ++w, --n) d[w] = s[w];
    }

    inline void xor_span(uint64_t *u, const uint64_t *b, uint32_t lo, uint32_t hi) {
        // never empty; an empty span stays a no-op, as the scalar loop it replaces.
        assert(hi >= lo);
        if (hi < lo) return;
        uint32_t n = hi - lo + 1, w = lo;
#if defined(__AVX512F__)
        if (FASTBP6_VEC_ON) {
            while (n >= kVecWords) {
                const __m512i a = _mm512_loadu_si512((const void *)(u + w));
                const __m512i c = _mm512_loadu_si512((const void *)(b + w));
                _mm512_storeu_si512((void *)(u + w), _mm512_xor_si512(a, c));
                w += kVecWords; n -= kVecWords;
            }
            if (n) {
                const __mmask8 k = span_tail_mask(n);
                const __m512i a = _mm512_maskz_loadu_epi64(k, (const void *)(u + w));
                const __m512i c = _mm512_maskz_loadu_epi64(k, (const void *)(b + w));
                _mm512_mask_storeu_epi64((void *)(u + w), k, _mm512_xor_si512(a, c));
            }
            return;
        }
#endif
        for (; n; ++w, --n) u[w] ^= b[w];
    }

    // Is any word of v[0..n-1] non-zero?  The trailing convergence test's scan (the same early exit:
    // the first non-zero word decides), n <= words_.
    inline bool any_nonzero_word(const uint64_t *v, uint32_t n) {
        uint32_t w = 0;
#if defined(__AVX512F__)
        if (FASTBP6_VEC_ON) {
            while (n - w >= kVecWords) {
                const __m512i a = _mm512_loadu_si512((const void *)(v + w));
                FASTBP6_ST(st_.tzero_words += kVecWords);
                if (_mm512_test_epi64_mask(a, a)) return true;
                w += kVecWords;
            }
        }
#endif
        for (; w < n; ++w) {
            FASTBP6_ST(++st_.tzero_words);
            if (v[w]) return true;
        }
        return false;
    }

    // Reduce u (in place) into the unique vector with no bit at an existing pivot row: the lowest
    // set bit first, XORing the basis vector of the pivot row it hits.  The scan starts at word
    // `lo` and never goes past *hi (a live bound: the XORs raise it to the highest word they can
    // touch).  Returns the lowest surviving (non-pivot) row, or kNone when u is zero.
    // `pbits`, when not null, records one bit per basis vector used (the parent bitset).
    uint32_t reduce_span(uint64_t *u, uint32_t lo, uint32_t *hi, const uint32_t *prow,
                         const uint64_t *basis, const uint32_t *bhi, uint64_t *pbits,
                         uint64_t cur0) {
        uint32_t wi = lo;
        // cur0 is u[lo] as the caller already holds it (the dense store's register, or the CSC
        // buffer's first word); the reference form re-reads it from the buffer.
        uint64_t cur = FASTBP6_CUR_ON ? cur0 : u[wi];
        uint32_t rlo = kNone;
        if (FASTBP6_CUR_ON && FASTBP6_CTZ_ON) {
            // the pivot-row bitmask (pivmask_, kept in step with prow_) splits a word into
            // its pivot hits and its free bits at once, so the loop runs once per pivot hit (each a
            // XOR) instead of once per set bit, with no per-bit "pivot or free" branch; the next
            // non-zero word is found with one vector test per 8 words instead of a word-by-word walk.
            // Exactness: a XOR at row r changes only bits above r, so the free bits below the lowest
            // hit are final and the lowest of them is exactly the reference's rlo; the hits are taken
            // lowest first, as the reference takes them; u in memory receives the same XORs.
            const uint64_t *pm = pivmask_.data();
            for (;;) {
                if (cur == 0) {
#if defined(__AVX512F__)
                    if (FASTBP6_VEC_ON) {
                        uint32_t w = wi + 1;
                        for (;;) {
                            if (w > *hi) return rlo;
                            const uint32_t left = *hi - w + 1;
                            const __mmask8 k = left >= kVecWords ? (__mmask8)0xff : span_tail_mask(left);
                            const __m512i v = _mm512_maskz_loadu_epi64(k, (const void *)(u + w));
                            const __mmask8 nz = _mm512_test_epi64_mask(v, v);
                            if (nz) { w += (uint32_t)__builtin_ctz((unsigned)nz); break; }
                            w += kVecWords;
                        }
                        wi = w;
                        cur = u[wi];
                        continue;
                    }
#endif
                    if (wi >= *hi) return rlo;
                    cur = u[++wi];
                    continue;
                }
                const uint64_t pmw = pm[wi];
                const uint64_t h = cur & pmw;
                const uint64_t f = cur & ~pmw;
                if (h == 0) {   // only free bits left in this word
                    if (rlo == kNone) rlo = (wi << 6) + lowest_set_bit64_fast(f);
                    cur = 0;
                    continue;
                }
                const uint32_t bit = lowest_set_bit64_fast(h);
                const uint64_t fb = f & ((1ull << bit) - 1ull);
                if (rlo == kNone && fb) rlo = (wi << 6) + lowest_set_bit64_fast(fb);
                const uint32_t bi = prow[(wi << 6) + bit] - 1;
#ifdef FASTBP6_OSD0_STATS
                if (record_) rec_basis_.push_back(bi);  // measurement-only (attribution mode): the basis rows read
#endif
                const uint64_t *b = basis + (size_t)bi * words_;
                const uint32_t hb = bhi[bi];
                FASTBP6_ST(++st_.xor_steps; st_.xor_words += hb - wi + 1);
                xor_span(u, b, wi, hb);
                if (hb > *hi) *hi = hb;
                if (pbits != nullptr) pbits[bi >> 6] ^= 1ull << (bi & 63u);
                cur = (cur ^ b[wi]) & (bit == 63 ? 0ull : (~0ull << (bit + 1)));
            }
        }
        for (;;) {
            FASTBP6_ST(++st_.scan_trips);
            if (cur == 0) {
                FASTBP6_ST(++st_.scan_words);
                if (wi >= *hi) return rlo;
                cur = u[++wi];
                continue;
            }
            const uint32_t bit = FASTBP6_CTZ_ON ? lowest_set_bit64_fast(cur)
                                                : lowest_set_bit64_bruijn(cur);
            FASTBP6_ST(++st_.scan_bits);
            const uint32_t r = (wi << 6) + bit;
            const uint32_t p1 = prow[r];
            if (p1 == 0) {
                FASTBP6_ST(++st_.scan_free);
                if (rlo == kNone) rlo = r;
                cur &= cur - 1;
                continue;
            }
            const uint32_t bi = p1 - 1;
#ifdef FASTBP6_OSD0_STATS
            if (record_) rec_basis_.push_back(bi);  // measurement-only (attribution mode): the basis rows read
#endif
            const uint64_t *b = basis + (size_t)bi * words_;
            const uint32_t hb = bhi[bi];
            FASTBP6_ST(++st_.xor_steps; st_.xor_words += hb - wi + 1);
            xor_span(u, b, wi, hb);
            if (hb > *hi) *hi = hb;
            if (pbits != nullptr) pbits[bi >> 6] ^= 1ull << (bi & 63u);
            if (FASTBP6_CUR_ON) {
                // cur agrees with the in-memory u[wi] on every bit at or above `bit` (the bits the
                // scan already cleared are below it), and the XOR flips u[wi] by b[wi]; so the
                // cursor's next value is cur ^ b[wi] masked above the pivot bit -- the same value
                // the reference gets by re-reading the freshly stored word.
                cur = (cur ^ b[wi]) & (bit == 63 ? 0ull : (~0ull << (bit + 1)));
            } else {
                cur = (bit == 63) ? 0 : (u[wi] & (~0ull << (bit + 1)));
            }
        }
    }

    // Instrumentation (never read by the library itself).
    uint64_t st_solved = 0, st_deferrals = 0;
    uint64_t st_resumed = 0, st_resume_fresh = 0;   // rung-two walks continued / restarted
    uint32_t st_last_walked = 0, st_last_pivots = 0, st_last_candidates = 0;

private:
    uint32_t m_ = 0, n_ = 0, words_ = 0, rank_h_ = 0;
    uint32_t rcap_ = 0, pwords_ = 0;
    std::vector<uint32_t> colptr_;       // n + 1
    std::vector<uint32_t> rows_;         // nnz: the rows of each column (CSC order)
    std::vector<uint64_t> dense_;        // n * kDenseWords (+ alignment padding): the dense table
    uint64_t *dense_base_ = nullptr;     // 64-byte aligned view of dense_ (block c at base + c*8)
    std::vector<uint32_t> lo_, hi_;      // per column: first and last word with a set bit
    std::vector<uint64_t> basis_;        // rcap * words: the reduced pivot vectors
    std::vector<uint32_t> basis_hi_;     // rcap: highest non-zero word of each pivot vector
    std::vector<uint64_t> parent_bits_;  // rcap * pwords: per-pivot reduction coefficients
    std::vector<uint64_t> acc_bits_;     // pwords: expansion of the syndrome in the pivot basis
    std::vector<uint32_t> prow_;         // m: pivot index + 1, 0 = not a pivot row
    std::vector<uint64_t> pivmask_;      // words + 8: bit r set iff prow_[r] != 0
    std::vector<uint64_t> tvec_, ubuf_;
    uint32_t twords_ = 1;
    std::vector<uint32_t> pivcol_;       // pivot id -> column
    std::vector<uint32_t> sup_;          // columns with x = 1, the walk order (observable masks)
    uint32_t nsup_ = 0;                  // size of sup_ after the last kSolved run
    std::vector<uint8_t> par_;           // pivot id -> coefficient of the syndrome expansion
    uint32_t dlo_ = 0, dhi_ = 0;
    // The two locals of a walk that ran out of its set, and whether one is live.
    uint32_t w_rank_ = 0, w_tvec_hi_ = 0;
    bool w_live_ = false;
    Osd0Order ord_;

public:
#ifdef FASTBP6_OSD0_STATS
    // The counters of the most recent run() / run_candidates() (replay harness).  Measurement
    // only: with the macro undefined nothing of this exists in the shipped engine.
    const Osd0Stats &stats() const { return st_; }
    // Measurement-only switch of the vector word operations (zero_span/copy_span/xor_span/
    // any_nonzero_word), so the replay harness can measure the two forms paired in one process.
    // The shipped engine has no such member and only ever runs the vector form.
    void set_scalar_walk(bool on) { scalar_walk_ = on; }
    // Measurement-only switch of the dense column table (the CSC materialisation form is the A/B
    // reference of the replay harness); the shipped engine always uses the dense table.
    void set_dense(bool on) { scalar_dense_ = on; }
    // Measurement-only switch of the restructured walk's parts: bit 0 the hardware bit scan, bit 1
    // the register-carried cursor, bit 2 the scalar parent-bitset clear, bit 3 the next-column
    // prefetch (replay harness, paired mode); the shipped engine runs all of them (0xf).
    void set_walk_fast(unsigned bits) { fast_bits_ = bits; }
    // Measurement-only per-column phase attribution: -1 = no timer, 0..6 = one phase, 7 = empty
    // phase (the TSC pair's own overhead).
    void set_phase(int k) { fastbp6_phase_ = k; }
    // Measurement-only recorder of a shot's walked columns and the basis rows its reduction reads
    // (the replay harness's attribution mode: its "touch" mode pre-reads exactly those lines).
    void set_record(bool on) { record_ = on; }
    const std::vector<uint32_t> &rec_columns() const { return rec_cols_; }
    const std::vector<uint32_t> &rec_basis() const { return rec_basis_; }
    // Measurement-only: read one word of each listed column's dense block / of each listed basis
    // row's live lines, bringing the lines into the cache without running any of the walk's code.
    void touch_columns(const std::vector<uint32_t> &cols) {
        for (uint32_t c : cols) {
            const volatile uint64_t *p = (const volatile uint64_t *)(dense_base_ + (size_t)c * kDenseWords);
            (void)p[0];
        }
    }
    void touch_basis(const std::vector<uint32_t> &bis) {
        for (uint32_t bi : bis) {
            const volatile uint64_t *p = (const volatile uint64_t *)(basis_.data() + (size_t)bi * words_);
            for (uint32_t w = 0; w < words_; w += 8) (void)p[w];
        }
    }

private:
    Osd0Stats st_;
    bool scalar_walk_ = false;
    bool scalar_dense_ = false;
    unsigned fast_bits_ = 0xfu;
    bool record_ = false;
    std::vector<uint32_t> rec_cols_, rec_basis_;
    int fastbp6_phase_ = -1;
#endif
};

}  // namespace of the OSD-0 engine

#endif  // FASTBPOSD_B_OSD0_ENGINE_HPP
