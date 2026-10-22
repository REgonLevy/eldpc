// fastbpgpu6g: exact min-sum BP engine -- the GPU half of the library.
//
// The engine keeps the API the ABI glue calls (setup / set_priors / run plus decoding, posterior and
// last_converged), so fastbpgpu6g.cpp, osd0_engine.hpp and the per-shot behaviour on the CPU side are
// unchanged from the CPU-only implementation.  What runs on the GPU is bp_decode_parallel() MINIMUM_SUM with the same
// double-precision addition orders and the same rounding as the caller's fp environment; the device
// side lives in bp_gpu.cu and this header only fixes the layout shared by the two translation units.
//
// Exactness:
//   * The check update is the row's minimum, second minimum and sign parity -- all three order-free --
//     so they may be reduced in parallel lanes; every outgoing message is
//         (|b2c_e| == m1 ? m2 : m1) * (+-alpha)
//     with the final sign as the operand of the single multiply, exactly bp.hpp:251-270.
//   * The variable update is one sequential walk per column in increasing row order, the reference's
//     exact addition order (b2c := t; t += c2b, then t = +0.0 back down), one thread per column.
//   * Every addition uses __dadd_rn/rd/ru/rz and the multiply __dmul_rn/rd/ru/rz picked from
//     fegetround() read in dbench_create: the GPU has no rounding-mode state, so the caller's mode is
//     compiled into the kernel (*four instantiations of the BP kernel exist; the handle launches one*).
//
// Layout of the handle's device memory: the CSC edge list of H (one double per edge), the CSR of H
// (row -> edge id), the priors, the messages b2c/c2b, the posterior, the hard decision, the syndrome
// and the mismatch counters (no separate candidate-syndrome array: the row pass derives it from the
// decisions).  Everything is allocated, pinned and uploaded in dbench_create; dbench_decode only
// signals, launches and reads back.
#ifndef FASTBPGPU_B_BP_ENGINE_HPP
#define FASTBPGPU_B_BP_ENGINE_HPP

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace fastbp {

// ---------------------------------------------------------------------------
// Aligned allocation of the hot host arrays.  Every array of this engine is at
// least 64 B aligned; large ones are additionally hinted as transparent huge pages, so their placement
// does not depend on where the caller's create order left the allocator.  All of it happens in setup()
// (dbench_create); the decode path allocates nothing.
// ---------------------------------------------------------------------------
template <class T>
struct AlignedAlloc {
    using value_type = T;
    static constexpr size_t kAlign = 64;
#if defined(__linux__)
    static constexpr size_t kHugeHintBytes = 512 * 1024;
#else
    static constexpr size_t kHugeHintBytes = static_cast<size_t>(-1);
#endif
    AlignedAlloc() = default;
    template <class U> AlignedAlloc(const AlignedAlloc<U> &) {}
    T *allocate(size_t n) {
        if (n == 0) n = 1;
        const size_t bytes = n * sizeof(T);
        void *p = nullptr;
        if (posix_memalign(&p, kAlign, bytes) != 0) throw std::bad_alloc();
#if defined(__linux__)
        if (bytes >= kHugeHintBytes) {
            // Best effort: an unaligned or unsupported range simply keeps its 4 KiB pages.
            madvise(p, bytes, MADV_HUGEPAGE);
        }
#endif
        return static_cast<T *>(p);
    }
    void deallocate(T *p, size_t) { std::free(p); }
    template <class U> bool operator==(const AlignedAlloc<U> &) const { return true; }
    template <class U> bool operator!=(const AlignedAlloc<U> &) const { return false; }
};

template <class T>
using AVec = std::vector<T, AlignedAlloc<T>>;

// GPU resources of one handle (bp_gpu.cu).  Opaque here.
struct GpuState;

// The create-time layout a handle took, so tools can see which kernel and
// shared layout a row ran (g->balanced / rslot / sum_mode were write-only).  Filled by setup(), read by
// Engine::layout(); no I/O anywhere.  Every field is a u32 so a test build can hand the record out as words.
struct LayoutStats {
    uint32_t words;          // sizeof(LayoutStats) / 4
    uint32_t sms, nblocks, nthreads;
    uint32_t sum_mode;       // 0 in-place, 1 fused single-barrier summary, 2 two-barrier summary
    uint32_t ilv, balanced, rslot, rec;   // rec: the 32 B record kernel (ILV + SHSUM + RSLOT, RN / RZ)
    uint32_t ns_max;         // the real largest block row count (the shared tables' size)
    uint32_t ns_uni, ns_bal, gate;          // the gate on the true footprints and its decision
    uint32_t ns_uni_6e, ns_bal_6e, gate_6e; // the same as the earlier gate computed it (test probe builds only; else 0)
    uint32_t bal_cost_ratio_ppm;            // bmax / umax (the predicted slowest-block ratio) x 1e6
    uint32_t rslot_bytes, meta_max, fused_shbytes, max_optin;
    uint32_t win_rounds, win_bytes, max_block_cols;   // the b2c window (0 when there is none)
    // The window guard (layout-probe builds only, else 0): the window form's own dynamic request
    // (tables_rec + rslot + meta + window, recorded whether or not it fits), and for the kernel this handle
    // launches (its rounding mode, formulation, layout) the static shared bytes and the dynamic bytes the
    // driver still has for a block at this grid's blocks per SM.  The window's margin = fused_avail_dyn -
    // fused_shbytes; a step that grows static or the tables past it turns the window off.
    uint32_t win_request, fused_static, fused_avail_dyn;
    // The launched kernel's max dynamic shared attribute as the gate set it (max_optin - sharedSizeBytes)
    uint32_t fused_max_dyn_attr;
    uint32_t reserved[4];
};

// The engine as the ABI glue sees it: the interface of the CPU engine it replaces.
class Engine {
public:
    uint32_t m = 0, n = 0;

    // Outputs of the most recent run(): the hard decision (n) and the posterior LLRs (n).  The
    // posterior is copied back lazily: run() fetches it only when the caller needs it
    // for the same shot (the OSD path), fetch_posterior() otherwise, and get_soft_output drives that.
    // Both point straight into the handle's pinned staging buffer
    // (the D2H destination), so the decode path copies neither of them on the host; they are set by
    // setup() and stay valid until the handle is destroyed.
    const uint8_t *decoding = nullptr;
    const double *posterior = nullptr;
    bool last_converged = false;
    bool posterior_valid = false;   // posterior holds the most recent run()'s values

    // The device-computed front of the OSD-0 column order of the most recent
    // run(), when the shot's kernel produced one.  order_records points at order_cand_n 16-byte (u64 key,
    // u32 column, u32 pad) records -- a *prefix* of the reference order, so the OSD-0 elimination
    // over them consumes exactly the columns the full order would (the contract's exact fallback
    // covers the walk running past the set).  order_cand_n is 0 when the shot converged, when the
    // kernel returned early, or when the first rung had no set; a handle that does not take the
    // OSD-0 path never fills either.
    //
    // The run() hand-off carries *rung one*, the small prefix
    // (kOsdPrefixKA / kOsdPrefixCapA).  osd_repick() replaces it with rung two -- the larger
    // (kOsdPrefixK / kOsdPrefixCap) prefix, re-picked on the device from the sample and histogram
    // run() left -- which the caller uses for the shots whose walk ran past rung one.
    //
    // The records stay in the pinned hand-off staging
    // block and the host walk reads them *there*; there is no host copy into a second buffer any more.
    // The pointer is valid from the return of the run()/osd_repick() that produced it until that
    // handle's next run() or osd_repick() (both overwrite the staging block on the handle's stream and
    // wait for it); null when order_cand_n is 0.
    const void *order_records = nullptr;
    uint32_t order_cand_n = 0;
    // The last osd_repick() hit a CUDA failure.  The decode returns an error then (the
    // visible failure-injection rule: a fired failure is an error return, never a silent recovery); false otherwise.
    bool repick_failed = false;
    // True when those records are already in the exact OSD-0 order (the device's P6 rank sort ran):
    // the host walk then consumes them with no selection (Osd0Order::init_candidates_sorted).
    bool order_sorted = false;

    Engine();
    ~Engine();
    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;

    // Build the handle's GPU resources and copy the problem in.  Throws std::runtime_error with a
    // human-readable message (the ABI glue turns it into the create error) when there is no usable GPU.
    // osd0: the handle's config is decoder=bposd with osd_method=osd0, i.e. this handle runs the
    // library's own OSD-0 on non-converged shots.  Only such a handle builds the device-side OSD-0
    // order prefix; a bp handle pays nothing for it.
    // wide_ok = false is setup's own retry without the wide-grid term (bp_gpu.cu).
    bool setup(uint32_t m_, uint32_t n_, const uint32_t *colptr, const uint32_t *rowind, bool osd0,
               bool wide_ok = true);
    // Upload the prior LLRs (n entries), on the handle's own stream and completed before returning.
    // Called from create and from dbench_set_priors only.  Returns false when any step failed; the
    // device may then hold part of the new priors, so the caller re-runs it with the previous priors
    // (dbench_set_priors does; a failed create deletes the handle).
    bool set_priors(const double *prior_llr_n);
    // One shot: syndrome (m bytes, values 0/1), max_iter >= 1, alpha > 0.  Returns the iteration count
    // (the iteration that converged, or max_iter) and fills decoding / last_converged (and posterior
    // when need_post).  Throws std::runtime_error on a CUDA failure: the caller turns that into an
    // error return and never into DBENCH_OK.
    uint32_t run(const uint8_t *syn01, uint32_t max_iter, double alpha, bool need_post);
    // Copy the last shot's posteriors into posterior[] (device -> host) if they are not there yet.
    // Returns false on a CUDA failure; true when posterior[] is valid.
    bool fetch_posterior();
    // Rung two: re-pick the OSD-0 prefix of the *most recent* run() at the larger
    // (kOsdPrefixK, kOsdPrefixCap) settings, from the sample and histogram that run left, and put it
    // in order_cand / order_cand_n.  Only valid for a decoder=bposd osd_method=osd0 handle and only
    // after a run() that took the OSD-0 path (a non-converged shot, the same condition the caller
    // checks before it walks); it launches one kernel on the handle's stream and waits for it.
    // Returns false on a CUDA failure (order_cand_n is 0 then).
    bool osd_repick();

    // The layout this handle's create took (see LayoutStats); zeros before setup().
    void layout(LayoutStats *out) const;

    // The GPU flag of fastbp_keep_warm: launch one empty 1-block kernel on the handle's stream and
    // return at once (no synchronisation, no allocation, no device state touched).  Returns false when the
    // launch failed (the error is consumed, as every other launch's is).
    bool keep_warm_gpu();

private:
    GpuState *gpu_ = nullptr;
    // The parity buffer (iteration & 1) the most recent run()'s final posterior and
    // decision were written to; fetch_posterior() copies from that buffer.  Set by run() from the
    // shot's own status word; 0 for a kernel without the double buffer.
    uint32_t last_post_parity = 0;
};

}  // namespace fastbp

#endif  // FASTBPGPU_B_BP_ENGINE_HPP
