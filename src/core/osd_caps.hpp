// fastbpgpu6g: the two ends of one coupling -- the candidate cap of the device-side OSD-0 order
// prefix (bp_gpu.cu's osd_prefix_pass collects at most kOsdCap (key, column)
// pairs) and the capacity of the host engine's candidate buffer (osd0_engine.hpp's Osd0Order::kCand,
// which init_candidates() truncates at).  Both are 2048 by construction; the assert below makes that
// a checked fact instead of a coincidence: a prefix larger than the host buffer would be truncated
// silently, and the host would then walk an order that is not the reference's.
#ifndef FASTBPGPU5_H_OSD_CAPS_HPP
#define FASTBPGPU5_H_OSD_CAPS_HPP

#include <cstdint>

namespace fastbp {

// Test-only: the four rung constants can be overridden at compile
// time so the failure-injection tests can build the forced-fallback variants
// ((KA, capA) = (24, 200) / (64, 400) and (24, 96) / (64, 200)): small rungs make most shots take the
// repick and the full-posterior paths.  The shipped build defines none of the macros below and gets
// exactly the values it had (384 / 640 / 1024 / 2048); the static asserts still apply to both.
#ifndef FASTBP_OSD_PREFIX_CAP
#define FASTBP_OSD_PREFIX_CAP 2048u
#endif
#ifndef FASTBP_OSD_PREFIX_K
#define FASTBP_OSD_PREFIX_K 1024u
#endif
#ifndef FASTBP_OSD_PREFIX_KA
#define FASTBP_OSD_PREFIX_KA 384u
#endif
#ifndef FASTBP_OSD_PREFIX_CAPA
#define FASTBP_OSD_PREFIX_CAPA 640u
#endif

inline constexpr uint32_t kOsdPrefixCap = FASTBP_OSD_PREFIX_CAP;   // candidates the device hands to the host (rung two)
inline constexpr uint32_t kOsdPrefixK = FASTBP_OSD_PREFIX_K;       // rung two's K

// The size of the OSD-0 candidate set.  The device passes publish the
// front of the OSD-0 order in two rungs.  Rung one, appended to the BP kernel, collects the *small*
// prefix -- the first kOsdPrefixKA entries of the order, or the longest prefix that fits
// kOsdPrefixCapA (measured on a development corpus: walk depth median 38, p99 180, max 588 of
// 2000 shots, so this covers most shots at a fraction of the collect, the copy and the host
// selection).  A shot whose walk runs past it is re-picked by a second, standalone kernel at the
// larger (kOsdPrefixK, kOsdPrefixCap) settings -- same sample, same histogram, only the threshold
// selection and the collect run again -- and only a shot that runs past *that* returns to the full
// posterior (the exact fallback of the contract).
//
// Both rungs hand over {key <= tau} for one sample value tau, i.e. a whole run of equal keys, so
// each set is a *prefix* of the reference order (Osd0Order::init_candidates sorts it lazily and the
// walk consumes it in the reference's order); a shorter prefix only means the walk reports the
// exhaustion sooner.  The host engine's candidate buffer (Osd0Order::kCand) must hold the larger
// rung, which is what the assert in osd0_engine.hpp checks.
inline constexpr uint32_t kOsdPrefixKA = FASTBP_OSD_PREFIX_KA;     // rung one's K
inline constexpr uint32_t kOsdPrefixCapA = FASTBP_OSD_PREFIX_CAPA; // rung one's cap

static_assert(kOsdPrefixCapA <= kOsdPrefixCap, "rung one must fit the host candidate buffer");
static_assert(kOsdPrefixKA <= kOsdPrefixCapA, "rung one's K must fit its own cap");
static_assert(kOsdPrefixK <= kOsdPrefixCap, "rung two's K must fit its own cap");
static_assert(kOsdPrefixCap >= 1u && kOsdPrefixCapA >= 1u, "a cap must leave room for one candidate");

}  // namespace fastbp

#endif  // FASTBPGPU5_H_OSD_CAPS_HPP
