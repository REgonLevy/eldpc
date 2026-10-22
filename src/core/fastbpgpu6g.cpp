// fastbpgpu6g: C ABI (decoder_abi.h) over the flat exact min-sum BP engine
// (bp_engine.hpp) plus the vendored ldpc::osd::OsdDecoder (osd.hpp of the vendored ldpc sources).
//
// Behaviour is the per-shot behaviour of ldpc-ref with
// bp_decode_parallel(MINIMUM_SUM) replaced by the engine: an all-zero syndrome short-circuits to
// zeros + BP_CONVERGED + 0 iterations, everything else runs BP; converged -> hard decision +
// BP_CONVERGED; bposd -> the vendored OSD on the BP posteriors (POSTPROC_INVOKED, HIGHER_ORDER_WIN
// when a higher-order candidate wins, SYNDROME_UNMATCHED when the result does not reproduce the
// syndrome); bp not converged -> the hard decision + SYNDROME_UNMATCHED.  The soft output is the
// prior LLRs after an all-zero shot and the BP posteriors otherwise, copied lazily by
// dbench_get_soft_output.
//
// Config keys are parsed exactly as by ldpc-ref (same keys, whitespace and duplicate rules, strtod
// for the float); everything ldpc-ref refuses is refused here too, plus decoder=bplsd,
// bp_method=ps, schedule=serial and ms_scaling_factor=0.
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "decoder_abi.h"

// Generated at configure time by CMakeLists.txt.
#include "dbench_candidate_build_info.h"

// Stage marks of the measurement build: nothing in any shipped build.
#define DBENCH_STAGES_DEFINE
#include "dbench_stages.h"

#include "bp.hpp"
#include "osd.hpp"

#include "bp_engine.hpp"
#include "osd0_engine.hpp"
#include "fastbp_ext.h"   // fastbp_keep_warm

namespace {

enum class Kind { kBp = 0, kBpOsd = 1 };

struct Config {
    Kind kind = Kind::kBp;
    int max_iter = 1;
    double ms_scaling_factor = 0.625;
    ldpc::osd::OsdMethod osd_method = ldpc::osd::OSD_0;
    int osd_order = 0;
};

// ---- config string parsing (identical rules to ldpc-ref) -----------

bool is_ascii_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

std::string strip(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && is_ascii_space(s[a])) ++a;
    while (b > a && is_ascii_space(s[b - 1])) --b;
    return s.substr(a, b - a);
}

bool valid_key(const std::string &k) {
    if (k.empty()) return false;
    for (char c : k) {
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) return false;
    }
    return true;
}

bool parse_config(const char *config, std::map<std::string, std::string> &out, std::string &err) {
    std::string s(config ? config : "");
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t semi = s.find(';', pos);
        std::string seg = s.substr(pos, semi == std::string::npos ? std::string::npos : semi - pos);
        pos = (semi == std::string::npos) ? s.size() + 1 : semi + 1;
        if (strip(seg).empty()) continue;
        size_t eq = seg.find('=');
        if (eq == std::string::npos || seg.find('=', eq + 1) != std::string::npos) {
            err = "malformed config segment '" + strip(seg) + "'";
            return false;
        }
        std::string key = strip(seg.substr(0, eq));
        std::string val = strip(seg.substr(eq + 1));
        if (!valid_key(key)) {
            err = "invalid config key '" + key + "'";
            return false;
        }
        if (out.count(key)) {
            err = "duplicate config key '" + key + "'";
            return false;
        }
        out[key] = val;
    }
    return true;
}

bool parse_int_value(const std::string &v, long long lo, long long hi, long long &out) {
    if (v.empty()) return false;
    size_t i = 0;
    if (v[0] == '+' || v[0] == '-') i = 1;
    if (i >= v.size()) return false;
    unsigned long long acc = 0;
    for (; i < v.size(); ++i) {
        if (v[i] < '0' || v[i] > '9') return false;
        acc = acc * 10u + static_cast<unsigned long long>(v[i] - '0');
        if (acc > 9000000000000000000ull) return false;
    }
    if (v[0] == '-') return false;
    long long sacc = static_cast<long long>(acc);
    if (sacc < lo || sacc > hi) return false;
    out = sacc;
    return true;
}

bool parse_float_value(const std::string &v, double &out) {
    if (v.empty()) return false;
    const char *begin = v.c_str();
    char *end = nullptr;
    errno = 0;
    double d = std::strtod(begin, &end);
    if (end != begin + v.size()) return false;
    if (!std::isfinite(d)) return false;
    out = d;
    return true;
}

bool build_config(const std::map<std::string, std::string> &kv, Config &cfg, std::string &err) {
    static const char *const kKeys[] = {"decoder",   "bp_method",  "max_iter",   "ms_scaling_factor",
                                        "schedule",  "osd_method", "osd_order",  "lsd_method",
                                        "lsd_order"};
    for (const auto &it : kv) {
        bool known = false;
        for (const char *k : kKeys) {
            if (it.first == k) known = true;
        }
        if (!known) {
            err = "unknown config key '" + it.first + "'";
            return false;
        }
    }

    auto need = [&](const char *key, std::string &val) -> bool {
        auto it = kv.find(key);
        if (it == kv.end()) {
            err = std::string("missing required config key '") + key + "'";
            return false;
        }
        val = it->second;
        return true;
    };

    std::string dec;
    if (!need("decoder", dec)) return false;
    if (dec == "bposd") cfg.kind = Kind::kBpOsd;
    else if (dec == "bp") cfg.kind = Kind::kBp;
    else if (dec == "bplsd") {
        // ldpc-ref accepts this; the support set of this library does not.
        err = "decoder 'bplsd' is not supported by fastbpgpu6g (expected bposd or bp)";
        return false;
    } else {
        err = "invalid decoder '" + dec + "' (expected bposd or bp)";
        return false;
    }

    std::string bpm;
    if (!need("bp_method", bpm)) return false;
    if (bpm == "ms") {
        // minimum-sum: the only supported method
    } else if (bpm == "ps") {
        err = "bp_method 'ps' is not supported by fastbpgpu6g (expected ms)";
        return false;
    } else {
        err = "invalid bp_method '" + bpm + "' (expected ms)";
        return false;
    }

    std::string mi;
    if (!need("max_iter", mi)) return false;
    long long mi_val = 0;
    if (!parse_int_value(mi, 1, std::numeric_limits<int>::max(), mi_val)) {
        err = "invalid max_iter '" + mi + "' (expected an integer >= 1)";
        return false;
    }
    cfg.max_iter = static_cast<int>(mi_val);

    if (kv.count("ms_scaling_factor")) {
        double msf = 0.0;
        if (!parse_float_value(kv.at("ms_scaling_factor"), msf) || !(msf > 0.0)) {
            err = "invalid ms_scaling_factor '" + kv.at("ms_scaling_factor") +
                  "' (expected a float > 0)";
            return false;
        }
        cfg.ms_scaling_factor = msf;
    }

    if (kv.count("schedule")) {
        const std::string &sch = kv.at("schedule");
        if (sch == "parallel") {
            // the default and the only supported schedule
        } else if (sch == "serial") {
            err = "schedule 'serial' is not supported by fastbpgpu6g (expected parallel)";
            return false;
        } else {
            err = "invalid schedule '" + sch + "' (expected parallel)";
            return false;
        }
    }

    if (cfg.kind == Kind::kBpOsd) {
        if (kv.count("lsd_method") || kv.count("lsd_order")) {
            err = "config key 'lsd_method'/'lsd_order' does not apply to decoder=bposd";
            return false;
        }
        std::string om;
        if (!need("osd_method", om)) return false;
        if (om == "osd0") cfg.osd_method = ldpc::osd::OSD_0;
        else if (om == "osd_e") cfg.osd_method = ldpc::osd::EXHAUSTIVE;
        else if (om == "osd_cs") cfg.osd_method = ldpc::osd::COMBINATION_SWEEP;
        else {
            err = "invalid osd_method '" + om + "' (expected osd0, osd_e or osd_cs)";
            return false;
        }
        if (kv.count("osd_order")) {
            long long oo = 0;
            if (!parse_int_value(kv.at("osd_order"), 0, std::numeric_limits<int>::max(), oo)) {
                err = "invalid osd_order '" + kv.at("osd_order") + "' (expected an integer >= 0)";
                return false;
            }
            cfg.osd_order = static_cast<int>(oo);
        }
        if (cfg.osd_method == ldpc::osd::OSD_0 && cfg.osd_order != 0) {
            err = "osd_order must be 0 for osd_method=osd0";
            return false;
        }
        if (cfg.osd_method == ldpc::osd::EXHAUSTIVE && cfg.osd_order > 15) {
            err = "osd_order must be <= 15 for osd_method=osd_e";
            return false;
        }
    } else {
        if (kv.count("osd_method") || kv.count("osd_order") || kv.count("lsd_method") ||
            kv.count("lsd_order")) {
            err = "config key 'osd_*'/'lsd_*' does not apply to decoder=bp";
            return false;
        }
    }
    return true;
}

// ---- problem validation (same rules as ldpc-ref) ---------------------------

bool validate_csc(const uint32_t *colptr, const uint32_t *rowind, uint32_t n, uint32_t rows,
                  const char *what, std::string &err) {
    if (colptr == nullptr) {
        err = std::string(what) + " colptr is NULL";
        return false;
    }
    if (colptr[0] != 0) {
        err = std::string(what) + " colptr[0] != 0";
        return false;
    }
    for (uint32_t j = 0; j < n; ++j) {
        if (colptr[j + 1] < colptr[j]) {
            err = std::string(what) + " colptr is not non-decreasing";
            return false;
        }
        if (colptr[j + 1] == colptr[j]) continue;
        if (rowind == nullptr) {
            err = std::string(what) + " rowind is NULL";
            return false;
        }
        uint32_t prev = rowind[colptr[j]];
        if (prev >= rows) {
            err = std::string(what) + " row index out of range";
            return false;
        }
        for (uint32_t k = colptr[j] + 1; k < colptr[j + 1]; ++k) {
            if (rowind[k] >= rows) {
                err = std::string(what) + " row index out of range";
                return false;
            }
            if (rowind[k] <= prev) {
                err = std::string(what) + " row indices are not strictly increasing";
                return false;
            }
            prev = rowind[k];
        }
    }
    return true;
}

void set_err(char *err, size_t err_len, const std::string &msg) {
    if (err != nullptr && err_len > 0) {
        std::snprintf(err, err_len, "%s", msg.c_str());
    }
}

std::string fail(const std::string &msg) { return "fastbpgpu6g: " + msg; }

}  // namespace

// The opaque handle of decoder_abi.h.
struct dbench_decoder {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    Config cfg;

    // Priors in the probability domain: the single source of truth the OSD weighs candidates
    // with (ldpc-ref's BpOsdDecoder holds the same reference).
    std::vector<double> channel_probs;
    // Prior LLRs log((1-p)/p), computed once at create and in set_priors (never per shot).  They
    // are the soft output of an all-zero-syndrome shot.
    std::vector<double> prior_llr;
    // True while the most recent dbench_decode() was the all-zero short circuit.
    bool last_shot_zero = true;
    // False after a decode that returned an error: the outputs of that call (and of the shot before
    // it) must not be handed out as if they were this shot's.
    bool last_decode_ok = false;
    // Instrumentation of the device prefix: the shots whose OSD-0 fell back to
    // the full posterior (no usable prefix) and the shots whose walk ran past the prefix.  Never
    // read by the library; a probe or a debugger can watch it.
    uint64_t fallbacks = 0;
    // Staging for dbench_set_priors: the new probabilities and prior LLRs are computed here and
    // committed to the handle only after the device upload succeeded (created at create).
    std::vector<double> prob_scratch, llr_scratch;

    fastbp::Engine engine;

    // Row-major edge list of H (ascending columns), for the syndrome re-check after OSD.
    std::vector<uint32_t> h_rowptr, h_rowcol;
    // O in CSC -- its native layout in the ABI problem, copied at create -- for the observables,
    // plus, for k <= 64, one 64-bit mask per column (bit r of o_colmask[j] is O[r][j]) so that a
    // set correction bit costs one XOR.  o_acc is the word accumulator of the k > 64 path.
    std::vector<uint32_t> o_csc_ptr, o_csc_row;
    std::vector<uint64_t> o_colmask, o_acc;

    ldpc::bp::BpSparse *pcm = nullptr;
    ldpc::osd::OsdDecoder *osd = nullptr;

    // The library's own exact OSD-0 (used for osd_method=osd0; the vendored decoder stays for
    // osd_cs/osd_e and for the shots the own engine defers as outside the image of H).
    fastbposd::Osd0Engine osd0;
    // Correction buffer for the own path when the caller passes no out_correction.
    std::vector<uint8_t> xscratch;

    // The masked syndrome of the last decode (OSD input, syndrome re-check).
    std::vector<uint8_t> syndrome;
    // Plain-vector copy of the BP posteriors for the vendored OsdDecoder, which takes a
    // std::vector<double>& (the engine's posterior lives in an aligned vector).  Filled only on the
    // shots that reach the vendored post-processing, from a buffer created here.
        std::vector<double> posterior_plain;

    ~dbench_decoder() {
        delete osd;  // holds a reference to *pcm
        delete pcm;
    }
};

namespace {

uint8_t row_parity(const uint32_t *ptr, const uint32_t *cols, uint32_t row, const uint8_t *bits) {
    uint8_t p = 0;
    for (uint32_t k = ptr[row]; k < ptr[row + 1]; ++k) p ^= static_cast<uint8_t>(bits[cols[k]] & 1u);
    return p;
}

// The observables are O x, the XOR of O's columns over the set bits of the correction:
// XOR is order-free, so this is exactly the row-by-row scan it replaces.  For k <= 64 every column
// of O is pre-reduced to one 64-bit mask at create and the pass is a *branchless masked scan of all
// n columns* (one AND+XOR per column, vectorized) -- its cost is n-proportional, not support-
// proportional; a correction with a one-bit support costs the same n-byte walk as a full one.
// k > 64 instead walks O's CSC row lists for the correction's set columns, so that path is
// support-proportional (O entries in the support of x).  k == 0 (or a NULL O, which leaves k == 0)
// writes nothing.  No allocation.
void compute_obs(dbench_decoder *h, const uint8_t *corr, uint8_t *out_obs) {
    const uint32_t k = h->k, n = h->n;
    if (k == 0 || out_obs == nullptr) return;
    if (!h->o_colmask.empty()) {
        // Walk the correction in 32-byte blocks and pay the masked XORs only for a block that
        // holds a set byte.  A column's mask is still the low bit of its correction byte
        // (0 - (corr[j] & 1)), so a byte that is not 0/1 is read exactly as before; the block test
        // only decides whether to look closer, and the XOR it skips is one of a zero mask.  The
        // shipped base walked all n columns branchlessly (23.2-24.0 us per shot at n = 76,032 on
        // the development machine); the measured own-OSD-0 support is 10.8-14.1 columns, so the scan is what the
        // block test removes, while the dense case keeps the vector step.
        const uint64_t *cm = h->o_colmask.data();
        uint64_t acc = 0;
        uint32_t j = 0;
#if defined(__AVX2__)
        __m256i vacc = _mm256_setzero_si256();
        for (; j + 32 <= n; j += 32) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(corr + j));
            if (_mm256_testz_si256(v, v)) continue;  // no byte set: none of the 32 masks is taken
            // Four of the block's 32 columns at a time: their 64-bit masks (the low bit of the
            // correction byte, as before) and the four mask words of those columns.
            for (uint32_t b = 0; b < 32; b += 4) {
                const uint64_t m0 = 0ull - (uint64_t)(corr[j + b + 0] & 1u);
                const uint64_t m1 = 0ull - (uint64_t)(corr[j + b + 1] & 1u);
                const uint64_t m2 = 0ull - (uint64_t)(corr[j + b + 2] & 1u);
                const uint64_t m3 = 0ull - (uint64_t)(corr[j + b + 3] & 1u);
                const __m256i mm = _mm256_set_epi64x((long long)m3, (long long)m2, (long long)m1,
                                                     (long long)m0);
                vacc = _mm256_xor_si256(
                    vacc, _mm256_and_si256(
                              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(cm + j + b)), mm));
            }
        }
        {
            const __m128i x = _mm_xor_si128(_mm256_castsi256_si128(vacc),
                                            _mm256_extracti128_si256(vacc, 1));
            acc = (uint64_t)_mm_cvtsi128_si64(x) ^ (uint64_t)_mm_extract_epi64(x, 1);
        }
#else
        for (; j + 8 <= n; j += 8) {
            uint64_t any = 0;
            for (uint32_t b = 0; b < 8; ++b) any |= corr[j + b];
            if (any == 0) continue;
            for (uint32_t b = 0; b < 8; ++b) {
                acc ^= cm[j + b] & (0ull - (uint64_t)(corr[j + b] & 1u));
            }
        }
#endif
        for (; j < n; ++j) acc ^= cm[j] & (0ull - (uint64_t)(corr[j] & 1u));
        for (uint32_t i = 0; i < k; ++i) out_obs[i] = static_cast<uint8_t>((acc >> i) & 1u);
        return;
    }
    for (uint64_t &w : h->o_acc) w = 0;
    const uint32_t *cp = h->o_csc_ptr.data(), *cr = h->o_csc_row.data();
    for (uint32_t j = 0; j < n; ++j) {
        if (!(corr[j] & 1u)) continue;
        for (uint32_t e = cp[j]; e < cp[j + 1]; ++e) {
            const uint32_t r = cr[e];
            h->o_acc[r >> 6] ^= 1ull << (r & 63u);
        }
    }
    for (uint32_t i = 0; i < k; ++i) {
        out_obs[i] = static_cast<uint8_t>((h->o_acc[i >> 6] >> (i & 63u)) & 1u);
    }
}

// The observables of a correction whose set columns are already known: the own OSD-0's support list
// (the columns the back-substitution loop set, in its walk order).  One XOR of O's column mask per
// set column -- nothing proportional to n at all.  The order of the list does not matter: XOR is
// commutative and associative and the contract makes the observables order-free.  Requires the
// per-column masks (k <= 64), which the caller checks.
void compute_obs_support(dbench_decoder *h, const uint32_t *sup, uint32_t nsup, uint8_t *out_obs) {
    const uint32_t k = h->k;
    const uint64_t *cm = h->o_colmask.data();
    uint64_t acc = 0;
    for (uint32_t t = 0; t < nsup; ++t) acc ^= cm[sup[t]];
    for (uint32_t i = 0; i < k; ++i) out_obs[i] = static_cast<uint8_t>((acc >> i) & 1u);
}

// One shot: the single implementation used by dbench_decode and dbench_decode_batch.
int decode_one(dbench_decoder *h, const uint8_t *syndrome, uint8_t *out_correction, uint8_t *out_obs,
               uint32_t *out_status) {
    DBENCH_STAGE_BEGIN();
    const uint32_t m = h->m, n = h->n, k = h->k;
    // A call rejected before any shot is decoded decodes nothing: the handle's soft output must not
    // keep pointing at an earlier shot.  Mark the outputs
    // invalid first, so dbench_get_soft_output after this error returns an error, not the previous
    // shot's posterior.
    if (m > 0 && syndrome == nullptr) {
        h->last_decode_ok = false;
        return DBENCH_EINVAL;
    }
    if (k > 0 && out_obs == nullptr) {
        h->last_decode_ok = false;
        return DBENCH_EINVAL;
    }

    // Scan the syndrome in place: the all-zero shot is detected without copying it anywhere.
    bool zero_syndrome = true;
    for (uint32_t i = 0; i < m; ++i) {
        if (syndrome[i] & 1u) {
            zero_syndrome = false;
            break;
        }
    }

    DBENCH_STAGE(DBENCH_STAGE_PREP);
    if (zero_syndrome) {
        // Zero correction, zero observables, BP_CONVERGED with 0 iterations, no BP, no LLR work.
        if (out_correction != nullptr && n > 0) std::memset(out_correction, 0, n);
        for (uint32_t i = 0; i < k; ++i) out_obs[i] = 0;
        if (out_status != nullptr) *out_status = DBENCH_ST_BP_CONVERGED;
        h->last_shot_zero = true;
        h->last_decode_ok = true;
        DBENCH_STAGE(DBENCH_STAGE_RET);
        return DBENCH_OK;
    }

    try {
        // An OSD-0 handle hands the caller's syndrome
        // bytes to the engine and to the walk directly -- both mask every byte with & 1 themselves
        // (Engine::run's parameter pack and h_syn staging; Osd0Engine's pack_syndrome), which is exactly
        // the copy below -- and fills the handle's masked vector only on the vendored path, which is
        // where it is read (`osd->decode`) and where the syndrome re-check reads it.  The copy leaves the
        // launch's critical path (the 0.5 us "syndrome copy into the handle" stage).
        const bool own_osd0 = (h->cfg.kind == Kind::kBpOsd && h->cfg.osd_method == ldpc::osd::OSD_0);
        h->syndrome.resize(m);
        if (!own_osd0) {
            for (uint32_t i = 0; i < m; ++i) h->syndrome[i] = static_cast<uint8_t>(syndrome[i] & 1u);
        }
        DBENCH_STAGE(DBENCH_STAGE_BP_BEGIN);

        // The posteriors are copied back only when this shot needs them (the OSD path) -- a
        // BP-converged shot of a decoder=bp handle makes no device-to-host copy of them at all.
        const uint32_t iterations = h->engine.run(own_osd0 ? syndrome : h->syndrome.data(),
                                                  (uint32_t)h->cfg.max_iter, h->cfg.ms_scaling_factor, false);
        DBENCH_STAGE(DBENCH_STAGE_BP_END);
        const uint8_t *corr = h->engine.decoding;

        uint32_t flags = 0;
        // True when the correction comes from the vendored post-processing, whose result may not
        // reproduce the syndrome and therefore has to be re-checked.  The own OSD-0 returns the
        // exact solution of H x = s by construction (its reduced syndrome reaches zero), so its
        // result is matched without re-deriving it.
        bool need_syndrome_check = false;
        // The own OSD-0 hands over the set columns of its correction (its back-substitution loop
        // emitted them), so the observables below need no scan of the correction at all.  Declared
        // here so the observables step after the branch chain can read them.
        const uint32_t *obs_sup = nullptr;
        uint32_t obs_nsup = 0;
        if (h->engine.last_converged) {
            flags = DBENCH_ST_BP_CONVERGED;
        } else if (h->cfg.kind == Kind::kBpOsd) {
            bool solved = false;
            if (h->cfg.osd_method == ldpc::osd::OSD_0) {
                uint8_t *dst = out_correction != nullptr ? out_correction : h->xscratch.data();
                // Device-side prefix rungs: this shot's kernel computed the front of the
                // OSD-0 column order on the device and handed over the *small* prefix (rung one,
                // kOsdK / kOsdCapA).  The elimination runs here on that prefix.  A walk that runs
                // past it (or a rung one that had no set at all) gets rung two -- the larger
                // 1024-2048 prefix, re-picked by the device from the same sample and histogram --
                // and only a walk that runs past rung two comes back as the full posterior, which is
                // the contract's fallback.  Every rung is a prefix of the same reference order, so
                // all of them give the same answer bit for bit.  (When the device sorted
                // rung one by the exact OSD-0 order -- its gate passed -- the walk consumes it
                // straight through with no selection; rung two always arrives unsorted.)
                bool need_larger = false;   // rung one could not finish the walk
                // When rung one's walk ran out, rung two continues it from that state
                // (Osd0Engine::resume_candidates) over its records above rung one's largest key,
                // instead of walking rung two from the first column again.
                bool resume1 = false;
                uint64_t after_key = 0;
                if (h->engine.order_cand_n > 0) {
                    // hand-off: the device's candidate records are on the host (they came back with BP)
                    DBENCH_STAGE(DBENCH_STAGE_POSTERIOR);
                    bool exhausted = false;
                    // When the device sorted the prefix (its gate passed), the walk
                    // consumes it directly with no selection; otherwise the records come over in the
                    // collection's order and the unchanged chunk walk sorts them.
                    const fastbposd::Osd0Engine::Result r = h->osd0.run_candidates(
                        h->engine.order_records, h->engine.order_cand_n, syndrome, dst,
                        &exhausted, h->engine.order_sorted);
                    if (r == fastbposd::Osd0Engine::kSolved) {
                        corr = dst;
                        solved = true;
                    } else if (exhausted) {
                        need_larger = true;  // the small prefix ran out before the walk stopped
                        // rung one's set is {key <= tau1}: its largest key is tau1 (the last record
                        // when the device sorted it).  Read it before osd_repick() replaces the records.
                        const unsigned char *rec = static_cast<const unsigned char *>(h->engine.order_records);
                        for (uint32_t i = 0; i < h->engine.order_cand_n; ++i) {
                            uint64_t k;
                            std::memcpy(&k, rec + (size_t)i * 16u, 8);
                            if (k > after_key) after_key = k;
                        }
                        resume1 = true;
                    }
                    // else: the syndrome is outside the image of H -- the vendored solver below.
                } else {
                    // No set at rung one: a converged shot (the kernel returned early), a shot whose
                    // BP converged at max_iter, a tie bucket larger than rung one's cap, n too small
                    // for a prefix, or a bp handle.  Rung two is still worth a look for the
                    // non-converged OSD-0 shots (its own cap may cover the tie bucket).
                    need_larger = true;
                }
                if (!solved && need_larger) {
                    bool exhausted = false;
                    if (h->engine.osd_repick() && h->engine.order_cand_n > 0) {
                        DBENCH_STAGE(DBENCH_STAGE_POSTERIOR);
                        const fastbposd::Osd0Engine::Result r =
                            resume1 ? h->osd0.resume_candidates(h->engine.order_records,
                                                                h->engine.order_cand_n, after_key,
                                                                syndrome, dst, &exhausted)
                                    : h->osd0.run_candidates(h->engine.order_records,
                                                             h->engine.order_cand_n, syndrome,
                                                             dst, &exhausted);
                        if (r == fastbposd::Osd0Engine::kSolved) {
                            corr = dst;
                            solved = true;
                        }
                        // else: rung two ran out too (exhausted) or the syndrome is outside the
                        // image of H (the walk reached rank(H)); both go the same way as before.
                    } else if (h->engine.repick_failed) {
                        // A CUDA failure inside osd_repick is the decode's error, as
                        // fetch_posterior's is below -- the caller's outputs are not written and the handle stays
                        // usable (the next run() starts the shot's state over).
                        h->last_decode_ok = false;
                        return DBENCH_EINTERNAL;
                    } else {
                        exhausted = true;  // no rung-two set: the full posterior is next
                    }
                    if (!solved && exhausted) {
                        // The last candidate rung ran out: continue on the full order.
                        h->fallbacks += 1;
                        if (!h->engine.fetch_posterior()) {
                            h->last_decode_ok = false;
                            return DBENCH_EINTERNAL;
                        }
                        DBENCH_STAGE(DBENCH_STAGE_POSTERIOR);
                        if (h->osd0.run(h->engine.posterior, syndrome, dst) ==
                            fastbposd::Osd0Engine::kSolved) {
                            corr = dst;
                            solved = true;
                        }
                    }
                }
                if (solved) {
                    obs_sup = h->osd0.support();
                    obs_nsup = h->osd0.support_size();
                }
            }
            if (!solved) {
                // osd_cs/osd_e, and the osd0 shots whose syndrome is outside the image of H (the
                // reference's own x there depends on its internal pivot-row choice).  The vendored
                // decoder takes a std::vector<double>&; the copy is into a buffer created at
                // create time, so these shots still allocate only inside the vendored decoder.
                // On an OSD-0 handle the masked vector is built here, the only
                // place that reads it (the own path above read the caller's bytes directly).
                if (own_osd0) {
                    for (uint32_t i = 0; i < m; ++i) h->syndrome[i] = static_cast<uint8_t>(syndrome[i] & 1u);
                }
                if (!h->engine.fetch_posterior()) {
                    h->last_decode_ok = false;
                    return DBENCH_EINTERNAL;
                }
                std::memcpy(h->posterior_plain.data(), h->engine.posterior,
                            (size_t)n * sizeof(double));
                corr = h->osd->decode(h->syndrome, h->posterior_plain).data();
                need_syndrome_check = true;
            }
            DBENCH_STAGE(DBENCH_STAGE_OSD_END);
            flags = DBENCH_ST_POSTPROC_INVOKED;
            if (h->cfg.osd_order > 0 && h->osd->osdw_decoding != h->osd->osd0_decoding) {
                flags |= DBENCH_ST_HIGHER_ORDER_WIN;
            }
        } else {
            // Plain BP that did not converge: its hard decision cannot reproduce the syndrome.
            flags = DBENCH_ST_SYNDROME_UNMATCHED;
        }

        // The H*correction re-check is needed only for the vendored post-processing, which can
        // return a syndrome-inconsistent vector (that is what ST_SYNDROME_UNMATCHED reports).
        if (flags & DBENCH_ST_POSTPROC_INVOKED && need_syndrome_check) {
            bool matched = true;
            for (uint32_t i = 0; i < m && matched; ++i) {
                if (row_parity(h->h_rowptr.data(), h->h_rowcol.data(), i, corr) != h->syndrome[i]) {
                    matched = false;
                }
            }
            if (!matched) flags |= DBENCH_ST_SYNDROME_UNMATCHED;
        }

        if (out_correction != nullptr && n > 0 && corr != out_correction) {
            std::memcpy(out_correction, corr, n);
        }

        if (obs_sup != nullptr && !h->o_colmask.empty()) {
            compute_obs_support(h, obs_sup, obs_nsup, out_obs);
        } else {
            compute_obs(h, corr, out_obs);
        }
        DBENCH_STAGE(DBENCH_STAGE_OBS);
        if (out_status != nullptr) {
            uint32_t it = iterations;
            if (it > DBENCH_ST_ITER_MAX) it = DBENCH_ST_ITER_MAX;
            *out_status = (flags & DBENCH_ST_FLAGS_MASK) | (it << DBENCH_ST_ITER_SHIFT);
        }
        h->last_shot_zero = false;
    } catch (const std::exception &) {
        h->last_decode_ok = false;
        return DBENCH_EINTERNAL;
    } catch (...) {
        h->last_decode_ok = false;
        return DBENCH_EINTERNAL;
    }
    h->last_decode_ok = true;
    DBENCH_STAGE(DBENCH_STAGE_RET);
    return DBENCH_OK;
}

}  // namespace

extern "C" {

DBENCH_EXPORT uint32_t dbench_abi_version(void) { return DBENCH_ABI_VERSION; }
DBENCH_EXPORT const char *dbench_decoder_name(void) { return DBENCH_CANDIDATE_NAME; }
DBENCH_EXPORT uint32_t dbench_capabilities(void) {
    return DBENCH_CAP_CORRECTION | DBENCH_CAP_SET_PRIORS | DBENCH_CAP_SOFT_OUTPUT;
}

DBENCH_EXPORT const char *dbench_decoder_build_info(void) { return DBENCH_CANDIDATE_BUILD_INFO; }

DBENCH_EXPORT dbench_decoder *dbench_create(const char *config, const dbench_problem *problem,
                                            char *err, size_t err_len) {
    if (problem == nullptr) {
        set_err(err, err_len, fail("problem is NULL").c_str());
        return nullptr;
    }
    if (problem->abi_version != DBENCH_ABI_VERSION) {
        set_err(err, err_len, fail("problem ABI version mismatch").c_str());
        return nullptr;
    }

    std::map<std::string, std::string> kv;
    std::string parse_err;
    if (!parse_config(config, kv, parse_err)) {
        set_err(err, err_len, fail(parse_err).c_str());
        return nullptr;
    }
    Config cfg;
    if (!build_config(kv, cfg, parse_err)) {
        set_err(err, err_len, fail(parse_err).c_str());
        return nullptr;
    }

    const uint32_t m = problem->num_detectors;
    const uint32_t n = problem->num_mechanisms;
    const uint32_t k = problem->num_observables;
    if (n == 0) {
        set_err(err, err_len, fail("problem has no mechanisms").c_str());
        return nullptr;
    }
    if (problem->priors == nullptr) {
        set_err(err, err_len, fail("priors is NULL").c_str());
        return nullptr;
    }
    if (!validate_csc(problem->h_colptr, problem->h_rowind, n, m, "H", parse_err)) {
        set_err(err, err_len, fail(parse_err).c_str());
        return nullptr;
    }
    if (k > 0 || problem->o_colptr != nullptr) {
        if (!validate_csc(problem->o_colptr, problem->o_rowind, n, k, "O", parse_err)) {
            set_err(err, err_len, fail(parse_err).c_str());
            return nullptr;
        }
    }
    for (uint32_t j = 0; j < n; ++j) {
        const double p = problem->priors[j];
        if (!std::isfinite(p) || !(p > 0.0) || !(p < 1.0)) {
            set_err(err, err_len, fail("priors out of range (0, 1)").c_str());
            return nullptr;
        }
    }

    dbench_decoder *h = nullptr;
    try {
        h = new dbench_decoder();
        h->m = m;
        h->n = n;
        h->k = k;
        h->cfg = cfg;
        h->channel_probs.assign(problem->priors, problem->priors + n);

        const uint32_t nnz = problem->h_colptr[n];
        h->h_rowptr.assign((size_t)m + 1, 0);
        for (uint32_t e = 0; e < nnz; ++e) h->h_rowptr[problem->h_rowind[e] + 1]++;
        for (uint32_t i = 0; i < m; ++i) h->h_rowptr[i + 1] += h->h_rowptr[i];
        h->h_rowcol.resize(nnz);
        {
            std::vector<uint32_t> fill(h->h_rowptr.begin(), h->h_rowptr.end() - 1);
            for (uint32_t j = 0; j < n; ++j) {
                for (uint32_t e = problem->h_colptr[j]; e < problem->h_colptr[j + 1]; ++e) {
                    h->h_rowcol[fill[problem->h_rowind[e]]++] = j;
                }
            }
        }
        // O stays in CSC (the layout the ABI problem already uses); for k <= 64 each column is
        // pre-reduced to one 64-bit mask here, otherwise the CSC is walked per shot.  k == 0 or a
        // NULL O leaves both empty and compute_obs writes nothing.
        if (k > 0 && problem->o_colptr != nullptr) {
            const uint32_t onz = problem->o_colptr[n];
            h->o_csc_ptr.assign(problem->o_colptr, problem->o_colptr + n + 1);
            h->o_csc_row.assign(problem->o_rowind, problem->o_rowind + onz);
            if (k <= 64) {
                h->o_colmask.assign(n, 0ull);
                for (uint32_t j = 0; j < n; ++j) {
                    uint64_t m = 0;
                    for (uint32_t e = h->o_csc_ptr[j]; e < h->o_csc_ptr[j + 1]; ++e) {
                        m ^= 1ull << h->o_csc_row[e];
                    }
                    h->o_colmask[j] = m;
                }
                h->o_csc_ptr.clear();
                h->o_csc_row.clear();
            } else {
                h->o_acc.assign((size_t)(k + 63) / 64, 0ull);
            }
        }

        const bool own_osd0 = (h->cfg.kind == Kind::kBpOsd && h->cfg.osd_method == ldpc::osd::OSD_0);
        if (!h->engine.setup(m, n, problem->h_colptr, problem->h_rowind, own_osd0)) {
            set_err(err, err_len, fail("create failed: engine setup").c_str());
            delete h;
            return nullptr;
        }

        if (cfg.kind == Kind::kBpOsd) {
            // Same construction as ldpc-ref: entries inserted row-major, ascending columns.
            h->pcm = new ldpc::bp::BpSparse((int)m, (int)n, (int)nnz);
            for (uint32_t i = 0; i < m; ++i) {
                for (uint32_t e = h->h_rowptr[i]; e < h->h_rowptr[i + 1]; ++e) {
                    h->pcm->insert_entry((int)i, (int)h->h_rowcol[e]);
                }
            }
            h->osd = new ldpc::osd::OsdDecoder(*h->pcm, cfg.osd_method, cfg.osd_order,
                                               h->channel_probs);
            if (cfg.osd_method == ldpc::osd::OSD_0) {
                // The library's own OSD-0: column bitsets of H, rank(H) and every per-shot buffer.
                if (!h->osd0.setup(m, n, problem->h_colptr, problem->h_rowind)) {
                    set_err(err, err_len, fail("create failed: OSD-0 setup").c_str());
                    delete h;
                    return nullptr;
                }
                h->xscratch.assign(n, 0);
            }
        }

        // Prior LLRs, cached once: the soft output of an all-zero-syndrome shot.
        h->prior_llr.resize(n);
        for (uint32_t j = 0; j < n; ++j) {
            const double p = h->channel_probs[j];
            h->prior_llr[j] = std::log((1.0 - p) / p);
        }
        if (!h->engine.set_priors(h->prior_llr.data())) {
            set_err(err, err_len, fail("create failed: prior LLR upload").c_str());
            delete h;
            return nullptr;
        }
        h->syndrome.assign(m, 0);
        h->posterior_plain.assign(n, 0.0);
        h->prob_scratch.assign(n, 0.0);
        h->llr_scratch.assign(n, 0.0);
    } catch (const std::exception &e) {
        set_err(err, err_len, fail(std::string("create failed: ") + e.what()).c_str());
        delete h;
        return nullptr;
    } catch (...) {
        set_err(err, err_len, fail("create failed").c_str());
        delete h;
        return nullptr;
    }
    return h;
}

DBENCH_EXPORT int dbench_set_priors(dbench_decoder *h, const double *llr, uint32_t n) {
    if (h == nullptr || llr == nullptr || n != h->n) return DBENCH_EINVAL;
    // Validate every element before writing anything.
    for (uint32_t j = 0; j < n; ++j) {
        if (!std::isfinite(llr[j])) return DBENCH_EINVAL;
        const double p = 1.0 / (1.0 + std::exp(llr[j]));
        if (!(p > 0.0) || !(p < 1.0) || !std::isfinite(p)) return DBENCH_EINVAL;
    }
    try {
        // Computed into staging first: the host probabilities and prior LLRs change only once the
        // device side has succeeded.  A failure inside Engine::set_priors after the upload was
        // enqueued can leave the device partly on the new priors, so the error path below re-runs it
        // with the committed priors: a failed set_priors leaves the handle on its
        // previous priors.
        for (uint32_t j = 0; j < n; ++j) {
            const double p = 1.0 / (1.0 + std::exp(llr[j]));
            h->prob_scratch[j] = p;
            h->llr_scratch[j] = std::log((1.0 - p) / p);
        }
        if (!h->engine.set_priors(h->llr_scratch.data())) {
            // Restore the device side to the committed priors (upload, position copy,
            // first-pass arrays, OSD window).  If the GPU is gone this fails too, and later decodes
            // return errors as well.
            (void)h->engine.set_priors(h->prior_llr.data());
            return DBENCH_EINTERNAL;
        }
        h->channel_probs.swap(h->prob_scratch);
        h->prior_llr.swap(h->llr_scratch);
        // The OSD references channel_probs, so it already sees the new values.
    } catch (...) {
        return DBENCH_EINTERNAL;
    }
    return DBENCH_OK;
}

DBENCH_EXPORT int dbench_decode(dbench_decoder *h, const uint8_t *syndrome, uint8_t *out_correction,
                                uint8_t *out_obs, uint32_t *out_status) {
    if (h == nullptr) return DBENCH_EINVAL;
    return decode_one(h, syndrome, out_correction, out_obs, out_status);
}

DBENCH_EXPORT int dbench_decode_batch(dbench_decoder *h, uint32_t num_shots, const uint8_t *syndromes,
                                      uint8_t *out_corrections, uint8_t *out_obs,
                                      uint32_t *out_status) {
    if (h == nullptr) return DBENCH_EINVAL;
    // Same rule as decode_one's argument rejection: a failed call decodes nothing, so the handle's
    // soft output must not keep pointing at an earlier shot.
    if (num_shots > 0 && syndromes == nullptr) {
        h->last_decode_ok = false;
        return DBENCH_EINVAL;
    }
    if (h->k > 0 && num_shots > 0 && out_obs == nullptr) {
        h->last_decode_ok = false;
        return DBENCH_EINVAL;
    }
    // A batch of 0 shots decodes nothing, so it must not leave an earlier shot's soft
    // output available either, which is the rule applied to the other argument
    // rejections above.  The
    // return code stays the reference's (DBENCH_OK: ldpc-ref accepts a 0-shot batch), and no decode's
    // outputs change, so the library stays bit-identical to ldpc-ref shot by shot.
    if (num_shots == 0) {
        h->last_decode_ok = false;
        return DBENCH_OK;
    }
    for (uint32_t s = 0; s < num_shots; ++s) {
        const uint8_t *syn = syndromes + (size_t)s * h->m;
        uint8_t *corr = out_corrections != nullptr ? out_corrections + (size_t)s * h->n : nullptr;
        uint8_t *obs = out_obs != nullptr ? out_obs + (size_t)s * h->k : nullptr;
        uint32_t *st = out_status != nullptr ? out_status + s : nullptr;
        int rc = decode_one(h, syn, corr, obs, st);
        if (rc != DBENCH_OK) return rc;
    }
    return DBENCH_OK;
}

DBENCH_EXPORT int dbench_get_soft_output(dbench_decoder *h, double *out_llr, uint32_t n) {
    if (h == nullptr || out_llr == nullptr || n != h->n) return DBENCH_EINVAL;
    if (!h->last_decode_ok) {
        // No decode, or the last one failed: there is no posterior of a current shot to hand out.
        return DBENCH_EINTERNAL;
    }
    if (h->last_shot_zero) {
        if (h->prior_llr.size() != n) return DBENCH_EINTERNAL;
        std::memcpy(out_llr, h->prior_llr.data(), (size_t)n * sizeof(double));
        return DBENCH_OK;
    }
    if (h->engine.posterior == nullptr) return DBENCH_EINTERNAL;
    // The posterior is copied back lazily: device -> host now, if this shot's decode did not need it.
    if (!h->engine.fetch_posterior()) return DBENCH_EINTERNAL;
    std::memcpy(out_llr, h->engine.posterior, (size_t)n * sizeof(double));
    return DBENCH_OK;
}

DBENCH_EXPORT void dbench_destroy(dbench_decoder *h) { delete h; }

// See fastbp_ext.h.  The GPU flag launches the engine's empty kernel, which changes no state a decode
// reads.  (A host flag -- a sweep of the handle's host tables -- was evaluated and removed: placed
// right it gained nothing, and alone it evicted more than it warmed.)
DBENCH_EXPORT int fastbp_keep_warm(dbench_decoder *h, uint32_t flags) {
    if (h == nullptr || (flags & ~FASTBP_KEEP_WARM_GPU) != 0u) return DBENCH_EINVAL;
    if ((flags & FASTBP_KEEP_WARM_GPU) != 0u) {
        if (!h->engine.keep_warm_gpu()) return DBENCH_EINTERNAL;
    }
    return DBENCH_OK;
}

#if defined(FASTBP_TEST_LAYOUT_PROBE) && FASTBP_TEST_LAYOUT_PROBE
// Test-only build (never shipped: the macro is set only by the layout-probe builds): the
// handle's create-time layout record (fastbp::LayoutStats) as u32 words into the caller's buffer.  Returns the
// record's word count.  No I/O.
DBENCH_EXPORT uint32_t fastbp_test_layout(const dbench_decoder *h, uint32_t *out, uint32_t n) {
    if (h == nullptr || out == nullptr) return 0u;
    fastbp::LayoutStats st{};
    h->engine.layout(&st);
    uint32_t w[sizeof(fastbp::LayoutStats) / 4u];
    std::memcpy(w, &st, sizeof(st));
    const uint32_t nw = (uint32_t)(sizeof(st) / 4u);
    for (uint32_t i = 0; i < n && i < nw; ++i) out[i] = w[i];
    return nw;
}
#endif

}  // extern "C"
