// SPDX-License-Identifier: Apache-2.0
// ldpc-ref: C ABI (include/decoder_abi.h) over the vendored ldpc 2.4.1 C++ core.
//
// Behaviour:
//  - name "ldpc-ref"; capabilities CORRECTION | SET_PRIORS | SOFT_OUTPUT.
//  - config keys: decoder=bposd|bplsd|bp, bp_method=ms|ps, max_iter=<int>=1>,
//    ms_scaling_factor=<float>0, default 0.625>, schedule=parallel|serial (default parallel),
//    osd_method=osd0|osd_e|osd_cs + osd_order=<int> (bposd only),
//    lsd_method=lsd0|lsd_e|lsd_cs + lsd_order=<int> (bplsd only). Anything else -> create fails.
//  - the prior LLRs log((1-p)/p) are computed once at create and in dbench_set_priors, never per
//    shot; dbench_set_priors validates every element before writing anything.
//  - an all-zero syndrome is scanned in place and never copied: it writes the zero correction and
//    the zero observables, reports BP_CONVERGED with 0 iterations, and runs no BP and no LLR work.
//  - otherwise run ldpc::bp::BpDecoder (SYNDROME input type, no random schedule); if it converges
//    return its hard decision (BP_CONVERGED); else bposd -> ldpc::osd::OsdDecoder on the posterior
//    LLRs (POSTPROC_INVOKED, plus HIGHER_ORDER_WIN when osd_order > 0 and the final solution differs
//    from the OSD-0 solution); bplsd -> ldpc::lsd::LsdDecoder::lsd_decode(..., bits_per_step=1,
//    on_the_fly=true) (POSTPROC_INVOKED); bp -> BP hard decision, and since BP did not converge that
//    decision cannot reproduce the syndrome by construction, so SYNDROME_UNMATCHED is set directly.
//    The H*correction re-check is done only after OSD/LSD; for the other paths its result is known
//    by construction (zero syndrome and BP_CONVERGED: matched). iterations = BpDecoder::iterations.
//  - the soft output is produced lazily by dbench_get_soft_output, which returns the cached prior
//    LLRs after an all-zero shot and BpDecoder::log_prob_ratios of the last decode otherwise.
//    dbench_decode() copies nothing for it.
//  - the only per-shot buffers are the reused syndrome vector and the references the decoders
//    return; dbench_decode does no allocation of its own.
// The deterministic tie-breaks themselves are changes to vendor/ldpc/src_cpp (sort.hpp, lsd.hpp).
//
// The column orders produced by the fork (ascending LLR, ties by ascending column index) are what
// the OSD/LSD post-processing sees; BP itself is untouched apart from the fixed schedule (no
// random serial schedule is reachable), so BP messages/iteration counts are upstream's.
#include <algorithm>
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

// Generated at configure time by this directory's CMakeLists.txt.
#include "ldpc_ref_build_info.hpp"

#include "bp.hpp"
#include "lsd.hpp"
#include "osd.hpp"

namespace {

enum class Kind { kBp = 0, kBpOsd = 1, kBpLsd = 2 };

struct Config {
    Kind kind = Kind::kBp;
    ldpc::bp::BpMethod bp_method = ldpc::bp::MINIMUM_SUM;
    int max_iter = 1;
    double ms_scaling_factor = 0.625;
    ldpc::bp::BpSchedule schedule = ldpc::bp::PARALLEL;
    ldpc::osd::OsdMethod osd_method = ldpc::osd::OSD_0;
    int osd_order = 0;
    ldpc::osd::OsdMethod lsd_method = ldpc::osd::OSD_0;
    int lsd_order = 0;
};

// ---- config string parsing (include/decoder_abi.h "Config") --------------------------------

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

// Split "k=v;k=v" with the ABI rules: empty segments ignored, exactly one '=' per segment,
// keys [a-z0-9_]+, surrounding whitespace stripped, duplicate keys rejected.
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
        if (acc > 9000000000000000000ull) return false;  // far beyond every documented range
    }
    if (v[0] == '-') return false;  // every documented integer is >= 0
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
    else if (dec == "bplsd") cfg.kind = Kind::kBpLsd;
    else if (dec == "bp") cfg.kind = Kind::kBp;
    else {
        err = "invalid decoder '" + dec + "' (expected bposd, bplsd or bp)";
        return false;
    }

    std::string bpm;
    if (!need("bp_method", bpm)) return false;
    if (bpm == "ms") cfg.bp_method = ldpc::bp::MINIMUM_SUM;
    else if (bpm == "ps") cfg.bp_method = ldpc::bp::PRODUCT_SUM;
    else {
        err = "invalid bp_method '" + bpm + "' (expected ms or ps)";
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
            err = "invalid ms_scaling_factor '" + kv.at("ms_scaling_factor") + "' (expected a float > 0)";
            return false;
        }
        cfg.ms_scaling_factor = msf;
    }

    if (kv.count("schedule")) {
        const std::string &sch = kv.at("schedule");
        if (sch == "parallel") cfg.schedule = ldpc::bp::PARALLEL;
        else if (sch == "serial") cfg.schedule = ldpc::bp::SERIAL;
        else {
            err = "invalid schedule '" + sch + "' (expected parallel or serial)";
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
    } else if (cfg.kind == Kind::kBpLsd) {
        if (kv.count("osd_method") || kv.count("osd_order")) {
            err = "config key 'osd_method'/'osd_order' does not apply to decoder=bplsd";
            return false;
        }
        std::string lm;
        if (!need("lsd_method", lm)) return false;
        if (lm == "lsd0") cfg.lsd_method = ldpc::osd::OSD_0;
        else if (lm == "lsd_e") cfg.lsd_method = ldpc::osd::EXHAUSTIVE;
        else if (lm == "lsd_cs") cfg.lsd_method = ldpc::osd::COMBINATION_SWEEP;
        else {
            err = "invalid lsd_method '" + lm + "' (expected lsd0, lsd_e or lsd_cs)";
            return false;
        }
        if (kv.count("lsd_order")) {
            long long lo = 0;
            if (!parse_int_value(kv.at("lsd_order"), 0, std::numeric_limits<int>::max(), lo)) {
                err = "invalid lsd_order '" + kv.at("lsd_order") + "' (expected an integer >= 0)";
                return false;
            }
            cfg.lsd_order = static_cast<int>(lo);
        }
        if (cfg.lsd_method == ldpc::osd::OSD_0 && cfg.lsd_order != 0) {
            err = "lsd_order must be 0 for lsd_method=lsd0";
            return false;
        }
    } else {
        if (kv.count("osd_method") || kv.count("osd_order") || kv.count("lsd_method") || kv.count("lsd_order")) {
            err = "config key 'osd_*'/'lsd_*' does not apply to decoder=bp";
            return false;
        }
    }
    return true;
}

// ---- problem validation --------------------------------------------------------------------

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

void copy_csc_rows(const uint32_t *colptr, const uint32_t *rowind, uint32_t n, uint32_t rows,
                   std::vector<std::vector<uint32_t>> &out) {
    out.assign(rows, std::vector<uint32_t>());
    for (uint32_t j = 0; j < n; ++j) {
        for (uint32_t k = colptr[j]; k < colptr[j + 1]; ++k) {
            out[rowind[k]].push_back(j);
        }
    }
}

uint8_t parity_of(const std::vector<uint32_t> &cols, const uint8_t *bits) {
    uint8_t p = 0;
    for (uint32_t j : cols) p ^= static_cast<uint8_t>(bits[j] & 1u);
    return p;
}

}  // namespace

// The opaque handle of include/decoder_abi.h.
struct dbench_decoder {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    Config cfg;

    // Priors in the probability domain, the single source of truth shared by BP (which copies it)
    // and OSD (which holds a reference to it, like upstream's BpOsdDecoder).
    std::vector<double> channel_probs;
    // Prior LLRs log((1-p)/p) of channel_probs, computed once at create and in set_priors
    // (never per shot). They are the soft output of an all-zero-syndrome shot.
    std::vector<double> prior_llr;
    // True while the most recent dbench_decode() was the all-zero shot, i.e. while the lazy soft
    // output is the prior LLRs rather than bp->log_prob_ratios.
    bool last_shot_zero = true;

    ldpc::bp::BpSparse *pcm = nullptr;
    ldpc::bp::BpDecoder *bp = nullptr;
    ldpc::osd::OsdDecoder *osd = nullptr;
    ldpc::lsd::LsdDecoder *lsd = nullptr;

    std::vector<std::vector<uint32_t>> h_rows;
    std::vector<std::vector<uint32_t>> o_rows;
    std::vector<uint8_t> syndrome;

    ~dbench_decoder() {
        delete bp;   // these hold references to *pcm, so they go first
        delete osd;
        delete lsd;
        delete pcm;
    }
};

namespace {

void set_err(char *err, size_t err_len, const std::string &msg) {
    if (err != nullptr && err_len > 0) {
        std::snprintf(err, err_len, "%s", msg.c_str());
    }
}

std::string fail(const std::string &msg) { return "ldpc-ref: " + msg; }

// One shot, the single implementation used by dbench_decode and dbench_decode_batch.
// The timed body of dbench_decode is exactly this function: no allocation of its own (the
// syndrome vector is reused and the decoders return references into their own buffers) and no
// work whose result is known by construction.
int decode_one(dbench_decoder *h, const uint8_t *syndrome, uint8_t *out_correction, uint8_t *out_obs,
               uint32_t *out_status) {
    const uint32_t m = h->m, n = h->n, k = h->k;
    if (m > 0 && syndrome == nullptr) return DBENCH_EINVAL;
    if (k > 0 && out_obs == nullptr) return DBENCH_EINVAL;

    // Scan the syndrome in place: the all-zero shot is detected without copying it anywhere.
    bool zero_syndrome = true;
    for (uint32_t i = 0; i < m; ++i) {
        if (syndrome[i] & 1u) {
            zero_syndrome = false;
            break;
        }
    }

    if (zero_syndrome) {
        // Short circuit: zero correction, zero observables, BP_CONVERGED with 0
        // iterations. No BP, no LLR work, no copies. The lazy soft output reports the prior LLRs.
        if (out_correction != nullptr && n > 0) std::memset(out_correction, 0, n);
        for (uint32_t i = 0; i < k; ++i) out_obs[i] = 0;
        if (out_status != nullptr) *out_status = DBENCH_ST_BP_CONVERGED;
        h->last_shot_zero = true;
        return DBENCH_OK;
    }

    // Non-zero shot: the C++ decoders take a uint8_t vector, so copy the (masked) syndrome into
    // the reused buffer.
    h->syndrome.resize(m);
    for (uint32_t i = 0; i < m; ++i) h->syndrome[i] = static_cast<uint8_t>(syndrome[i] & 1u);

    uint32_t flags = 0;
    int iterations = 0;
    const std::vector<uint8_t> *corr = nullptr;
    try {
        // BpDecoder::decode() returns its internal buffer by value; calling the schedule directly
        // returns a reference to the same buffer, so the shot adds no copy of its own.
        std::vector<uint8_t> &dec = h->cfg.schedule == ldpc::bp::SERIAL
                                            ? h->bp->bp_decode_serial(h->syndrome)
                                            : h->bp->bp_decode_parallel(h->syndrome);
        iterations = h->bp->iterations;
        if (h->bp->converge) {
            corr = &dec;
            flags = DBENCH_ST_BP_CONVERGED;
        } else if (h->cfg.kind == Kind::kBpOsd) {
            corr = &h->osd->decode(h->syndrome, h->bp->log_prob_ratios);
            flags = DBENCH_ST_POSTPROC_INVOKED;
            if (h->cfg.osd_order > 0 && h->osd->osdw_decoding != h->osd->osd0_decoding) {
                flags |= DBENCH_ST_HIGHER_ORDER_WIN;
            }
        } else if (h->cfg.kind == Kind::kBpLsd) {
            // ldpc::lsd::LsdDecoder::lsd_decode() grows clusters until every one of them is
            // valid, so it does not terminate when the syndrome is outside the column space of
            // H (an inconsistent syndrome: e.g. a detector with no error mechanism set to 1).
            // That is upstream ldpc 2.4.1 behaviour. It is
            // unreachable from a corpus shot set, whose syndromes are
            // sampled from the DEM and therefore lie in the image of H by construction. Not
            // guarded here: the documented per-shot behaviour for bplsd is exactly this call.
            corr = &h->lsd->lsd_decode(h->syndrome, h->bp->log_prob_ratios, 1, true);
            flags = DBENCH_ST_POSTPROC_INVOKED;
        } else {
            // Plain BP that did not converge: its hard decision cannot reproduce the syndrome
            // (bp_decode_parallel/serial set `converge` exactly when H*decoding == syndrome), so
            // the flag is set directly instead of re-deriving it.
            corr = &dec;
            flags = DBENCH_ST_SYNDROME_UNMATCHED;
        }
    } catch (const std::exception &) {
        return DBENCH_EINTERNAL;
    } catch (...) {
        return DBENCH_EINTERNAL;
    }

    // The H*correction re-check is needed only after OSD/LSD, which can return a
    // syndrome-inconsistent vector (that is exactly what ST_SYNDROME_UNMATCHED reports).
    if (flags & DBENCH_ST_POSTPROC_INVOKED) {
        bool matched = true;
        for (uint32_t i = 0; i < m && matched; ++i) {
            if (parity_of(h->h_rows[i], corr->data()) != h->syndrome[i]) matched = false;
        }
        if (!matched) flags |= DBENCH_ST_SYNDROME_UNMATCHED;
    }

    if (out_correction != nullptr && n > 0) {
        std::memcpy(out_correction, corr->data(), n);
    }
    for (uint32_t i = 0; i < k; ++i) {
        out_obs[i] = parity_of(h->o_rows[i], corr->data());
    }
    if (out_status != nullptr) {
        uint32_t it = iterations < 0 ? 0u : static_cast<uint32_t>(iterations);
        if (it > DBENCH_ST_ITER_MAX) it = DBENCH_ST_ITER_MAX;
        *out_status = (flags & DBENCH_ST_FLAGS_MASK) | (it << DBENCH_ST_ITER_SHIFT);
    }
    h->last_shot_zero = false;
    return DBENCH_OK;
}

}  // namespace

extern "C" {

DBENCH_EXPORT uint32_t dbench_abi_version(void) { return DBENCH_ABI_VERSION; }
DBENCH_EXPORT const char *dbench_decoder_name(void) { return "ldpc-ref"; }
DBENCH_EXPORT uint32_t dbench_capabilities(void) {
    return DBENCH_CAP_CORRECTION | DBENCH_CAP_SET_PRIORS | DBENCH_CAP_SOFT_OUTPUT;
}

// Optional ABI symbol (include/decoder_abi.h): a static, NUL
// terminated build description naming the compiler, the flags this target is built with, the build
// type and the SHA-256 of the fork's patch file.
DBENCH_EXPORT const char *dbench_decoder_build_info(void) { return DBENCH_LDPCREF_BUILD_INFO; }

DBENCH_EXPORT dbench_decoder *dbench_create(const char *config, const dbench_problem *problem, char *err,
                                            size_t err_len) {
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
    // With no observables (k == 0) the O CSC is empty; callers may pass NULL column pointers.
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
        copy_csc_rows(problem->h_colptr, problem->h_rowind, n, m, h->h_rows);
        if (k > 0 && problem->o_colptr != nullptr) {
            copy_csc_rows(problem->o_colptr, problem->o_rowind, n, k, h->o_rows);
        } else {
            h->o_rows.assign(k, std::vector<uint32_t>());
        }

        // Same construction as upstream ldpc 2.4.1's Python wrappers: Bp with a channel-probability
        // copy, then OSD/LSD over the same pcm. The entries are inserted in the same order as
        // upstream's `pcm.nonzero()` (row index ascending, then column index ascending), because
        // SparseMatrixBase allocates its entries from one contiguous block (entry_block_size =
        // m + n + nnz) and the insertion order therefore fixes the memory layout of the linked
        // entries, which is worth several percent of BP latency.
        uint32_t nnz = problem->h_colptr[n];
        h->pcm = new ldpc::bp::BpSparse(static_cast<int>(m), static_cast<int>(n), static_cast<int>(nnz));
        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t j : h->h_rows[i]) {
                h->pcm->insert_entry(static_cast<int>(i), static_cast<int>(j));
            }
        }
        const std::vector<int> no_schedule = ldpc::bp::NULL_INT_VECTOR;
        h->bp = new ldpc::bp::BpDecoder(*h->pcm, h->channel_probs, cfg.max_iter, cfg.bp_method, cfg.schedule,
                                        cfg.ms_scaling_factor, 1, no_schedule, 0, false,
                                        ldpc::bp::BpInputType::SYNDROME);
        if (cfg.kind == Kind::kBpOsd) {
            // upstream BpOsdDecoder keeps a reference to the BP decoder's channel probabilities; here
            // both see the handle's vector, which dbench_set_priors() updates in place.
            h->osd = new ldpc::osd::OsdDecoder(*h->pcm, cfg.osd_method, cfg.osd_order, h->channel_probs);
        } else if (cfg.kind == Kind::kBpLsd) {
            h->lsd = new ldpc::lsd::LsdDecoder(*h->pcm, cfg.lsd_method, cfg.lsd_order);
        }
        // Prior LLRs, cached once, used as the soft output of an all-zero-syndrome shot.
        h->prior_llr.resize(n);
        for (uint32_t j = 0; j < n; ++j) {
            const double p = h->channel_probs[j];
            h->prior_llr[j] = std::log((1.0 - p) / p);
        }
        h->syndrome.assign(m, 0);
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
    // Validate every element before writing any of them: a rejected call leaves the handle (and
    // therefore subsequent decodes) exactly as it was.
    for (uint32_t j = 0; j < n; ++j) {
        if (!std::isfinite(llr[j])) return DBENCH_EINVAL;
        const double p = 1.0 / (1.0 + std::exp(llr[j]));
        if (!(p > 0.0) || !(p < 1.0) || !std::isfinite(p)) return DBENCH_EINVAL;
    }
    try {
        for (uint32_t j = 0; j < n; ++j) {
            const double p = 1.0 / (1.0 + std::exp(llr[j]));
            h->channel_probs[j] = p;
            h->prior_llr[j] = std::log((1.0 - p) / p);
        }
        // BpDecoder copies its channel probabilities; the OSD decoder references the vector above
        // (upstream's BpOsdDecoder does the same), so it already sees the new values.
        h->bp->channel_probabilities = h->channel_probs;
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
                                      uint8_t *out_corrections, uint8_t *out_obs, uint32_t *out_status) {
    if (h == nullptr) return DBENCH_EINVAL;
    if (num_shots > 0 && syndromes == nullptr) return DBENCH_EINVAL;
    if (h->k > 0 && num_shots > 0 && out_obs == nullptr) return DBENCH_EINVAL;
    // Exactly "dbench_decode() on each shot in order".
    for (uint32_t s = 0; s < num_shots; ++s) {
        const uint8_t *syn = syndromes + static_cast<size_t>(s) * h->m;
        uint8_t *corr = out_corrections != nullptr ? out_corrections + static_cast<size_t>(s) * h->n : nullptr;
        uint8_t *obs = out_obs != nullptr ? out_obs + static_cast<size_t>(s) * h->k : nullptr;
        uint32_t *st = out_status != nullptr ? out_status + s : nullptr;
        int rc = decode_one(h, syn, corr, obs, st);
        if (rc != DBENCH_OK) return rc;
    }
    return DBENCH_OK;
}

DBENCH_EXPORT int dbench_get_soft_output(dbench_decoder *h, double *out_llr, uint32_t n) {
    if (h == nullptr || out_llr == nullptr || n != h->n) return DBENCH_EINVAL;
    // Lazy: the prior LLRs after an all-zero-syndrome shot, the BP posteriors of the
    // last decode otherwise. dbench_decode() itself copies nothing for this.
    const std::vector<double> &src = h->last_shot_zero ? h->prior_llr : h->bp->log_prob_ratios;
    if (src.size() != n) return DBENCH_EINTERNAL;
    std::memcpy(out_llr, src.data(), static_cast<size_t>(n) * sizeof(double));
    return DBENCH_OK;
}

DBENCH_EXPORT void dbench_destroy(dbench_decoder *h) { delete h; }

}  // extern "C"
