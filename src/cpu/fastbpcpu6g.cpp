// fastbpcpu6g: C ABI (decoder_abi.h) over the multi-threaded exact min-sum BP engine
// (bp_engine.hpp) plus the exact bit-packed host OSD-0 engine of the GPU core
// (osd0_engine.hpp and osd_caps.hpp, the same files as src/core's), with the vendored
// ldpc::osd::OsdDecoder (osd.hpp of the vendored ldpc sources) kept for osd_cs/osd_e and for the OSD-0 shots
// whose syndrome is outside the image of H.
//
// Behaviour is the per-shot behaviour of ldpc-ref
// with bp_decode_parallel(MINIMUM_SUM) replaced by the engine:
// an all-zero syndrome short-circuits to zeros + BP_CONVERGED + 0 iterations, everything else runs
// BP; converged -> hard decision + BP_CONVERGED; bposd -> OSD on the BP posteriors
// (POSTPROC_INVOKED, HIGHER_ORDER_WIN when a higher-order candidate wins, SYNDROME_UNMATCHED when
// the result does not reproduce the syndrome); bp not converged -> the hard decision +
// SYNDROME_UNMATCHED.  The soft output is the prior LLRs after an all-zero shot and the BP
// posteriors otherwise, copied lazily by dbench_get_soft_output.
//
// Only the glue lives here: the config parsing, thread team, per-shot status word and
// observable handling are the multi-threaded engine's; the OSD-0 call goes to the bit-packed host engine
// (Osd0Engine) instead of that engine's own.  Both engines take the same inputs
// (the BP posterior as doubles, the syndrome as 0/1 bytes) and implement the same OSD-0 reference
// order, so no adaptation of either engine is needed.
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
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "decoder_abi.h"

// Generated at configure time by CMakeLists.txt.
#include "dbench_candidate_build_info.h"

#include "bp.hpp"
#include "osd.hpp"

#include "bp_engine.hpp"
#include "osd0_engine.hpp"

namespace {

enum class Kind { kBp = 0, kBpOsd = 1 };

struct Config {
    Kind kind = Kind::kBp;
    int max_iter = 1;
    double ms_scaling_factor = 0.625;
    ldpc::osd::OsdMethod osd_method = ldpc::osd::OSD_0;
    int osd_order = 0;
    // threads= (1..64, default 1) and the resolved CPU list (exactly `threads` entries).
    int threads = 1;
    std::vector<int> cpus;
    // counter_base= (test hook, 0 .. 2^63, default 0): the value every counter the engine
    // carries across shots starts from (Engine::set_counter_base).
    uint64_t counter_base = 0;
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

// `counter_base=N`, digits only, 0 <= N <= 2^63 (a sign, an exponent, hex or anything above
// 2^63 is malformed).
bool parse_counter_base(const std::string &v, uint64_t &out) {
    if (v.empty() || v.size() > 19) return false;
    unsigned __int128 acc = 0;
    for (char c : v) {
        if (c < '0' || c > '9') return false;
        acc = acc * 10u + (unsigned)(c - '0');
    }
    if (acc > ((unsigned __int128)1 << 63)) return false;
    out = (uint64_t)acc;
    return true;
}

// `cpus=LIST`, a list of CPU ids: comma-separated items `a` or `a-b`, decimal, a <= b,
// every id below 1024, repeats allowed.  A range stands for all its ids.  Malformed items, empty
// items, a non-decimal id or an id >= 1024 make the whole list malformed.
bool parse_cpu_list(const std::string &v, std::vector<int> &out, std::string &err) {
    out.clear();
    if (v.empty()) {
        err = "invalid cpus '" + v + "' (expected a comma-separated list of CPUs)";
        return false;
    }
    size_t pos = 0;
    while (pos <= v.size()) {
        const size_t comma = v.find(',', pos);
        const std::string item = v.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        pos = (comma == std::string::npos) ? v.size() + 1 : comma + 1;
        const size_t dash = item.find('-');
        if (dash == std::string::npos) {
            long long a = 0;
            if (!parse_int_value(item, 0, 1023, a)) {
                err = "invalid cpus item '" + item + "' (expected 'a' or 'a-b', 0 <= a <= b < 1024)";
                return false;
            }
            out.push_back((int)a);
        } else {
            long long a = 0, b = 0;
            if (!parse_int_value(item.substr(0, dash), 0, 1023, a) ||
                !parse_int_value(item.substr(dash + 1), 0, 1023, b) || b < a) {
                err = "invalid cpus range '" + item + "' (expected 'a-b', 0 <= a <= b < 1024)";
                return false;
            }
            for (long long c = a; c <= b; ++c) out.push_back((int)c);
        }
    }
    return true;
}

// The CPU list of a handle without an explicit cpus= key: the creating thread's affinity
// mask in ascending order, repeated to `T` entries.
bool default_cpu_list(int T, std::vector<int> &out, std::string &err) {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (::sched_getaffinity(0, sizeof(set), &set) != 0) {
        err = "cannot read the creating thread's CPU affinity";
        return false;
    }
    std::vector<int> mask;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &set)) mask.push_back(c);
    }
    if (mask.empty()) {
        err = "the creating thread's CPU affinity mask is empty";
        return false;
    }
    out.clear();
    for (int i = 0; i < T; ++i) out.push_back(mask[(size_t)i % mask.size()]);
    return true;
}

bool build_config(const std::map<std::string, std::string> &kv, Config &cfg, std::string &err) {
    static const char *const kKeys[] = {"decoder",   "bp_method",  "max_iter",   "ms_scaling_factor",
                                        "schedule",  "osd_method", "osd_order",  "lsd_method",
                                        "lsd_order", "threads",    "cpus",      "counter_base"};
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

    if (kv.count("threads")) {
        long long t = 0;
        if (!parse_int_value(kv.at("threads"), 1, 64, t)) {
            err = "invalid threads '" + kv.at("threads") + "' (expected an integer 1..64)";
            return false;
        }
        cfg.threads = (int)t;
    }

    if (kv.count("counter_base")) {
        if (!parse_counter_base(kv.at("counter_base"), cfg.counter_base)) {
            err = "invalid counter_base '" + kv.at("counter_base") +
                  "' (expected a decimal integer 0 .. 9223372036854775808)";
            return false;
        }
    }

    if (kv.count("cpus")) {
        if (!parse_cpu_list(kv.at("cpus"), cfg.cpus, err)) return false;
    } else if (!default_cpu_list(cfg.threads, cfg.cpus, err)) {
        return false;
    }
    if ((int)cfg.cpus.size() != cfg.threads) {
        err = "cpus list has " + std::to_string(cfg.cpus.size()) + " entries, threads=" +
              std::to_string(cfg.threads) + " needs " + std::to_string(cfg.threads);
        return false;
    }

    std::string dec;
    if (!need("decoder", dec)) return false;
    if (dec == "bposd") cfg.kind = Kind::kBpOsd;
    else if (dec == "bp") cfg.kind = Kind::kBp;
    else if (dec == "bplsd") {
        // ldpc-ref accepts this; the support set of this library does not.
        err = "decoder 'bplsd' is not supported by fastbpcpu6g (expected bposd or bp)";
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
        err = "bp_method 'ps' is not supported by fastbpcpu6g (expected ms)";
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
            err = "schedule 'serial' is not supported by fastbpcpu6g (expected parallel)";
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

std::string fail(const std::string &msg) { return "fastbpcpu6g: " + msg; }

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

    fastbp::Engine engine;

    // Row-major edge lists of H and O (ascending columns), for the syndrome re-check after OSD,
    // the observable prediction and the observable matrix of the OSD problem.
    std::vector<uint32_t> h_rowptr, h_rowcol;
    std::vector<uint32_t> o_rowptr, o_rowcol;

    ldpc::bp::BpSparse *pcm = nullptr;
    ldpc::osd::OsdDecoder *osd = nullptr;

    // The library's own exact OSD-0 (used for osd_method=osd0; the vendored decoder stays for
    // osd_cs/osd_e and for the shots the own engine defers as outside the image of H).
    fastbposd::Osd0Engine osd0;
    // Correction buffer for the own path when the caller passes no out_correction.
    std::vector<uint8_t> xscratch;
    // Per-column observable masks (kMaxObsMaskWords 64-bit words each; o_kw == 0 when they are not
    // built, i.e. k == 0 or k > 1024).  For a sparse correction O*x is then one XOR per set column.
    // kMaxObsMaskWords is a real bound: the accumulator in decode_one is
    // uint64_t acc[kMaxObsMaskWords], and only k <= 1024 builds masks.
    static constexpr uint32_t kMaxObsMaskWords = 16;   // k <= 1024
    std::vector<uint64_t> o_mask;
    uint32_t o_kw = 0;

    // The masked syndrome of the last decode (OSD input, syndrome re-check).
    std::vector<uint8_t> syndrome;
    // Plain-vector copies handed to the vendored OsdDecoder, which takes both inputs by non-const
    // reference (the engine's posterior lives in an aligned vector, and the syndrome re-check reads
    // h->syndrome after decode()): the vendored code never gets a reference to either array.  They
    // are allocated at create, so a shot that reaches the vendored post-processing allocates
    // nothing itself.
    std::vector<double> posterior_plain;
    std::vector<uint8_t> osd_syn;

    // The library's threads (threads - 1 of them; the calling thread computes as thread 0).
    // Started and pinned in dbench_create, stopped and joined in dbench_destroy (before the engine
    // and the rest of the handle are destroyed).
    std::vector<std::thread> team;
    void stop_team() {
        if (team.empty()) return;
        engine.job_seq.store(fastbp::Engine::kStopSeq, std::memory_order_release);
        for (auto &t : team) t.join();
        team.clear();
    }

    ~dbench_decoder() {
        stop_team();
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

// One shot: the single implementation used by dbench_decode and dbench_decode_batch.
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
        // Zero correction, zero observables, BP_CONVERGED with 0 iterations, no BP, no LLR work.
        if (out_correction != nullptr && n > 0) std::memset(out_correction, 0, n);
        for (uint32_t i = 0; i < k; ++i) out_obs[i] = 0;
        if (out_status != nullptr) *out_status = DBENCH_ST_BP_CONVERGED;
        h->last_shot_zero = true;
        return DBENCH_OK;
    }

    try {
        h->syndrome.resize(m);
        for (uint32_t i = 0; i < m; ++i) h->syndrome[i] = static_cast<uint8_t>(syndrome[i] & 1u);

        const uint32_t iterations = h->engine.decode_shot(h->syndrome.data(),
                                                          (uint32_t)h->cfg.max_iter,
                                                          h->cfg.ms_scaling_factor);
        const uint8_t *corr = h->engine.decoding.data();

        uint32_t flags = 0;
        // True when the correction comes from the vendored post-processing, whose result may not
        // reproduce the syndrome and therefore has to be re-checked.  The own OSD-0 returns the
        // exact solution of H x = s by construction (its reduced syndrome reaches zero), so its
        // result is matched without re-deriving it.
        bool need_syndrome_check = false;
        bool own_osd = false;
        if (h->engine.last_converged) {
            flags = DBENCH_ST_BP_CONVERGED;
        } else if (h->cfg.kind == Kind::kBpOsd) {
            bool solved = false;
            if (h->cfg.osd_method == ldpc::osd::OSD_0) {
                uint8_t *dst = out_correction != nullptr ? out_correction : h->xscratch.data();
                if (h->osd0.run(h->engine.posterior.data(), h->syndrome.data(), dst) ==
                    fastbposd::Osd0Engine::kSolved) {
                    corr = dst;
                    solved = true;
                    own_osd = true;
                }
            }
            if (!solved) {
                // osd_cs/osd_e, and the osd0 shots whose syndrome is outside the image of H (the
                // reference's own x there depends on its internal pivot-row choice).  The vendored
                // decoder takes a std::vector<double>&; the copy is into a buffer created at
                // create time, so these shots still allocate only inside the vendored decoder.
                std::memcpy(h->posterior_plain.data(), h->engine.posterior.data(),
                            (size_t)n * sizeof(double));
                if (m > 0) std::memcpy(h->osd_syn.data(), h->syndrome.data(), m);
                corr = h->osd->decode(h->osd_syn, h->posterior_plain).data();
                need_syndrome_check = true;
            }
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

        // Observable prediction.  When the correction is a sparse set of columns the per-column
        // masks turn O*x into one XOR per set column.  That is kept for the own OSD-0's support
        // (measured faster than the row-major loop on bb144 it1: 48.1 vs 52.9 us/shot); the BP hard
        // decision keeps the row-major loop, because a non-converged hard decision can be dense and
        // the mask form was not measured faster there (so the masks are kept for it
        // only if they are).  FASTBPOSD_OBS_MASK_CONVERGED turns them on for the hard decision too
        // (measurement only); FASTBPOSD_OBS_ROW_PARITY drops the masks entirely.
        bool mask_ok = h->o_kw > 0 && own_osd;
#ifdef FASTBPOSD_OBS_MASK_CONVERGED
        if (h->o_kw > 0 && !own_osd && corr == h->engine.decoding.data()) mask_ok = true;
#endif
        if (mask_ok) {
            uint64_t acc[dbench_decoder::kMaxObsMaskWords];
            std::memset(acc, 0, sizeof(acc));
            if (own_osd) {
                const uint32_t nsup = h->osd0.support_size();
                const uint32_t *sup = h->osd0.support();
                for (uint32_t t = 0; t < nsup; ++t) {
                    const uint64_t *mask = &h->o_mask[(size_t)sup[t] * h->o_kw];
                    for (uint32_t w = 0; w < h->o_kw; ++w) acc[w] ^= mask[w];
                }
            } else {
                for (uint32_t j = 0; j < n; ++j) {
                    if ((corr[j] & 1u) == 0) continue;
                    const uint64_t *mask = &h->o_mask[(size_t)j * h->o_kw];
                    for (uint32_t w = 0; w < h->o_kw; ++w) acc[w] ^= mask[w];
                }
            }
            for (uint32_t i = 0; i < k; ++i) {
                out_obs[i] = static_cast<uint8_t>((acc[i >> 6] >> (i & 63)) & 1u);
            }
        } else {
            for (uint32_t i = 0; i < k; ++i) {
                out_obs[i] = row_parity(h->o_rowptr.data(), h->o_rowcol.data(), i, corr);
            }
        }
        if (out_status != nullptr) {
            uint32_t it = iterations;
            if (it > DBENCH_ST_ITER_MAX) it = DBENCH_ST_ITER_MAX;
            *out_status = (flags & DBENCH_ST_FLAGS_MASK) | (it << DBENCH_ST_ITER_SHIFT);
        }
        h->last_shot_zero = false;
    } catch (const std::exception &) {
        return DBENCH_EINTERNAL;
    } catch (...) {
        return DBENCH_EINTERNAL;
    }
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
        h->o_rowptr.assign((size_t)k + 1, 0);
        if (k > 0 && problem->o_colptr != nullptr) {
            const uint32_t onz = problem->o_colptr[n];
            for (uint32_t e = 0; e < onz; ++e) h->o_rowptr[problem->o_rowind[e] + 1]++;
            for (uint32_t i = 0; i < k; ++i) h->o_rowptr[i + 1] += h->o_rowptr[i];
            h->o_rowcol.resize(h->o_rowptr[k]);
            std::vector<uint32_t> fill(h->o_rowptr.begin(), h->o_rowptr.end() - 1);
            for (uint32_t j = 0; j < n; ++j) {
                for (uint32_t e = problem->o_colptr[j]; e < problem->o_colptr[j + 1]; ++e) {
                    h->o_rowcol[fill[problem->o_rowind[e]]++] = j;
                }
            }
        }

        if (!h->engine.setup(m, n, problem->h_colptr, problem->h_rowind, (uint32_t)cfg.threads)) {
            set_err(err, err_len, fail("create failed: engine setup (problem beyond the int32 index "
                                       "limit, or the arena could not be allocated)").c_str());
            delete h;
            return nullptr;
        }
        // Before any thread starts: the threads read the starting sequence when they start.
        h->engine.set_counter_base(cfg.counter_base);

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

        // Per-column observable masks: the own OSD-0's support is then predicted with one XOR per
        // set column.  They are built only for the configurations that have the own OSD-0 (a bp-only
        // handle never uses them; the masks would cost n * kw * 8 bytes for nothing).  Larger k
        // keeps the row-parity path (the accumulator bound), as does the vendored decoder's
        // correction, whose support is not tracked here.  FASTBPOSD_OBS_ROW_PARITY forces the
        // row-major path and FASTBPOSD_OBS_MASK_CONVERGED needs the masks in every config: both are
        // measurement-only (a standalone benchmark).
        const uint32_t kw = (uint32_t)(((uint64_t)k + 63) / 64);
#if defined(FASTBPOSD_OBS_ROW_PARITY)
        if (false) {
#elif defined(FASTBPOSD_OBS_MASK_CONVERGED)
        if (kw > 0 && kw <= dbench_decoder::kMaxObsMaskWords) {
#else
        if (cfg.kind == Kind::kBpOsd && kw > 0 && kw <= dbench_decoder::kMaxObsMaskWords) {
#endif
            h->o_kw = kw;
            h->o_mask.assign((size_t)n * kw, 0);
            for (uint32_t j = 0; j < n; ++j) {
                for (uint32_t e = problem->o_colptr[j]; e < problem->o_colptr[j + 1]; ++e) {
                    const uint32_t r = problem->o_rowind[e];
                    h->o_mask[(size_t)j * kw + (r >> 6)] |= 1ull << (r & 63);
                }
            }
        }

        // Prior LLRs, cached once: the soft output of an all-zero-syndrome shot.
        h->prior_llr.resize(n);
        for (uint32_t j = 0; j < n; ++j) {
            const double p = h->channel_probs[j];
            h->prior_llr[j] = std::log((1.0 - p) / p);
        }
        h->engine.set_priors(h->prior_llr.data());
        h->syndrome.assign(m, 0);
        h->posterior_plain.assign(n, 0.0);
        h->osd_syn.assign(m, 0);

        // The threads that compute a shot are started and pinned here, before
        // dbench_create returns, and dbench_destroy stops and joins them.  A listed CPU is accepted
        // exactly when the kernel will pin a thread to it (online and inside this process's cpuset);
        // the creating thread's own affinity mask is not the criterion (the kernel lets a thread be
        // pinned to a CPU its creator is not currently allowed to run on).  Each started thread's
        // pthread_setaffinity_np result is checked, and with threads=1 -- which starts no thread --
        // the single entry is checked with a short-lived probe thread that is joined before
        // returning.  On any failure the threads started so far are stopped and joined (delete h)
        // and create returns NULL with a message.
        if (cfg.threads == 1) {
            int rc = 0;
            std::thread probe([&rc, &cfg]() {
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(cfg.cpus[0], &set);
                rc = ::pthread_setaffinity_np(::pthread_self(), sizeof(set), &set);
            });
            probe.join();
            if (rc != 0) {
                set_err(err, err_len,
                        fail("create failed: CPU " + std::to_string(cfg.cpus[0]) +
                             " is not one a thread can be pinned to").c_str());
                delete h;
                return nullptr;
            }
        }
        if (cfg.threads > 1) {
            try {
                h->team.reserve((size_t)cfg.threads - 1);
                for (int i = 0; i < cfg.threads - 1; ++i) {
                    dbench_decoder *hh = h;
                    const int tid = i + 1;
                    h->team.emplace_back([hh, tid]() { hh->engine.thread_loop(tid); });
                    cpu_set_t set;
                    CPU_ZERO(&set);
                    CPU_SET(cfg.cpus[i], &set);
                    if (::pthread_setaffinity_np(h->team.back().native_handle(), sizeof(set), &set) != 0) {
                        set_err(err, err_len,
                                fail("create failed: cannot pin a thread to CPU " +
                                     std::to_string(cfg.cpus[i])).c_str());
                        delete h;
                        return nullptr;
                    }
                }
            } catch (const std::exception &e) {
                set_err(err, err_len, fail(std::string("create failed: ") + e.what()).c_str());
                delete h;
                return nullptr;
            }
        }
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
        for (uint32_t j = 0; j < n; ++j) {
            const double p = 1.0 / (1.0 + std::exp(llr[j]));
            h->channel_probs[j] = p;
            h->prior_llr[j] = std::log((1.0 - p) / p);
        }
        // The OSD references channel_probs, so it already sees the new values.
        h->engine.set_priors(h->prior_llr.data());
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
    if (num_shots > 0 && syndromes == nullptr) return DBENCH_EINVAL;
    if (h->k > 0 && num_shots > 0 && out_obs == nullptr) return DBENCH_EINVAL;
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
    if (h->last_shot_zero) {
        if (h->prior_llr.size() != n) return DBENCH_EINTERNAL;
        std::memcpy(out_llr, h->prior_llr.data(), (size_t)n * sizeof(double));
        return DBENCH_OK;
    }
    if (h->engine.posterior.size() != n) return DBENCH_EINTERNAL;
    std::memcpy(out_llr, h->engine.posterior.data(), (size_t)n * sizeof(double));
    return DBENCH_OK;
}

DBENCH_EXPORT void dbench_destroy(dbench_decoder *h) { delete h; }

}  // extern "C"
