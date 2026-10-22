// SPDX-License-Identifier: Apache-2.0
// libeldpc.so: the C API front of the eldpc package.
//
// The package ships four builds of the GPU decoder core: the default build and the "ada64" build
// of each host ISA (AVX2 and AVX-512). The two builds of one ISA are compiled from the same six
// sources with the same flags; their only difference is DBENCH_GPU_SM_BUDGET=64 in the ada64
// build, which caps the SM count the create path plans for (src/core/bp_gpu.cu, g->sms). It also
// ships the optional CPU core fastbpcpu6g, one build per host ISA (libeldpc_cpu_avx2.so,
// libeldpc_cpu_avx512.so).
//
// This front chooses three things:
//   - the host ISA, once, here, at load time: ELDPC_ISA=avx2|avx512 forces one, otherwise the
//     AVX-512 build is used when the CPU and the OS support every instruction set it was compiled
//     with, else the AVX2 build. The default GPU build of the chosen ISA is loaded here
//     (eldpc_isa() and eldpc_build_info() report it); the ada64 build of the same ISA and the CPU
//     core are loaded on the first handle that needs them.
//   - preset names, per handle, in eldpc_create(): a config string may start with a preset name of
//     the Python package ("a03125c3", or "a03125c3;threads=8"), which this front expands to the
//     preset's key=value segments, extra keys overriding or adding to them (expand_preset below).
//   - the target device, per handle, in eldpc_create(): the config key device=gpu|cpu|auto, which
//     this front reads and removes before the core parses the config (it therefore never reaches a
//     core). device=gpu is the default. A device=cpu create loads the CPU core of the chosen host
//     ISA and makes no CUDA call at all (no device query, no driver or runtime call). device=auto
//     is device=gpu when device 0 is usable by this build's GPU cores (a driver, a device, a
//     driver new enough for the cores' CUDA version and a compute capability the cores carry an
//     image for, gpu_arch.h) and device=cpu otherwise, including in a CPU-only build.
//   - the GPU core variant, per handle, in eldpc_create(): from ELDPC_DEVICE_CLASS when it is set
//     ("default" or "ada64"), else from the target device through the table of core_variant.h
//     (compute capability and SM-count range). The target device is device 0 of the process's
//     visible devices, which is the device the decoder core plans for (Engine::setup calls
//     cudaSetDevice(0)). The device properties are read through the CUDA driver API at create
//     time and cached; no decode path reads the environment, opens a file or queries a device.
//     ELDPC_DEVICE_CLASS applies to GPU handles only.
//
// Cores are loaded with dlopen(RTLD_NOW | RTLD_LOCAL) from this library's own directory. Handles
// of both variants, of both device kinds (GPU and CPU) and of both ISAs (one per process) can live
// in one process.
//
// A CPU-only build (CMake option ELDPC_WITHOUT_CUDA, or no CUDA compiler found) compiles this front
// with ELDPC_CPU_ONLY: no GPU cores exist, so the CPU core is the process's core loaded at load
// time, and a device=gpu create fails with a message that says the build has no GPU cores and
// names device="cpu". In the normal build this file has no knowledge of that option.
//
// The AVX-512 core is compiled with -mavx2 -mavx512f -mavx512vl -mavx512dq -mavx512bw, so the
// check tests exactly those CPUID bits plus the OS's XMM/YMM and opmask/ZMM state in XCR0
// (XGETBV). The AVX2 core is -mavx2: CPUID AVX2 plus the OS's XMM/YMM state.

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

#include "eldpc.h"

#include "core_variant.h"
#include "gpu_arch.h"

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

// The core's ABI (decoder_abi.h, decoder-bench ABI v1). The problem structs are field for field
// the same; eldpc_create copies the fields into a dbench_problem below.
#include "decoder_abi.h"

namespace {

using core_abi_version_fn = uint32_t (*)();
using core_str_fn = const char *(*)();
using core_caps_fn = uint32_t (*)();
using core_create_fn = dbench_decoder *(*)(const char *, const dbench_problem *, char *, size_t);
using core_set_priors_fn = int (*)(dbench_decoder *, const double *, uint32_t);
using core_decode_fn = int (*)(dbench_decoder *, const uint8_t *, uint8_t *, uint8_t *, uint32_t *);
using core_decode_batch_fn = int (*)(dbench_decoder *, uint32_t, const uint8_t *, uint8_t *, uint8_t *,
                                     uint32_t *);
using core_soft_fn = int (*)(dbench_decoder *, double *, uint32_t);
using core_keep_warm_fn = int (*)(dbench_decoder *, uint32_t);
using core_destroy_fn = void (*)(dbench_decoder *);

// A constant-initialized aggregate: it lives in .bss and needs no dynamic initialization, so the
// load-time constructor below cannot run before it is ready.
struct CoreApi {
    void *handle;
    bool loaded;
    const char *isa;
    core_abi_version_fn abi_version;
    core_str_fn decoder_name;
    core_caps_fn capabilities;
    core_create_fn create;
    core_set_priors_fn set_priors;
    core_decode_fn decode;
    core_decode_batch_fn decode_batch;
    core_soft_fn get_soft_output;
    core_keep_warm_fn keep_warm;
    core_destroy_fn destroy;
    core_str_fn build_info;
};

enum class Isa { kAvx2 = 0, kAvx512 = 1 };
constexpr int kIsaCount = 2;
constexpr int kVariantCount = 2;

// One slot per (host ISA, GPU core variant). The default core of the host ISA is loaded by the
// constructor; the others are loaded on first use, under g_core_mutex. The CPU cores (one slot per
// host ISA, no variant) are loaded on the first device=cpu create.
CoreApi g_cores[kIsaCount][kVariantCount] = {};
CoreApi g_cpu_cores[kIsaCount] = {};
CoreApi *g_default_core = nullptr;
Isa g_isa = Isa::kAvx2;
bool g_isa_valid = false;
std::mutex g_core_mutex;

char g_load_error[2048];
bool g_load_error_set = false;
char g_build_info[8192];

// The target device's properties, read once through the CUDA driver API at create time.
struct DeviceInfo {
    int cc_major;
    int cc_minor;
    int sm_count;
    int driver_version;  // cuDriverGetVersion (e.g. 13000 for CUDA 13.0), 0 when unknown
    char name[256];
    bool valid;
};
DeviceInfo g_device_info = {0, 0, 0, 0, {0}, false};
bool g_device_queried = false;
std::mutex g_device_mutex;

void set_load_error(const std::string &msg) {
    std::snprintf(g_load_error, sizeof(g_load_error), "%s", msg.c_str());
    g_load_error_set = true;
}

const char *isa_name(Isa isa) { return isa == Isa::kAvx512 ? "avx512" : "avx2"; }
int isa_index(Isa isa) { return isa == Isa::kAvx512 ? 1 : 0; }
int variant_index(eldpc::Variant variant) { return variant == eldpc::Variant::kAda64 ? 1 : 0; }

std::string core_file_name(Isa isa, eldpc::Variant variant) {
    std::string name = "libeldpc_core_";
    name += isa_name(isa);
    if (variant == eldpc::Variant::kAda64) name += "_ada64";
    name += ".so";
    return name;
}

std::string cpu_core_file_name(Isa isa) {
    std::string name = "libeldpc_cpu_";
    name += isa_name(isa);
    name += ".so";
    return name;
}

#if defined(__x86_64__) || defined(__i386__)
uint64_t xgetbv0() {
    uint32_t lo = 0, hi = 0;
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

bool wc_cpuid(unsigned leaf, unsigned subleaf, unsigned out[4]) {
    if (__get_cpuid_max(0, nullptr) < leaf) return false;
    __cpuid_count(leaf, subleaf, out[0], out[1], out[2], out[3]);
    return true;
}
#endif

// AVX2: CPUID.1 AVX and OSXSAVE, XCR0 XMM|YMM, CPUID.7.0 EBX.AVX2.
bool cpu_has_avx2_os() {
#if defined(__x86_64__) || defined(__i386__)
    unsigned r1[4];
    if (!wc_cpuid(1, 0, r1)) return false;
    if (!((r1[2] >> 27) & 1u)) return false;  // OSXSAVE
    if (!((r1[2] >> 28) & 1u)) return false;  // AVX
    if ((xgetbv0() & 0x6ull) != 0x6ull) return false;
    unsigned r7[4];
    if (!wc_cpuid(7, 0, r7)) return false;
    return ((r7[1] >> 5) & 1u) != 0;  // AVX2
#else
    return false;
#endif
}

// What the AVX-512 core was compiled with: AVX2 plus AVX512F, AVX512VL, AVX512DQ, AVX512BW, and
// the OS's opmask/ZMM state (XCR0 bits 5, 6, 7).
bool cpu_has_avx512_os() {
#if defined(__x86_64__) || defined(__i386__)
    if (!cpu_has_avx2_os()) return false;
    unsigned r7[4];
    if (!wc_cpuid(7, 0, r7)) return false;
    const unsigned b = r7[1];
    const unsigned need = (1u << 16) | (1u << 17) | (1u << 30) | (1u << 31);  // F, DQ, BW, VL
    if ((b & need) != need) return false;
    return (xgetbv0() & 0xE6ull) == 0xE6ull;
#else
    return false;
#endif
}

// What this CPU and OS can run, for the message of an ELDPC_ISA value that cannot be used: the
// automatic choice (ELDPC_ISA unset) and the ISAs it could load.
std::string host_isa_text() {
    if (cpu_has_avx512_os()) {
        return "this CPU and OS support both cores (avx512 and avx2); unset ELDPC_ISA to let eldpc "
               "choose (the automatic choice is avx512)";
    }
    if (cpu_has_avx2_os()) {
        return "this CPU and OS support the avx2 core only; unset ELDPC_ISA (or set it to avx2) to "
               "use it";
    }
    return "this CPU and OS support neither core";
}

// Directory of the loaded front library (the cores are installed next to it).
bool front_dir(std::string &dir) {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(&eldpc_api_version)), &info)) return false;
    if (info.dli_fname == nullptr || info.dli_fname[0] == '\0') return false;
    std::string p(info.dli_fname);
    const size_t slash = p.rfind('/');
    dir = (slash == std::string::npos) ? std::string(".") : p.substr(0, slash);
    return true;
}

// Load one core library into a slot. `file` is the library file name (the cores live next to this
// front), `what` names the core in messages, and `require_keep_warm` says whether fastbp_keep_warm
// is a required symbol: the GPU cores export it; the CPU core does not, and the front implements
// eldpc_keep_warm() for a CPU handle itself.
bool load_core(CoreApi &slot, Isa isa, const std::string &file, const std::string &what, bool require_keep_warm,
               std::string &error) {
    std::string dir;
    if (!front_dir(dir)) {
        error = "eldpc: cannot determine the directory of libeldpc.so (dladdr failed)";
        return false;
    }
    const std::string path = dir + "/" + file;
    void *h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (h == nullptr) {
        const char *e = dlerror();
        error = "eldpc: cannot load the " + what + " " + path + ": " +
                (e != nullptr ? e : "unknown dlopen error");
        return false;
    }
    CoreApi api = {};
    api.handle = h;
    std::string missing;
    auto symbol = [&](auto &fn, const char *name) {
        void *sym = dlsym(h, name);
        if (sym == nullptr) {
            if (!missing.empty()) missing += ", ";
            missing += name;
            return false;
        }
        fn = reinterpret_cast<typename std::remove_reference<decltype(fn)>::type>(sym);
        return true;
    };
    bool ok = true;
    ok &= symbol(api.abi_version, "dbench_abi_version");
    ok &= symbol(api.decoder_name, "dbench_decoder_name");
    ok &= symbol(api.capabilities, "dbench_capabilities");
    ok &= symbol(api.create, "dbench_create");
    ok &= symbol(api.set_priors, "dbench_set_priors");
    ok &= symbol(api.decode, "dbench_decode");
    ok &= symbol(api.decode_batch, "dbench_decode_batch");
    ok &= symbol(api.get_soft_output, "dbench_get_soft_output");
    ok &= symbol(api.destroy, "dbench_destroy");
    if (require_keep_warm) ok &= symbol(api.keep_warm, "fastbp_keep_warm");
    symbol(api.build_info, "dbench_decoder_build_info");  // optional
    if (!ok) {
        error = "eldpc: the core " + path + " is missing required symbols: " + missing;
        dlclose(h);
        return false;
    }
    const uint32_t v = api.abi_version();
    if (v != DBENCH_ABI_VERSION) {
        error = "eldpc: the core " + path + " has decoder ABI version " + std::to_string(v) +
                ", expected " + std::to_string(DBENCH_ABI_VERSION);
        dlclose(h);
        return false;
    }
    api.isa = isa_name(isa);
    api.loaded = true;
    slot = api;
    return true;
}

// The core slot of (isa, variant), loading it on first use. Returns nullptr on failure. The lock is
// taken on every call (create path only): the slot's fields are written while loading, and a
// lock-free read of them would race with a concurrent first create of the same variant.
const CoreApi *core_for(Isa isa, eldpc::Variant variant, std::string &error) {
    CoreApi &slot = g_cores[isa_index(isa)][variant_index(variant)];
    std::lock_guard<std::mutex> lock(g_core_mutex);
    if (!slot.loaded) {
        const std::string what =
            std::string(isa_name(isa)) + (variant == eldpc::Variant::kAda64 ? " ada64" : "") + " core";
        if (!load_core(slot, isa, core_file_name(isa, variant), what, true, error)) return nullptr;
    }
    return &slot;
}

// The CPU core slot of the host ISA, loaded on the first device=cpu create. The CPU core exports
// the decoder ABI and no fastbp_keep_warm (a CPU handle's eldpc_keep_warm() is the front's no-op).
const CoreApi *cpu_core_for(Isa isa, std::string &error) {
    CoreApi &slot = g_cpu_cores[isa_index(isa)];
    std::lock_guard<std::mutex> lock(g_core_mutex);
    if (!slot.loaded) {
        const std::string what = std::string(isa_name(isa)) + " CPU core";
        if (!load_core(slot, isa, cpu_core_file_name(isa), what, false, error)) return nullptr;
    }
    return &slot;
}

// The CUDA driver API's device-property calls, loaded from libcuda.so.1 (the front does not link
// the CUDA driver). CUdevice_attribute values of the fields read here are stable across CUDA
// versions: 16 multiprocessor count, 75/76 compute capability major/minor.
constexpr int kAttrMultiprocessorCount = 16;
constexpr int kAttrComputeCapabilityMajor = 75;
constexpr int kAttrComputeCapabilityMinor = 76;

using cu_init_fn = int (*)(unsigned);
using cu_device_get_fn = int (*)(int *, int);
using cu_device_get_attribute_fn = int (*)(int *, int, int);
using cu_device_get_name_fn = int (*)(char *, int, int);
using cu_driver_get_version_fn = int (*)(int *);
using cu_get_error_name_fn = int (*)(int, const char **);
constexpr int kCudaErrorNoDevice = 100;

// "CUDA_ERROR_NO_DEVICE (100)" from cuGetErrorName when the driver has it, else the number.
std::string cu_error_text(cu_get_error_name_fn name_fn, int rc) {
    const char *name = nullptr;
    if (name_fn != nullptr && name_fn(rc, &name) == 0 && name != nullptr) {
        return std::string(name) + " (" + std::to_string(rc) + ")";
    }
    return "CUresult " + std::to_string(rc);
}

// Reads device 0's properties. On failure returns false with `reason`, a clause that says what is
// missing (no driver library, a failed cuInit, no device).
bool query_device(DeviceInfo &info, std::string &reason) {
    void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) lib = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) {
        reason = "the NVIDIA driver library libcuda.so.1 cannot be loaded (no NVIDIA driver is "
                 "installed, or it is not visible to this process)";
        return false;
    }
    auto init = reinterpret_cast<cu_init_fn>(dlsym(lib, "cuInit"));
    auto get = reinterpret_cast<cu_device_get_fn>(dlsym(lib, "cuDeviceGet"));
    auto attr = reinterpret_cast<cu_device_get_attribute_fn>(dlsym(lib, "cuDeviceGetAttribute"));
    auto get_name = reinterpret_cast<cu_device_get_name_fn>(dlsym(lib, "cuDeviceGetName"));
    auto driver_version = reinterpret_cast<cu_driver_get_version_fn>(dlsym(lib, "cuDriverGetVersion"));
    auto error_name = reinterpret_cast<cu_get_error_name_fn>(dlsym(lib, "cuGetErrorName"));
    if (init == nullptr || get == nullptr || attr == nullptr) {
        dlclose(lib);
        reason = "the NVIDIA driver library libcuda.so.1 lacks the device-query entry points";
        return false;
    }
    int device = 0;
    int major = 0, minor = 0, sms = 0;
    // cuInit(0) is idempotent and needed before any device query; the decoder core initialises the
    // same driver when it creates its context.
    int rc = init(0);
    if (rc != 0) {
        reason = "cuInit failed with " + cu_error_text(error_name, rc);
        if (rc == kCudaErrorNoDevice) {
            reason += ": the driver sees no CUDA device (check CUDA_VISIBLE_DEVICES)";
        }
        dlclose(lib);
        return false;
    }
    rc = get(&device, 0);
    if (rc == 0) rc = attr(&major, kAttrComputeCapabilityMajor, device);
    if (rc == 0) rc = attr(&minor, kAttrComputeCapabilityMinor, device);
    if (rc == 0) rc = attr(&sms, kAttrMultiprocessorCount, device);
    if (rc != 0 || sms <= 0) {
        reason = "the CUDA driver could not query device 0 (" +
                 (rc != 0 ? cu_error_text(error_name, rc) : std::string("no multiprocessors")) + ")";
        dlclose(lib);
        return false;
    }
    info.driver_version = 0;
    int version = 0;
    if (driver_version != nullptr && driver_version(&version) == 0 && version > 0) {
        info.driver_version = version;
    }
    info.name[0] = '\0';
    if (get_name != nullptr && get_name(info.name, static_cast<int>(sizeof(info.name)), device) != 0) {
        info.name[0] = '\0';
    }
    info.name[sizeof(info.name) - 1] = '\0';
    dlclose(lib);
    info.cc_major = major;
    info.cc_minor = minor;
    info.sm_count = sms;
    info.valid = true;
    return true;
}

// Properties of the device a new handle will decode on: device 0 of the process's visible devices.
// Cached after the first successful query (device properties cannot change while a process runs);
// a failed query is retried, so a create that finds no GPU does not pin the answer. When the query
// fails, `reason` says why (see query_device).
DeviceInfo target_device(std::string &reason) {
    std::lock_guard<std::mutex> lock(g_device_mutex);
    if (!g_device_queried) {
        DeviceInfo info = {0, 0, 0, 0, {0}, false};
        if (query_device(info, reason)) {
            g_device_info = info;
            g_device_queried = true;
        }
    }
    return g_device_info;
}

DeviceInfo target_device() {
    std::string reason;
    return target_device(reason);
}

// Whether device 0 can run this build's GPU cores. When it cannot, `error` is the message of a
// device=gpu create: what is missing and that device="cpu" (or device="auto") runs on the CPU core.
// The checks: a CUDA device (driver library, cuInit, device 0); a driver that supports the CUDA
// major version the cores were compiled with (their build-info field cuda=); a compute capability
// the cores carry SASS or PTX for (their build-info field cuda_arch=, gpu_arch.h). A field the
// core does not report skips its check.
bool gpu_usable(std::string &error) {
    std::string reason;
    const DeviceInfo device = target_device(reason);
    if (!device.valid) {
        error = "eldpc: no usable CUDA device for device=gpu: " + reason + "; " +
                eldpc::cpu_core_hint();
        return false;
    }
    const char *core_info = (g_default_core != nullptr && g_default_core->build_info != nullptr)
                                ? g_default_core->build_info()
                                : nullptr;
    const std::string info = core_info != nullptr ? core_info : "";
    // cuda=NVIDIA 13.3.73: the toolkit's major version is the one the driver must support.
    const std::string cuda = eldpc::build_info_field(info, "cuda");
    const size_t space = cuda.rfind(' ');
    const std::string cuda_version = space == std::string::npos ? cuda : cuda.substr(space + 1);
    const int cuda_major = eldpc::leading_decimal(cuda_version);
    if (cuda_major > 0 && device.driver_version > 0 && device.driver_version / 1000 < cuda_major) {
        const int v = device.driver_version;
        error = "eldpc: the NVIDIA driver supports CUDA " + std::to_string(v / 1000) + "." +
                std::to_string((v % 1000) / 10) +
                ", but the GPU cores of this build were compiled with CUDA " + cuda_version +
                " and need a driver that supports CUDA " + std::to_string(cuda_major) +
                ".x; update the NVIDIA driver, or " + eldpc::cpu_core_hint();
        return false;
    }
    const std::vector<eldpc::CudaImage> images =
        eldpc::parse_cuda_arch(eldpc::build_info_field(info, "cuda_arch"));
    if (!images.empty() && !eldpc::device_supported(images, device.cc_major, device.cc_minor)) {
        error = eldpc::unsupported_device_message(device.name, device.cc_major, device.cc_minor, images);
        return false;
    }
    return true;
}

// The variant this create gets: ELDPC_DEVICE_CLASS when set (read per create, so one process can
// hold handles of both variants), else the device-class table of core_variant.h.
bool select_variant(eldpc::Variant &variant, std::string &error) {
    const char *force = std::getenv("ELDPC_DEVICE_CLASS");
    if (force != nullptr && force[0] != '\0') {
        if (std::strcmp(force, "default") == 0) {
            variant = eldpc::Variant::kDefault;
            return true;
        }
        if (std::strcmp(force, "ada64") == 0) {
            variant = eldpc::Variant::kAda64;
            return true;
        }
        error = "eldpc: ELDPC_DEVICE_CLASS must be \"default\" or \"ada64\" (got \"" +
                std::string(force) + "\")";
        return false;
    }
    const DeviceInfo device = target_device();
    variant = device.valid
                  ? eldpc::variant_for_device(device.cc_major, device.cc_minor, device.sm_count)
                  : eldpc::Variant::kDefault;
    return true;
}

// ---- the front's own config key: device=gpu|cpu|auto ----------------------------------------

bool is_ascii_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

std::string strip(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && is_ascii_space(s[a])) ++a;
    while (b > a && is_ascii_space(s[b - 1])) --b;
    return s.substr(a, b - a);
}

enum class DeviceRequest { kGpu, kCpu, kAuto };

// Read and remove the front's own config key device=gpu|cpu|auto. Every other segment is passed on
// to the core unchanged, in order, so the core parses exactly the caller's string minus that one
// key (the core never sees "device" and would reject it as unknown). A missing key means
// device=gpu. Returns false, with a message, for a duplicate key or another value.
// `cpu_rest` is `rest`; `gpu_rest` is `rest` without the CPU core's keys threads= and cpus=, the
// string a device=auto create that chose the GPU passes on (those keys apply to the CPU core only).
bool split_device(const char *config, DeviceRequest &request, std::string &rest, std::string &gpu_rest,
                  std::string &error) {
    request = DeviceRequest::kGpu;
    rest.clear();
    gpu_rest.clear();
    const std::string s(config != nullptr ? config : "");
    bool seen = false;
    size_t pos = 0;
    while (pos <= s.size()) {
        const size_t semi = s.find(';', pos);
        const std::string seg = strip(s.substr(pos, semi == std::string::npos ? std::string::npos : semi - pos));
        pos = (semi == std::string::npos) ? s.size() + 1 : semi + 1;
        if (seg.empty()) continue;  // empty segments are ignored by the cores too
        const size_t eq = seg.find('=');
        const std::string key = eq == std::string::npos ? seg : strip(seg.substr(0, eq));
        if (key == "device") {
            if (seen) {
                error = "eldpc: duplicate config key \"device\"";
                return false;
            }
            seen = true;
            const std::string value = eq == std::string::npos ? std::string() : strip(seg.substr(eq + 1));
            if (value == "gpu") {
                request = DeviceRequest::kGpu;
            } else if (value == "cpu") {
                request = DeviceRequest::kCpu;
            } else if (value == "auto") {
                request = DeviceRequest::kAuto;
            } else {
                error = "eldpc: device must be \"gpu\", \"cpu\" or \"auto\" (got \"" + value + "\")";
                return false;
            }
            continue;  // removed from the string the core sees
        }
        if (!rest.empty()) rest += ";";
        rest += seg;
        if (key == "threads" || key == "cpus") continue;
        if (!gpu_rest.empty()) gpu_rest += ";";
        gpu_rest += seg;
    }
    return true;
}

// ---- preset names in config strings -----------------------------------------------------------

// The package's presets (python/eldpc/_config.py, presets()) as ABI strings, keys in sorted order:
// the strings the Python package passes for them (tests/test_config_presets.py checks the two
// tables against each other). Each min-sum BP + OSD-0 preset has a ".bp" variant (decoder=bp).
struct Preset {
    const char *name;
    const char *config;
};

const Preset kPresets[] = {
    {"ms30",
     "bp_method=ms;decoder=bposd;max_iter=30;ms_scaling_factor=0.625;osd_method=osd0;osd_order=0;schedule=parallel"},
    {"ms100",
     "bp_method=ms;decoder=bposd;max_iter=100;ms_scaling_factor=0.625;osd_method=osd0;osd_order=0;schedule=parallel"},
    {"ms30a075",
     "bp_method=ms;decoder=bposd;max_iter=30;ms_scaling_factor=0.75;osd_method=osd0;osd_order=0;schedule=parallel"},
    {"a05ms31",
     "bp_method=ms;decoder=bposd;max_iter=31;ms_scaling_factor=0.5;osd_method=osd0;osd_order=0;schedule=parallel"},
    {"a05c15",
     "bp_method=ms;decoder=bposd;max_iter=15;ms_scaling_factor=0.5;osd_method=osd0;osd_order=0;schedule=parallel"},
    {"a05c11",
     "bp_method=ms;decoder=bposd;max_iter=11;ms_scaling_factor=0.5;osd_method=osd0;osd_order=0;schedule=parallel"},
    {"a0375c7",
     "bp_method=ms;decoder=bposd;max_iter=7;ms_scaling_factor=0.375;osd_method=osd0;osd_order=0;schedule=parallel"},
    {"a04375c3",
     "bp_method=ms;decoder=bposd;max_iter=3;ms_scaling_factor=0.4375;osd_method=osd0;osd_order=0;schedule=parallel"},
    {"a03125c3",
     "bp_method=ms;decoder=bposd;max_iter=3;ms_scaling_factor=0.3125;osd_method=osd0;osd_order=0;schedule=parallel"},
    {"ms30.bp", "bp_method=ms;decoder=bp;max_iter=30;ms_scaling_factor=0.625;schedule=parallel"},
    {"ms100.bp", "bp_method=ms;decoder=bp;max_iter=100;ms_scaling_factor=0.625;schedule=parallel"},
    {"ms30a075.bp", "bp_method=ms;decoder=bp;max_iter=30;ms_scaling_factor=0.75;schedule=parallel"},
    {"a05ms31.bp", "bp_method=ms;decoder=bp;max_iter=31;ms_scaling_factor=0.5;schedule=parallel"},
    {"a05c15.bp", "bp_method=ms;decoder=bp;max_iter=15;ms_scaling_factor=0.5;schedule=parallel"},
    {"a05c11.bp", "bp_method=ms;decoder=bp;max_iter=11;ms_scaling_factor=0.5;schedule=parallel"},
    {"a0375c7.bp", "bp_method=ms;decoder=bp;max_iter=7;ms_scaling_factor=0.375;schedule=parallel"},
    {"a04375c3.bp", "bp_method=ms;decoder=bp;max_iter=3;ms_scaling_factor=0.4375;schedule=parallel"},
    {"a03125c3.bp", "bp_method=ms;decoder=bp;max_iter=3;ms_scaling_factor=0.3125;schedule=parallel"},
};

// A config string may start with a preset name, alone or followed by extra key=value segments:
// "a03125c3", "a03125c3;threads=8;device=cpu", "ms30;max_iter=50". The name is replaced by the
// preset's segments; an extra segment whose key the preset has replaces that segment (the extra
// value wins), every other extra segment is appended in order, and a key repeated among the extra
// segments reaches the core twice, which rejects it as a duplicate. A string whose first non-empty
// segment holds a '=' (every ABI string) is returned unchanged, byte for byte. Returns false, with
// a message, for a first segment without '=' that names no preset.
bool expand_preset(const char *config, std::string &out, std::string &error) {
    const std::string s(config != nullptr ? config : "");
    std::vector<std::string> segs;
    size_t pos = 0;
    while (pos <= s.size()) {
        const size_t semi = s.find(';', pos);
        const std::string seg = strip(s.substr(pos, semi == std::string::npos ? std::string::npos : semi - pos));
        pos = (semi == std::string::npos) ? s.size() + 1 : semi + 1;
        if (!seg.empty()) segs.push_back(seg);
    }
    if (segs.empty() || segs[0].find('=') != std::string::npos) {
        out = s;
        return true;
    }
    const Preset *preset = nullptr;
    for (const Preset &p : kPresets) {
        if (segs[0] == p.name) preset = &p;
    }
    if (preset == nullptr) {
        std::string names;
        for (const Preset &p : kPresets) names += std::string(names.empty() ? "" : ", ") + p.name;
        error = "eldpc: unknown preset \"" + segs[0] + "\" at the start of the config string (a config " +
                "string is key=value segments, optionally after a preset name; presets: " + names + ")";
        return false;
    }
    std::vector<std::string> keys, values;
    std::vector<bool> replaced;
    const std::string base(preset->config);
    pos = 0;
    while (pos <= base.size()) {
        const size_t semi = base.find(';', pos);
        const std::string seg = base.substr(pos, semi == std::string::npos ? std::string::npos : semi - pos);
        pos = (semi == std::string::npos) ? base.size() + 1 : semi + 1;
        keys.push_back(seg.substr(0, seg.find('=')));
        values.push_back(seg);
        replaced.push_back(false);
    }
    for (size_t i = 1; i < segs.size(); ++i) {
        const size_t eq = segs[i].find('=');
        const std::string key = eq == std::string::npos ? segs[i] : strip(segs[i].substr(0, eq));
        bool done = false;
        for (size_t j = 0; j < keys.size() && !done; ++j) {
            if (keys[j] == key && !replaced[j]) {
                values[j] = segs[i];
                replaced[j] = true;
                done = true;
            }
        }
        if (!done) {  // a new key, or a repeated one (which the core then reports as a duplicate)
            keys.push_back(key);
            values.push_back(segs[i]);
            replaced.push_back(true);
        }
    }
    out.clear();
    for (const std::string &v : values) out += (out.empty() ? "" : ";") + v;
    return true;
}

void init_front() {
    const char *force = std::getenv("ELDPC_ISA");  // read once, at load
    Isa isa = Isa::kAvx2;
    if (force != nullptr && force[0] != '\0') {
        if (std::strcmp(force, "avx2") == 0) {
            if (!cpu_has_avx2_os()) {
                set_load_error("eldpc: ELDPC_ISA=avx2 was requested, but this CPU and OS do not "
                               "support AVX2 and the OS's YMM state; " + host_isa_text());
                return;
            }
            isa = Isa::kAvx2;
        } else if (std::strcmp(force, "avx512") == 0) {
            if (!cpu_has_avx512_os()) {
                set_load_error("eldpc: ELDPC_ISA=avx512 was requested, but this CPU and OS do not "
                               "support AVX2 with AVX-512F, AVX-512VL, AVX-512DQ, AVX-512BW and the "
                               "OS's opmask/ZMM state; " + host_isa_text());
                return;
            }
            isa = Isa::kAvx512;
        } else {
            set_load_error("eldpc: ELDPC_ISA must be avx2 or avx512 (got \"" + std::string(force) + "\"); " +
                           host_isa_text());
            return;
        }
    } else if (cpu_has_avx512_os()) {
        isa = Isa::kAvx512;
    } else if (cpu_has_avx2_os()) {
        isa = Isa::kAvx2;
    } else {
        set_load_error("eldpc: this CPU and OS support neither AVX2 (with the OS's YMM state) nor "
                       "AVX-512F/VL/DQ/BW (with the OS's opmask/ZMM state); libeldpc has no core to run");
        return;
    }
    g_isa = isa;
    g_isa_valid = true;
    // The process's core: in a normal build the default GPU core of the chosen ISA, which
    // eldpc_isa() and eldpc_build_info() report and a GPU create needs; the ada64 core of the same
    // ISA and the CPU core load on first use. In a CPU-only build (no GPU cores were built) the
    // CPU core is the process's core: it reports the ISA and the build info, and a GPU create fails
    // with the CPU-only message below.
    std::string error;
#if defined(ELDPC_CPU_ONLY)
    CoreApi &slot = g_cpu_cores[isa_index(isa)];
    if (!load_core(slot, isa, cpu_core_file_name(isa), std::string(isa_name(isa)) + " CPU core", false,
                   error)) {
        set_load_error(error);
        return;
    }
#else
    CoreApi &slot = g_cores[isa_index(isa)][variant_index(eldpc::Variant::kDefault)];
    if (!load_core(slot, isa, core_file_name(isa, eldpc::Variant::kDefault), std::string(isa_name(isa)) + " core",
                   true, error)) {
        set_load_error(error);
        return;
    }
#endif
    g_default_core = &slot;
    const char *core_info = slot.build_info != nullptr ? slot.build_info() : nullptr;
    std::snprintf(g_build_info, sizeof(g_build_info),
                  "eldpc %s; front=%s; host_isa=%s; core=%s", ELDPC_VERSION, ELDPC_FRONT_COMPILER,
                  slot.isa, core_info != nullptr ? core_info : "unknown");
}

__attribute__((constructor)) void eldpc_front_constructor() { init_front(); }

}  // namespace

// The C API's opaque handle: the core's decoder, the core it came from, and the variant it was
// created with. Keeping the core next to the handle lets handles of both devices, both GPU
// variants and both ISAs (one per process) be dispatched to the right entry points.
struct eldpc_decoder {
    dbench_decoder *core;
    const CoreApi *api;
    const char *variant;  // "default", "ada64" or "cpu"
    bool cpu;             // true for a handle of the CPU core
};

extern "C" {

uint32_t eldpc_api_version(void) { return ELDPC_API_VERSION; }

const char *eldpc_version(void) { return ELDPC_VERSION; }

const char *eldpc_build_info(void) {
    if (g_default_core != nullptr) return g_build_info;
    return g_load_error_set ? g_load_error : "eldpc: no core loaded";
}

const char *eldpc_isa(void) { return g_default_core != nullptr ? g_default_core->isa : ""; }

const char *eldpc_core_variant(const eldpc_decoder *h) {
    return h != nullptr ? h->variant : "";
}

const char *eldpc_core_variant_for_device(int cc_major, int cc_minor, int sm_count) {
    return eldpc::variant_name(eldpc::variant_for_device(cc_major, cc_minor, sm_count));
}

eldpc_decoder *eldpc_create(const char *config, const eldpc_problem *p, char *err, size_t err_len) {
    if (g_default_core == nullptr || !g_isa_valid) {
        if (err != nullptr && err_len > 0) std::snprintf(err, err_len, "%s", eldpc_build_info());
        return nullptr;
    }
    if (p == nullptr) {
        if (err != nullptr && err_len > 0) std::snprintf(err, err_len, "eldpc_create: problem is NULL");
        return nullptr;
    }
    // device=gpu|cpu|auto is this front's key: read it here and pass the rest of the string to the
    // core.
    DeviceRequest request = DeviceRequest::kGpu;
    std::string core_config;
    std::string gpu_config;
    std::string error;
    // A leading preset name ("a03125c3;threads=8") is expanded to its key=value segments first.
    std::string expanded;
    if (!expand_preset(config, expanded, error)) {
        if (err != nullptr && err_len > 0) std::snprintf(err, err_len, "%s", error.c_str());
        return nullptr;
    }
    if (!split_device(expanded.c_str(), request, core_config, gpu_config, error)) {
        if (err != nullptr && err_len > 0) std::snprintf(err, err_len, "%s", error.c_str());
        return nullptr;
    }
    bool cpu = request == DeviceRequest::kCpu;
    if (request == DeviceRequest::kAuto) {
        // device=auto: the GPU when device 0 is usable by this build's GPU cores, else the CPU core.
        // The C API has no warning channel: eldpc_core_variant() reports the choice ("cpu").
#if defined(ELDPC_CPU_ONLY)
        cpu = true;
#else
        std::string why;
        cpu = !gpu_usable(why);
        if (!cpu) core_config = gpu_config;
#endif
    }
    eldpc::Variant variant = eldpc::Variant::kDefault;
    const CoreApi *api = nullptr;
    if (cpu) {
        // The CPU path: no CUDA call at all (no device query, no driver or runtime call).
        api = cpu_core_for(g_isa, error);
        if (api == nullptr) {
            if (err != nullptr && err_len > 0) std::snprintf(err, err_len, "%s", error.c_str());
            return nullptr;
        }
    } else {
#if defined(ELDPC_CPU_ONLY)
        error = std::string("eldpc: device=gpu is not available: this build has no GPU cores (built "
                            "without a CUDA toolkit); ") +
                eldpc::cpu_core_hint() +
                "; for the GPU cores, install the CUDA 13 toolkit "
                "(https://developer.nvidia.com/cuda-downloads), put its nvcc on PATH or set CUDACXX, "
                "and rebuild eldpc (pip install .)";
        if (err != nullptr && err_len > 0) std::snprintf(err, err_len, "%s", error.c_str());
        return nullptr;
#endif
        // The GPU path needs a usable CUDA device: a driver and a device, a driver new enough for
        // the cores' CUDA version, and a compute capability the cores carry an image for. When one
        // is missing, say which and name the CPU core instead of passing the driver's own message
        // through alone.
        if (!gpu_usable(error)) {
            if (err != nullptr && err_len > 0) std::snprintf(err, err_len, "%s", error.c_str());
            return nullptr;
        }
        if (!select_variant(variant, error)) {
            if (err != nullptr && err_len > 0) std::snprintf(err, err_len, "%s", error.c_str());
            return nullptr;
        }
        api = core_for(g_isa, variant, error);
        if (api == nullptr) {
            if (err != nullptr && err_len > 0) std::snprintf(err, err_len, "%s", error.c_str());
            return nullptr;
        }
    }
    dbench_problem dp;
    dp.abi_version = p->abi_version;
    dp.num_detectors = p->num_detectors;
    dp.num_mechanisms = p->num_mechanisms;
    dp.num_observables = p->num_observables;
    dp.h_colptr = p->h_colptr;
    dp.h_rowind = p->h_rowind;
    dp.o_colptr = p->o_colptr;
    dp.o_rowind = p->o_rowind;
    dp.priors = p->priors;
    dp.dem_text = p->dem_text;
    dbench_decoder *core = api->create(core_config.c_str(), &dp, err, err_len);
    if (core == nullptr) return nullptr;
    eldpc_decoder *h = new (std::nothrow) eldpc_decoder;
    if (h == nullptr) {
        api->destroy(core);
        if (err != nullptr && err_len > 0) std::snprintf(err, err_len, "eldpc_create: out of memory");
        return nullptr;
    }
    h->core = core;
    h->api = api;
    h->cpu = cpu;
    h->variant = cpu ? "cpu" : eldpc::variant_name(variant);
    return h;
}

int eldpc_set_priors(eldpc_decoder *h, const double *llr, uint32_t n) {
    if (h == nullptr || h->api == nullptr) return ELDPC_EINVAL;
    return h->api->set_priors(h->core, llr, n);
}

int eldpc_decode(eldpc_decoder *h, const uint8_t *syndrome, uint8_t *out_correction, uint8_t *out_obs,
                 uint32_t *out_status) {
    if (h == nullptr || h->api == nullptr) return ELDPC_EINVAL;
    return h->api->decode(h->core, syndrome, out_correction, out_obs, out_status);
}

int eldpc_decode_batch(eldpc_decoder *h, uint32_t num_shots, const uint8_t *syndromes, uint8_t *out_corrections,
                       uint8_t *out_obs, uint32_t *out_status) {
    if (h == nullptr || h->api == nullptr) return ELDPC_EINVAL;
    return h->api->decode_batch(h->core, num_shots, syndromes, out_corrections, out_obs, out_status);
}

int eldpc_get_soft_output(eldpc_decoder *h, double *out_llr, uint32_t n) {
    if (h == nullptr || h->api == nullptr) return ELDPC_EINVAL;
    return h->api->get_soft_output(h->core, out_llr, n);
}

int eldpc_keep_warm(eldpc_decoder *h, uint32_t flags) {
    if (h == nullptr || h->api == nullptr) return ELDPC_EINVAL;
    // The CPU core has no GPU stream and runs no idle device work: keeping it warm is a no-op.
    if (h->cpu) return ELDPC_OK;
    return h->api->keep_warm(h->core, flags);
}

void eldpc_destroy(eldpc_decoder *h) {
    if (h == nullptr || h->api == nullptr) return;
    h->api->destroy(h->core);
    delete h;
}

}  // extern "C"
