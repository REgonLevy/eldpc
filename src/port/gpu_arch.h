/* SPDX-License-Identifier: Apache-2.0 */
/* gpu_arch.h -- which GPUs the GPU cores of this build can run on, and the messages that say so.
 *
 * The GPU cores report their CUDA images in their build-info string, in the field
 * cuda_arch=sm_89,sm_120,compute_120 (CMakeLists.txt writes it from the same call that sets the
 * cores' CUDA_ARCHITECTURES): sm_XY is SASS for compute capability X.Y, compute_XY is PTX for the
 * virtual architecture X.Y. The front reads that field from the core it loads, so the check below
 * is the build's own list, not a second copy of it. python/eldpc/_auto.py mirrors these functions
 * for device="auto" (tests/test_device_auto.py compares the two).
 *
 * The rules are CUDA's: SASS for sm_XY runs on a device of compute capability X.Z with Z >= Y (an
 * arch-specific "a" image, sm_XYa, only on X.Y itself); PTX for compute_XY is compiled by the
 * driver for any device of compute capability X.Y or newer.
 *
 * Header-only, with no CUDA dependency, so that a test can compile it on its own.
 */
#ifndef ELDPC_GPU_ARCH_H
#define ELDPC_GPU_ARCH_H

#include <string>
#include <vector>

namespace eldpc {

/* The value of the leading decimal digits of `text` (13 for "13.3.73"), or 0 when it starts with
 * none; at most nine digits are read. Written out instead of std::atoi or strtol: built with g++
 * (which defines _GNU_SOURCE) against glibc 2.38 or newer, <cstdlib> redirects strtol to
 * __isoc23_strtol, a GLIBC_2.38 symbol, and libeldpc.so must load on glibc 2.34
 * (docs/SUPPORT.md; tests/test_library_deps.py checks the built libraries). */
inline int leading_decimal(const std::string &text) {
    int value = 0;
    for (size_t i = 0; i < text.size() && i < 9; ++i) {
        const char c = text[i];
        if (c < '0' || c > '9') break;
        value = value * 10 + (c - '0');
    }
    return value;
}

struct CudaImage {
    std::string name;    // as listed, e.g. "sm_89" or "compute_120"
    int major;
    int minor;
    bool ptx;            // compute_XY (PTX) rather than sm_XY (SASS)
    bool arch_specific;  // an "a" suffix: SASS for exactly this compute capability
};

/* The value of `key=` in a build-info string ("key=value; key=value"); "" when absent. */
inline std::string build_info_field(const std::string &info, const std::string &key) {
    const std::string needle = key + "=";
    size_t pos = 0;
    while ((pos = info.find(needle, pos)) != std::string::npos) {
        if (pos == 0 || info[pos - 1] == ' ' || info[pos - 1] == ';') {
            const size_t start = pos + needle.size();
            const size_t end = info.find(';', start);
            return info.substr(start, end == std::string::npos ? std::string::npos : end - start);
        }
        pos += needle.size();
    }
    return std::string();
}

/* "sm_89,sm_120,compute_120" -> the images. Entries of another form are skipped. */
inline std::vector<CudaImage> parse_cuda_arch(const std::string &list) {
    std::vector<CudaImage> out;
    size_t pos = 0;
    while (pos <= list.size()) {
        const size_t comma = list.find(',', pos);
        const size_t len = comma == std::string::npos ? std::string::npos : comma - pos;
        std::string item = list.substr(pos, len);
        pos = comma == std::string::npos ? list.size() + 1 : comma + 1;
        while (!item.empty() && item.front() == ' ') item.erase(item.begin());
        while (!item.empty() && item.back() == ' ') item.pop_back();
        bool ptx;
        size_t digits;
        if (item.compare(0, 3, "sm_") == 0) {
            ptx = false;
            digits = 3;
        } else if (item.compare(0, 8, "compute_") == 0) {
            ptx = true;
            digits = 8;
        } else {
            continue;
        }
        size_t end = digits;
        while (end < item.size() && item[end] >= '0' && item[end] <= '9') ++end;
        if (end - digits < 2) continue;
        const int number = leading_decimal(item.substr(digits, end - digits));
        const std::string suffix = item.substr(end);
        if (!suffix.empty() && suffix != "a" && suffix != "f") continue;
        out.push_back(CudaImage{item, number / 10, number % 10, ptx, suffix == "a"});
    }
    return out;
}

/* Whether a device of compute capability major.minor can run one of the images. */
inline bool device_supported(const std::vector<CudaImage> &images, int major, int minor) {
    for (const CudaImage &im : images) {
        if (im.ptx) {
            if (major > im.major || (major == im.major && minor >= im.minor)) return true;
        } else if (im.arch_specific) {
            if (major == im.major && minor == im.minor) return true;
        } else if (major == im.major && minor >= im.minor) {
            return true;
        }
    }
    return false;
}

/* "SASS for sm_89, sm_120 and PTX for compute_120 (compute capability 8.9; 12.0 or newer)":
 * the images and the compute capabilities they cover. A SASS image whose compute capability a PTX
 * image already covers is not listed again in the parenthesis. */
inline std::string images_text(const std::vector<CudaImage> &images) {
    std::string sass, ptx, covers;
    for (const CudaImage &im : images) {
        std::string &list = im.ptx ? ptx : sass;
        if (!list.empty()) list += ", ";
        list += im.name;
    }
    for (const CudaImage &im : images) {
        if (im.ptx) continue;
        bool by_ptx = false;
        for (const CudaImage &p : images) {
            if (p.ptx && (im.major > p.major || (im.major == p.major && im.minor >= p.minor))) {
                by_ptx = true;
            }
        }
        if (by_ptx) continue;
        if (!covers.empty()) covers += "; ";
        covers += std::to_string(im.major) + "." + std::to_string(im.minor);
    }
    for (const CudaImage &im : images) {
        if (!im.ptx) continue;
        if (!covers.empty()) covers += "; ";
        covers += std::to_string(im.major) + "." + std::to_string(im.minor) + " or newer";
    }
    std::string text;
    if (!sass.empty()) text += "SASS for " + sass;
    if (!ptx.empty()) text += std::string(text.empty() ? "" : " and ") + "PTX for " + ptx;
    if (text.empty()) return "no CUDA images";
    return text + " (compute capability " + covers + ")";
}

/* What a GPU create error tells the caller to do instead, in the words of every API: the message
 * reaches Python, C, the benchmark and the tuner alike. */
inline const char *cpu_core_hint() {
    return "to run on the CPU core, create the decoder with device=cpu, or with device=auto to use "
           "the GPU only when it is usable (Python: device=\"cpu\" or device=\"auto\"; C: device=cpu "
           "or device=auto in the config string)";
}

/* The create error for device=gpu on a device the build cannot run on. */
inline std::string unsupported_device_message(const std::string &device_name, int major, int minor,
                                              const std::vector<CudaImage> &images) {
    std::string who = "device 0";
    if (!device_name.empty()) who += " (" + device_name + ")";
    return "eldpc: " + who + " has compute capability " + std::to_string(major) + "." +
           std::to_string(minor) + ", which the GPU cores of this build do not support: they carry " +
           images_text(images) + "; " + cpu_core_hint();
}

}  // namespace eldpc

#endif  // ELDPC_GPU_ARCH_H
