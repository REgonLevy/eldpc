/* SPDX-License-Identifier: Apache-2.0 */
/* core_variant.h -- the device-class table of the eldpc front.
 *
 * A decoder handle runs one of two builds of the decoder core: "default" (the build the package
 * has always shipped) or "ada64" (the same sources and the same flags, with DBENCH_GPU_SM_BUDGET=64
 * the only difference: the create path plans for at most 64 SMs, which is faster on the 142-SM Ada
 * parts and slower on smaller GPUs, so it is used only there).
 *
 * Which build a handle gets is decided at create time from the target device, through this table.
 * A row names a compute capability (major.minor) and an SM-count range; the first matching row
 * wins and a device that matches no row gets the default build. The table is the only place a
 * device class is named: changing a measured choice means editing (or adding) a row here and
 * rebuilding the package; the Python mirror in python/eldpc/_variant.py, which eldpc.devices()
 * uses to report the same value, has to be edited with it (a test compares the two).
 *
 * To name a device class whose SM-count range has an upper bound, set sm_max; 0 means "no upper
 * bound". Rows are tried in order, so put the narrower row first if ranges overlap.
 */
#ifndef ELDPC_CORE_VARIANT_H
#define ELDPC_CORE_VARIANT_H

namespace eldpc {

enum class Variant { kDefault = 0, kAda64 = 1 };

struct VariantRule {
    int cc_major;
    int cc_minor;
    int sm_min;  /* inclusive */
    int sm_max;  /* inclusive; 0 = no upper bound */
    Variant variant;
};

/* Compute capability 8.9 with at least 128 SMs is the Ada 142-SM class (RTX 6000 Ada, L40S: 142
 * SMs); every other device keeps the default build. */
inline constexpr VariantRule kDeviceClassTable[] = {
    {8, 9, 128, 0, Variant::kAda64},
};

/* The variant a handle created for this device gets from the table alone (no ELDPC_DEVICE_CLASS
 * override). Invalid device properties (negative, or no SMs) match no row and give the default. */
inline Variant variant_for_device(int cc_major, int cc_minor, int sm_count) {
    if (cc_major < 0 || cc_minor < 0 || sm_count <= 0) return Variant::kDefault;
    for (const VariantRule &rule : kDeviceClassTable) {
        if (rule.cc_major == cc_major && rule.cc_minor == cc_minor && sm_count >= rule.sm_min &&
            (rule.sm_max == 0 || sm_count <= rule.sm_max)) {
            return rule.variant;
        }
    }
    return Variant::kDefault;
}

inline const char *variant_name(Variant variant) {
    return variant == Variant::kAda64 ? "ada64" : "default";
}

}  // namespace eldpc

#endif /* ELDPC_CORE_VARIANT_H */
