#pragma once
// HydraGen fork: knobs introduced by this fork's chain read HG_* first and fall
// back to the legacy STRATA_* spelling. Upstream-derived knobs stay plain
// STRATA_* reads and never go through this helper.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>

inline const char* hg_getenv(const char* strata_name) {
    if (strata_name && std::strncmp(strata_name, "STRATA_", 7) == 0 &&
        std::strlen(strata_name) < 88) {
        char alt[96];
        std::snprintf(alt, sizeof alt, "HG_%s", strata_name + 7);
        if (const char* v = std::getenv(alt)) return v;
    }
    return std::getenv(strata_name);
}

inline std::optional<uint64_t> hg_env_u64(const char* strata_name) {
    const char* v = hg_getenv(strata_name);
    if (!v || !*v) return std::nullopt;
    char* end = nullptr;
    const unsigned long long n = std::strtoull(v, &end, 10);
    if (end == v) return std::nullopt;
    return n;
}
