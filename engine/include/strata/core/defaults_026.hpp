// include/strata/core/defaults_026.hpp - STRATA_DEFAULTS=026: the master switch of the Qwen engine's 0.2.7 defaults.
//
// Every default the Qwen engine (`strata`) changed since v0.2.6 goes back to what v0.2.6 did when STRATA_DEFAULTS=026 is in the
// environment (the app's "off" for its speed switches sends it): the wide
// hyper-connection up read (STRATA_GR_UP_WIDE), the stream-split hyper-connection read (STRATA_GR_V3), the int8 prompt attention
// (STRATA_PROMPT_ATTN_IMMA), the four gfx12 decode switches of the arch table, the PLE residency policy (STRATA_PLE_AUTO_RAM), and the
// gfx12 decode miss split of 0.15 (back to 0.55).  KVA is off by default in 0.2.7 and stays whatever STRATA_KVA says.
// The residency-biased routing (STRATA_ROUTE_PRIOR) was in this set in 0.2.7; it is off by default again (owner, 2026-10-29), so the master switch
// has nothing to take back there, and route_prior_default() answers 0 with the switch on or off.
// An explicit per-feature variable (STRATA_ROUTE_PRIOR=2, STRATA_GR_V3=1, STRATA_PLE_AUTO_RAM=1, ...) or flag (--pcie-frac, --ple-io)
// still wins over the master switch.  Each rule is a pure function that takes the answer of qwen_defaults_026() as its `v026`
// argument, so a test can check both settings without touching the environment (src/core/arch_defaults_test.cpp,
// src/core/ple_residency_test.cpp).
#pragma once

#include <cstdlib>
#include <cstring>

namespace strata {

/// "026", "0.2.6" and "v0.2.6" name version 0.2.6; anything else (unset, "", "0", "027") does not.
inline bool is_026_token(const char* v) {
    return v != nullptr && (std::strcmp(v, "026") == 0 || std::strcmp(v, "0.2.6") == 0 || std::strcmp(v, "v0.2.6") == 0);
}

/// STRATA_DEFAULTS=026 (read once).
inline bool qwen_defaults_026() {
    static const bool on = is_026_token(std::getenv("STRATA_DEFAULTS"));
    return on;
}

/// The decode miss split's start value (the share of a native pack's missing experts the CPU pool takes) when --pcie-frac is not given:
/// NVFP4 packs 0.25, other native packs 0.55 (the x16 share), and 0.15 on gfx12 (0.2.7) unless STRATA_PCIE_FRAC_DEFAULT=old or the master
/// switch is 026; a canonical pack 0.2.
inline double pcie_start_frac(bool nvfp4_pack, bool native_pack, bool gfx12_default, bool pf_old, bool v026 = false) {
    if (nvfp4_pack) return 0.25;
    if (!native_pack) return 0.2;
    return gfx12_default && !pf_old && !v026 ? 0.15 : 0.55;
}

/// Phase 3 Track 3C: the cap on that start share for a batched stage (--batch N, N > 0).  A batched stage
/// puts every row's misses on one link, so the share measured for a single stream over-fills it, and PcieAuto
/// never refines a batched stage (generate.cpp gates it on o.batch == 0) - the start value is the value.
/// Measured on the 12 GiB RTX 3080 Ti bench box (--batch 8, 3,711 expert slots, IQ3_XXS native pack,
/// 2026-11-25): 0.50 -> 0.25 lifted bench_quick N=1 64.9 -> 75.4, N=2 65.8 -> 73.8, N=4 64.3 -> 70.6,
/// N=8 59.7 -> 67.2 and the agentic fan-out share 63.9 -> 72.8 tok/s (+10..16%), with the VRAM hit% flat
/// (86.6 -> 88.0); 0.75 was worse everywhere (N=8 50.3) and 0.0 slightly below 0.25.  cap <= 0 or
/// batch <= 0 leaves the share alone (STRATA_PCIE_FRAC_BATCH_MAX=0), and an explicit --pcie-frac never
/// reaches this path.
inline double pcie_batch_frac_cap(double frac, int batch, double cap) {
    if (batch <= 0 || cap <= 0.0) return frac;
    return frac > cap ? cap : frac;
}

}  // namespace strata
