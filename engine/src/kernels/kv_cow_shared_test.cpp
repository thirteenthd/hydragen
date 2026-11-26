// src/kernels/kv_cow_shared_test.cpp - the §235 shared-region tail-slot pin law (GPU, synthetic, no model).
//
// Three streamed maps share one arena and one shared region at its far end: A and B are the SAME
// lineage (same conv: one hint tab, peer joins legal), C is a different lineage (different tab).
// They decode OVERLAPPING block sets while the region stays far smaller than the joint working
// set, and each map periodically detaches (cow_release + registry conv=-1 + range-clean + reset:
// exactly qsa_cow_state_release, hints surviving: the lineage lives on) and is re-admitted. That
// is the §235 failure shape (probe222x shared=2048, the region rolling): a detach frees region
// slots, another lineage re-arms them under new blocks, and the detached lineage's surviving hints
// dangle at slots whose bytes just moved. Pre-fix, the re-arm cleared the old block's hint in the
// CLAIMER's tab (the dangling hint survived in the owner's tab and the owner adopted foreign KV),
// and a blind atomicAdd could bump a free (0) or in-flight (-2) slot_use. Post-fix the re-arm
// kills the hint in the PREVIOUS OWNER's tab (sconv) and every join pin is a live-only CAS with a
// full sblk/sconv revalidation.
//
// After every round the test asserts:
//   1. CONTENT: every block any table names holds exactly the host-copy bytes (0 cross-lineage
//      corruption - the §235 symptom);
//   2. PIN LAW: slot_use[s] == (tables naming s) + (1 if s sits in a zombie range: an ended owner's
//      seed still under a live sibling's pin), exactly - no leaks, no underflow, no -2 left after
//      the copies, no slot named while free;
//   3. HINTS: no hint in any tab names a region slot that does not name it back (sblk/sconv
//      revalidate), and no claim is stuck at -2 after the copies;
//   4. LIVE: the region served fetches (sh_ctr > 0), joins happened (adoption or peer join), the
//      region actually rolled (sblk re-armed), and nothing overflowed.
// Phase 2 launches all three maps' resolve+copy on separate streams so the claim-vs-adopt race is
// genuinely concurrent. A final full detach + zombie sweep must return every refcount to zero.
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
}  // namespace

int main() {
    std::printf("kv_cow_shared_test: rolling shared region, three maps (two one lineage), §235 pin law\n");
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t H = s.n_head_kv, D = s.head_dim, PS = s.page_size;
    const int64_t n_blocks = 600;                 // blocks of content in the host copy
    const int64_t n_win = 1024;                   // per-map window (>= the resolve block: the host wrapper guards this)
    const int64_t n_shared = 16;                  // the region: far smaller than the joint working set -> it rolls
    const int NM = 3;                             // A, B (same lineage, conv 5), C (conv 6)
    const int64_t shared_base = NM * n_win;       // region anchored at the arena's far end
    const int64_t n_arena = NM * n_win + n_shared;
    const size_t GS = (size_t) (D / k::KV_Q8_GROUP);

    // Host copy: canonical bytes for every block (the copy kernel is a pure memcpy of these).
    k::KvHostPools host{};
    const size_t hrows = (size_t) n_blocks * H * PS, arows = (size_t) n_arena * H * PS;
    {
        void *hk, *hv, *hks, *hvs;
        ck(cudaHostAlloc(&hk, hrows * D, cudaHostAllocMapped), "host k");
        ck(cudaHostAlloc(&hv, hrows * D, cudaHostAllocMapped), "host v");
        ck(cudaHostAlloc(&hks, hrows * GS * 2, cudaHostAllocMapped), "host ks");
        ck(cudaHostAlloc(&hvs, hrows * GS * 2, cudaHostAllocMapped), "host vs");
        host.k_q = (int8_t*) hk; host.v_q = (int8_t*) hv;
        host.k_scale = (uint16_t*) hks; host.v_scale = (uint16_t*) hvs;
    }
    {
        std::mt19937 rng(235);
        std::uniform_int_distribution<int> byte_d(1, 255);   // never all-zero: a zero-filled slot must fail the check
        auto* hq = (int8_t*) host.k_q; auto* hv = (int8_t*) host.v_q;
        auto* hks = (uint16_t*) host.k_scale; auto* hvs = (uint16_t*) host.v_scale;
        for (size_t i = 0; i < hrows * D; ++i) { hq[i] = (int8_t) byte_d(rng); hv[i] = (int8_t) byte_d(rng); }
        for (size_t i = 0; i < hrows * GS; ++i) { hks[i] = (uint16_t) byte_d(rng); hvs[i] = (uint16_t) byte_d(rng); }
    }

    // The arena: one pool set for all slots; slot_use spans the whole arena.
    k::QsaAttnPools arena{};
    int8_t *ark = nullptr, *arv = nullptr; uint16_t *arks = nullptr, *arvs = nullptr;
    ck(cudaMalloc(&ark, arows * D), "arena k");
    ck(cudaMalloc(&arv, arows * D), "arena v");
    ck(cudaMalloc(&arks, arows * GS * 2), "arena ks");
    ck(cudaMalloc(&arvs, arows * GS * 2), "arena vs");
    ck(cudaMemset(ark, 0, arows * D), "arena k0");
    ck(cudaMemset(arv, 0, arows * D), "arena v0");
    ck(cudaMemset(arks, 0, arows * GS * 2), "arena ks0");
    ck(cudaMemset(arvs, 0, arows * GS * 2), "arena vs0");
    arena.k_q = ark; arena.v_q = arv; arena.k_scale = arks; arena.v_scale = arvs;
    int32_t* slot_use = nullptr;
    ck(cudaMalloc(&slot_use, n_arena * 4), "use");
    ck(cudaMemset(slot_use, 0, n_arena * 4), "use0");
    auto seed_range = [&](int64_t base) {   // cow-m1b: the owner holds its own range: use = 1 across it
        std::vector<int32_t> u((size_t) n_win, 1);
        ck(cudaMemcpy(slot_use + base, u.data(), (size_t) n_win * 4, cudaMemcpyHostToDevice), "seed");
    };

    // Region arrays (per arena, shared by all maps) + hint tabs + counters.
    auto dalloc = [](size_t n, bool ones) {
        int32_t* p = nullptr;
        ck(cudaMalloc(&p, n * 4), "alloc");
        ck(cudaMemset(p, ones ? 0xFF : 0, n * 4), "init");
        return p;
    };
    int32_t *sblk = dalloc(n_shared, true), *sst = dalloc(n_shared, true),
            *srf = dalloc(n_shared, false), *sconv = dalloc(n_shared, true);
    int32_t* lin_all = nullptr;
    ck(cudaMalloc(&lin_all, (size_t) k::kCowLinTabs * n_blocks * 4), "lin");
    ck(cudaMemset(lin_all, 0xFF, (size_t) k::kCowLinTabs * n_blocks * 4), "lin0");
    unsigned long long *join_ctr = nullptr, *sh_ctr = nullptr;
    ck(cudaMalloc(&join_ctr, 8), "jctr"); ck(cudaMemset(join_ctr, 0, 8), "jctr0");
    ck(cudaMalloc(&sh_ctr, 8), "sctr"); ck(cudaMemset(sh_ctr, 0, 8), "sctr0");

    // Three maps: windows [0,n_win), [n_win,2*n_win), [2*n_win,3*n_win); the same region.
    k::KvStreamMap M[NM];
    for (int li = 0; li < NM; ++li) {
        k::KvStreamMap& m = M[li];
        m.page_table = dalloc(n_blocks, true);
        m.slot_block = dalloc(n_win, true); m.slot_stamp = dalloc(n_win, true); m.slot_ref = dalloc(n_win, false);
        m.miss_block = dalloc(n_win, false); m.miss_slot = dalloc(n_win, false);
        m.miss_sh = dalloc(n_win, false); m.miss_sh_slot = dalloc(n_win, false);
        m.ctl = dalloc(k::kKvCtlInts, false);
        m.n_blocks = n_blocks; m.n_slots = n_win; m.slot_base = li * n_win;
        m.slot_use = slot_use;
        m.lin_all = lin_all; m.shared_base = (int32_t) shared_base; m.n_shared = (int32_t) n_shared;
        m.sblk = sblk; m.sst = sst; m.srf = srf; m.sconv = sconv;
        m.join_ctr = join_ctr; m.sh_ctr = sh_ctr;
        k::kv_stream_reset(m, nullptr);
    }

    // Registry: A and B conv 5 (one tab, peer joins legal), C conv 6 (its own tab: the cross-tab
    // re-arm is the §235 corruption shape). floor stays at max: the synthetic bytes are identical
    // per block across the "lineage" by construction; hhead generous (joins uncapped for the test).
    const int convs[NM] = {5, 5, 6};
    std::vector<k::CowPeer> regh(NM);
    for (int li = 0; li < NM; ++li)
        regh[li] = {convs[li], 64, 0, (int32_t) (li * n_win), (int32_t) n_win, (int32_t) n_blocks,
                    M[li].page_table, 0x7FFFFFFF};
    k::CowPeer* dreg = nullptr;
    ck(cudaMalloc(&dreg, NM * sizeof(k::CowPeer)), "reg");
    ck(cudaMemcpy(dreg, regh.data(), NM * sizeof(k::CowPeer), cudaMemcpyHostToDevice), "reg0");
    for (int li = 0; li < NM; ++li) { M[li].reg = dreg; M[li].reg_n = NM; }

    const int cap = 512;
    int32_t *ids[NM], *st[NM];
    cudaStream_t strm[NM];
    for (int li = 0; li < NM; ++li) {
        ids[li] = dalloc(cap, false); st[li] = dalloc(k::kStepCount, false);
        ck(cudaStreamCreateWithFlags(&strm[li], cudaStreamNonBlocking), "stream");
    }

    auto lo_of = [](int r, int li) {   // overlapping shifting windows over 24 blocks
        return 8 + 24 * ((r + 3 * li) % 20) + 7 * li;
    };
    auto launch = [&](int li, int lo, bool concurrent) {
        std::vector<int32_t> h(cap, 0);
        for (int i = 0; i < 96; ++i) h[i] = (int32_t) (4 * lo + i);
        ck(cudaMemcpy(ids[li], h.data(), 96 * 4, cudaMemcpyHostToDevice), "ids");
        const int32_t hst[k::kStepCount] = {0, 4 * (lo + 24), lo + 24, 96};
        ck(cudaMemcpy(st[li], hst, sizeof hst, cudaMemcpyHostToDevice), "st");
        k::QsaAttnPools ap = arena; ap.page_table = M[li].page_table;
        k::kv_stream_resolve(M[li], ap, host, k::kKvInt8, ids[li], st[li], 1, cap, s, concurrent ? strm[li] : nullptr);
    };

    // The engine's release protocol (qsa_cow_state_release): the state's own foreign pins go first
    // (cow_release device atomics: region pins AND sibling-window pins), the registry entry resets
    // (conv=-1: no joins to or from an ended state), then the range turns free ONLY if clean (no
    // live sibling still names it), else it zombies: the owner's seed stays until the sibling ends.
    // The table resets; the conv's hint tab survives (the lineage lives on) - those hints are what
    // §235 corrupted.
    bool live[NM] = {true, true, true}, freed[NM] = {false, false, false};
    auto detach = [&](int li) {
        k::kv_stream_cow_release(M[li], nullptr);
        regh[li].conv = -1; regh[li].floor = 0x7FFFFFFF;
        ck(cudaMemcpy(dreg + li, &regh[li], sizeof(k::CowPeer), cudaMemcpyHostToDevice), "reg-");
        std::vector<int32_t> u((size_t) n_win);
        ck(cudaMemcpy(u.data(), slot_use + li * n_win, (size_t) n_win * 4, cudaMemcpyDeviceToHost), "rd");
        bool clean = true;
        for (const int32_t x : u) if (x > 1) clean = false;
        if (clean) { ck(cudaMemset(slot_use + li * n_win, 0, (size_t) n_win * 4), "free"); freed[li] = true; }
        k::kv_stream_reset(M[li], nullptr);
        live[li] = false;
    };
    auto settle = [&]() {   // the zombie sweep: an ended range turns free once the last sibling pin drops
        for (int li = 0; li < NM; ++li) {
            if (live[li] || freed[li]) continue;
            std::vector<int32_t> u((size_t) n_win);
            ck(cudaMemcpy(u.data(), slot_use + li * n_win, (size_t) n_win * 4, cudaMemcpyDeviceToHost), "rd");
            bool clean = true;
            for (const int32_t x : u) if (x > 1) clean = false;
            if (clean) { ck(cudaMemset(slot_use + li * n_win, 0, (size_t) n_win * 4), "free"); freed[li] = true; }
        }
    };
    auto readmit = [&](int li) {   // admission only from the FREE list (cow_free_push): a zombie range
                                    // is not free - the map stays detached until the sweep cleans it
        settle();
        if (!freed[li]) return;
        seed_range(li * n_win);
        freed[li] = false;
        regh[li].conv = convs[li];
        ck(cudaMemcpy(dreg + li, &regh[li], sizeof(k::CowPeer), cudaMemcpyHostToDevice), "reg+");
        live[li] = true;
    };
    for (int li = 0; li < NM; ++li) seed_range(li * n_win);

    std::vector<int32_t> pt(NM * n_blocks), use(n_arena), sbk(n_shared), scv(n_shared);
    std::vector<int32_t> lin((size_t) k::kCowLinTabs * n_blocks);
    std::vector<int8_t> ak, av; std::vector<uint16_t> aks, avs;
    ak.resize(arows * D); av.resize(arows * D);
    aks.resize(arows * GS); avs.resize(arows * GS);

    int bad = 0, checks = 0;
    long long re_arms = 0;
    std::vector<int> prev_sblk(n_shared, -1);
    auto fail = [&](const char* phase, const char* what) {
        if (bad++ < 5) std::fprintf(stderr, "  %s: %s\n", phase, what);
    };
    auto verify = [&](const char* phase) {
        for (int li = 0; li < NM; ++li)
            ck(cudaMemcpy(pt.data() + li * n_blocks, M[li].page_table, n_blocks * 4, cudaMemcpyDeviceToHost), "pt");
        ck(cudaMemcpy(use.data(), slot_use, n_arena * 4, cudaMemcpyDeviceToHost), "use");
        ck(cudaMemcpy(lin.data(), lin_all, lin.size() * 4, cudaMemcpyDeviceToHost), "lin");
        ck(cudaMemcpy(sbk.data(), sblk, n_shared * 4, cudaMemcpyDeviceToHost), "sblk");
        ck(cudaMemcpy(scv.data(), sconv, n_shared * 4, cudaMemcpyDeviceToHost), "sconv");
        ck(cudaMemcpy(ak.data(), arena.k_q, ak.size(), cudaMemcpyDeviceToHost), "ak");
        ck(cudaMemcpy(av.data(), arena.v_q, av.size(), cudaMemcpyDeviceToHost), "av");
        ck(cudaMemcpy(aks.data(), arena.k_scale, aks.size() * 2, cudaMemcpyDeviceToHost), "aks");
        ck(cudaMemcpy(avs.data(), arena.v_scale, avs.size() * 2, cudaMemcpyDeviceToHost), "avs");
        ++checks;
        for (int j = 0; j < n_shared; ++j) if (prev_sblk[j] >= 0 && sbk[j] != prev_sblk[j]) ++re_arms;
        for (int j = 0; j < n_shared; ++j) prev_sblk[j] = sbk[j];
        char msg[192];
        // 2a. no table entry sits at -2 after the copies; no refcount is -2 or negative
        for (int64_t b = 0; b < n_blocks; ++b)
            for (int li = 0; li < NM; ++li)
                if (pt[li * n_blocks + b] == -2) {
                    std::snprintf(msg, sizeof msg, "table entry stuck at -2 (map %d block %lld)", li, (long long) b);
                    fail(phase, msg); return;
                }
        for (int64_t sl = 0; sl < n_arena; ++sl)
            if (use[sl] < 0) {
                std::snprintf(msg, sizeof msg, "slot_use[%lld] = %d after the copies (claim stuck / underflow)",
                              (long long) sl, use[sl]);
                fail(phase, msg); return;
            }
        // 1. + 2b. content equality and the exact refcount invariant: tables naming the slot, plus
        // one for a zombie range (the ended owner's seed still under a live sibling's pin).
        std::vector<int> named(n_arena, 0), owner_named(n_arena, 0);
        for (int64_t b = 0; b < n_blocks; ++b)
            for (int li = 0; li < NM; ++li) {
                const int32_t sl = pt[li * n_blocks + b];
                if (sl >= 0) ++named[sl];
                if (sl >= 0 && sl < NM * n_win && li == (int) (sl / n_win)) owner_named[sl] = 1;
            }
        for (int64_t sl = 0; sl < n_arena; ++sl) {
            // The admission seed (cow-m1b: the owner holds its whole range, use = 1 across it) is
            // the owner's pin for its OWN table entry: a live-range slot the owner's table names
            // needs no extra; one it does not name still carries the seed. A zombie range keeps the
            // seed under a live sibling's pin. A freed range must be bare.
            int extra = 0;
            if (sl < NM * n_win) {
                const int li = (int) (sl / n_win);
                if (!live[li]) extra = freed[li] ? 0 : 1;
                else if (!owner_named[sl]) extra = 1;
            }
            if (use[sl] != named[sl] + extra) {
                std::snprintf(msg, sizeof msg, "PIN LAW: slot_use[%lld] = %d, %d tables name it (%s)",
                              (long long) sl, use[sl], named[sl],
                              sl < NM * n_win ? (live[sl / n_win] ? "live range" : (freed[sl / n_win] ? "freed range" : "zombie range")) : "region");
                fail(phase, msg); return;
            }
        }
        for (int64_t b = 0; b < n_blocks; ++b) {
            const size_t rb = (size_t) b * H * PS;
            for (int li = 0; li < NM; ++li) {
                const int32_t sl = pt[li * n_blocks + b];
                if (sl < 0) continue;
                const size_t rs = (size_t) sl * H * PS;
                if (std::memcmp(ak.data() + rs * D, ((int8_t*) host.k_q) + rb * D, H * PS * D) != 0 ||
                    std::memcmp(av.data() + rs * D, ((int8_t*) host.v_q) + rb * D, H * PS * D) != 0 ||
                    std::memcmp(aks.data() + rs * GS, ((uint16_t*) host.k_scale) + rb * GS, H * PS * GS * 2) != 0 ||
                    std::memcmp(avs.data() + rs * GS, ((uint16_t*) host.v_scale) + rb * GS, H * PS * GS * 2) != 0) {
                    std::snprintf(msg, sizeof msg, "CORRUPTION: map %d block %lld slot %d holds foreign bytes",
                                  li, (long long) b, sl);
                    fail(phase, msg); return;
                }
            }
        }
        // 3. no dangling hint: every hint names a region slot that names it back (sblk/sconv)
        for (int tab = 0; tab < k::kCowLinTabs; ++tab)
            for (int64_t b = 0; b < n_blocks; ++b) {
                const int32_t L = lin[(size_t) tab * n_blocks + b];
                if (L == -2) {
                    std::snprintf(msg, sizeof msg, "hint stuck at -2 (tab %d block %lld)", tab, (long long) b);
                    fail(phase, msg); return;
                }
                if (L < 0) continue;
                const int j = L - (int) shared_base;
                if (j < 0 || j >= (int) n_shared || sbk[j] != (int32_t) b || scv[j] != tab) {
                    std::snprintf(msg, sizeof msg, "DANGLING HINT: tab %d block %lld -> slot %d (sblk=%d sconv=%d)",
                                  tab, (long long) b, L, j >= 0 ? sbk[j] : -9, j >= 0 ? scv[j] : -9);
                    fail(phase, msg); return;
                }
            }
    };

    // Phase 1: sequential rounds (the §235 corruption was deterministic: wrong-tab clearing needs
    // no concurrency). Phase 2: all three maps' resolve+copy on separate streams (the pin-vs-claim
    // race genuinely in flight). Detach/re-admit cycles keep the region rolling; a detached map
    // never resolves (the engine never streams a dying state).
    for (int phase = 0; phase < 2; ++phase) {
        const int rounds = phase == 0 ? 48 : 32;
        for (int r = 0; r < rounds; ++r) {
            if (r % 6 == 0) readmit(2);
            if (r % 6 == 3) detach(0);
            if (r % 6 == 4) { detach(1); readmit(0); }
            if (r % 6 == 5) { detach(2); readmit(1); }
            for (int li = 0; li < NM; ++li)
                if (live[li]) launch(li, lo_of(r, li), phase == 1);
            ck(cudaDeviceSynchronize(), "round");
            settle();
            verify(phase == 0 ? "seq" : "conc");
            if (bad) break;
        }
        if (bad) break;
    }

    // Read the per-map ctl counters BEFORE the final detach (reset zeroes ctl).
    unsigned long long lookups = 0, fetches = 0; int ovf = 0;
    for (int li = 0; li < NM; ++li) {
        const k::KvStreamCounters c = k::kv_stream_counters(M[li]);
        lookups += c.lookups; fetches += c.misses; if (c.overflow) ovf = 1;
    }
    // Final settle: detach all three; the zombie sweep must return every refcount to zero (leak check).
    for (int li = 0; li < NM; ++li) detach(li);
    ck(cudaDeviceSynchronize(), "final");
    settle();
    ck(cudaMemcpy(use.data(), slot_use, n_arena * 4, cudaMemcpyDeviceToHost), "use");
    for (int64_t sl = 0; sl < n_arena; ++sl)
        if (use[sl] != 0) {
            std::fprintf(stderr, "  FINAL: slot_use[%lld] = %d after full detach (leak)\n", (long long) sl, use[sl]);
            ++bad; break;
        }

    unsigned long long ctr[2] = {0, 0};
    ck(cudaMemcpy(ctr, join_ctr, 8, cudaMemcpyDeviceToHost), "ctr");
    unsigned long long shc = 0;
    ck(cudaMemcpy(&shc, sh_ctr, 8, cudaMemcpyDeviceToHost), "shc");
    std::printf("  rounds checked: %d, joins: %llu, shared-region fetches: %llu, region re-arms: %lld, "
                "lookups: %llu, window fetches: %llu\n",
                checks, ctr[0], shc, re_arms, lookups, fetches);
    if (shc == 0) { std::fprintf(stderr, "  the shared region served no fetches: the test proved nothing\n"); ++bad; }
    if (ctr[0] == 0) { std::fprintf(stderr, "  no join/adoption ever happened: the pin path was never exercised\n"); ++bad; }
    if (re_arms < 8) { std::fprintf(stderr, "  the region barely rolled (%lld re-arms): the test proved nothing\n", re_arms); ++bad; }
    if (ovf) { std::fprintf(stderr, "  a resolve overflowed\n"); ++bad; }
    if (bad) ++g_fail;
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
