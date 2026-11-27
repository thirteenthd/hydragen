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
//
// P5-5C.9 adds a second, independent arm (grouped_arm): kv_stream_resolve_groups() over three CowArena
// windows - the C>=2 batch-decode path whose copy kernel forgot the + slot_base offset, the page_table
// publish, and the aborted-victim guard (prompt-canary cross-talk at 9k/22k C>=4, never at C=1 or 3k).
// P5-5C.10 adds a third (restore_rewind_arm): an in-place conversation_checkpoint_restore over a
// CoW-forked main state - the restore must release the lineage (conv=-1: no joins-at-fetch for the
// new conversation) and reset the streamed page table, while the sibling slot's fork pins (slot_use
// > 1) and pages survive untouched (the 9k C=8 stream-7 <- stream-6 cross-talk shape).
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

// P5-5C.9 regression arm: kv_stream_resolve_groups() over THREE distinct CowArena windows (slot_base =
// 0, n_win, 2*n_win) sharing one arena and one slot_use array - the C>=2 batch-decode shape that
// corrupted main_ss. Pre-fix, copy_groups_kernel wrote at the LOCAL sl (no + slot_base): groups 1 and
// 2's tail KV bytes landed in group 0's range, overwriting its pinned forked pages (the prompt-canary
// cross-talk at 9k/22k C>=4), and it never published the entries resolve_body left at -2 (CLAIMED).
// Group 0 holds a pinned page and takes NO misses; groups 1 and 2 miss 40 blocks each from their OWN
// host copies (distinct bytes per slot). Asserts:
//   1. group 0's arena range [0, n_win) is bit-for-bit untouched (no cross-slot overwrite),
//   2. every missed block's page_table entry == sl + slot_base (never -2, never in [0, n_win)),
//   3. the bytes at (sl + slot_base) match that group's own host copy bit-for-bit.
int grouped_arm() {
    std::printf("  grouped arm: kv_stream_resolve_groups over three CowArena windows (P5-5C.9)\n");
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t H = s.n_head_kv, D = s.head_dim, PS = s.page_size;
    const int NG = 3;
    const int64_t n_blocks = 600, n_win = 1024;          // n_win >= the resolve block (the wrapper guards this)
    const int64_t n_arena = NG * n_win;
    const size_t GS = (size_t) (D / k::KV_Q8_GROUP);
    const size_t hrows = (size_t) n_blocks * H * PS, arows = (size_t) n_arena * H * PS;
    int bad = 0;

    // Distinct host copies per group (group 1's bytes must never look like group 2's).
    k::KvHostPools host[NG];
    for (int i = 0; i < NG; ++i) {
        void *hk, *hv, *hks, *hvs;
        ck(cudaHostAlloc(&hk, hrows * D, cudaHostAllocMapped), "gk host k");
        ck(cudaHostAlloc(&hv, hrows * D, cudaHostAllocMapped), "gk host v");
        ck(cudaHostAlloc(&hks, hrows * GS * 2, cudaHostAllocMapped), "gk host ks");
        ck(cudaHostAlloc(&hvs, hrows * GS * 2, cudaHostAllocMapped), "gk host vs");
        host[i].k_q = (int8_t*) hk; host[i].v_q = (int8_t*) hv;
        host[i].k_scale = (uint16_t*) hks; host[i].v_scale = (uint16_t*) hvs;
        std::mt19937 rng(909 + 17 * i);
        std::uniform_int_distribution<int> byte_d(1, 255);   // never all-zero
        auto* hq = (int8_t*) host[i].k_q; auto* hv2 = (int8_t*) host[i].v_q;
        auto* hks2 = (uint16_t*) host[i].k_scale; auto* hvs2 = (uint16_t*) host[i].v_scale;
        for (size_t j = 0; j < hrows * D; ++j) { hq[j] = (int8_t) byte_d(rng); hv2[j] = (int8_t) byte_d(rng); }
        for (size_t j = 0; j < hrows * GS; ++j) { hks2[j] = (uint16_t) byte_d(rng); hvs2[j] = (uint16_t) byte_d(rng); }
    }

    // One shared CowArena + slot_use; three maps with slot_base = i * n_win. No registry, no shared
    // region: the arm isolates the copy kernel's arena duties (offset, publish, victim guard).
    k::QsaAttnPools arena{};
    int8_t *ark = nullptr, *arv = nullptr; uint16_t *arks = nullptr, *arvs = nullptr;
    ck(cudaMalloc(&ark, arows * D), "gk arena k");
    ck(cudaMalloc(&arv, arows * D), "gk arena v");
    ck(cudaMalloc(&arks, arows * GS * 2), "gk arena ks");
    ck(cudaMalloc(&arvs, arows * GS * 2), "gk arena vs");
    ck(cudaMemset(ark, 0xA5, arows * D), "gk arena k0");      // canary fill: a zeroed or moved slot fails
    ck(cudaMemset(arv, 0xA5, arows * D), "gk arena v0");
    ck(cudaMemset(arks, 0xA5, arows * GS * 2), "gk arena ks0");
    ck(cudaMemset(arvs, 0xA5, arows * GS * 2), "gk arena vs0");
    arena.k_q = ark; arena.v_q = arv; arena.k_scale = arks; arena.v_scale = arvs;
    int32_t* slot_use = nullptr;
    ck(cudaMalloc(&slot_use, n_arena * 4), "gk use");
    ck(cudaMemset(slot_use, 0, n_arena * 4), "gk use0");
    auto dalloc = [](size_t n, bool ones) {
        int32_t* p = nullptr;
        ck(cudaMalloc(&p, n * 4), "gk alloc");
        ck(cudaMemset(p, ones ? 0xFF : 0, n * 4), "gk init");
        return p;
    };
    k::KvStreamMap M[NG];
    for (int i = 0; i < NG; ++i) {
        k::KvStreamMap& m = M[i];
        m.page_table = dalloc(n_blocks, true);
        m.slot_block = dalloc(n_win, true); m.slot_stamp = dalloc(n_win, true); m.slot_ref = dalloc(n_win, false);
        m.miss_block = dalloc(n_win, false); m.miss_slot = dalloc(n_win, false);
        m.miss_sh = dalloc(n_win, false); m.miss_sh_slot = dalloc(n_win, false);
        m.ctl = dalloc(k::kKvCtlInts, false);
        m.n_blocks = n_blocks; m.n_slots = n_win; m.slot_base = i * n_win;
        m.slot_use = slot_use;   // the CowArena shape: slot_use != nullptr -> resolve leaves entries at -2
        k::kv_stream_reset(m, nullptr);
        std::vector<int32_t> u((size_t) n_win, 1);   // the admission seed: the owner holds its range
        ck(cudaMemcpy(slot_use + i * n_win, u.data(), (size_t) n_win * 4, cudaMemcpyHostToDevice), "gk seed");
    }

    // Group 0's pinned forked page: block b0 resident in slot j0, its bytes = host[0]'s block b0.
    const int64_t b0 = 7, j0 = 5;
    {
        const size_t rb = (size_t) b0 * H * PS, rs = (size_t) j0 * H * PS;
        ck(cudaMemcpy(ark + rs * D, host[0].k_q + rb * D, H * PS * D, cudaMemcpyDeviceToDevice), "gk pin k");
        ck(cudaMemcpy(arv + rs * D, host[0].v_q + rb * D, H * PS * D, cudaMemcpyDeviceToDevice), "gk pin v");
        ck(cudaMemcpy(arks + rs * GS, host[0].k_scale + rb * GS, H * PS * GS * 2, cudaMemcpyDeviceToDevice), "gk pin ks");
        ck(cudaMemcpy(arvs + rs * GS, host[0].v_scale + rb * GS, H * PS * GS * 2, cudaMemcpyDeviceToDevice), "gk pin vs");
        const int32_t ptb = (int32_t) j0, sbb = (int32_t) b0;
        ck(cudaMemcpy(M[0].page_table + b0, &ptb, 4, cudaMemcpyHostToDevice), "gk pin pt");
        ck(cudaMemcpy(M[0].slot_block + j0, &sbb, 4, cudaMemcpyHostToDevice), "gk pin sb");
    }
    // Snapshot group 0's whole range: it must be bit-for-bit untouched after the grouped resolve.
    std::vector<int8_t> snap_k((size_t) n_win * H * PS * D), snap_v(snap_k.size());
    std::vector<uint16_t> snap_ks((size_t) n_win * H * PS * GS), snap_vs(snap_ks.size());
    ck(cudaMemcpy(snap_k.data(), ark, snap_k.size(), cudaMemcpyDeviceToHost), "gk snap k");
    ck(cudaMemcpy(snap_v.data(), arv, snap_v.size(), cudaMemcpyDeviceToHost), "gk snap v");
    ck(cudaMemcpy(snap_ks.data(), arks, snap_ks.size() * 2, cudaMemcpyDeviceToHost), "gk snap ks");
    ck(cudaMemcpy(snap_vs.data(), arvs, snap_vs.size() * 2, cudaMemcpyDeviceToHost), "gk snap vs");

    // One query per group: group 0 names only its resident block (zero misses); groups 1 and 2 miss
    // W blocks each from their own host copies.
    const int cap = 64, W = 40;
    std::vector<int32_t> ids((size_t) NG * cap, 0), steps((size_t) NG * k::kStepCount, 0);
    ids[0 * cap] = (int32_t) (PS * b0);
    steps[0 * k::kStepCount + 3] = 1;
    for (int i = 1; i < NG; ++i) {
        const int64_t bs = 100 + 200 * i;
        for (int t = 0; t < W; ++t) ids[i * cap + t] = (int32_t) (PS * (bs + t));
        steps[i * k::kStepCount + 3] = W;
    }
    int32_t *dids = nullptr, *dsteps = nullptr;
    ck(cudaMalloc(&dids, ids.size() * 4), "gk ids");
    ck(cudaMalloc(&dsteps, steps.size() * 4), "gk steps");
    ck(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "gk ids0");
    ck(cudaMemcpy(dsteps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "gk steps0");

    k::KvResolveGroups rg{};
    rg.ng = NG;
    for (int i = 0; i < NG; ++i) {
        rg.map[i] = M[i];
        rg.slots[i] = arena;                 // the SAME arena arrays: slot_base is the only difference
        rg.host[i] = host[i];
        rg.fmt[i] = k::kKvInt8;
        rg.first[i] = i;
        rg.ntok[i] = 1;
    }
    k::kv_stream_resolve_groups(rg, dids, dsteps, cap, s, nullptr);
    ck(cudaGetLastError(), "gk resolve");
    ck(cudaDeviceSynchronize(), "gk sync");

    // 1. group 0's range untouched.
    std::vector<int8_t> now_k(snap_k.size()), now_v(snap_v.size());
    std::vector<uint16_t> now_ks(snap_ks.size()), now_vs(snap_vs.size());
    ck(cudaMemcpy(now_k.data(), ark, now_k.size(), cudaMemcpyDeviceToHost), "gk rd k");
    ck(cudaMemcpy(now_v.data(), arv, now_v.size(), cudaMemcpyDeviceToHost), "gk rd v");
    ck(cudaMemcpy(now_ks.data(), arks, now_ks.size() * 2, cudaMemcpyDeviceToHost), "gk rd ks");
    ck(cudaMemcpy(now_vs.data(), arvs, now_vs.size() * 2, cudaMemcpyDeviceToHost), "gk rd vs");
    if (now_k != snap_k || now_v != snap_v || now_ks != snap_ks || now_vs != snap_vs) {
        std::fprintf(stderr, "  gk: CORRUPTION: slot_base=0's arena range was overwritten by a grouped copy\n");
        ++bad;
    }
    // The whole arena for the per-slot byte checks below (the snapshot above only covers group 0's range).
    std::vector<int8_t> ak(arows * D), av(arows * D);
    std::vector<uint16_t> aks(arows * GS), avs(arows * GS);
    ck(cudaMemcpy(ak.data(), ark, ak.size(), cudaMemcpyDeviceToHost), "gk full k");
    ck(cudaMemcpy(av.data(), arv, av.size(), cudaMemcpyDeviceToHost), "gk full v");
    ck(cudaMemcpy(aks.data(), arks, aks.size() * 2, cudaMemcpyDeviceToHost), "gk full ks");
    ck(cudaMemcpy(avs.data(), arvs, avs.size() * 2, cudaMemcpyDeviceToHost), "gk full vs");
    // 2. + 3. per-group tables and bytes.
    std::vector<int32_t> pt(n_blocks), ctl(k::kKvCtlInts);
    for (int i = 0; i < NG; ++i) {
        ck(cudaMemcpy(pt.data(), M[i].page_table, n_blocks * 4, cudaMemcpyDeviceToHost), "gk pt");
        ck(cudaMemcpy(ctl.data(), M[i].ctl, ctl.size() * 4, cudaMemcpyDeviceToHost), "gk ctl");
        if (ctl[2] != (i == 0 ? 0 : W)) {
            std::fprintf(stderr, "  gk: group %d placed %d misses, expected %d (the arm proved nothing)\n",
                         i, ctl[2], W);
            ++bad;
        }
        if (ctl[3] != 0) { std::fprintf(stderr, "  gk: group %d resolve overflowed\n", i); ++bad; }
        if (i == 0) {
            if (pt[b0] != (int32_t) j0) {
                std::fprintf(stderr, "  gk: group 0's pinned entry moved: page_table[%lld] = %d\n",
                             (long long) b0, pt[b0]);
                ++bad;
            }
            continue;
        }
        int named = 0;
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int32_t gs = pt[b];
            if (gs == -2) {
                std::fprintf(stderr, "  gk: group %d block %lld stuck at -2 (copy kernel never published)\n",
                             i, (long long) b);
                ++bad; break;
            }
            if (gs < 0) continue;
            ++named;
            if (gs < i * n_win || gs >= (i + 1) * n_win) {
                std::fprintf(stderr, "  gk: group %d block %lld published slot %d OUTSIDE its arena range\n",
                             i, (long long) b, gs);
                ++bad; break;
            }
            const size_t rb = (size_t) b * H * PS, rs = (size_t) gs * H * PS;
            if (std::memcmp(ak.data() + rs * D, ((int8_t*) host[i].k_q) + rb * D,
                            H * PS * D) != 0 ||
                std::memcmp(av.data() + rs * D, ((int8_t*) host[i].v_q) + rb * D,
                            H * PS * D) != 0 ||
                std::memcmp(aks.data() + rs * GS, ((uint16_t*) host[i].k_scale) + rb * GS,
                            H * PS * GS * 2) != 0 ||
                std::memcmp(avs.data() + rs * GS, ((uint16_t*) host[i].v_scale) + rb * GS,
                            H * PS * GS * 2) != 0) {
                std::fprintf(stderr, "  gk: group %d block %lld slot %d holds foreign bytes\n",
                             i, (long long) b, gs);
                ++bad; break;
            }
        }
        if (named != W) {
            std::fprintf(stderr, "  gk: group %d table names %d blocks, expected %d\n", i, named, W);
            ++bad;
        }
    }
    std::printf("  grouped arm: %s\n", bad ? "FAILED" : "ok");
    return bad;
}

// P5-5C.10 regression arm: an IN-PLACE conversation_checkpoint_restore over a CoW-forked main state.
// Shape (the 9k C=8 cross-talk, stream 7 <- stream 6): main (M0) holds conversation A - blocks
// [0, P+T) resident in its own window. Slot 1 (M1) is admitted with qsa_cow_fork(main, slot1,
// upto = A): M1's table names M0's pages (kv_stream_cow_fork_join: slot_use 1 -> 2 on [0, P+T))
// and BOTH registry entries carry conv = lineage_A, floor = P + T. Slot 1 then decodes A's reply:
// blocks [P+T, P+2T) resident in M1's OWN slots. Conversation B rewinds M0 IN PLACE (it shares
// only A's first P blocks; its tail [P, P+T) diverges). Pre-fix, the restore left M0's stale table
// (hits naming the pinned pages: B attends to A's tail bytes, and B's appends overwrite them) and
// M0's stale conv (a cleared table would JOIN M1's spliced entries - joins-at-fetch). The fix runs
// the release pair inside the restore (exactly what qsa_cow_release + kv_stream_reset do):
//   1. after the rewind: M0's table is all -1 and its registry entry reads conv=-1 / floor=max,
//   2. slot 1's fork pins survive: slot_use stays 2 on [0, P+T) and every pinned page's bytes are
//      bit-for-bit intact, as are M1's own reply pages,
//   3. B's re-resolve NEVER joins (the join counter stays flat: conv=-1 gates joins-at-fetch),
//      NEVER names a pinned slot (cow_hold keeps use>1 out of the victim sweep), names only its
//      own range, and every fetched page matches B's OWN host copy - A's tail bytes never appear.
int restore_rewind_arm() {
    std::printf("  rewind arm: in-place checkpoint restore over a CoW fork (P5-5C.10)\n");
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t H = s.n_head_kv, D = s.head_dim, PS = s.page_size;
    const int64_t n_blocks = 600, n_win = 1024;
    const int64_t n_arena = 2 * n_win;
    const int64_t P = 400, T = 100;   // A: prefix [0,P) shared with B, tail [P,P+T) diverges, reply [P+T,P+2T)
    const size_t GS = (size_t) (D / k::KV_Q8_GROUP);
    const size_t hrows = (size_t) n_blocks * H * PS, arows = (size_t) n_arena * H * PS;
    int bad = 0;

    // Host copies: A's bytes, and B's - blocks [0,P) identical (the shared prefix), [P, n_blocks) distinct.
    k::KvHostPools hostA{}, hostB{};
    {
        void *ak, *av, *aks, *avs, *bk, *bv, *bks, *bvs;
        ck(cudaHostAlloc(&ak, hrows * D, cudaHostAllocMapped), "rw hostA k");
        ck(cudaHostAlloc(&av, hrows * D, cudaHostAllocMapped), "rw hostA v");
        ck(cudaHostAlloc(&aks, hrows * GS * 2, cudaHostAllocMapped), "rw hostA ks");
        ck(cudaHostAlloc(&avs, hrows * GS * 2, cudaHostAllocMapped), "rw hostA vs");
        ck(cudaHostAlloc(&bk, hrows * D, cudaHostAllocMapped), "rw hostB k");
        ck(cudaHostAlloc(&bv, hrows * D, cudaHostAllocMapped), "rw hostB v");
        ck(cudaHostAlloc(&bks, hrows * GS * 2, cudaHostAllocMapped), "rw hostB ks");
        ck(cudaHostAlloc(&bvs, hrows * GS * 2, cudaHostAllocMapped), "rw hostB vs");
        hostA.k_q = (int8_t*) ak; hostA.v_q = (int8_t*) av; hostA.k_scale = (uint16_t*) aks; hostA.v_scale = (uint16_t*) avs;
        hostB.k_q = (int8_t*) bk; hostB.v_q = (int8_t*) bv; hostB.k_scale = (uint16_t*) bks; hostB.v_scale = (uint16_t*) bvs;
        std::mt19937 ra(509), rb(1010);
        std::uniform_int_distribution<int> byte_d(1, 255);   // never all-zero
        auto fill = [&](k::KvHostPools& h, std::mt19937& r) {
            for (size_t i = 0; i < hrows * D; ++i) { h.k_q[i] = (int8_t) byte_d(r); h.v_q[i] = (int8_t) byte_d(r); }
            for (size_t i = 0; i < hrows * GS; ++i) { h.k_scale[i] = (uint16_t) byte_d(r); h.v_scale[i] = (uint16_t) byte_d(r); }
        };
        fill(hostA, ra); fill(hostB, rb);
        const size_t pc = (size_t) P * H * PS;   // the shared prefix: byte-identical across A and B
        std::memcpy(hostB.k_q, hostA.k_q, pc * D); std::memcpy(hostB.v_q, hostA.v_q, pc * D);
        std::memcpy(hostB.k_scale, hostA.k_scale, pc * GS); std::memcpy(hostB.v_scale, hostA.v_scale, pc * GS);
    }

    // One arena, one slot_use, two maps: M0 = main (range [0, n_win)), M1 = slot 1 (range [n_win, 2*n_win)).
    k::QsaAttnPools arena{};
    int8_t *ark = nullptr, *arv = nullptr; uint16_t *arks = nullptr, *arvs = nullptr;
    ck(cudaMalloc(&ark, arows * D), "rw arena k");
    ck(cudaMalloc(&arv, arows * D), "rw arena v");
    ck(cudaMalloc(&arks, arows * GS * 2), "rw arena ks");
    ck(cudaMalloc(&arvs, arows * GS * 2), "rw arena vs");
    ck(cudaMemset(ark, 0xA5, arows * D), "rw arena k0");
    ck(cudaMemset(arv, 0xA5, arows * D), "rw arena v0");
    ck(cudaMemset(arks, 0xA5, arows * GS * 2), "rw arena ks0");
    ck(cudaMemset(arvs, 0xA5, arows * GS * 2), "rw arena vs0");
    arena.k_q = ark; arena.v_q = arv; arena.k_scale = arks; arena.v_scale = arvs;
    int32_t* slot_use = nullptr;
    ck(cudaMalloc(&slot_use, n_arena * 4), "rw use");
    ck(cudaMemset(slot_use, 0, n_arena * 4), "rw use0");
    auto dalloc = [](size_t n, bool ones) {
        int32_t* p = nullptr;
        ck(cudaMalloc(&p, n * 4), "rw alloc");
        ck(cudaMemset(p, ones ? 0xFF : 0, n * 4), "rw init");
        return p;
    };
    k::KvStreamMap M[2];
    for (int i = 0; i < 2; ++i) {
        k::KvStreamMap& m = M[i];
        m.page_table = dalloc(n_blocks, true);
        m.slot_block = dalloc(n_win, true); m.slot_stamp = dalloc(n_win, true); m.slot_ref = dalloc(n_win, false);
        m.miss_block = dalloc(n_win, false); m.miss_slot = dalloc(n_win, false);
        m.miss_sh = dalloc(n_win, false); m.miss_sh_slot = dalloc(n_win, false);
        m.ctl = dalloc(k::kKvCtlInts, false);
        m.n_blocks = n_blocks; m.n_slots = n_win; m.slot_base = i * n_win;
        m.slot_use = slot_use;
        std::vector<int32_t> u((size_t) n_win, 1);   // the admission seed: each owner holds its range
        ck(cudaMemcpy(slot_use + i * n_win, u.data(), (size_t) n_win * 4, cudaMemcpyHostToDevice), "rw seed");
    }
    // The registry: both states in lineage 5 (conv), floor = P+T (the fork's complete pages), same ord.
    std::vector<k::CowPeer> regh(2);
    regh[0] = {5, 64, 0, 0, (int32_t) n_win, (int32_t) n_blocks, M[0].page_table, (int32_t) (P + T)};
    regh[1] = {5, 64, 0, (int32_t) n_win, (int32_t) n_win, (int32_t) n_blocks, M[1].page_table, (int32_t) (P + T)};
    k::CowPeer* dreg = nullptr;
    ck(cudaMalloc(&dreg, 2 * sizeof(k::CowPeer)), "rw reg");
    ck(cudaMemcpy(dreg, regh.data(), 2 * sizeof(k::CowPeer), cudaMemcpyHostToDevice), "rw reg0");
    for (int i = 0; i < 2; ++i) { M[i].reg = dreg; M[i].reg_n = 2; }
    unsigned long long* join_ctr = nullptr;
    ck(cudaMalloc(&join_ctr, 8), "rw jctr"); ck(cudaMemset(join_ctr, 0, 8), "rw jctr0");
    M[0].join_ctr = join_ctr; M[1].join_ctr = join_ctr;

    // A resident in main: blocks [0, P+T) in slots [0, P+T), bytes = hostA.
    for (int64_t b = 0; b < P + T; ++b) {
        const size_t rb = (size_t) b * H * PS, rs = (size_t) b * H * PS;
        ck(cudaMemcpy(ark + rs * D, hostA.k_q + rb * D, H * PS * D, cudaMemcpyDeviceToDevice), "rw A k");
        ck(cudaMemcpy(arv + rs * D, hostA.v_q + rb * D, H * PS * D, cudaMemcpyDeviceToDevice), "rw A v");
        ck(cudaMemcpy(arks + rs * GS, hostA.k_scale + rb * GS, H * PS * GS * 2, cudaMemcpyDeviceToDevice), "rw A ks");
        ck(cudaMemcpy(arvs + rs * GS, hostA.v_scale + rb * GS, H * PS * GS * 2, cudaMemcpyDeviceToDevice), "rw A vs");
        const int32_t ptb = (int32_t) b, sbb = (int32_t) b;
        ck(cudaMemcpy(M[0].page_table + b, &ptb, 4, cudaMemcpyHostToDevice), "rw A pt");
        ck(cudaMemcpy(M[0].slot_block + b, &sbb, 4, cudaMemcpyHostToDevice), "rw A sb");
    }
    // The admission fork: slot 1's table names main's pages; the join kernel revalidates and pins (use 1 -> 2).
    {
        std::vector<int32_t> pt1((size_t) n_blocks, -1);
        for (int64_t b = 0; b < P + T; ++b) pt1[b] = (int32_t) b;
        ck(cudaMemcpy(M[1].page_table, pt1.data(), (size_t) n_blocks * 4, cudaMemcpyHostToDevice), "rw fork pt");
        k::kv_stream_cow_fork_join(M[1], M[0], nullptr);
        ck(cudaGetLastError(), "rw fork join");
        ck(cudaDeviceSynchronize(), "rw fork sync");
    }
    // Slot 1 decodes A's reply: blocks [P+T, P+2T) in M1's own slots, bytes = hostA.
    for (int64_t b = P + T; b < P + 2 * T; ++b) {
        const int64_t j = n_win + (b - (P + T));
        const size_t rb = (size_t) b * H * PS, rs = (size_t) j * H * PS;
        ck(cudaMemcpy(ark + rs * D, hostA.k_q + rb * D, H * PS * D, cudaMemcpyDeviceToDevice), "rw R k");
        ck(cudaMemcpy(arv + rs * D, hostA.v_q + rb * D, H * PS * D, cudaMemcpyDeviceToDevice), "rw R v");
        ck(cudaMemcpy(arks + rs * GS, hostA.k_scale + rb * GS, H * PS * GS * 2, cudaMemcpyDeviceToDevice), "rw R ks");
        ck(cudaMemcpy(arvs + rs * GS, hostA.v_scale + rb * GS, H * PS * GS * 2, cudaMemcpyDeviceToDevice), "rw R vs");
        const int32_t ptj = (int32_t) j, sbb = (int32_t) b;
        ck(cudaMemcpy(M[1].page_table + b, &ptj, 4, cudaMemcpyHostToDevice), "rw R pt");
        ck(cudaMemcpy(M[1].slot_block + (j - n_win), &sbb, 4, cudaMemcpyHostToDevice), "rw R sb");
    }
    ck(cudaDeviceSynchronize(), "rw setup");

    // Snapshot the bytes that must survive: main's pinned range and slot 1's reply range.
    const size_t span = (size_t) (P + T) * H * PS;
    std::vector<int8_t> pin_k(span * D), pin_v(span * D);
    std::vector<uint16_t> pin_ks(span * GS), pin_vs(span * GS);
    ck(cudaMemcpy(pin_k.data(), ark, pin_k.size(), cudaMemcpyDeviceToHost), "rw snap k");
    ck(cudaMemcpy(pin_v.data(), arv, pin_v.size(), cudaMemcpyDeviceToHost), "rw snap v");
    ck(cudaMemcpy(pin_ks.data(), arks, pin_ks.size() * 2, cudaMemcpyDeviceToHost), "rw snap ks");
    ck(cudaMemcpy(pin_vs.data(), arvs, pin_vs.size() * 2, cudaMemcpyDeviceToHost), "rw snap vs");
    const size_t rep = (size_t) T * H * PS;
    std::vector<int8_t> rep_k(rep * D), rep_v(rep * D);
    std::vector<uint16_t> rep_ks(rep * GS), rep_vs(rep * GS);
    ck(cudaMemcpy(rep_k.data(), ark + (size_t) n_win * H * PS * D, rep_k.size(), cudaMemcpyDeviceToHost), "rw rsnap k");
    ck(cudaMemcpy(rep_v.data(), arv + (size_t) n_win * H * PS * D, rep_v.size(), cudaMemcpyDeviceToHost), "rw rsnap v");
    ck(cudaMemcpy(rep_ks.data(), arks + (size_t) n_win * H * PS * GS, rep_ks.size() * 2, cudaMemcpyDeviceToHost), "rw rsnap ks");
    ck(cudaMemcpy(rep_vs.data(), arvs + (size_t) n_win * H * PS * GS, rep_vs.size() * 2, cudaMemcpyDeviceToHost), "rw rsnap vs");

    // THE REWIND: the pair conversation_checkpoint_restore now runs (qsa_cow_release's kernel half +
    // its registry reset, then the map reset). M0's table names only its own slots, so the release
    // drops no pins here; the registry entry leaves the lineage; the table empties. slot_use - and
    // with it slot 1's fork pins - is untouched by the reset.
    k::kv_stream_cow_release(M[0], nullptr);
    ck(cudaGetLastError(), "rw release");
    regh[0].conv = -1; regh[0].floor = 0x7FFFFFFF;
    ck(cudaMemcpy(dreg, &regh[0], sizeof(k::CowPeer), cudaMemcpyHostToDevice), "rw reg-");
    k::kv_stream_reset(M[0], nullptr);
    ck(cudaGetLastError(), "rw reset");
    ck(cudaDeviceSynchronize(), "rw rewind");

    // 1. the rewind cleared main's residency and left the lineage.
    std::vector<int32_t> pt0(n_blocks);
    ck(cudaMemcpy(pt0.data(), M[0].page_table, n_blocks * 4, cudaMemcpyDeviceToHost), "rw pt0");
    for (int64_t b = 0; b < n_blocks; ++b)
        if (pt0[b] != -1) {
            std::fprintf(stderr, "  rw: main's page_table[%lld] = %d after the rewind (expected -1)\n",
                         (long long) b, pt0[b]);
            ++bad; break;
        }
    k::CowPeer reg0[2];
    ck(cudaMemcpy(reg0, dreg, sizeof reg0, cudaMemcpyDeviceToHost), "rw reg rd");
    if (reg0[0].conv != -1 || reg0[0].floor != 0x7FFFFFFF) {
        std::fprintf(stderr, "  rw: main's registry entry after the rewind: conv=%d floor=%d (expected -1 / max)\n",
                     reg0[0].conv, reg0[0].floor);
        ++bad;
    }
    // 2. slot 1's pins and pages survived the rewind.
    std::vector<int32_t> use(n_arena);
    ck(cudaMemcpy(use.data(), slot_use, n_arena * 4, cudaMemcpyDeviceToHost), "rw use rd");
    for (int64_t b = 0; b < P + T; ++b)
        if (use[b] != 2) {
            std::fprintf(stderr, "  rw: slot_use[%lld] = %d after the rewind (slot 1's fork pin lost)\n",
                         (long long) b, use[b]);
            ++bad; break;
        }
    {
        std::vector<int8_t> k2(pin_k.size()), v2(pin_v.size());
        std::vector<uint16_t> ks2(pin_ks.size()), vs2(pin_vs.size());
        ck(cudaMemcpy(k2.data(), ark, k2.size(), cudaMemcpyDeviceToHost), "rw rd k");
        ck(cudaMemcpy(v2.data(), arv, v2.size(), cudaMemcpyDeviceToHost), "rw rd v");
        ck(cudaMemcpy(ks2.data(), arks, ks2.size() * 2, cudaMemcpyDeviceToHost), "rw rd ks");
        ck(cudaMemcpy(vs2.data(), arvs, vs2.size() * 2, cudaMemcpyDeviceToHost), "rw rd vs");
        if (k2 != pin_k || v2 != pin_v || ks2 != pin_ks || vs2 != pin_vs) {
            std::fprintf(stderr, "  rw: main's pinned range changed across the rewind\n");
            ++bad;
        }
    }

    // B's prefill: resolve blocks [0, P+T) from B's OWN host copy. Pre-fix (stale conv + stale table)
    // this either hit A's tail bytes or joined slot 1's spliced entries; post-fix every block is a
    // cold fetch into an unpinned slot of main's own range.
    const int cap = 512;
    std::vector<int32_t> ids((size_t) cap, 0), steps(k::kStepCount, 0);
    for (int64_t t = 0; t < P + T; ++t) ids[t] = (int32_t) (PS * t);
    steps[k::kStepWidth] = (int32_t) (P + T);
    int32_t *dids = nullptr, *dsteps = nullptr;
    ck(cudaMalloc(&dids, ids.size() * 4), "rw ids");
    ck(cudaMalloc(&dsteps, steps.size() * 4), "rw steps");
    ck(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "rw ids0");
    ck(cudaMemcpy(dsteps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "rw steps0");
    k::QsaAttnPools ap = arena; ap.page_table = M[0].page_table;
    k::kv_stream_resolve(M[0], ap, hostB, k::kKvInt8, dids, dsteps, 1, cap, s, nullptr);
    ck(cudaGetLastError(), "rw resolve");
    ck(cudaDeviceSynchronize(), "rw resolve sync");

    // 3. no join fired (conv=-1 gated joins-at-fetch), every miss was placed, and the table is clean.
    unsigned long long jc = 0;
    ck(cudaMemcpy(&jc, join_ctr, 8, cudaMemcpyDeviceToHost), "rw jctr rd");
    if (jc != 0) {
        std::fprintf(stderr, "  rw: B's resolve JOINED %llu lineage pages after the rewind (cross-talk)\n", jc);
        ++bad;
    }
    int32_t ctl[k::kKvCtlInts];
    ck(cudaMemcpy(ctl, M[0].ctl, sizeof ctl, cudaMemcpyDeviceToHost), "rw ctl");
    if (ctl[2] != (int32_t) (P + T)) {
        std::fprintf(stderr, "  rw: B's resolve placed %d misses, expected %lld (the arm proved nothing)\n",
                     ctl[2], (long long) (P + T));
        ++bad;
    }
    if (ctl[3] != 0) { std::fprintf(stderr, "  rw: B's resolve overflowed\n"); ++bad; }
    ck(cudaMemcpy(pt0.data(), M[0].page_table, n_blocks * 4, cudaMemcpyDeviceToHost), "rw pt0 rd");
    ck(cudaMemcpy(use.data(), slot_use, n_arena * 4, cudaMemcpyDeviceToHost), "rw use rd2");
    std::vector<int8_t> ak(arows * D), av(arows * D);
    std::vector<uint16_t> aks(arows * GS), avs(arows * GS);
    ck(cudaMemcpy(ak.data(), ark, ak.size(), cudaMemcpyDeviceToHost), "rw full k");
    ck(cudaMemcpy(av.data(), arv, av.size(), cudaMemcpyDeviceToHost), "rw full v");
    ck(cudaMemcpy(aks.data(), arks, aks.size() * 2, cudaMemcpyDeviceToHost), "rw full ks");
    ck(cudaMemcpy(avs.data(), arvs, avs.size() * 2, cudaMemcpyDeviceToHost), "rw full vs");
    for (int64_t b = 0; b < P + T && !bad; ++b) {
        const int32_t sl = pt0[b];
        if (sl < 0 || sl >= n_win) {
            std::fprintf(stderr, "  rw: B's block %lld names slot %d (expected inside main's own range)\n",
                         (long long) b, sl);
            ++bad; break;
        }
        if (use[sl] > 1) {
            std::fprintf(stderr, "  rw: B's block %lld took a PINNED slot %d (use=%d): cow_hold failed\n",
                         (long long) b, sl, use[sl]);
            ++bad; break;
        }
        const size_t rb = (size_t) b * H * PS, rs = (size_t) sl * H * PS;
        if (std::memcmp(ak.data() + rs * D, hostB.k_q + rb * D, H * PS * D) != 0 ||
            std::memcmp(av.data() + rs * D, hostB.v_q + rb * D, H * PS * D) != 0 ||
            std::memcmp(aks.data() + rs * GS, hostB.k_scale + rb * GS, H * PS * GS * 2) != 0 ||
            std::memcmp(avs.data() + rs * GS, hostB.v_scale + rb * GS, H * PS * GS * 2) != 0) {
            std::fprintf(stderr, "  rw: B's block %lld slot %d holds FOREIGN bytes (A's tail leaked into B)\n",
                         (long long) b, sl);
            ++bad; break;
        }
    }
    // Slot 1's reply pages and its table survived B's prefill untouched.
    {
        std::vector<int8_t> k2(rep_k.size()), v2(rep_v.size());
        std::vector<uint16_t> ks2(rep_ks.size()), vs2(rep_vs.size());
        ck(cudaMemcpy(k2.data(), ark + (size_t) n_win * H * PS * D, k2.size(), cudaMemcpyDeviceToHost), "rw rd R k");
        ck(cudaMemcpy(v2.data(), arv + (size_t) n_win * H * PS * D, v2.size(), cudaMemcpyDeviceToHost), "rw rd R v");
        ck(cudaMemcpy(ks2.data(), arks + (size_t) n_win * H * PS * GS, ks2.size() * 2, cudaMemcpyDeviceToHost), "rw rd R ks");
        ck(cudaMemcpy(vs2.data(), arvs + (size_t) n_win * H * PS * GS, vs2.size() * 2, cudaMemcpyDeviceToHost), "rw rd R vs");
        if (k2 != rep_k || v2 != rep_v || ks2 != rep_ks || vs2 != rep_vs) {
            std::fprintf(stderr, "  rw: slot 1's reply range was overwritten by B's prefill\n");
            ++bad;
        }
        std::vector<int32_t> pt1(n_blocks);
        ck(cudaMemcpy(pt1.data(), M[1].page_table, n_blocks * 4, cudaMemcpyDeviceToHost), "rw pt1 rd");
        for (int64_t b = 0; b < P + 2 * T && !bad; ++b) {
            const int32_t want = b < P + T ? (int32_t) b : (int32_t) (n_win + (b - (P + T)));
            if (pt1[b] != want) {
                std::fprintf(stderr, "  rw: slot 1's page_table[%lld] = %d, expected %d\n",
                             (long long) b, pt1[b], want);
                ++bad; break;
            }
        }
    }
    std::printf("  rewind arm: %s\n", bad ? "FAILED" : "ok");
    return bad;
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

    // P5-5C.9: the grouped-resolve regression arm (its own arena; runs even when the phases above failed).
    bad += grouped_arm();
    // P5-5C.10: the in-place checkpoint-restore rewind arm (its own arena; runs likewise).
    bad += restore_rewind_arm();

    if (bad) ++g_fail;
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
