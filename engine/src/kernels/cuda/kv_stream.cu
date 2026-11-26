// src/kernels/cuda/kv_stream.cu - see include/strata/kernels/kv_stream.hpp.
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_stream: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int RT = 1024;   // the resolve block

// The per-block byte runs of the (up to four) pool arrays: block b of array i is bytes [b * len, (b + 1) * len).
struct Runs {
    const uint8_t* src[4];
    uint8_t* dst[4];
    int len[4];
    int n;
};

Runs runs_of(const QsaAttnPools& slots, const KvHostPools& host, int fmt, const QsaShapes& s) {
    const int rows = (int) (s.n_head_kv * s.page_size);
    Runs r{};
    if (fmt == kKvHybrid) {   // K8V4: int8 K codes, their scales, rotated q4_0 V
        const int codes = rows * (int) s.head_dim, scales = rows * (int) (s.head_dim / KV_Q8_GROUP) * 2;
        const int v = rows * (int) kv_q4_bytes_per_head((int) s.head_dim);
        r.src[0] = (const uint8_t*) host.k_q;     r.dst[0] = (uint8_t*) slots.k_q;     r.len[0] = codes;
        r.src[1] = (const uint8_t*) host.k_scale; r.dst[1] = (uint8_t*) slots.k_scale; r.len[1] = scales;
        r.src[2] = (const uint8_t*) host.v_q4;    r.dst[2] = (uint8_t*) slots.v_q4;    r.len[2] = v;
        r.n = 3;
    } else if (fmt == kKvQ4) {
        const int bytes = rows * (int) kv_q4_bytes_per_head((int) s.head_dim);
        r.src[0] = (const uint8_t*) host.k_q4; r.dst[0] = (uint8_t*) slots.k_q4; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_q4; r.dst[1] = (uint8_t*) slots.v_q4; r.len[1] = bytes;
        r.n = 2;
    } else if (fmt == kKvInt8) {
        const int codes = rows * (int) s.head_dim, scales = rows * (int) (s.head_dim / KV_Q8_GROUP) * 2;
        r.src[0] = (const uint8_t*) host.k_q;     r.dst[0] = (uint8_t*) slots.k_q;     r.len[0] = codes;
        r.src[1] = (const uint8_t*) host.v_q;     r.dst[1] = (uint8_t*) slots.v_q;     r.len[1] = codes;
        r.src[2] = (const uint8_t*) host.k_scale; r.dst[2] = (uint8_t*) slots.k_scale; r.len[2] = scales;
        r.src[3] = (const uint8_t*) host.v_scale; r.dst[3] = (uint8_t*) slots.v_scale; r.len[3] = scales;
        r.n = 4;
    } else {
        const int bytes = rows * (int) s.head_dim * 2;
        r.src[0] = (const uint8_t*) host.k_pool; r.dst[0] = (uint8_t*) slots.k_pool; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_pool; r.dst[1] = (uint8_t*) slots.v_pool; r.len[1] = bytes;
        r.n = 2;
    }
    return r;
}

// Block-wide exclusive prefix sum of one int per thread (RT threads); `total` is the sum over the block.
__device__ int block_scan(int v, int* warp_sums, int& total) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_sums[w] = x;
    __syncthreads();
    if (w == 0) {
        int t = warp_sums[lane];
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, t, o);
            if (lane >= o) t += y;
        }
        warp_sums[lane] = t;
    }
    __syncthreads();
    total = warp_sums[31];
    const int excl = x - v + (w > 0 ? warp_sums[w - 1] : 0);
    __syncthreads();
    return excl;
}

// §235 tail-slot pin law: a join pin may only land on a LIVE slot (use >= 1). A blind atomicAdd
// could bump a free 0 to 1 (the sweeper then can never claim it: its CAS expects 0) or an
// in-flight -2 claim to -1 (the sweeper's hand-back CAS(-2, 0) then fails forever). CAS from the
// observed value: 0 (free) and -2 (claimed) are NEVER perturbed.
__device__ __forceinline__ bool pin_if_live(int* p) {
    int cur = *reinterpret_cast<volatile int*>(p);
    for (;;) {
        if (cur < 1) return false;
        const int prev = atomicCAS(p, cur, cur + 1);
        if (prev == cur) return true;
        cur = prev;
    }
}

__device__ __forceinline__ void resolve_body(KvStreamMap m, const int32_t* __restrict__ ids,
                                             const int32_t* __restrict__ steps, int n_q, int cap, int page_size,
                                             const bool grouped) {
    __shared__ int s_nmiss, s_lookups, s_nsh, s_cut;
    __shared__ int warp_sums[32];
    const int epoch = m.ctl[0] + 1;
    if (threadIdx.x == 0) { s_nmiss = 0; s_lookups = 0; s_nsh = 0; }
    // cow-m1e: this map's lineage hint table and byte-identity floor (found once in the registry)
    int32_t* lin = nullptr;
    int my_conv_slot = -1;   // §235: which hint tab (conv % kCowLinTabs) this map arms and revalidates against
    int32_t fl = 0x7FFFFFFF;
    // dyn-conc §222: the floor gates joins-at-fetch (cow-m1d) and must be read whenever the registry
    // exists; only the cow-m1e lineage hint table needs the shared region (lin_all) to be mapped.
    if (m.reg != nullptr)
        for (int e = 0; e < m.reg_n; ++e)
            if (m.reg[e].pt == m.page_table) {
                if (m.reg[e].conv >= 0 && m.n_shared > 0 && m.lin_all != nullptr) {  // §234: hint path is a shared-region mechanism; the floor (fl) below stays live regardless
                    lin = m.lin_all + (size_t) (m.reg[e].conv % kCowLinTabs) * (size_t) m.n_blocks;
                    my_conv_slot = m.reg[e].conv % kCowLinTabs;
                }
                fl = m.reg[e].floor;
                break;
            }
    __syncthreads();
    // 1. hits take this epoch and their reference bit; a missing block is claimed exactly once (-1 -> -2)
    int lookups = 0;
    for (int q = 0; q < n_q; ++q) {
        const int width = steps[q * kStepCount + kStepWidth];
        const int32_t* qi = ids + (long long) q * cap;
        for (int i = threadIdx.x; i < width; i += RT) {
            const int b = qi[i] / page_size;
            if (i > 0 && qi[i - 1] / page_size == b) continue;   // ids are ascending: one lookup per block
            ++lookups;
            const int sl = m.page_table[b];
            if (sl >= 0) {
                // cow-m1: an entry may name a slot outside this map's own arena range (a shared
                // page from another session): the stamp/ref arrays are local, the owner's clock governs.
                if (sl >= m.slot_base && sl < m.slot_base + (int) m.n_slots) {
                    m.slot_stamp[sl - m.slot_base] = epoch;
                    m.slot_ref[sl - m.slot_base] = 1;
                } else if (m.n_shared > 0 && sl >= m.shared_base && sl < m.shared_base + m.n_shared) {
                    m.sst[sl - m.shared_base] = epoch;   // cow-m1e: a shared page sits in the region's clock, not this window's
                    m.srf[sl - m.shared_base] = 1;
                }
            } else if (sl == -1) {
                bool joined = false, shpend = false;
                if (m.reg != nullptr && m.slot_use != nullptr) {   // cow-m1d: join a lineage sibling's resident page instead of fetching a duplicate
                    int se = -1;
                    for (int e = 0; e < m.reg_n && se < 0; ++e) if (m.reg[e].pt == m.page_table) se = e;
                    if (se >= 0 && m.reg[se].conv >= 0) {
                        // cow-m1e: check the region before paying for ANY fetch: an adoption costs this
                        // window nothing (PIN LAW safe), and one lineage fetch serves every sibling from
                        // now on. Only blocks below the fork floor are byte-identical across the lineage.
                        if (lin != nullptr && b < fl) {
                            volatile int32_t* vlin = lin;
                            const int32_t L = vlin[b];
                            // arm-227 KILL restored (§234): adoption is only reachable with a properly mapped+0xFF'd
                            // hint tab (resolve wires lin only when n_shared > 0 && lin_all != nullptr, and the arena
                            // leaves the tab unmapped/null when the shared region is off), so the zero-init aliasing
                            // that armed-227 for cannot recur.
                            if (L >= 0) {
                                // §235 tail-slot pin law: pin_if_live never perturbs a free (0) or
                                // in-flight-claim (-2) slot. After the pin, revalidate ALL THREE:
                                // the hint still names L, and L's region slot still names this block
                                // in THIS lineage's tab (sblk/sconv). A slot whose bytes moved (or
                                // moved to another lineage) fails the revalidation and unpins: the
                                // miss falls through, never a foreign adoption.
                                if (pin_if_live(&m.slot_use[L])) {
                                    __threadfence();
                                    volatile int32_t* vsblk = m.sblk;
                                    volatile int32_t* vsconv = m.sconv;
                                    const int j = L - m.shared_base;
                                    const bool live = vlin[b] == L && j >= 0 && j < (int) m.n_shared &&
                                                      vsblk[j] == b && vsconv[j] == my_conv_slot;
                                    if (live) {
                                        m.page_table[b] = L;
                                        joined = true;
                                        if (m.join_ctr) atomicAdd(m.join_ctr, 1ull);
                                    } else atomicSub(&m.slot_use[L], 1);
                                }
                            } else if (L == -1 && !grouped && atomicCAS(&lin[b], -1, -2) == -1) {
                                // §235: the grouped copy kernel (copy_groups_kernel) does not publish
                                // region fetches, so the grouped resolve never CLAIMS one: a claim it
                                // made would sit at -2 forever and permanently shrink the region.
                                // Adoption (the L >= 0 branch above) needs no copy and stays live
                                // in both paths.
                                const int k = atomicAdd(&s_nsh, 1);
                                if (k < (int) m.n_slots) { m.miss_sh[k] = b; shpend = true; }   // the entry stays -1: host read this step
                                else (void) atomicCAS(&lin[b], -2, -1);                         // the claim list is full: hand the hint back
                            }
                            // L == -2: a sibling is fetching right now: read host this step, adopt next.
                        }
                        for (int e = 0; e < m.reg_n && !joined; ++e) {
                            if (e == se || m.reg[e].pt == nullptr || m.reg[e].pt == m.page_table ||
                                m.reg[e].ord != m.reg[se].ord || m.reg[e].conv != m.reg[se].conv ||
                                b >= m.reg[e].n_blocks || b >= fl) continue;   // cow-m1e: at/above the floor the bytes diverge
                            volatile int32_t* vpt = m.reg[e].pt;   // L2 reads: the entry is written by another kernel, possibly another SM
                            const int32_t s2 = vpt[b];
                            if (s2 < 0) continue;
                            // cow-m1d join protocol: ANY number of holders share a page, so the pin is an
                            // unconditional add; prev >= 1 says a holder was there when we landed. An add that
                            // hit the owner's evict window (prev == 0) takes itself back; the revalidation
                            // catches entries that moved. A rollback landing after the owner re-published
                            // (store use = 1) can undercount that slot forever: the sweep then can never take
                            // it and joins to it degrade to duplicate fetches - a leak, never a corruption.
                            // A joined pin is un-evictable by its owner, so it spends the fetcher's window:
                            // hhead caps how many foreign joins one range accepts, keeping every window
                            // turning. Over the cap the miss falls through to a cold fetch instead.
                            if (atomicSub(&m.reg[e].hhead, 1) <= 0) { atomicAdd(&m.reg[e].hhead, 1); continue; }
                            // §235 tail-slot pin law: the same live-only pin over peer ranges.
                            if (pin_if_live(&m.slot_use[s2])) {
                                if (vpt[b] == s2) {
                                    m.page_table[b] = s2;
                                    joined = true;
                                    if (m.join_ctr) atomicAdd(m.join_ctr, 1ull);
                                } else {
                                    atomicSub(&m.slot_use[s2], 1);
                                    atomicAdd(&m.reg[e].hhead, 1);
                                }
                            } else atomicAdd(&m.reg[e].hhead, 1);
                        }
                    }
                }
                if (!joined && !shpend && atomicCAS(&m.page_table[b], -1, -2) == -1) {
                    m.miss_block[atomicAdd(&s_nmiss, 1)] = b;
                }
            }
        }
    }
    if (lookups > 0) atomicAdd(&s_lookups, lookups);   // (#783, stuchapin909) most threads see none: skip the shared atomic
    __syncthreads();
    // 2. one victim per miss: a clock sweep from the hand. A slot this call uses (stamp == epoch) is never taken;
    //    a referenced one loses its bit as the hand passes it and is taken on the next pass.
    const int need = s_nmiss, n = (int) m.n_slots;
    const int need_sh = s_nsh < (int) m.n_slots ? s_nsh : (int) m.n_slots;   // cow-m1e
    int hand = m.ctl[1], got = 0, hand_sh = m.ctl[10], got_sh = 0;
    for (int scanned = 0; got < need && scanned < 3 * n; scanned += RT) {
        const int j = (int) (((long long) hand + threadIdx.x) % n);
        const bool mine = m.slot_stamp[j] == epoch;
        // cow-m1: a page another session forked onto (use > 1) is never taken as this owner's victim.
        const bool cow_hold = m.slot_use != nullptr && m.slot_use[j + (int) m.slot_base] > 1;
        const bool cand = !cow_hold && !mine && (m.slot_block[j] < 0 || m.slot_ref[j] == 0);
        int total = 0;
        const int rank = block_scan(cand ? 1 : 0, warp_sums, total);
        const int want = need - got;
        if (threadIdx.x == 0) s_cut = RT;
        __syncthreads();
        if (cand && rank == want - 1) s_cut = threadIdx.x + 1;   // the hand stops just past the last slot taken
        __syncthreads();
        const int cut = s_cut;
        if (cand && rank < want) {
            m.miss_slot[got + rank] = j;
            m.slot_stamp[j] = epoch;   // taken: a sweep that wraps around must not take it twice
        } else if (threadIdx.x < cut && !mine) {
            m.slot_ref[j] = 0;
        }
        got += total < want ? total : want;
        hand = (int) (((long long) hand + cut) % n);
        __syncthreads();
    }
    // cow-m1e: the shared region's own clock sweep. Candidates are slots NO live table names
    // (use == 0 - the PIN LAW boundary: nothing can be gathering from them) whose region clock
    // passed. Two sessions' sweeps can overlap, so the claim CAS runs inside the candidate pass
    // and claims past the rank cut are handed straight back (-2 -> 0 is claimer-private).
    if (need_sh > 0 && lin != nullptr) {
        const int sn = m.n_shared;
        for (int scanned = 0; got_sh < need_sh && scanned < 3 * sn; scanned += RT) {
            const int j = (int) (((long long) hand_sh + threadIdx.x) % sn);
            const int gs = j + m.shared_base;
            const bool mine = m.sst[j] == epoch;
            int v = 0;
            if (!mine && m.slot_use[gs] == 0 && (m.sblk[j] < 0 || m.srf[j] == 0) &&
                atomicCAS(&m.slot_use[gs], 0, -2) == 0)
                v = 1;
            int total = 0;
            const int rank = block_scan(v, warp_sums, total);
            const int want = need_sh - got_sh;
            if (threadIdx.x == 0) s_cut = RT;
            __syncthreads();
            if (v == 1 && rank == want - 1) s_cut = threadIdx.x + 1;
            __syncthreads();
            const int cut = s_cut;
            if (v == 1 && rank < want) {
                m.miss_sh_slot[got_sh + rank] = gs - (int) m.slot_base;   // same slot_base basis as miss_slot
                m.sst[j] = epoch;
            } else if (v == 1) {
                (void) atomicCAS(&m.slot_use[gs], -2, 0);                  // claimed past the cut: hand it back
            } else if (threadIdx.x < cut && !mine) {
                m.srf[j] = 0;
            }
            got_sh += total < want ? total : want;
            hand_sh = (int) (((long long) hand_sh + cut) % sn);
            __syncthreads();
        }
    }
    const int placed_sh = got_sh < need_sh ? got_sh : need_sh;
    // 3. re-point the table; the copy kernel fills the slots
    const int placed = got < need ? got : need;
    for (int k = threadIdx.x; k < need; k += RT) {
        const int b = m.miss_block[k];
        if (k >= placed) { m.page_table[b] = -1; continue; }   // overflow: never happens with a legal n_slots
        const int sl = m.miss_slot[k];
        const int gs = sl + (int) m.slot_base;
        const int old = m.slot_block[sl];
        if (m.slot_use != nullptr) {
            // cow-m1d: take the old page's entry OFF the table, fenced, before the commit CAS: a sibling can
            // then never latch a join onto a slot whose bytes are about to move. The CAS(1->0) contends with
            // any join that beat us: it fails, we restore the entry exactly as it was, and the miss turns back
            // cold (a host read-through, stall-free).
            if (old >= 0) m.page_table[old] = -1;
            __threadfence();
            if (atomicCAS(&m.slot_use[gs], 1, 0) != 1) {
                if (old >= 0) m.page_table[old] = gs;
                m.miss_slot[k] = -1;
                m.page_table[b] = -1;
                continue;
            }
            m.slot_block[sl] = b;
            m.slot_stamp[sl] = epoch;
            m.slot_ref[sl] = 1;
            m.slot_use[gs] = 1;
            m.page_table[b] = -2;   // stays CLAIMED: the copy kernel publishes it once the bytes are in, so no sibling can join a half-copied page
        } else {
            if (old >= 0) m.page_table[old] = -1;
            m.slot_block[sl] = b;
            m.slot_stamp[sl] = epoch;
            m.slot_ref[sl] = 1;
            m.page_table[b] = gs;   // cow-m1: the table names global arena pages
        }
    }
    // cow-m1e: arm the claimed region slots: the block map lands now (the bytes follow in the copy
    // kernel), and the old block's hint dies before its bytes can move. The copy kernel publishes
    // the entry, the fetcher's refcount (-2 claim -> held 1) and the fresh hint.
    for (int k = threadIdx.x; k < need_sh && lin != nullptr; k += RT) {
        const int b = m.miss_sh[k];
        if (k >= placed_sh) { (void) atomicCAS(&lin[b], -2, -1); continue; }   // region full: release the claim, host read this step
        const int gs = (int) m.miss_sh_slot[k] + (int) m.slot_base;
        const int j = gs - m.shared_base;
        // §235 root cause 1: the old block's hint dies in the tab of the LINEAGE that armed the
        // slot (sconv[j]), not in this claimer's tab. Clearing it here (lin[oldb] = -1) left the
        // previous owner's hint pointing at this slot; once the copy kernel re-published the slot
        // for the new block, the old lineage adopted the new lineage's KV bytes.
        const int oldb = m.sblk[j];
        const int oldc = m.sconv[j];
        if (oldb >= 0 && oldc >= 0 && m.lin_all != nullptr)
            m.lin_all[(size_t) oldc * (size_t) m.n_blocks + (size_t) oldb] = -1;
        __threadfence();
        m.sblk[j] = b;
        m.sconv[j] = my_conv_slot;
        m.sst[j] = epoch;
        m.srf[j] = 1;
    }
    if (threadIdx.x == 0) {
        m.ctl[0] = epoch;
        m.ctl[1] = hand;
        m.ctl[2] = placed;
        if (placed < need) m.ctl[3] += 1;
        m.ctl[10] = hand_sh;      // cow-m1e: region sweep hand and last placed count
        m.ctl[11] = placed_sh;
        unsigned long long* c = reinterpret_cast<unsigned long long*>(m.ctl + 4);
        c[0] += (unsigned long long) placed;
        c[1] += (unsigned long long) s_lookups;
        c[2] += 1ull;
        c[3] += (unsigned long long) placed_sh;   // cow-m1e: committed lineage fetches (ctl[12,13])
    }
}

__global__ void __launch_bounds__(RT) resolve_kernel(KvStreamMap m, const int32_t* __restrict__ ids,
                                                     const int32_t* __restrict__ steps, int n_q, int cap,
                                                     int page_size) {
    resolve_body(m, ids, steps, n_q, cap, page_size, false);
}

// Lean per-kernel parameter structs to stay well within CUDA's 4 KB pass-by-value kernel argument limit.
struct KvResolveMapGroups {
    KvStreamMap map[kKvResolveMaxGroups];
    int first[kKvResolveMaxGroups];
    int ntok[kKvResolveMaxGroups];
    int ng;
};
static_assert(sizeof(KvResolveMapGroups) <= 2048, "sm_86 4KB kernel param limit");

struct KvCopyGroups {
    int32_t* ctl[kKvResolveMaxGroups];
    int32_t* miss_block[kKvResolveMaxGroups];
    int32_t* miss_slot[kKvResolveMaxGroups];
    Runs runs[kKvResolveMaxGroups];
    int ng;
};
static_assert(sizeof(KvCopyGroups) <= 2048, "sm_86 4KB kernel param limit");

// One block per distinct streamed slot: resolves all ntok[g] contiguous rows of slot g in a single epoch.
__global__ void __launch_bounds__(RT) resolve_groups_kernel(KvResolveMapGroups mg, const int32_t* __restrict__ ids,
                                                            const int32_t* __restrict__ steps, int cap, int page_size) {
    const unsigned g = blockIdx.x;
    if ((int) g >= mg.ng) return;
    const KvStreamMap m = mg.map[g];
    const int first = mg.first[g];
    const int n_q = mg.ntok[g];
    resolve_body(m, ids + (long long) first * cap, steps + (long long) first * kStepCount, n_q, cap, page_size, true);
}

// One block-row per distinct streamed slot (blockIdx.y = g): copies that slot's missed blocks from host into VRAM.
__global__ void copy_groups_kernel(KvCopyGroups cg) {
    const unsigned g = blockIdx.y;
    if ((int) g >= cg.ng) return;
    const int need = cg.ctl[g][2];
    if (need <= 0) return;
    const int32_t* miss_block = cg.miss_block[g];
    const int32_t* miss_slot = cg.miss_slot[g];
    const Runs r = cg.runs[g];
    for (int k = blockIdx.x; k < need; k += gridDim.x) {
        const long long b = miss_block[k], sl = miss_slot[k];
        for (int a = 0; a < r.n; ++a) {
            const uint4* src = reinterpret_cast<const uint4*>(r.src[a] + b * r.len[a]);
            uint4* dst = reinterpret_cast<uint4*>(r.dst[a] + sl * r.len[a]);
            for (int i = threadIdx.x; i < r.len[a] / 16; i += blockDim.x) dst[i] = src[i];
        }
    }
}

// One block per missed block (grid-stride): copy its runs from the host copy into its slot, 16 B per thread.
__global__ void copy_kernel(KvStreamMap m, Runs r) {
    const int need = m.ctl[2], nsh = m.ctl[11];   // cow-m1e: + region fetches
    int32_t* lin = nullptr;
    int my_conv_slot = -1;   // §235: the tab this map's lineage owns (see resolve_body)
    if (m.n_shared > 0 && m.reg != nullptr && m.lin_all != nullptr)
        for (int e = 0; e < m.reg_n; ++e)
            if (m.reg[e].pt == m.page_table && m.reg[e].conv >= 0) {
                lin = m.lin_all + (size_t) (m.reg[e].conv % kCowLinTabs) * (size_t) m.n_blocks;
                my_conv_slot = m.reg[e].conv % kCowLinTabs;
                break;
            }
    for (int k = blockIdx.x; k < need + nsh; k += gridDim.x) {
        const bool sh = k >= need;
        const int kk = sh ? k - need : k;
        const long long b = sh ? m.miss_sh[kk] : m.miss_block[kk];
        const long long sl = sh ? m.miss_sh_slot[kk] : m.miss_slot[kk];
        if (b < 0 || sl < 0) continue;   // cow-m1d: an aborted victim must not have its (now foreign-owned) slot overwritten
        for (int a = 0; a < r.n; ++a) {
            const uint4* src = reinterpret_cast<const uint4*>(r.src[a] + b * r.len[a]);
            uint4* dst = reinterpret_cast<uint4*>(r.dst[a] + (sl + m.slot_base) * r.len[a]);   // cow-m1
            for (int i = threadIdx.x; i < r.len[a] / 16; i += blockDim.x) dst[i] = src[i];
        }
        // cow-m1d: publish AFTER the bytes: an arena map left the entry at -2 (claimed); a legacy map
        // published the same value in resolve, so this store is idempotent there.
        __syncthreads();
        if (threadIdx.x == 0) {
            const int32_t gs = (int32_t) (sl + m.slot_base);
            if (sh) {   // cow-m1e + §235: publish the slot's block map and owning tab FIRST (fenced),
                        // then take the refcount live (-2 claim -> held 1), name the page, and let
                        // the lineage see the hint LAST: any joiner that can pin the slot already
                        // sees sblk/sconv that pass its revalidation, and any joiner that reads the
                        // hint is guaranteed to find a refcount it can pin.
                const int j = gs - m.shared_base;
                m.sblk[j] = (int32_t) b;
                m.sconv[j] = my_conv_slot;
                __threadfence();
                (void) atomicExch(&m.slot_use[gs], 1);
                if (m.sh_ctr) atomicAdd(m.sh_ctr, 1ull);
                m.page_table[b] = gs;
                __threadfence();
                if (lin != nullptr) lin[b] = gs;
            } else m.page_table[b] = gs;
        }
    }
}

__global__ void reset_kernel(KvStreamMap m) {
    const long long i0 = (long long) blockIdx.x * blockDim.x + threadIdx.x, st = (long long) gridDim.x * blockDim.x;
    for (long long i = i0; i < m.n_blocks; i += st) m.page_table[i] = -1;
    for (long long i = i0; i < m.n_slots; i += st) {
        m.slot_block[i] = -1;
        m.slot_stamp[i] = -1;
        m.slot_ref[i] = 0;
    }
    if (i0 < kKvCtlInts) m.ctl[i0] = 0;
}

__global__ void ring_kernel(int32_t* table, long long n_blocks, long long n_slots) {
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n_blocks;
         i += (long long) gridDim.x * blockDim.x)
        table[i] = (int32_t) (i % n_slots);
}

__global__ void cow_fork_join_kernel(KvStreamMap to, KvStreamMap from) {
    const long long fb = from.slot_base, fn = from.n_slots;
    for (long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x; b < to.n_blocks;
         b += (long long) gridDim.x * blockDim.x) {
        const int32_t s = to.page_table[b];
        if (s < fb || s >= fb + fn) continue;
        if (from.page_table[b] == s) atomicAdd(&to.slot_use[s], 1);
        else to.page_table[b] = -1;   // cow-m1d: the source evicted between the host splice and here: revert the entry cold
    }
}
__global__ void cow_release_kernel(KvStreamMap m) {
    const long long b0 = m.slot_base, n = m.n_slots;
    for (long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x; b < m.n_blocks;
         b += (long long) gridDim.x * blockDim.x) {
        const int32_t s = m.page_table[b];
        if (s >= 0 && !(s >= b0 && s < b0 + n))
            atomicSub(&m.slot_use[s], 1);   // cow-m1d: drop a foreign pin (a fork source or a shared-region page).
                                            // No headroom refund: a fork splice never spent it and reset re-seeds it.
    }
}

}  // namespace

uint64_t kv_block_bytes(const QsaShapes& s, int fmt) {
    const uint64_t rows = (uint64_t) (s.n_head_kv * s.page_size);
    if (fmt == kKvHybrid)
        return rows * (uint64_t) s.head_dim + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 +
               rows * kv_q4_bytes_per_head((int) s.head_dim);
    if (fmt == kKvQ4) return rows * kv_q4_bytes_per_head((int) s.head_dim) * 2;
    return fmt == kKvInt8 ? rows * (uint64_t) s.head_dim * 2 + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 * 2
                : rows * (uint64_t) s.head_dim * 2 * 2;
}

void kv_stream_cow_fork_join(const KvStreamMap& to, const KvStreamMap& from, void* stream) {
    if (to.slot_use == nullptr || from.slot_use != to.slot_use) return;
    cow_fork_join_kernel<<<128, 512, 0, (cudaStream_t) stream>>>(to, from);
}

void kv_stream_cow_release(const KvStreamMap& m, void* stream) {
    if (m.slot_use == nullptr) return;
    cow_release_kernel<<<128, 512, 0, (cudaStream_t) stream>>>(m);
}

void kv_stream_reset(const KvStreamMap& m, void* stream) {
    reset_kernel<<<128, 256, 0, (cudaStream_t) stream>>>(m);
    check("reset");
}

void kv_stream_resolve(const KvStreamMap& m, const QsaAttnPools& slots, const KvHostPools& host, int fmt,
                       const int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap, const QsaShapes& s,
                       void* stream) {
    if (n_q <= 0) return;
    if (s.n_head_kv * s.page_size * (s.head_dim / KV_Q8_GROUP) * 2 % 16 != 0) {
        std::fprintf(stderr, "kv_stream: a block's scale run must be a multiple of 16 bytes\n");
        std::exit(1);
    }
    // one sweep step looks at RT consecutive slots `(hand + thread) % n_slots`; with fewer slots than RT two
    // threads see the same slot and may both take it for two different misses.  The engine never streams with fewer
    // than qsa_kv_resident_min() / page_size = 5,120 slots, so this is a guard, not a limit.
    if (m.n_slots < RT) {
        std::fprintf(stderr, "kv_stream: %lld slots is fewer than the resolve block (%d): the clock sweep would take a "
                             "slot twice\n", (long long) m.n_slots, RT);
        std::exit(1);
    }
    resolve_kernel<<<1, RT, 0, (cudaStream_t) stream>>>(m, ids, steps, (int) n_q, (int) cap, (int) s.page_size);
    check("resolve");
    copy_kernel<<<96, 128, 0, (cudaStream_t) stream>>>(m, runs_of(slots, host, fmt, s));
    check("copy");
}

void kv_stream_resolve_groups(const KvResolveGroups& g, const int32_t* ids, const int32_t* steps, int64_t cap,
                              const QsaShapes& s, void* stream) {
    if (g.ng <= 0) return;
    if (g.ng > kKvResolveMaxGroups) {
        std::fprintf(stderr, "kv_stream_resolve_groups: ng %d exceeds %d\n", g.ng, kKvResolveMaxGroups);
        std::exit(1);
    }
    if (s.n_head_kv * s.page_size * (s.head_dim / KV_Q8_GROUP) * 2 % 16 != 0) {
        std::fprintf(stderr, "kv_stream_resolve_groups: a block's scale run must be a multiple of 16 bytes\n");
        std::exit(1);
    }
    KvResolveMapGroups mg{};
    KvCopyGroups cg{};
    mg.ng = cg.ng = g.ng;
    for (int i = 0; i < g.ng; ++i) {
        if (g.ntok[i] < 1 || g.map[i].n_slots < RT || !g.map[i].page_table || !g.map[i].ctl) {
            std::fprintf(stderr, "kv_stream_resolve_groups: invalid group %d (ntok=%d, n_slots=%lld)\n", i, g.ntok[i],
                         (long long) g.map[i].n_slots);
            std::exit(1);
        }
        // Hard invariant: every group MUST target a physically distinct KvStreamMap so CTAs never race on one map.
        for (int j = 0; j < i; ++j) {
            if (g.map[i].page_table == g.map[j].page_table || g.map[i].ctl == g.map[j].ctl) {
                std::fprintf(stderr, "kv_stream_resolve_groups: duplicate KvStreamMap between groups %d and %d\n", j, i);
                std::exit(1);
            }
        }
        mg.map[i] = g.map[i];
        mg.first[i] = g.first[i];
        mg.ntok[i] = g.ntok[i];
        cg.ctl[i] = g.map[i].ctl;
        cg.miss_block[i] = g.map[i].miss_block;
        cg.miss_slot[i] = g.map[i].miss_slot;
        cg.runs[i] = runs_of(g.slots[i], g.host[i], g.fmt[i], s);
    }
    const cudaStream_t cs = (cudaStream_t) stream;
    resolve_groups_kernel<<<(unsigned) g.ng, RT, 0, cs>>>(mg, ids, steps, (int) cap, (int) s.page_size);
    check("resolve groups");
    const unsigned copy_x = g.ng <= 1 ? 96u : (g.ng == 2 ? 48u : 32u);
    copy_groups_kernel<<<dim3(copy_x, (unsigned) g.ng), 128, 0, cs>>>(cg);
    check("copy groups");
}

void kv_ring_table(int32_t* page_table, int64_t n_blocks, int64_t n_slots, void* stream) {
    ring_kernel<<<64, 256, 0, (cudaStream_t) stream>>>(page_table, n_blocks, n_slots);
    check("ring table");
}

void kv_ring_restore(const QsaAttnPools& slots, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                     int64_t n_slots, const QsaShapes& s, void* stream) {
    const Runs r = runs_of(slots, host, fmt, s);
    for (int64_t b = b0; b < b1;) {
        const int64_t sl = b % n_slots, run = std::min<int64_t>(b1 - b, n_slots - sl);   // up to the ring's end
        for (int a = 0; a < r.n; ++a)
            if (cudaMemcpyAsync(r.dst[a] + sl * r.len[a], r.src[a] + b * r.len[a], (size_t) (run * r.len[a]),
                                cudaMemcpyDefault, (cudaStream_t) stream) != cudaSuccess)
                check("ring restore");
        b += run;
    }
}

void kv_stage_from_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t n_blocks,
                        const QsaShapes& s, void* stream) {
    if (n_blocks <= 0) return;
    const Runs r = runs_of(stage, host, fmt, s);
    for (int a = 0; a < r.n; ++a)
        if (cudaMemcpyAsync(r.dst[a], r.src[a], (size_t) (n_blocks * r.len[a]), cudaMemcpyDefault,
                            (cudaStream_t) stream) != cudaSuccess)
            check("stage");
}

void kv_unstage_to_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                        const QsaShapes& s, void* stream) {
    if (b1 <= b0) return;
    const Runs r = runs_of(stage, host, fmt, s);   // src: the host copy, dst: the staging pool (identity layout both)
    for (int a = 0; a < r.n; ++a)
        if (cudaMemcpyAsync((void*) (r.src[a] + b0 * r.len[a]), r.dst[a] + b0 * r.len[a], (size_t) ((b1 - b0) * r.len[a]),
                            cudaMemcpyDefault, (cudaStream_t) stream) != cudaSuccess)
            check("unstage");
}

KvStreamCounters kv_stream_counters(const KvStreamMap& m) {
    int32_t c[kKvCtlInts] = {};
    KvStreamCounters r;
    if (m.ctl == nullptr || cudaMemcpy(c, m.ctl, sizeof(c), cudaMemcpyDeviceToHost) != cudaSuccess) return r;
    const unsigned long long* u = reinterpret_cast<const unsigned long long*>(c + 4);
    r.misses = u[0];
    r.lookups = u[1];
    r.calls = u[2];
    r.overflow = c[3] != 0;
    r.overflow_count = (uint32_t) c[3];
    return r;
}

}  // namespace strata::kernels
