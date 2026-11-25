// fanout_prefix_test.cpp: unit tests for the HydraGen 2A fan-out prefix cut (header-only, no GPU, no model).
//
// The case that started it: an agent harness sends one seed and three siblings that share an 11,751-token header.
// The seed's read saved its only checkpoint at the last turn boundary, 11,784 tokens - past where the siblings
// diverge - so every sibling re-read the whole header (cached_tokens 0).  The cut puts a checkpoint at 11,751.
#include "strata/program/fanout_prefix.hpp"

#include <cstdio>
#include <vector>

using strata::program::FanoutPeer;
using strata::program::fanout_lcp;
using strata::program::fanout_prefix_cut;

static int fails = 0;
static int checks = 0;
#define CHECK(c)                                                             \
    do {                                                                     \
        ++checks;                                                            \
        if (!(c)) {                                                          \
            std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c);          \
            ++fails;                                                         \
        }                                                                    \
    } while (0)

// a prompt of `n` tokens whose first `head` tokens are the shared header
static std::vector<int32_t> prompt(int64_t n, int64_t head, int32_t tail_seed) {
    std::vector<int32_t> ids;
    ids.reserve((size_t) n);
    for (int64_t i = 0; i < head; ++i) ids.push_back((int32_t) (i + 1));
    for (int64_t i = head; i < n; ++i) ids.push_back((int32_t) (100000 + tail_seed + i));
    return ids;
}

int main() {
    const int64_t head = 11751, n = 12743, min_len = 2048, near_tol = 256;
    const std::vector<int32_t> ids = prompt(n, head, 7);
    const std::vector<int32_t> none;

    // ---- fanout_lcp -----------------------------------------------------------------
    CHECK(fanout_lcp(ids, {FanoutPeer{none}}) == 0);
    // the seed's checkpoint: the shared header plus its own 33-token tail
    std::vector<int32_t> seed = prompt(n, head, 1);
    seed.resize(head + 33);
    CHECK(fanout_lcp(ids, {FanoutPeer{seed}}) == head);
    // the deepest peer wins, and a peer that diverges at the first token counts for nothing
    std::vector<int32_t> shallow = ids;
    shallow[(size_t) 4] = -1;
    CHECK(fanout_lcp(ids, {FanoutPeer{shallow}, FanoutPeer{seed}}) == head);
    std::vector<int32_t> unrelated = prompt(n, 0, 3);
    CHECK(fanout_lcp(ids, {FanoutPeer{unrelated}}) == 0);
    // a shared prefix is counted from token 0: a peer that diverges early but agrees later must not extend it
    std::vector<int32_t> holed = ids;
    holed[(size_t) 3] = -1;
    CHECK(fanout_lcp(ids, {FanoutPeer{holed}}) == 3);
    CHECK(fanout_lcp(ids, {FanoutPeer{seed}, FanoutPeer{holed}}) == head);
    // a peer longer than this prompt still stops at this prompt's end
    std::vector<int32_t> longer = prompt(n + 500, head, 1);
    CHECK(fanout_lcp(ids, {FanoutPeer{longer}}) == head);
    // the serve path's prompt is int64_t while every stored state is int32_t: same decision, both widths
    const std::vector<int64_t> wide(ids.begin(), ids.end());
    CHECK(fanout_lcp(wide, {FanoutPeer{seed}}) == head);
    CHECK(fanout_prefix_cut(wide, head, 0, min_len, near_tol, {n - 1}, {}) == head);

    // ---- the cut itself -------------------------------------------------------------
    // a sibling of a fan-out already under way: cut at the shared length
    CHECK(fanout_prefix_cut(ids, head, 0, min_len, near_tol, {n - 1}, {}) == head);
    // solo traffic: no peer, so no lcp and no cut
    CHECK(fanout_prefix_cut(ids, 0, 0, min_len, near_tol, {n - 1}, {}) == -1);
    // a header shorter than HG_FANOUT_PREFIX_MIN is cheaper to read again than the extra checkpoint costs
    CHECK(fanout_prefix_cut(ids, 1000, 0, min_len, near_tol, {n - 1}, {}) == -1);
    // ... and so is a gain too small over where the request already resumes
    CHECK(fanout_prefix_cut(ids, head, head - 100, min_len, near_tol, {n - 1}, {}) == -1);
    CHECK(fanout_prefix_cut(ids, head, head, min_len, near_tol, {n - 1}, {}) == -1);
    // a checkpoint already saved at or just before the shared length covers it: no second one
    CHECK(fanout_prefix_cut(ids, head, 0, min_len, near_tol, {n - 1}, {head - near_tol}) == -1);
    CHECK(fanout_prefix_cut(ids, head, 0, min_len, near_tol, {n - 1}, {head - near_tol - 1}) == head);
    // a checkpoint DEEPER than the shared length is on another branch: this prompt cannot resume from it, so it
    // does not suppress the cut.  The G9 shape: the seed's only checkpoint at head+33, siblings diverging at head.
    CHECK(fanout_prefix_cut(ids, head, 0, min_len, near_tol, {n - 1}, {head + near_tol}) == head);
    CHECK(fanout_prefix_cut(ids, head, 0, min_len, near_tol, {n - 1}, {head + 33}) == head);
    CHECK(fanout_prefix_cut(ids, head, 0, min_len, near_tol, {n - 1}, {head}) == -1);
    // the read already stops there (the turn boundary, the root, the pin, the prompt's end)
    CHECK(fanout_prefix_cut(ids, head, 0, min_len, near_tol, {head, n - 1}, {}) == -1);
    CHECK(fanout_prefix_cut(ids, n - 1, 0, min_len, near_tol, {n - 1}, {}) == -1);
    // bounds: a cut that would hold the last prompt token, and a degenerate prompt
    CHECK(fanout_prefix_cut(ids, n, 0, min_len, near_tol, {}, {}) == -1);
    CHECK(fanout_prefix_cut(std::vector<int32_t>{1, 2}, 2, 0, 1, near_tol, {}, {}) == -1);
    CHECK(fanout_prefix_cut(std::vector<int32_t>{1, 2, 3}, 2, 0, 1, near_tol, {}, {}) == 2);
    // HG_FANOUT_PREFIX_MIN=0 still needs a real shared prefix
    CHECK(fanout_prefix_cut(ids, 0, 0, 0, near_tol, {n - 1}, {}) == -1);
    CHECK(fanout_prefix_cut(ids, 1, 0, 0, near_tol, {n - 1}, {}) == 1);

    if (fails == 0) std::printf("fanout_prefix_test: %d checks passed\n", checks);
    return fails != 0;
}
