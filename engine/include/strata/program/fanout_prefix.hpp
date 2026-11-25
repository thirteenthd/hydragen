// HydraGen 2A: where a fan-out's shared header gets a checkpoint of its own.
//
// An agent harness farms several requests off one long shared header.  A hybrid GDN+QSA model can only resume from a
// saved conversation checkpoint whose length sits at or before the point where the siblings diverge, and at
// --prefill auto:N a header shorter than N runs as ONE chunk that saves its checkpoint at the last turn boundary -
// *after* the siblings have already diverged.  The next sibling then matches nothing and reads the whole header
// again: the G9 fan-out probe measured cached_tokens 0 for S1..S3 against the control's 6144.
//
// The fix is a cut: split the prompt read at the length another stream of the same conversation really shares with
// this prompt, so the header gets a checkpoint of its own and the siblings after it resume from there.  These two
// helpers are the whole decision; the caller feeds them the state the request already walks (the live session, the
// checkpoints, the batch slots, the parked conversations), so solo or unrelated traffic finds no peer and pays
// nothing, and the cut only moves where a checkpoint that would be saved anyway is saved.
#ifndef STRATA_PROGRAM_FANOUT_PREFIX_HPP
#define STRATA_PROGRAM_FANOUT_PREFIX_HPP

#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::program {

/// Another stream's state to compare this prompt against, not owned.  `ids` are the tokens that state holds (a
/// checkpoint's prefix, a batch slot's prompt, a parked conversation's live tokens) - stored as int32_t everywhere
/// in the cache, while the incoming request prompt is int64_t, hence the templated query below.  A state that
/// carries pictures is not a peer: reusing it needs the picture keys, and a fan-out's header has none, so the caller
/// only offers states with no pictures on them.  (`near` is a Windows macro from windef.h: never a name here.)
struct FanoutPeer {
    const std::vector<int32_t>& ids;
};

/// The longest token prefix this prompt shares with any peer, counted from token 0: the siblings of a fan-out start
/// at the same first token, and a state that diverges earlier cannot be resumed from deeper than that.
template <typename Token>
inline int64_t fanout_lcp(const std::vector<Token>& ids, const std::vector<FanoutPeer>& peers) {
    int64_t best = 0;
    for (const FanoutPeer& p : peers) {
        const int64_t n = std::min<int64_t>((int64_t) ids.size(), (int64_t) p.ids.size());
        if (n <= best) continue;                // too short to beat a match already found
        int64_t i = 0;                          // a prefix is counted from token 0: no resuming past `best`
        while (i < n && ids[(size_t) i] == p.ids[(size_t) i]) ++i;
        if (i > best) best = i;
    }
    return best;
}

/// Where to cut the prompt read so the shared header gets its own checkpoint, or -1 for "no cut".
///   lcp         - the longest prefix shared with another stream (fanout_lcp, or a parked conversation's match)
///   resume      - where this request already resumes: a cut at or below it buys nothing
///   min_len     - the shortest shared header worth a checkpoint of its own, and the shortest gain over `resume`
///                 (HG_FANOUT_PREFIX_MIN)
///   near_tol    - a checkpoint already saved within this many tokens of the shared length covers it (256)
///   cuts        - the lengths this prompt read already stops at (root / message / turn / pin / reread / end)
///   checkpoints - the lengths checkpoints already exist at, in every state this request could resume from
template <typename Token>
inline int64_t fanout_prefix_cut(const std::vector<Token>& ids, int64_t lcp, int64_t resume, int64_t min_len,
                                 int64_t near_tol, const std::vector<int64_t>& cuts,
                                 const std::vector<int64_t>& checkpoints) {
    if (ids.size() < 2 || lcp < 1 || min_len < 0) return -1;
    if (lcp < min_len || lcp < resume + min_len) return -1;   // nothing worth having, or nothing this request gains
    const int64_t n = (int64_t) ids.size();
    if (lcp > n - 1) return -1;   // the last prompt token always starts the first verify window
    for (const int64_t c : cuts)
        if (c == lcp) return -1;                                  // the read already stops there
    for (const int64_t c : checkpoints)
        if (c <= lcp && c + near_tol >= lcp) return -1;   // a checkpoint at or before the shared length covers it
    // (A checkpoint DEEPER than the shared length is on another branch - this prompt diverges before it, so it can
    // never resume from there and it must not suppress the cut.  G9 is that case: the seed's only checkpoint sat at
    // 11,784, 33 tokens past where the siblings diverge at 11,751, and every sibling still re-read the header.)
    return lcp;
}

}  // namespace strata::program

#endif
