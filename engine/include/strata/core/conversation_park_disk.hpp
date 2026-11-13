// M-C3a: an optional disk tier under the parked-conversation cache (STRATA_PARK_DISK=DIR).
//
// The parked cache is host RAM and its LRU throws a conversation's whole image away: at 100k depth an
// evicted sibling pays a ~44 s re-prefill (or ~20 s remount + ~1 s read once tiered). Eviction spills the
// image to a STRSES file (conversation_file.hpp: identity-bound, hash-verified, temp-file-and-rename publish);
// an admission that finds its best prefix on disk reads it back and restores it through the normal
// conversation-snapshot path. Files are trusted exactly as disk sessions are: identity and payload hashes
// decide, and any refusal falls back to reading the prompt (fail-closed, no partial state).
//
// The RAM cache's own rules are untouched: pinned prefixes and the guarded donor are never evicted, so they
// never reach the tier; drop_superseded and clear() drop without spilling (stale copies / poisoned state).
// A spilled image is a FULL copy: M-C2 segment views serialize to their logical bytes, and a restored
// conversation re-anchors against the RAM cache the next time it parks (M-C3b replaces this with deduped
// segment-view files).
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

class ParkDisk {
public:
    struct Options {
        std::string dir;            // created (0700) if missing; the tier is off when empty
        size_t budget = 0;          // STRATA_PARK_DISK_MIB; 0: no cap, the free-space guard alone
        uint64_t min_free = 0;      // STRATA_PARK_DISK_MIN_FREE_MIB: keep this much disk free
    };

    // The knob triple, from the environment. nullopt when the tier is off.
    static std::optional<Options> from_env();

    // `identity`: what files are bound to (model + resolved config fingerprints; cached after the first call
    // under one lock, because the spill worker and serve-thread loads both reach it).
    // `limits`: this engine's read limits (geometry, layer range, per-part bounds) INCLUDING the RAM preflight,
    // so a 1.35 GB image passes the same gate the RAM cache uses before it is ever allocated. Called on the
    // serve thread only (init and load).
    using IdentityFn = std::function<bool(SessionFileIdentity&, std::string&)>;
    using LimitsFn = std::function<bool(SessionReadLimits&, std::string&)>;

    // Creates the folder, rebuilds the index from files already there (full validated reads: mismatches and
    // corruption are refused and the file deleted), and starts the writer. Called once, before serving.
    void init(const Options& o, IdentityFn identity, LimitsFn limits);

    ~ParkDisk();   // drains queued spills, then joins the writer

    bool enabled() const { return !dir_.empty(); }

    // A RAM-cache victim goes here (conversation_cache.hpp spill_hook). The image is queued for the writer;
    // its bytes count against the disk budget from the moment it is queued, and it becomes matchable when
    // its rename lands. Disk room (budget + free guard) is enforced here, oldest entry first.
    void spill(SavedConversation&& image);

    struct Match {
        uint64_t serial = 0;    // entries leave through the writer's failure drain as well, so an index would
                                // shift under a pending load; a serial never does
        int64_t tokens = 0;     // the shared token prefix length
    };

    // The deepest disk prefix reaching this prompt. Fail-closed like common_best: picture-carrying requests
    // never match (their entries' pictures would have to re-verify), and cvec must agree. An entry LONGER
    // than the prompt can still be a match: a parked sibling that grew past this prompt shares its head.
    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        Match best;
        if (dir_.empty() || prompt.empty() || !images.empty()) return best;   // pictures: fail closed
        for (const auto& e : entries_) {
            if (!e.done || e.cvec != cvec) continue;
            const size_t m = std::min(e.ids.size(), prompt.size());
            int64_t n = 0;
            while (n < (int64_t) m && (int64_t) e.ids[(size_t) n] == (int64_t) prompt[(size_t) n]) ++n;
            if (n > best.tokens) best = {e.serial, n};
        }
        return best;
    }

    // Re-reads the matched file (identity, limits, payload hash - all of it). Null on any failure; a file that
    // fails VALIDATION is deleted from the tier, a file that merely could not be read (RAM preflight, io) is
    // kept for a later turn. The caller restores the image through conversation_snapshot_validate/restore.
    std::optional<SavedConversation> load(Match m, std::string& error);

    // The matched entry's tokens: for the caller's own covers/prefix arithmetic.
    std::vector<int32_t> ids(uint64_t serial) const;

    size_t entries() const { return entries_.size(); }
    size_t bytes() const { return bytes_; }
    uint64_t spills() const { return spills_; }
    uint64_t loads() const { return loads_; }
    uint64_t drops() const { return drops_; }

private:
    struct Entry {
        uint64_t serial = 0;
        std::string path;
        size_t bytes = 0;
        std::vector<int32_t> ids;   // the spilled live state's tokens: the prefix-match key
        bool cvec = true;
        bool done = false;          // the rename landed: matchable
        std::chrono::steady_clock::time_point spilled_at;
    };

    struct Write {                  // one queued spill: path paired at spill time (entries_ is serve-side only)
        uint64_t serial = 0;
        std::string path;
        SavedConversation image;
    };

    struct Finished {               // the writer reports every write here; the serve thread applies it
        uint64_t serial = 0;
        bool ok = false;
        std::string error;
        double ms = 0;
    };

    void drain_finished();          // mark done / drop failed, at the top of every serve-thread call
    void writer();                  // one file at a time; serialized for LRU truth and disk bandwidth
    bool free_guard_ok(size_t incoming) const;   // statvfs against min_free
    void drop_oldest();             // free disk room (file + index + any still-queued image)
    std::optional<SessionFileIdentity> identity_cached(std::string& error);

    std::string dir_;
    size_t budget_ = 0;
    uint64_t min_free_ = 0;
    IdentityFn identity_fn_;
    LimitsFn limits_fn_;

    std::vector<Entry> entries_;    // serve thread: spill(), best(), load(), drain_finished()
    size_t bytes_ = 0;              // queued + written bytes on disk
    uint64_t next_serial_ = 1;
    uint64_t spills_ = 0, loads_ = 0, drops_ = 0;

    std::mutex write_mtx_;          // identity cache only (worker + load)
    std::optional<SessionFileIdentity> identity_;

    std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<Write> queued_;
    std::deque<Finished> finished_;
    std::thread worker_;
    bool stop_ = false;
};

} // namespace strata::core
