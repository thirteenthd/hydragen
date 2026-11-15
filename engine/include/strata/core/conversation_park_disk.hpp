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
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::core {

class ParkDisk {
public:
    struct Options {
        std::string dir;            // created (0700) if missing; the tier is off when empty
        size_t budget = 0;          // STRATA_PARK_DISK_MIB; 0: no cap, the free-space guard alone
        uint64_t min_free = 0;      // STRATA_PARK_DISK_MIN_FREE_MIB: keep this much disk free
        bool v2 = false;            // STRATA_PARK_DISK_V2=1: deduped block files (POSIX; v1 stays the fallback)
    };

    // The knob triple, from the environment. nullopt when the tier is off.
    static std::optional<Options> from_env();

    // M-C3b3: one parent image materialized per assembly read (public: the read_v2 default argument uses it).
    struct ViewFeed {
        std::unordered_map<uint64_t, SavedConversation> images;
        std::vector<uint64_t> parents_seen;   // M-C6b: parents this file's tables fed from (boot re-pins)
        uint64_t cur_par = 0, cur_aux = UINT64_MAX, cur_off = 0;
        const SavedConversation* cur_img = nullptr;
        int depth = 0;
    };

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
    void spill(SavedConversation&& image, int64_t cache_serial);

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
    size_t blobs() const { return blobs_.size(); }         // v2: distinct chunk files held
    uint64_t blob_bytes() const { return blob_bytes_; }
    uint64_t saved() const { return saved_bytes_; }        // v2: spill bytes skipped as existing chunks
    uint64_t spills() const { return spills_; }
    uint64_t loads() const { return loads_; }
    uint64_t drops() const { return drops_; }

private:
    // M-C3b v2: a shared chunk key - a 128-bit content digest over one 16 MiB block of the payload
    // stream (SessionHasher rounds, two seeds; corruption checks, not authentication, same as STRSES).
    struct PdKey {
        uint64_t k0 = 0, k1 = 0;
        bool operator==(const PdKey& o) const { return k0 == o.k0 && k1 == o.k1; }
    };
    struct PdKeyHash {
        size_t operator()(const PdKey& k) const { return (size_t) (k.k0 ^ (k.k1 >> 32) ^ k.k1); }
    };
    struct Entry {
        uint64_t serial = 0;
        std::string path;
        size_t bytes = 0;
        std::vector<int32_t> ids;   // the spilled live state's tokens: the prefix-match key
        bool cvec = true;
        std::vector<PdKey> blob_keys;   // v2: the file's blob blocks with duplicates (one refcount step per occurrence)
        size_t file_bytes = 0;          // v2: the real .ses footprint once written (the spill estimate is RAM-sized)
        bool done = false;          // the rename landed: matchable
        std::vector<uint64_t> view_parents;   // M-C3b3: disk serials this file views (kept off the eviction path)
        uint32_t view_refs = 0;               // M-C3b3: files that view THIS entry; eviction prefers refs==0
        int64_t cache_serial = -1;            // M-C3b3: RAM-cache slot serial at spill (the donor lookup key)
        std::chrono::steady_clock::time_point spilled_at;
    };

    struct Write {                  // one queued spill: path paired at spill time (entries_ is serve-side only)
        uint64_t serial = 0;
        std::string path;
        SavedConversation image;
        // M-C3b3: payload spans standing in for the PARENT's bytes: (memory span, parent disk serial,
        // (kv_layer << 3) | part). Pointers stay valid: the segment stores live in the moved image.
        struct ViewSpan { const void* p; size_t len; uint64_t parent; uint32_t aux; };
        std::vector<ViewSpan> views;
    };

    struct Finished {               // the writer reports every write here; the serve thread applies it
        uint64_t serial = 0;
        bool ok = false;
        std::string error;
        double ms = 0;
        // v2: the block outcome, applied by the serve thread when the write is drained
        std::vector<PdKey> all_keys;                       // every blob key in the table, with duplicates
        std::vector<std::pair<PdKey, uint64_t>> created;   // chunk files this write published (key, len)
        size_t file_bytes = 0;                             // the published .ses size (0: this was a v1 write)
        uint64_t saved = 0;                                // bytes skipped because the chunk already existed
        std::vector<uint64_t> view_parents;                // M-C3b3: parents this file views (refs applied on drain)
    };

    void write_v2(const Write& w, const SessionFileIdentity& id, Finished& f);   // falls through to a v1 write itself
    bool read_v2(const std::string& path, const SessionFileIdentity& id, const SessionReadLimits& limits,
                 SavedConversation& image, size_t& bytes, std::string& error, SessionStatus& st,
                 std::vector<PdKey>* keys, ViewFeed* vf = nullptr);
    // M-C5 pass 1: header/table scan of a v2 file: the chunk keys it names, no payload read, no
    // identity check: a quarantined foreign file's keys shield shared bytes from the boot sweep.
    bool scan_v2_keys(const std::string& path, std::vector<PdKey>& keys, std::string& error) const;
    // M-C3b3: one parent image materialized per assembly read (kept while the child builds; b3a copies the
    // bytes, b3b turns them into adopt_shared views). Depth-capped chains; a missing parent fails the child
    // closed (the caller deletes the orphan, the conversation re-prefills).
    const SavedConversation* view_image(uint64_t serial, const SessionFileIdentity& id,
                                        const SessionReadLimits& limits, ViewFeed& vf,
                                        std::string& error, SessionStatus& st);
    static bool file_is_v2(const std::string& path);
    std::string blob_path(const PdKey& k) const;
    void release_entry(const Entry& e);   // drop this entry's chunk refs (unlink at the last holder)
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
    bool v2_ = false;
    std::string chunks_dir_;                            // <dir>/chunks: the shared chunk store (v2)
    struct BlobRef { uint32_t refs = 0; uint64_t len = 0; };
    std::unordered_map<PdKey, BlobRef, PdKeyHash> blobs_;   // serve thread only (init/drain/drop/load)
    uint64_t blob_bytes_ = 0, saved_bytes_ = 0;
    std::unordered_map<int64_t, uint64_t> cache_to_park_;  // M-C3b3: RAM slot serial -> disk serial (lineage)

    std::thread worker_;
    bool stop_ = false;
};

} // namespace strata::core
