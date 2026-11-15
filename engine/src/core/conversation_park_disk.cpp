// M-C3a implementation. All file I/O goes through conversation_file.cpp (STRSES v1): identity binding,
// bounded parse, payload hashes, temp-file-and-rename publish, mode 0600 files under a 0700 folder.
#include "strata/core/conversation_park_disk.hpp"
#include "strata/core/conversation_snapshot.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <unordered_map>

#ifndef _WIN32
#include <cerrno>
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unordered_set>
#include <unistd.h>
#else
#include <windows.h>
#endif

namespace strata::core {

namespace {

std::optional<uint64_t> pd_env_u64(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) return std::nullopt;
    char* end = nullptr;
    const unsigned long long n = std::strtoull(v, &end, 10);
    if (end == v) return std::nullopt;
    return n;
}

std::string pd_tier_path(const std::string& dir, uint64_t serial) {
    // fixed-width serial: a lexicographic folder scan then keeps spill (LRU) order across a restart
    char buf[24];
    std::snprintf(buf, sizeof buf, "/park-%09llu.ses", (unsigned long long) serial);
    return dir + buf;
}

// ---- M-C3b v2 envelope: 128 B header, a 64 KiB fixed table area (32 B per block, so at most 2048
// blocks per file = an 8 GiB payload ceiling), then the INLINE block body. Blocks >= 64 KiB are chunk
// files under <dir>/chunks/<32 hex>.bin instead; the table says which, per block, in stream order.
constexpr char kPd2Magic[8] = {'S', 'T', 'R', 'P', 'K', 'D', '4', 0};
constexpr uint64_t kPd2Hdr = 128, kPd2TblArea = 65536, kPd2Ent = 32;
constexpr uint64_t kPd2Body = kPd2Hdr + kPd2TblArea, kPd2MaxBlocks = kPd2TblArea / kPd2Ent;
constexpr size_t kPd2Block = 16u << 20;              // one block's hard ceiling (also the table's max len)
constexpr uint64_t kPd2HashSeed1 = 0x9E3779B97F4A7C15ull;
// M-C3b1c: content-defined chunking. A fixed 16 MiB grid never repeats its cuts across generations of
// one conversation (the ids list and every layer's KV region each grow internally, so all downstream
// boundaries shift and no chunk key is ever seen twice: 0 skips in 142 spills of the first v2 sweep).
// Gear-hash cuts (h = (h << 1) + G[byte]; cut when the low bits drop to zero) depend only on local
// bytes, so an unchanged span between two growth points re-chunks byte-identically: re-parking a deep
// session then ships only its new tails. Sizes: cut checks start at 2 MiB, the expected cut is 4 MiB
// (mask 22 bits) and 16 MiB forces one - so a session's blocks are ~2048 at worst for a 4 GiB image,
// inside the 2048-entry table; larger images overflow and fall back to v1, fail-closed as always.
constexpr uint64_t kPd2CutMask = (1ull << 21) - 1;   // M-C3b1d: mean cut 2 MiB; keeps a 131k image under the 2048-entry table
constexpr size_t kPd2CutMin = 512u << 10;   // M-C3b1d: 6 MiB equal spans need cuts well below 2 MiB
inline const uint64_t* pd_gear_table() {
    static const uint64_t* g = [] {
        static uint64_t tab[256];
        uint64_t s = 0x2545F4914F6CDD1Dull;            // fixed: keys must survive restarts
        for (int i = 0; i < 256; ++i) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;  // xorshift64*, deterministic across processes
            tab[i] = s;
        }
        return tab;
    }();
    return g;
}

std::string pd_hexkey(uint64_t k0, uint64_t k1) {
    char b[33];
    std::snprintf(b, sizeof b, "%016llx%016llx", (unsigned long long) k0, (unsigned long long) k1);
    return b;
}

bool pd_key_from_hex(const std::string& hex, uint64_t& k0, uint64_t& k1) {
    if (hex.size() != 32) return false;
    char* e = nullptr;
    k0 = std::strtoull(hex.substr(0, 16).c_str(), &e, 16); if (!e || *e) return false;
    k1 = std::strtoull(hex.substr(16).c_str(), &e, 16); if (!e || *e) return false;
    return true;
}

} // namespace

std::optional<ParkDisk::Options> ParkDisk::from_env() {
    const char* dir = std::getenv("STRATA_PARK_DISK");
    if (!dir || !*dir) return std::nullopt;
    Options o;
    o.dir = dir;
    o.budget = (size_t) pd_env_u64("STRATA_PARK_DISK_MIB").value_or(0) * 1024 * 1024;
    o.min_free = pd_env_u64("STRATA_PARK_DISK_MIN_FREE_MIB").value_or(4096) * 1024ull * 1024ull;
    o.v2 = pd_env_u64("STRATA_PARK_DISK_V2").value_or(0) != 0;
    return o;
}

void ParkDisk::init(const Options& o, IdentityFn identity, LimitsFn limits) {
    dir_ = o.dir;
    budget_ = o.budget;
    min_free_ = o.min_free;
    identity_fn_ = std::move(identity);
    limits_fn_ = std::move(limits);
    v2_ = o.v2;
#ifdef _WIN32
    if (v2_) { std::fprintf(stderr, "strata serve: park disk: STRATA_PARK_DISK_V2 is POSIX-only for now: staying on v1\n"); v2_ = false; }
#endif
    std::string error;
#ifndef _WIN32
    if (mkdir(dir_.c_str(), 0700) != 0 && errno != EEXIST) {
        std::fprintf(stderr, "strata serve: park disk: DISABLED: mkdir %s: %s\n",
                     dir_.c_str(), std::strerror(errno));
        dir_.clear();
        return;
    }
#else
    if (!CreateDirectoryA(dir_.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS) {
        std::fprintf(stderr, "strata serve: park disk: DISABLED: CreateDirectory %s\n", dir_.c_str());
        dir_.clear();
        return;
    }
#endif
    if (v2_) {
        chunks_dir_ = dir_ + "/chunks";
        if (mkdir(chunks_dir_.c_str(), 0700) != 0 && errno != EEXIST) {
            std::fprintf(stderr, "strata serve: park disk: v2 DISABLED: mkdir %s: %s\n",
                         chunks_dir_.c_str(), std::strerror(errno));
            v2_ = false;
        }
    }
    const auto boot_serial = [](const std::string& path) -> uint64_t {
        size_t base = path.rfind('/') + 1;
        const size_t bs = path.rfind('\\');
        if (bs != std::string::npos && bs + 1 > base) base = bs + 1;
        if (path.compare(base, 5, "park-") != 0) return 0;
        return std::strtoull(path.c_str() + base + 5, nullptr, 10);   // park-000000081.ses -> 81
    };
    // M-C5: TWO-PHASE boot rebuild. Pass 1 (this walk) indexes and classifies every file WITHOUT touching
    // chunk bytes: `invalid` (corrupt: the file's own bytes) unlink is per-file safe; `foreign` (another
    // model/configuration/geometry: permanent) QUARANTINES the file in place and phantom-references the
    // chunk keys it names, so no sweep can eat bytes still shared (park v2 dedup + b3a lineage make chunks
    // belong to MANY files); io/memory stops the walk and keeps everything. Pass 2 (the sweep below) runs
    // ONLY if the walk completed, and unlinks ONLY chunks unreferenced by accepted entries and quarantined
    // files alike. Before this, one foreign refuse mid-walk cascaded through shared chunks and wiped the
    // whole store on every boot (CHANGES 151).
    const auto t0 = std::chrono::steady_clock::now();
    size_t found = 0, refused = 0, quarantined = 0;
    bool walk_ok = true;
    std::vector<std::string> paths;
#ifndef _WIN32
    if (DIR* d = opendir(dir_.c_str())) {
        while (const dirent* e = readdir(d)) {
            const std::string name = e->d_name;
            if (name.rfind("park-", 0) == 0 && name.size() > 8 && name.compare(name.size() - 8, 8, ".ses.tmp") == 0) {
                ::unlink((dir_ + "/" + name).c_str());   // half-written spill of an old boot: no live writer can publish it
                continue;
            }
            if (name.rfind("park-", 0) == 0 && name.size() > 4 && name.compare(name.size() - 4, 4, ".ses") == 0)
                paths.push_back(dir_ + "/" + name);
        }
        closedir(d);
    }
#else
    WIN32_FIND_DATAA fd;
    if (HANDLE h = FindFirstFileA((dir_ + "\\park-*.ses").c_str(), &fd); h != INVALID_HANDLE_VALUE) {
        do paths.push_back(dir_ + "\\" + fd.cFileName); while (FindNextFileA(h, &fd));
        FindClose(h);
    }
#endif
    std::sort(paths.begin(), paths.end());   // serial order: the LRU keeps its spill order across a restart
    for (const auto& path : paths) {
        const auto id = identity_cached(error);   // the fingerprint is computed once; a failure is permanent
        if (!id) { ++refused; walk_ok = false; break; }
        // v2 files index through the block read even when the v2 WRITE knob is off: readers speak both.
        if (file_is_v2(path)) {
            SavedConversation v2img;
            size_t v2bytes = 0;
            SessionReadLimits v2lim;
            if (limits_fn_ && !limits_fn_(v2lim, error)) {
                std::fprintf(stderr, "strata serve: park disk: index stop at %s (%s)\n", path.c_str(), error.c_str());
                walk_ok = false;
                break;
            }
            SessionStatus v2st;
            std::vector<PdKey> v2keys;
            ViewFeed v2vf;   // M-C3b3: boot rebuild assembles view children through their parents too
            if (!read_v2(path, *id, v2lim, v2img, v2bytes, error, v2st, &v2keys, &v2vf)) {
                std::fprintf(stderr, "strata serve: park disk: index refuse %s (%s)\n", path.c_str(), error.c_str());
                ++refused;
                const uint64_t ser = boot_serial(path);
                if (ser >= next_serial_) next_serial_ = ser + 1;   // never rename a new spill over this file
                if (v2st.error == SessionError::foreign) {
                    // quarantine in place: keep the file, shield the chunk keys it names (a later boot with
                    // the matching geometry reads them back: content-addressed bytes, shared by design)
                    std::vector<PdKey> qkeys;
                    std::string qerr;
                    if (scan_v2_keys(path, qkeys, qerr)) {
                        for (const auto& k : qkeys) {
                            BlobRef& br = blobs_[k];
                            if (br.refs == 0) {
                                struct stat cs;
                                br.len = (::stat(blob_path(k).c_str(), &cs) == 0 && cs.st_size > 0) ? (uint64_t) cs.st_size : 0;
                                bytes_ += br.len;
                                blob_bytes_ += br.len;
                            }
                            br.refs += 1;   // phantom ref: released only when a future boot reclassifies the file
                        }
                    } else if (!qerr.empty())
                        std::fprintf(stderr, "strata serve: park disk: quarantine scan %s (%s)\n", path.c_str(), qerr.c_str());
                    ++quarantined;
                    continue;
                }
                if (v2st.error == SessionError::memory || v2st.error == SessionError::io) {
                    walk_ok = false;   // not the file's fault: the rest waits for a later boot, nothing churns
                    break;
                }
                ::unlink(path.c_str());   // corrupt: the file's own bytes; shared chunks are protected by refs, swept in pass 2 only
                continue;
            }
            if (v2img.live.ids.empty() || v2img.kv.empty()) { ++refused; continue; }
            Entry en;
            en.serial = boot_serial(path);   // M-C5: keep the file's own serial: view tables name parents by it
            if (en.serial >= next_serial_) next_serial_ = en.serial + 1;
            en.path = path;
            en.file_bytes = v2bytes;
            en.bytes = v2bytes;
            en.blob_keys = std::move(v2keys);
            en.ids = std::move(v2img.live.ids);
            en.view_parents = v2vf.parents_seen;   // M-C6b: boot derives the pins from the table itself.
                       // Without this an INHERITED parent carries zero protection: view_refs only ever
                       // counted children spilled during THIS boot, so the first budget drop after a
                       // restart orphaned the children of every cross-boot lineage (run37's refuse).
            std::sort(en.view_parents.begin(), en.view_parents.end());
            en.view_parents.erase(std::unique(en.view_parents.begin(), en.view_parents.end()),
                                  en.view_parents.end());
            en.cvec = v2img.cvec;
            en.done = true;
            en.spilled_at = std::chrono::steady_clock::now();
            size_t add = v2bytes;
            for (const auto& k : en.blob_keys) {
                BlobRef& br = blobs_[k];
                if (br.refs == 0) {
                    struct stat cs;
                    br.len = (::stat(blob_path(k).c_str(), &cs) == 0 && cs.st_size > 0) ? (uint64_t) cs.st_size : 0;
                    add += br.len;
                    blob_bytes_ += br.len;
                }
                br.refs += 1;
            }
            bytes_ += add;
            entries_.push_back(std::move(en));
            ++found;
            continue;
        }
        SavedConversation image;
        size_t bytes = 0;
        SessionReadLimits limits;
        if (limits_fn_ && !limits_fn_(limits, error)) {
            std::fprintf(stderr, "strata serve: park disk: index stop at %s (%s)\n", path.c_str(), error.c_str());
            walk_ok = false;
            break;
        }
        SessionStatus st;
        if (!session_file_read(path, *id, image, bytes, error, limits, &st)) {
            std::fprintf(stderr, "strata serve: park disk: index refuse %s (%s)\n", path.c_str(), error.c_str());
            ++refused;
            const uint64_t ser = boot_serial(path);
            if (ser >= next_serial_) next_serial_ = ser + 1;   // M-C5: a quarantined name is never renamed over
            if (st.error == SessionError::foreign) { ++quarantined; continue; }   // kept: a v1 file owns its bytes
            if (st.error == SessionError::memory || st.error == SessionError::io) {
                walk_ok = false;
                break;   // io/memory right now: the rest waits for a later boot rather than churn the disk
            }
#ifndef _WIN32
            ::unlink(path.c_str());
#else
            ::DeleteFileA(path.c_str());
#endif
            continue;
        }
        if (image.live.ids.empty() || image.kv.empty()) {   // nothing restorable to key on
            ++refused;
            continue;
        }
        Entry en;
        en.serial = boot_serial(path);   // M-C5: same for v1 files (a v1 parent can feed v2 children too)
        if (en.serial >= next_serial_) next_serial_ = en.serial + 1;
        en.path = path;
        en.bytes = bytes;
        en.ids = std::move(image.live.ids);
        en.cvec = image.cvec;
        en.done = true;
        en.spilled_at = std::chrono::steady_clock::now();
        bytes_ += en.bytes;
        entries_.push_back(std::move(en));
        ++found;
    }
    // M-C6b: repin pass. The spill side bumps view_refs when a child's write completes, but a boot
    // rebuilds the index from files alone: until now inherited parents entered service unprotected.
    for (auto& e : entries_)
        for (uint64_t vp : e.view_parents) {
            const auto pe = std::find_if(entries_.begin(), entries_.end(),
                                         [&](const Entry& x) { return x.serial == vp; });
            if (pe != entries_.end()) pe->view_refs += 1;   // serials are unique: self-reference impossible
        }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const std::string budget_note = budget_ ? std::to_string(budget_ >> 20) + " MiB" : std::string("free-space only");
    std::fprintf(stderr, "strata serve: park disk: ON %s (index %zu files, %.1f GiB, %zu refused, %zu quarantined, rebuilt in "
                         "%.0f ms; budget %s, keep-free %llu MiB)%s\n",
                 dir_.c_str(), found, (double) bytes_ / (1024.0 * 1024.0 * 1024.0), refused, quarantined, ms,
                 budget_note.c_str(),
                 (unsigned long long) (min_free_ >> 20),
                 walk_ok ? "" : " [walk stopped early]");
    if (v2_ && !walk_ok)
        std::fprintf(stderr, "strata serve: park disk: boot sweep SKIPPED (the walk did not finish; nothing is known unreferenced until it does)\n");
    if (v2_ && walk_ok) {
        // orphan sweep: chunk files no surviving entry references (and half-written .tmp publishes) leave now
        size_t swept = 0;
        if (DIR* d = opendir(chunks_dir_.c_str())) {
            while (const dirent* e = readdir(d)) {
                const std::string name = e->d_name;
                const std::string full = chunks_dir_ + "/" + name;
                if (name.find(".tmp") != std::string::npos) { ::unlink(full.c_str()); ++swept; continue; }
                if (name.size() == 36 && name.compare(name.size() - 4, 4, ".bin") == 0) {
                    uint64_t k0 = 0, k1 = 0;
                    if (!pd_key_from_hex(name.substr(0, 32), k0, k1)) { ::unlink(full.c_str()); ++swept; continue; }
                    if (!blobs_.count(PdKey{k0, k1})) { ::unlink(full.c_str()); ++swept; }
                }
            }
            closedir(d);
        }
        if (swept) std::fprintf(stderr, "strata serve: park disk: swept %zu orphan chunk file(s)\n", swept);
        std::fprintf(stderr, "strata serve: park disk: v2 dedup tier active (store %zu chunks, %.1f GiB)\n",
                     blobs_.size(), (double) blob_bytes_ / (1024.0 * 1024.0 * 1024.0));
    }
    worker_ = std::thread([this] { writer(); });
}

ParkDisk::~ParkDisk() {
    if (!worker_.joinable()) return;
    {
        std::lock_guard<std::mutex> g(mtx_);
        stop_ = true;
    }
    cv_.notify_all();
    worker_.join();   // the writer finishes everything already queued before it leaves
}

void ParkDisk::spill(SavedConversation&& image, int64_t cache_serial) {
    if (dir_.empty()) return;
    drain_finished();
    const size_t n = image.bytes();
    if (n == 0) return;
    // Room on disk: budget first, then the keep-free guard (oldest entries leave until the incoming fits).
    while (budget_ != 0 && bytes_ + n > budget_ && !entries_.empty()) drop_oldest();
    while (!entries_.empty() && !free_guard_ok(n)) drop_oldest();
    if (!free_guard_ok(n)) {
        std::fprintf(stderr, "strata serve: park disk: skip spill of %zu tokens, %.1f MiB: the free space minus "
                             "the %llu MiB keep-free guard cannot hold it\n",
                     image.live.ids.size(), (double) n / (1024.0 * 1024.0), (unsigned long long) (min_free_ >> 20));
        return;
    }
    Entry en;
    en.serial = next_serial_++;
    en.path = pd_tier_path(dir_, en.serial);
    en.bytes = n;
    en.ids = image.live.ids;      // the key stays serve-side; the worker owns its own copy of the image
    en.cvec = image.cvec;
    en.spilled_at = std::chrono::steady_clock::now();
    bytes_ += n;
    spills_ += 1;
    Write w;
    w.serial = en.serial;
    w.path = en.path;
    en.cache_serial = cache_serial;
    // M-C3b3: lineage spill. The cache spilled this image's park donor HERE first (evict_slot), so when the
    // donor's file record is present, the leading borrowed runs of each K/V part become view ENTRIES instead
    // of a second copy of the donor's bytes. Any doubt (no donor record, a small run under the chunk floor)
    // keeps the bytes inline: both forms reassemble identical payload and hash - the view is size, not risk.
    if (v2_ && image.view_donor >= 0) {
        const auto dp = cache_to_park_.find(image.view_donor);
        if (dp != cache_to_park_.end()) {
            for (size_t i = 0; i < image.kv.size(); ++i) {
                ConversationKv& k = image.kv[i];
                ConversationBuffer* parts[5] = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
                for (size_t pp = 0; pp < 5; ++pp)
                    parts[pp]->head_spans([&](const uint8_t* q, size_t len) {
                        if (len < (64u << 10)) return;
                        w.views.push_back({q, len, dp->second, (uint32_t) ((i << 3) | pp)});
                    });
            }
            if (!w.views.empty())
                std::fprintf(stderr, "strata serve: park disk: lineage spill: %zu view runs of cache donor %lld\n",
                             w.views.size(), (long long) image.view_donor);
        }
    }
    if (cache_serial >= 0) cache_to_park_[cache_serial] = en.serial;
    w.image = std::move(image);
    entries_.push_back(std::move(en));
    {
        std::lock_guard<std::mutex> g(mtx_);
        queued_.push_back(std::move(w));
    }
    cv_.notify_one();
}

std::vector<int32_t> ParkDisk::ids(uint64_t serial) const {
    for (const auto& e : entries_)
        if (e.serial == serial) return e.ids;
    return {};
}

std::optional<SavedConversation> ParkDisk::load(Match m, std::string& error) {
    drain_finished();
    const auto it = std::find_if(entries_.begin(), entries_.end(),
                                 [&](const Entry& e) { return e.serial == m.serial && e.done; });
    if (it == entries_.end()) return std::nullopt;
    const auto id = identity_cached(error);
    if (!id) return std::nullopt;
    SavedConversation image;
    size_t bytes = 0;
    SessionReadLimits limits;
    if (limits_fn_ && !limits_fn_(limits, error)) {   // RAM preflight inside: the file stays for a later turn
        error = "read refused: " + error;
        return std::nullopt;
    }
    SessionStatus st;
    const auto t0 = std::chrono::steady_clock::now();
    const bool v2_file = file_is_v2(it->path);
    ViewFeed vf;   // M-C3b3: a view child assembles through its parent (transients die with this load)
    const bool read_ok = v2_file ? read_v2(it->path, *id, limits, image, bytes, error, st, nullptr, &vf)
                                 : session_file_read(it->path, *id, image, bytes, error, limits, &st);
    if (!read_ok) {
        if (st.error == SessionError::invalid) {   // corrupt: never again
            std::fprintf(stderr, "strata serve: park disk: refuse %s (%s); file deleted\n",
                         it->path.c_str(), error.c_str());
            release_entry(*it);
            bytes_ -= it->bytes;
            entries_.erase(it);
            ++drops_;
        } else if (st.error == SessionError::foreign) {
            // M-C5: this runtime can never read it: quarantine the file. Phantom-reference its chunk keys
            // FIRST, so dropping this entry cannot cascade into deleting bytes another entry - now or at a
            // later boot with the matching identity - still needs (b3a lineage shares chunks; CHANGES 151).
            std::fprintf(stderr, "strata serve: park disk: refuse %s (%s); file quarantined\n",
                         it->path.c_str(), error.c_str());
            for (const auto& k : it->blob_keys) blobs_[k].refs += 1;
            release_entry(*it);
            bytes_ -= it->bytes;
            entries_.erase(it);
            ++drops_;
        }
        return std::nullopt;                        // io/memory: the file stays, a later turn retries
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    loads_ += 1;
    if (!v2_file) {   // v1: swap the spill estimate for the real file size; v2 already holds the real footprint
        bytes_ += bytes - it->bytes;
        it->bytes = bytes;
    }
    std::fprintf(stderr, "strata serve: park disk: restored %zu tokens, %.1f MiB read in %.0f ms "
                         "(%zu entries, %.1f GiB on disk)\n",
                 image.live.ids.size(), (double) bytes / (1024.0 * 1024.0), ms,
                 entries_.size(), (double) bytes_ / (1024.0 * 1024.0 * 1024.0));
    return image;
}

void ParkDisk::drain_finished() {
    std::deque<Finished> done;
    {
        std::lock_guard<std::mutex> g(mtx_);
        done.swap(finished_);
    }
    for (const auto& f : done) {
        const auto it = std::find_if(entries_.begin(), entries_.end(),
                                     [&](const Entry& e) { return e.serial == f.serial; });
        if (it == entries_.end()) {
            // dropped while its write ran (room loop): the rename may still have published a file nobody
            // indexes - take it back (a boot scan would otherwise resurrect the stale copy)
            if (f.ok) {
#ifndef _WIN32
                ::unlink(pd_tier_path(dir_, f.serial).c_str());
                for (const auto& [k, len] : f.created) ::unlink(blob_path(k).c_str());
#endif
            }
            continue;
        }
        if (f.ok) {
            it->done = true;
            for (uint64_t vp : f.view_parents) {
                const auto pe = std::find_if(entries_.begin(), entries_.end(),
                                             [&](const Entry& x) { return x.serial == vp; });
                if (pe != entries_.end()) { pe->view_refs += 1; it->view_parents.push_back(vp); }
                else {   // the parent left while this write ran: rewrite inline next time, orphan-guard here
                    it->view_parents.push_back(vp);   // load() fails closed on a missing parent and deletes the file
                }
            }
            if (f.file_bytes) {   // v2: swap the RAM-sized spill estimate for the real footprint; index the chunks
                for (const auto& [k, len] : f.created) {
                    BlobRef& br = blobs_[k];
                    br.refs = 0;
                    br.len = len;
                    bytes_ += len;
                    blob_bytes_ += len;
                }
                for (const auto& k : f.all_keys) blobs_[k].refs += 1;
                bytes_ += (long long) f.file_bytes - (long long) it->bytes;
                it->file_bytes = f.file_bytes;
                it->bytes = f.file_bytes;
                it->blob_keys = f.all_keys;
                saved_bytes_ += f.saved;
            }
            std::fprintf(stderr, "strata serve: park disk: spilled %zu tokens, %.1f MiB in %.0f ms "
                                 "(%zu entries, %.1f GiB on disk)\n",
                         it->ids.size(), (double) it->bytes / (1024.0 * 1024.0), f.ms,
                         entries_.size(), (double) bytes_ / (1024.0 * 1024.0 * 1024.0));
            if (f.saved)
                std::fprintf(stderr, "strata serve: park disk: dedup: %.1f GiB skipped as existing chunks "
                                     "(store %zu chunks, %.1f GiB)\n",
                             (double) f.saved / (1024.0 * 1024.0 * 1024.0), blobs_.size(),
                             (double) blob_bytes_ / (1024.0 * 1024.0 * 1024.0));
        } else {
            std::fprintf(stderr, "strata serve: park disk: spill failed for %zu tokens (%s)\n",
                         it->ids.size(), f.error.c_str());
            for (const auto& [k, len] : f.created) ::unlink(blob_path(k).c_str());   // published, never indexed
            bytes_ -= it->bytes;
            entries_.erase(it);
            ++drops_;
        }
    }
}

void ParkDisk::writer() {
    for (;;) {
        Write w;
        {
            std::unique_lock<std::mutex> g(mtx_);
            cv_.wait(g, [this] { return stop_ || !queued_.empty(); });
            if (queued_.empty()) return;   // stop_ with nothing left: everything queued got written
            w = std::move(queued_.front());
            queued_.pop_front();
        }
        const auto t0 = std::chrono::steady_clock::now();
        Finished f;
        f.serial = w.serial;
        const auto id = identity_cached(f.error);
        if (!id) {
            f.error = "identity: " + f.error;
        } else if (v2_) {
            write_v2(w, *id, f);   // a v2 write; falls through to the v1 path itself when the block view refuses
        } else {
            SessionWriteOptions wo;
            wo.min_free_bytes = min_free_;
            SessionStatus st;
            size_t bytes = 0;
            if (!session_file_write(w.path, w.image, *id, bytes, f.error, wo, &st) && !st.published) {
                if (f.error.empty()) f.error = "write failed";
            } else {
                f.ok = true;   // st.published with a folder-flush failure: the file is there; the next boot checks it
            }
        }
        f.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::lock_guard<std::mutex> g(mtx_);
        finished_.push_back(std::move(f));
    }
}

bool ParkDisk::free_guard_ok(size_t incoming) const {
#ifdef _WIN32
    ULARGE_INTEGER avail;
    if (!GetDiskFreeSpaceExA(dir_.c_str(), &avail, nullptr, nullptr)) return true;   // unknown: the write's own preflight guards
    return avail.QuadPart >= (unsigned long long) min_free_ + incoming;
#else
    struct statvfs st;
    if (statvfs(dir_.c_str(), &st) != 0) return true;   // unknown: the write's own preflight guards
    return (uint64_t) st.f_bavail * st.f_frsize >= min_free_ + incoming;
#endif
}

void ParkDisk::drop_oldest() {
    if (entries_.empty()) return;
    // M-C3b3: a parent whose file other entries view leaves only when no unviewed entry can pay - children
    // (and everything else) go first, so lineage normally ages out before it can orphan anything.
    size_t pick = 0;
    for (size_t i = 0; i < entries_.size(); ++i)
        if (entries_[i].view_refs == 0) { pick = i; break; }
    const Entry e = std::move(entries_[pick]);
    entries_.erase(entries_.begin() + (long long) pick);
    release_entry(e);
    bytes_ -= e.bytes;
    ++drops_;
#ifndef _WIN32
    ::unlink(e.path.c_str());
#else
    ::DeleteFileA(e.path.c_str());
#endif
    std::fprintf(stderr, "strata serve: park disk: drop %s (%.1f MiB, %zu entries, %.1f GiB on disk)\n",
                 e.path.c_str(), (double) e.bytes / (1024.0 * 1024.0), entries_.size(),
                 (double) bytes_ / (1024.0 * 1024.0 * 1024.0));
    // a not-yet-started write for this serial leaves the queue with it (its rename would republish the file);
    // one already in flight is caught by drain_finished(), which unlinks a file nobody indexes anymore
    std::lock_guard<std::mutex> g(mtx_);
    for (auto q = queued_.begin(); q != queued_.end(); ++q)
        if (q->serial == e.serial) { queued_.erase(q); break; }
}

// M-C3b3: materialize one view parent for an assembly read (kept in the feed until the child is built).
// A missing, unfinished or over-deep parent fails the child closed: the caller deletes the orphan file and
// the conversation re-prefills - never wrong bytes, never a partial child. Depth-capped at 3 generations.
const SavedConversation* ParkDisk::view_image(uint64_t serial, const SessionFileIdentity& id,
                                              const SessionReadLimits& limits, ViewFeed& vf,
                                              std::string& error, SessionStatus& st) {
    if (vf.depth >= 3) { error = "park v2: view chain too deep"; st.error = SessionError::invalid; return nullptr; }
    const auto got = vf.images.find(serial);
    if (got != vf.images.end()) return &got->second;
    const auto en = std::find_if(entries_.begin(), entries_.end(),
                                 [&](const Entry& x) { return x.serial == serial && x.done; });
    if (en == entries_.end()) return nullptr;
    SavedConversation par;
    size_t pbytes = 0;
    SessionStatus pst;
    std::string perr;
    const bool ok = file_is_v2(en->path)
        ? read_v2(en->path, id, limits, par, pbytes, perr, pst, nullptr, &vf)
        : session_file_read(en->path, id, par, pbytes, perr, limits, &pst);
    if (!ok) { error = perr; st.error = pst.error; return nullptr; }
    vf.depth += 1;
    return &vf.images.emplace(serial, std::move(par)).first->second;
}

std::optional<SessionFileIdentity> ParkDisk::identity_cached(std::string& error) {
    std::lock_guard<std::mutex> g(write_mtx_);
    if (!identity_) {
        SessionFileIdentity id;
        if (!identity_fn_ || !identity_fn_(id, error)) return std::nullopt;
        identity_ = id;
    }
    return identity_;
}


// ---- M-C3b v2: deduped block files. The payload walk is STRSES v1's own (session_payload_blocks);
// the envelope and the chunk store are ParkDisk's. Any structural refusal falls back to a plain v1
// write for the same image, so a v2 problem can never lose a spill. POSIX-only for now (the v2 knob
// forces v1 on Windows); v1 files keep working under v2 unchanged.
std::string ParkDisk::blob_path(const PdKey& k) const {
    return chunks_dir_ + "/" + pd_hexkey(k.k0, k.k1) + ".bin";
}

bool ParkDisk::file_is_v2(const std::string& path) {
#ifndef _WIN32
    int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW);
    if (fd < 0) return false;
    char m[8];
    const bool v2 = ::read(fd, m, 8) == 8 && std::memcmp(m, kPd2Magic, 8) == 0;
    ::close(fd);
    return v2;
#else
    return false;
#endif
}

void ParkDisk::release_entry(const Entry& e) {
    if (e.cache_serial >= 0) cache_to_park_.erase(e.cache_serial);   // M-C3b3: never resolve a stale donor key
    for (uint64_t vp : e.view_parents) {                             // M-C3b3: this child stops pinning its parents
        const auto pe = std::find_if(entries_.begin(), entries_.end(),
                                     [&](const Entry& x) { return x.serial == vp; });
        if (pe != entries_.end() && pe->view_refs) pe->view_refs -= 1;
    }
    for (const auto& k : e.blob_keys) {
        const auto it = blobs_.find(k);
        if (it == blobs_.end()) continue;
        if (--it->second.refs == 0) {
            bytes_ -= it->second.len;
            blob_bytes_ -= it->second.len;
#ifndef _WIN32
            ::unlink(blob_path(k).c_str());
#endif
            blobs_.erase(it);
        }
    }
}

void ParkDisk::write_v2(const Write& w, const SessionFileIdentity& id, Finished& f) {
    const auto to_v1 = [&] {
        f.error.clear();
        f.created.clear();
        f.all_keys.clear();
        f.saved = 0;
        f.file_bytes = 0;
        SessionWriteOptions wo;
        wo.min_free_bytes = min_free_;
        SessionStatus st;
        size_t bytes = 0;
        if (!session_file_write(w.path, w.image, id, bytes, f.error, wo, &st) && !st.published) {
            if (f.error.empty()) f.error = "write failed";
        } else {
            f.ok = true;
        }
    };
#ifndef _WIN32
    const std::string tmp = w.path + ".tmp";
    int fd = ::open(tmp.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
    if (fd < 0) { to_v1(); return; }
    std::vector<uint8_t> front(kPd2Body, 0);   // header + zero-filled table area; the body streams after
    if (::write(fd, front.data(), front.size()) != (ssize_t) front.size()) {
        ::close(fd);
        ::unlink(tmp.c_str());
        to_v1();
        return;
    }

    struct Ctl {
        std::vector<std::pair<const uint8_t*, size_t>> slices;   // the block being accumulated
        std::deque<std::string> arena;   // M-C5: spans <= 256 B point at put_* temporaries that are dead by
                                         // flush time (scalar fields, counts): copy them at once or the chunk
                                         // file stores whatever the stack held later, not the payload
        size_t slen = 0;
        uint64_t gear = 0;                                       // rolling hash over the open block
        uint64_t inline_bytes = 0, saved = 0;
        std::vector<uint8_t> table;
        std::vector<PdKey> keys;
        std::vector<std::pair<PdKey, uint64_t>> created;
        bool failed = false, overflow = false;
        std::vector<uint64_t> view_parents;
    } c;
    std::unordered_map<const void*, const Write::ViewSpan*> vmap;
    for (const auto& v : w.views) vmap[v.p] = &v;
    const auto abort_v2 = [&](void) {
        ::close(fd);
        ::unlink(tmp.c_str());
        for (const auto& [k, len] : c.created) ::unlink(blob_path(k).c_str());   // published, never will be indexed
        c.created.clear();
        to_v1();
    };

    const auto flush_block = [&](void) {
        if (c.slices.empty()) return;
        if (c.table.size() + kPd2Ent > kPd2TblArea) { c.overflow = true; return; }
        const size_t len = c.slen;
        SessionHasher h0(0), h1(kPd2HashSeed1);
        for (const auto& [p, n] : c.slices) { h0.update(p, n); h1.update(p, n); }
        const PdKey key{h0.digest(), h1.digest()};
        bool blobbed = false;
        if (len >= (64u << 10)) {           // worth a chunk file: below this the table cost beats the dedup
            const std::string bp = blob_path(key);
            struct stat sb;
            if (::stat(bp.c_str(), &sb) == 0) {
                blobbed = true;
                c.saved += len;
            } else {
                const std::string bt = bp + ".tmp";
                int bfd = ::open(bt.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
                if (bfd < 0 && errno == EEXIST) blobbed = true;   // raced with an identical publish: same bytes
                else if (bfd >= 0) {
                    bool ok = true;
                    for (const auto& [p, n] : c.slices)
                        if (::write(bfd, p, n) != (ssize_t) n) { ok = false; break; }
                    if (ok) ok = ::fsync(bfd) == 0;
                    ::close(bfd);
                    if (ok && ::rename(bt.c_str(), bp.c_str()) == 0) {
                        blobbed = true;
                        c.created.emplace_back(key, len);
                    } else {
                        ::unlink(bt.c_str());
                    }
                }
            }
        }
        if (blobbed) {
            c.keys.push_back(key);
        } else {
            for (const auto& [p, n] : c.slices)
                if (::write(fd, p, n) != (ssize_t) n) { c.failed = true; return; }
            c.inline_bytes += len;
        }
        uint8_t ent[kPd2Ent] = {};
        ent[0] = blobbed ? 1 : 0;
        std::memcpy(ent + 8, &len, 8);
        std::memcpy(ent + 16, &key.k0, 8);
        std::memcpy(ent + 24, &key.k1, 8);
        c.table.insert(c.table.end(), ent, ent + kPd2Ent);
        c.slices.clear();
        c.slen = 0;
        c.gear = 0;
    };

    // M-C7: a view entry may only name a parent that HAS a disk file. The donor is cache-side;
    // if it was never spilled (pinned donors never evict), a view entry would dangle: the next
    // boot's table cannot resolve the parent, the file is refused, and the refuse cascades.
    // Snapshot the published set here (brief lock); unpublished view spans fall through to the
    // normal content-defined chunk path: bytes dedup by hash, correctness over lineage.
    std::unordered_set<uint64_t> published;
    {
        std::lock_guard<std::mutex> g(mtx_);
        for (const auto& cp : cache_to_park_) published.insert(static_cast<uint64_t>(cp.first));
    }
    uint64_t payload = 0;
    SessionHasher phash(0);
    std::string werr;
    const uint64_t* gear_tab = pd_gear_table();
    bool walk_ok = session_payload_blocks(w.image, [&](uint64_t, const uint8_t* p, size_t n) -> bool {
        phash.update(p, n);
        // M-C3b3: a view span: the payload and its hash cover the bytes either way, a table entry just stands
        // in for them; the block stream never mixes a view with chunked bytes (flush first, entry, continue).
        if (const auto m = vmap.find(p); m != vmap.end() && m->second->len == n && published.count(m->second->parent)) {
            flush_block();
            if (c.failed || c.overflow) return false;
            if (c.table.size() + kPd2Ent > kPd2TblArea) { c.overflow = true; return false; }
            uint8_t ent[kPd2Ent] = {};
            ent[0] = 2;
            std::memcpy(ent + 8, &n, 8);
            std::memcpy(ent + 16, &m->second->parent, 8);
            const uint64_t aux = m->second->aux;
            std::memcpy(ent + 24, &aux, 8);
            c.table.insert(c.table.end(), ent, ent + kPd2Ent);
            c.saved += n;
            if (std::find(c.view_parents.begin(), c.view_parents.end(), m->second->parent) == c.view_parents.end())
                c.view_parents.push_back(m->second->parent);
            return true;
        }
        if (n <= 256) {   // M-C5: copy short spans now: their source is a put_* temporary that dies before flush
            c.arena.emplace_back(reinterpret_cast<const char*>(p), n);
            p = reinterpret_cast<const uint8_t*>(c.arena.back().data());
        }
        while (n) {
            size_t run = 0;                        // bytes of this span to take before the next cut decision
            if (c.slen < kPd2CutMin) {
                run = std::min(n, kPd2CutMin - c.slen);
            } else {
                const size_t room = kPd2Block - c.slen;
                run = n < room ? n : room;         // scan at most to the forced cut
                for (size_t i = 0; i < run; ++i) {
                    c.gear = (c.gear << 1) + gear_tab[p[i]];
                    if (!(c.gear & kPd2CutMask)) { run = i + 1; break; }
                }
            }
            c.slices.emplace_back(p, run);
            c.slen += run;
            p += run;
            n -= run;
            if (c.slen == kPd2Block) flush_block();
            if (c.failed || c.overflow) return false;
        }
        return true;
    }, payload, werr);
    if (walk_ok) flush_block();   // the last, short block
    if (!walk_ok || c.failed || c.overflow || c.table.empty()) { abort_v2(); return; }

    if (::pwrite(fd, c.table.data(), c.table.size(), (off_t) kPd2Hdr) != (ssize_t) c.table.size()) { abort_v2(); return; }
    uint8_t h[kPd2Hdr] = {};
    std::memcpy(h, kPd2Magic, 8);
    const uint32_t ver = 2, hdr = (uint32_t) kPd2Body;
    std::memcpy(h + 8, &ver, 4);
    std::memcpy(h + 12, &hdr, 4);
    std::memcpy(h + 16, &id.model, 8);
    std::memcpy(h + 24, &id.config, 8);
    const uint64_t serial = w.serial, tokens = w.image.live.ids.size(),
                   ph = phash.digest(), nblocks = c.table.size() / kPd2Ent;
    std::memcpy(h + 32, &serial, 8);
    std::memcpy(h + 40, &tokens, 8);
    std::memcpy(h + 48, &payload, 8);
    std::memcpy(h + 56, &ph, 8);
    std::memcpy(h + 64, &nblocks, 8);
    std::memcpy(h + 72, &c.inline_bytes, 8);
    const uint64_t hh = session_hash64(h, 80, 0);
    std::memcpy(h + 80, &hh, 8);
    if (::pwrite(fd, h, kPd2Hdr, 0) != (ssize_t) kPd2Hdr) { abort_v2(); return; }
    if (::fsync(fd) != 0) { abort_v2(); return; }
    ::close(fd);
    if (::rename(tmp.c_str(), w.path.c_str()) != 0) {
        ::unlink(tmp.c_str());
        for (const auto& [k, len] : c.created) ::unlink(blob_path(k).c_str());
        f.error = std::string("park v2 rename: ") + std::strerror(errno);
        return;
    }
    struct stat sb;
    if (::stat(w.path.c_str(), &sb) != 0) sb.st_size = 0;
    f.ok = true;
    f.file_bytes = (size_t) sb.st_size;
    f.all_keys = std::move(c.keys);
    f.created = std::move(c.created);
    f.saved = c.saved;
    f.view_parents = std::move(c.view_parents);
#else
    to_v1();
#endif
}

bool ParkDisk::read_v2(const std::string& path, const SessionFileIdentity& id, const SessionReadLimits& limits,
                       SavedConversation& image, size_t& bytes, std::string& error, SessionStatus& st,
                       std::vector<PdKey>* keys, ViewFeed* vf) {
#ifndef _WIN32
    struct Ctx {
        int fd = -1, bfd = -1;
        ~Ctx() { if (bfd >= 0) ::close(bfd); if (fd >= 0) ::close(fd); }
    } ctx;
    ctx.fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW);
    if (ctx.fd < 0) { st.error = SessionError::io; error = "park v2: open: " + std::string(std::strerror(errno)); return false; }
    struct stat sb;
    if (::fstat(ctx.fd, &sb) != 0 || !S_ISREG(sb.st_mode) || sb.st_nlink != 1 || (uint64_t) sb.st_size < kPd2Body) {
        st.error = SessionError::invalid; error = "park v2: not a single-linked regular file of the minimum size"; return false;
    }
    uint8_t h[kPd2Hdr];
    if (::pread(ctx.fd, h, kPd2Hdr, 0) != (ssize_t) kPd2Hdr) { st.error = SessionError::io; error = "park v2: header read"; return false; }
    if (std::memcmp(h, kPd2Magic, 8) != 0) { st.error = SessionError::invalid; error = "park v2: bad magic"; return false; }
    uint32_t ver = 0, hdr = 0;
    std::memcpy(&ver, h + 8, 4);
    std::memcpy(&hdr, h + 12, 4);
    uint64_t model = 0, config = 0, serial = 0, tokens = 0, payload = 0, phash = 0, nblocks = 0, inline_total = 0, hh = 0;
    std::memcpy(&model, h + 16, 8); std::memcpy(&config, h + 24, 8); std::memcpy(&serial, h + 32, 8);
    std::memcpy(&tokens, h + 40, 8); std::memcpy(&payload, h + 48, 8); std::memcpy(&phash, h + 56, 8);
    std::memcpy(&nblocks, h + 64, 8); std::memcpy(&inline_total, h + 72, 8); std::memcpy(&hh, h + 80, 8);
    if (ver != 2 || hdr != (uint32_t) kPd2Body || session_hash64(h, 80, 0) != hh) {
        st.error = SessionError::invalid; error = "park v2: bad header"; return false;
    }
    if (model != id.model || config != id.config) {
        st.error = SessionError::foreign; error = "park v2: bound to another model or configuration"; return false;
    }
    if (nblocks == 0 || nblocks > kPd2MaxBlocks) { st.error = SessionError::invalid; error = "park v2: bad block count"; return false; }
    std::vector<uint8_t> tbl((size_t) nblocks * kPd2Ent);
    if (::pread(ctx.fd, tbl.data(), tbl.size(), (off_t) kPd2Hdr) != (ssize_t) tbl.size()) { st.error = SessionError::io; error = "park v2: table read"; return false; }
    struct Ent2 { uint8_t kind; uint64_t len, k0, k1; };
    std::vector<Ent2> ents;
    ents.resize((size_t) nblocks);
    uint64_t sum = 0, itot = 0;
    for (uint64_t i = 0; i < nblocks; ++i) {
        const uint8_t* e = tbl.data() + (size_t) i * kPd2Ent;
        Ent2& en = ents[(size_t) i];
        en.kind = e[0];
        std::memcpy(&en.len, e + 8, 8);
        std::memcpy(&en.k0, e + 16, 8);
        std::memcpy(&en.k1, e + 24, 8);
        if (en.kind > 2 || en.len == 0 || en.len > kPd2Block) { st.error = SessionError::invalid; error = "park v2: bad table entry"; return false; }
        if (sum > UINT64_MAX - en.len) { st.error = SessionError::invalid; error = "park v2: table sum overflows"; return false; }
        sum += en.len;
        if (en.kind == 0) itot += en.len;
    }
    if (sum != payload || itot != inline_total || (uint64_t) sb.st_size != kPd2Body + inline_total) {
        st.error = SessionError::invalid; error = "park v2: table and file size disagree"; return false;
    }
    if (limits.max_file_bytes != UINT64_MAX && payload > limits.max_file_bytes) {
        st.error = SessionError::invalid; error = "park v2: over this engine's read limit"; return false;
    }
    if (limits.admit) {
        std::string why;
        if (!limits.admit(session_read_peak_bytes(payload), why)) {
            st.error = SessionError::memory; error = "park v2: RAM preflight: " + why; return false;
        }
    }
    // the assembled read: inline blocks stream from this file, chunks from the store, and every chunk's
    // own key is RECOMPUTED while its bytes serve the parser: a mutated chunk fails closed here
    struct Feed {
        std::vector<Ent2>::const_iterator it;
        uint64_t body_off = kPd2Body;
        int* bfp;
        bool missing = false, bad_chunk = false, no_parent = false;
        uint64_t bad_parent = 0;   // M-C6b: the serial that went missing (refuse line)
    } feed{ents.begin(), kPd2Body, &ctx.bfd, false, false, false};
    uint64_t block_left = 0, block_off = 0;
    bool hashing = false, inline_blk = false, view_blk = false;
    // M-C3b3: view ranges serve from the parent's live buffers (b3a copy; b3b turns this into shared views)
    const auto take_view = [&](uint8_t* dst, size_t n) -> bool {
        if (vf == nullptr || vf->cur_img == nullptr || vf->cur_aux == UINT64_MAX) return false;
        const size_t i = (size_t) (vf->cur_aux >> 3), part = (size_t) (vf->cur_aux & 7);
        if (i >= vf->cur_img->kv.size()) return false;
        const ConversationKv& k = vf->cur_img->kv[i];
        const ConversationBuffer* b = part == 0 ? &k.k : part == 1 ? &k.v : part == 2 ? &k.k_scale
                                  : part == 3 ? &k.v_scale : part == 4 ? &k.pooled : nullptr;
        if (b == nullptr || vf->cur_off + n > b->size()) return false;
        bool ok = b->visit(vf->cur_off, n, [&](const uint8_t* q, size_t c, size_t) {
            std::memcpy(dst, q, c); dst += c; return true; });
        if (ok) vf->cur_off += n;
        return ok;
    };
    PdKey cur{};
    SessionHasher bh(0), bh1(kPd2HashSeed1);
    const auto pull = [&](uint8_t* dst, size_t n) -> bool {
        while (n) {
            if (block_left == 0) {
                if (feed.it == ents.end()) return false;   // the payload asks for more than the table holds
                const Ent2& e = *feed.it;
                inline_blk = e.kind == 0;
                view_blk = e.kind == 2;
                block_left = e.len;
                block_off = 0;
                if (inline_blk) hashing = false;
                else if (view_blk) {
                    hashing = false;   // M-C6a: a view block is not chunk bytes: stop hashing the PREVIOUS
                                       // chunk here. Left armed, its parent-fed bytes digested against the
                                       // prior key and any chunk->view interleave self-flagged bad_chunk,
                                       // fail-closed deleting whole lineage families at boot.
                    if (vf != nullptr && (vf->cur_par != e.k0 || vf->cur_aux != e.k1)) {
                        vf->cur_img = view_image(e.k0, id, limits, *vf, error, st);
                        vf->cur_par = e.k0;
                        vf->parents_seen.push_back(e.k0);   // M-C6b: remember for the boot pin pass
                        vf->cur_aux = e.k1;
                        vf->cur_off = 0;
                    }
                    if (vf == nullptr || vf->cur_img == nullptr) { feed.no_parent = true; feed.bad_parent = e.k0; return false; }
                }
                else {
                    if (*feed.bfp >= 0) ::close(*feed.bfp);
                    *feed.bfp = ::open(blob_path(PdKey{e.k0, e.k1}).c_str(), O_RDONLY | O_NOFOLLOW);
                    if (*feed.bfp < 0) { std::fprintf(stderr, "park disk: dbg miss open %s errno %d\n", blob_path(PdKey{e.k0, e.k1}).c_str(), errno); feed.missing = true; return false; }
                    cur = PdKey{e.k0, e.k1};
                    bh = SessionHasher(0);
                    bh1 = SessionHasher(kPd2HashSeed1);
                    hashing = true;
                }
            }
            const size_t take = std::min(n, block_left);
            const ssize_t r = view_blk ? (take_view(dst, take) ? (ssize_t) take : -1)
                                       : ::pread(inline_blk ? ctx.fd : *feed.bfp, dst, take,
                                                (off_t) (inline_blk ? feed.body_off : block_off));
            if (r != (ssize_t) take) {
                if (inline_blk) { /* short body read below */ }
                else if (view_blk) feed.no_parent = true;
                else feed.missing = true;
                return false;
            }
            if (hashing) { bh.update(dst, take); bh1.update(dst, take); }
            if (inline_blk) feed.body_off += take;
            block_off += take;
            block_left -= take;
            dst += take;
            n -= take;
            if (block_left == 0) {
                if (hashing && keys) keys->push_back(cur);
                if (hashing && (bh.digest() != cur.k0 || bh1.digest() != cur.k1)) { feed.bad_chunk = true; return false; }
                ++feed.it;
            }
        }
        return true;
    };
    SessionError pkind = SessionError::invalid;
    if (!session_payload_unblocks(payload, phash, pull, image, limits, error, &pkind)) {
        if (feed.no_parent) { st.error = SessionError::invalid; error = "park v2: a view's parent file is gone (orphan, parent " + std::to_string(feed.bad_parent) + ")"; }
        else if (feed.missing) { st.error = SessionError::invalid; error = "park v2: a chunk file is missing or truncated"; }
        else if (feed.bad_chunk) { st.error = SessionError::invalid; error = "park v2: a chunk failed its key (corrupt)"; }
        else { st.error = pkind; if (error.empty()) error = "park v2: parse failed"; }
        return false;   // M-C5: classify foreign (permanent) vs invalid (corrupt) vs io/memory (retry later)
    }
    if (vf != nullptr && !vf->images.empty()) {
        // M-C3b3b: a view-mounted child re-joins CoW right after assembly: each viewed region is verified
// against the materialized parent and swapped for shared views (tails stay owned, byte-verified).
        // Steady-state host RAM then holds ONE physical copy of a lineage head no matter how many siblings
        // restore. The read's own peak is unchanged (the copy is assembled first: direct-view assembly is the
        // follow-up), so the RAM preflight stays priced at the full logical child - honest for what it peaks.
        std::unordered_map<uint64_t, std::vector<std::array<size_t, 5>>> per_parent;
        for (const auto& e : ents)
            if (e.kind == 2) {
                const size_t li = (size_t) (e.k1 >> 3), part = (size_t) (e.k1 & 7);
                if (li < image.kv.size() && part < 5) {
                    auto& sp = per_parent[e.k0];
                    if (sp.empty()) sp.resize(image.kv.size());   // M-C8: the spans table was indexed [li][part]
                    sp[li][part] += (size_t) e.len;               // while still EMPTY: operator[] on an empty
                }                                                 // vector derefs null data: every view child
                                                                  // that reached the adopt pass wrote off the
                                                                  // page (boot SEGV loop, run53 forensics).

            }
        size_t parents_hit = 0, reused = 0, nview = 0, vbytes = 0;
        for (const auto& pp : per_parent) {
            size_t tot = 0; for (const auto& ar : pp.second) for (size_t v : ar) tot += v;
            nview += 1; vbytes += tot;
            const auto got = vf->images.find(pp.first);
            if (got == vf->images.end()) {
                std::fprintf(stderr, "strata serve: park disk: ADOPT-DEBUG miss parent=%llu feed_size=%zu\n",
                             (unsigned long long) pp.first, vf->images.size());
                continue;
            }
            size_t r = 0;
            if (conversation_kv_share_views(image, got->second, pp.second, &r)) { parents_hit += 1; reused += r; }
            else
                std::fprintf(stderr, "strata serve: park disk: ADOPT-DEBUG verify-fail parent=%llu keeps=%zu bytes=%zu\n",
                             (unsigned long long) pp.first, pp.second.size(), tot);
        }
        if (nview && !reused)
            std::fprintf(stderr, "strata serve: park disk: ADOPT-DEBUG no-adopt views=%zu vbytes=%zu MiB=%zu\n",
                         nview, vbytes, reused >> 20);
        if (reused)
            std::fprintf(stderr, "strata serve: park disk: view-mounted %zu MiB shared with %zu parent(s)\n",
                         reused >> 20, parents_hit);
    }
    bytes = (size_t) sb.st_size;
    st.error = SessionError::none;
    return true;
#else
    st.error = SessionError::io; error = "park v2 files are POSIX-only"; return false;
#endif
}

// M-C5 pass 1: collect the chunk keys a v2 file names, WITHOUT reading the payload or consulting identity:
// the header authenticates itself (magic, version, header hash) and the table its own recorded counts.
// A foreign file's payload may be unreadable by this runtime, but its chunk bytes are content-addressed
// and may be shared with files this boot DOES accept: the caller phantom-references the returned keys so
// no sweep can unlink them (a later boot with the matching geometry reads the file back whole).
bool ParkDisk::scan_v2_keys(const std::string& path, std::vector<PdKey>& keys, std::string& error) const {
#ifndef _WIN32
    int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW);
    if (fd < 0) { error = "park v2 scan: open: " + std::string(std::strerror(errno)); return false; }
    struct stat sb;
    bool ok = ::fstat(fd, &sb) == 0 && S_ISREG(sb.st_mode) && (uint64_t) sb.st_size >= kPd2Body;
    uint8_t h[kPd2Hdr];
    if (ok && ::pread(fd, h, kPd2Hdr, 0) != (ssize_t) kPd2Hdr) ok = false;
    uint32_t ver = 0, hdr = 0;
    uint64_t nblocks = 0, hh = 0;
    if (ok) {
        std::memcpy(&ver, h + 8, 4); std::memcpy(&hdr, h + 12, 4);
        std::memcpy(&nblocks, h + 64, 8); std::memcpy(&hh, h + 80, 8);
        ok = std::memcmp(h, kPd2Magic, 8) == 0 && ver == 2 && hdr == (uint32_t) kPd2Body &&
             session_hash64(h, 80, 0) == hh && nblocks >= 1 && nblocks <= kPd2MaxBlocks;
    }
    std::vector<uint8_t> tbl;
    if (ok) {
        tbl.resize((size_t) nblocks * kPd2Ent);
        if (::pread(fd, tbl.data(), tbl.size(), (off_t) kPd2Hdr) != (ssize_t) tbl.size()) ok = false;
    }
    ::close(fd);
    if (!ok) { error = "park v2 scan: not a structurally valid v2 file"; return false; }
    for (uint64_t i = 0; i < nblocks; ++i) {
        const uint8_t* e = tbl.data() + (size_t) i * kPd2Ent;
        const uint8_t kind = e[0];
        uint64_t len = 0, k0 = 0, k1 = 0;
        std::memcpy(&len, e + 8, 8); std::memcpy(&k0, e + 16, 8); std::memcpy(&k1, e + 24, 8);
        if (kind > 2 || len == 0 || len > kPd2Block) { error = "park v2 scan: bad table entry"; return false; }
        if (kind == 1) keys.push_back(PdKey{k0, k1});   // kind 2 names a parent SERIAL: that file shields its own keys
    }
    return true;
#else
    error = "park v2 files are POSIX-only";
    return false;
#endif
}

} // namespace strata::core
