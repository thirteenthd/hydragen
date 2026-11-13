// M-C3a implementation. All file I/O goes through conversation_file.cpp (STRSES v1): identity binding,
// bounded parse, payload hashes, temp-file-and-rename publish, mode 0600 files under a 0700 folder.
#include "strata/core/conversation_park_disk.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef _WIN32
#include <cerrno>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
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

} // namespace

std::optional<ParkDisk::Options> ParkDisk::from_env() {
    const char* dir = std::getenv("STRATA_PARK_DISK");
    if (!dir || !*dir) return std::nullopt;
    Options o;
    o.dir = dir;
    o.budget = (size_t) pd_env_u64("STRATA_PARK_DISK_MIB").value_or(0) * 1024 * 1024;
    o.min_free = pd_env_u64("STRATA_PARK_DISK_MIN_FREE_MIB").value_or(4096) * 1024ull * 1024ull;
    return o;
}

void ParkDisk::init(const Options& o, IdentityFn identity, LimitsFn limits) {
    dir_ = o.dir;
    budget_ = o.budget;
    min_free_ = o.min_free;
    identity_fn_ = std::move(identity);
    limits_fn_ = std::move(limits);
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
    // Rebuild the index: every file already in the folder is fully validated (identity, this engine's read
    // limits INCLUDING the RAM preflight, payload hash) by the read itself; the image is then dropped except
    // its token key. A read that REFUSES the file (foreign, corrupt, over-limits) deletes it: fail-closed, a
    // bad file must never sit there for a later turn to retry forever. A read that merely could not run
    // (RAM preflight right now, io) keeps it and stops the scan rather than churn.
    const auto t0 = std::chrono::steady_clock::now();
    size_t found = 0, refused = 0;
    std::vector<std::string> paths;
#ifndef _WIN32
    if (DIR* d = opendir(dir_.c_str())) {
        while (const dirent* e = readdir(d)) {
            const std::string name = e->d_name;
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
        if (!id) { ++refused; break; }
        SavedConversation image;
        size_t bytes = 0;
        SessionReadLimits limits;
        if (limits_fn_ && !limits_fn_(limits, error)) {
            std::fprintf(stderr, "strata serve: park disk: index stop at %s (%s)\n", path.c_str(), error.c_str());
            break;
        }
        SessionStatus st;
        if (!session_file_read(path, *id, image, bytes, error, limits, &st)) {
            std::fprintf(stderr, "strata serve: park disk: index refuse %s (%s)\n", path.c_str(), error.c_str());
            if (st.error == SessionError::invalid) {
#ifndef _WIN32
                ::unlink(path.c_str());
#else
                ::DeleteFileA(path.c_str());
#endif
            } else {
                ++refused;
                break;   // io/memory right now: the rest waits for a later boot rather than churn the disk
            }
            ++refused;
            continue;
        }
        if (image.live.ids.empty() || image.kv.empty()) {   // nothing restorable to key on
            ++refused;
            continue;
        }
        Entry en;
        en.serial = next_serial_++;
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
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const std::string budget_note = budget_ ? std::to_string(budget_ >> 20) + " MiB" : std::string("free-space only");
    std::fprintf(stderr, "strata serve: park disk: ON %s (index %zu files, %.1f GiB, %zu refused, rebuilt in "
                         "%.0f ms; budget %s, keep-free %llu MiB)\n",
                 dir_.c_str(), found, (double) bytes_ / (1024.0 * 1024.0 * 1024.0), refused, ms,
                 budget_note.c_str(),
                 (unsigned long long) (min_free_ >> 20));
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

void ParkDisk::spill(SavedConversation&& image) {
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
    if (!session_file_read(it->path, *id, image, bytes, error, limits, &st)) {
        if (st.error == SessionError::invalid) {   // corrupt or foreign: never again
            std::fprintf(stderr, "strata serve: park disk: refuse %s (%s); file deleted\n",
                         it->path.c_str(), error.c_str());
            bytes_ -= it->bytes;
            entries_.erase(it);
            ++drops_;
        }
        return std::nullopt;                        // io/memory: the file stays, a later turn retries
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    loads_ += 1;
    bytes_ += bytes - it->bytes;
    it->bytes = bytes;
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
#else
                ::DeleteFileA(pd_tier_path(dir_, f.serial).c_str());
#endif
            }
            continue;
        }
        if (f.ok) {
            it->done = true;
            std::fprintf(stderr, "strata serve: park disk: spilled %zu tokens, %.1f MiB in %.0f ms "
                                 "(%zu entries, %.1f GiB on disk)\n",
                         it->ids.size(), (double) it->bytes / (1024.0 * 1024.0), f.ms,
                         entries_.size(), (double) bytes_ / (1024.0 * 1024.0 * 1024.0));
        } else {
            std::fprintf(stderr, "strata serve: park disk: spill failed for %zu tokens (%s)\n",
                         it->ids.size(), f.error.c_str());
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
    const Entry e = std::move(entries_.front());
    entries_.erase(entries_.begin());
    bytes_ -= e.bytes;
    ++drops_;
#ifndef _WIN32
    ::unlink(e.path.c_str());
#else
    ::DeleteFileA(e.path.c_str());
#endif
    // a not-yet-started write for this serial leaves the queue with it (its rename would republish the file);
    // one already in flight is caught by drain_finished(), which unlinks a file nobody indexes anymore
    std::lock_guard<std::mutex> g(mtx_);
    for (auto q = queued_.begin(); q != queued_.end(); ++q)
        if (q->serial == e.serial) { queued_.erase(q); break; }
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

} // namespace strata::core
