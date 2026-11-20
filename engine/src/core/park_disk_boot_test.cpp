// M-C5: boot-rebuild GC cascade regression for the park-disk tier (CHANGES 151/152). CPU-only.
// Properties under test (the b3a story: chunk files are SHARED between lineage files, so a boot that
// deletes a refusing .ses mid-walk drops bytes a KEPT file still needs, and the cascade wipes the store):
//   - a file bound to ANOTHER identity or geometry is QUARANTINED (kept on disk, never deleted), and the
//     chunk keys its own table names stay shielded from the orphan sweep;
//   - a fresh boot with the matching identity re-indexes those files and RESTORES them (byte-identical);
//   - a kept file's serial stays out of the new-spill range (a new spill never overwrites a kept file);
//   - corrupt files still fail closed (deleted, and genuinely unreferenced chunks swept) - one family,
//     not the store: shared bytes stay while a kept file names them;
//   - a walk that could not finish (limits cannot run) skips the chunk sweep entirely: unvisited files
//     keep their bytes for a later boot;
//   - v1 files (no chunks) quarantine on foreign identity too.
#include "strata/core/conversation_park_disk.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

// The park-disk view-adoption call is engine-side (conversation_snapshot.cpp, which links GPU restore code);
// since container v3 (S5b/186) NO writer emits kind==2 view table entries (sibling sharing is structural:
// salted content-addressed chunks dedup identically-hashed blocks), so this branch never runs. A link stub
// keeps the test CPU-only.
namespace strata::core {
bool conversation_kv_share_views(SavedConversation& img, const SavedConversation& donor,
                                 const std::vector<std::array<size_t, 5>>& keep, size_t* reused_bytes) {
    static_cast<void>(img); static_cast<void>(donor); static_cast<void>(keep);
    if (reused_bytes) *reused_bytes = 0;
    return false;
}
} // namespace strata::core

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}

std::vector<uint8_t> bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}

ConversationCheckpoint checkpoint(size_t tokens, uint8_t seed) {
    ConversationCheckpoint c;
    for (size_t i = 0; i < tokens; ++i) c.ids.push_back(int32_t(1000 + i));
    c.imgs = {};
    c.gdn = bytes_of(4099, seed);
    c.ple = bytes_of(77, seed + 1);
    c.tails = bytes_of(301, seed + 2);
    c.dead = bytes_of(64, seed + 3);
    c.block_pos = bytes_of(8, seed + 4);
    c.used = 42 + seed;
    return c;
}

// four kv layers, each with a 3 MiB k buffer: one >= 2 MiB region per layer -> several chunk files in the v2 store
SavedConversation make_image(uint8_t seed, size_t ntok) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0; s.layer_hi = 48;
    s.cvec = false;
    s.live = checkpoint(ntok, seed);
    for (int layer = 0; layer < 4; ++layer) {
        ConversationKv kv;
        kv.format = 2; kv.cells = 1024; kv.heads = 2; kv.head_dim = 256;
        kv.page_size = 64; kv.pooled_rows = 0; kv.idx_dim = 128;
        kv.k = pattern(3u << 20, uint8_t(seed + layer * 8));
        kv.v = pattern(4096, uint8_t(seed + 1 + layer * 8));
        kv.k_scale = pattern(512, uint8_t(seed + 2 + layer * 8));
        kv.v_scale = pattern(512, uint8_t(seed + 3 + layer * 8));
        s.kv.push_back(std::move(kv));
    }
    return s;
}

// live-shape geometry: 200k cells, 64 B/cell k/v, 4 B/cell scales (non-uniform size on
// purpose: exercises the whole-buffer block path), 6 layers -> ~600 salted blocks/spill.
SavedConversation make_live_image(uint8_t seed, size_t ntok) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0; s.layer_hi = 48;
    s.cvec = false;
    s.live = checkpoint(ntok, seed);
    for (int layer = 0; layer < 6; ++layer) {
        ConversationKv kv;
        kv.format = 2; kv.cells = (int64_t) ntok; kv.heads = 2; kv.head_dim = 32;
        kv.page_size = 64; kv.pooled_rows = 0; kv.idx_dim = 128;
        kv.k = pattern(ntok * 64, uint8_t(seed + layer * 8));
        kv.v = pattern(ntok * 64, uint8_t(seed + 1 + layer * 8));
        kv.k_scale = pattern(ntok * 4 - 17, uint8_t(seed + 2 + layer * 8));   // non-uniform
        kv.v_scale = pattern(ntok * 4, uint8_t(seed + 3 + layer * 8));
        s.kv.push_back(std::move(kv));
    }
    return s;
}


bool same_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used && b.stage_parts.empty();
}

bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        !same_checkpoint(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size())
        return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i]; const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale) ||
            !(x.pooled == y.pooled)) return false;
    }
    return true;
}

std::vector<std::string> list_ses(const fs::path& dir) {
    std::vector<std::string> out;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().filename().string().rfind("park-", 0) == 0 && e.path().extension() == ".ses")
            out.push_back(e.path().filename().string());
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<std::string> list_chunks(const fs::path& dir) {
    std::vector<std::string> out;
    const fs::path ch = dir / "chunks";
    if (!fs::exists(ch)) return out;
    for (const auto& e : fs::directory_iterator(ch)) out.push_back(e.path().filename().string());
    std::sort(out.begin(), out.end());
    return out;
}

void corrupt_file(const fs::path& p) {
    std::fstream f(p, std::ios::binary | std::ios::in | std::ios::out);
    const auto len = fs::file_size(p);
    f.seekg(std::streamoff(len / 2));
    char c = 0;
    f.read(&c, 1);
    f.seekp(std::streamoff(len / 2));
    f.put(char(c ^ 0xFF));
    f.flush();
}

// a v2 .ses keeps a 64 KiB fixed table area: flipping bytes in its UNUSED padding is undetected (nothing
// hashes it) - corruption arms must hit HASHED bytes: the v2 header itself, or a chunk file.
void corrupt_v2_header(const fs::path& p) {
    std::fstream f(p, std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(std::streamoff(8));   // 'ver' word: inside the header hash
    char c = 0;
    f.seekg(std::streamoff(8));
    f.read(&c, 1);
    f.seekp(std::streamoff(8));
    f.put(char(c ^ 0xFF));
    f.flush();
}

// the chunk key of the first kind-1 (chunk) table entry of a v2 file
std::string first_chunk_of(const fs::path& ses) {
    std::ifstream f(ses, std::ios::binary);
    uint8_t h[128];
    f.read(reinterpret_cast<char*>(h), 128);
    uint64_t nblocks = 0;
    std::memcpy(&nblocks, h + 64, 8);
    if (nblocks == 0 || nblocks > 2048) return "";
    for (uint64_t i = 0; i < nblocks; ++i) {
        uint8_t e[32];
        f.read(reinterpret_cast<char*>(e), 32);
        uint64_t k0 = 0, k1 = 0;
        std::memcpy(&k0, e + 16, 8);
        std::memcpy(&k1, e + 24, 8);
        if (e[0] == 1) {
            char hex[49];
            std::snprintf(hex, sizeof hex, "%016llx%016llx", (unsigned long long) k0, (unsigned long long) k1);
            return std::string(hex) + ".bin";
        }
    }
    return "";
}

ParkDisk::IdentityFn ident(uint64_t model, uint64_t config) {
    return [model, config](SessionFileIdentity& id, std::string&) { id.model = model; id.config = config; return true; };
}
ParkDisk::LimitsFn ok_limits() {
    return [](SessionReadLimits& l, std::string&) {
        std::array<int64_t, 18> g{};
        for (size_t i = 0; i < g.size(); ++i) g[i] = int64_t(100 + i);   // what make_image stamps
        l.geometry = g;
        l.layer_range = std::pair<int64_t, int64_t>{0, 48};
        return true;
    };
}
ParkDisk::LimitsFn dead_limits() {   // simulates the preflight that cannot answer right now: the walk stops
    return [](SessionReadLimits&, std::string& e) { e = "limits cannot run right now"; return false; };
}
ParkDisk::LimitsFn foreign_geometry_limits() {
    return [](SessionReadLimits& l, std::string&) {
        std::array<int64_t, 18> g{};
        for (size_t i = 0; i < g.size(); ++i) g[i] = int64_t(100 + i);
        g[0] = 999;   // this runtime's geometry: no stored file matches it
        l.geometry = g;
        l.layer_range = std::pair<int64_t, int64_t>{0, 48};
        return true;
    };
}

// flip one hashed byte of a chunk file: size unchanged, content no longer matches its key
void flip_byte(const fs::path& p) {
    std::fstream f(p, std::ios::binary | std::ios::in | std::ios::out);
    const auto at = std::streamoff(fs::file_size(p) > 1000 ? 1000 : 0);
    char c = 0;
    f.seekg(at);
    f.read(&c, 1);
    f.seekp(at);
    f.put(char(c ^ 0xFF));
    f.flush();
}

ParkDisk* g_pd = nullptr;  // wait_for pump target

// waits until `fn` holds, up to ~10 s of 20 ms polls (the writer is async; completions flip at the next
// drain: an empty spill drains finished writes and returns without queueing anything: that is the pump)
template<class Fn> void wait_for(Fn&& fn, const char* label) {
    for (int i = 0; i < 500; ++i) {
        if (g_pd) g_pd->spill(SavedConversation{}, -1);
        if (fn()) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    check(false, label);
}
} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "mc5_boot_test_store";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const uint64_t M1 = 0x1111111111111111ull, C1 = 0x2222222222222222ull;
    const uint64_t M2 = 0x3333333333333333ull, C2 = 0x4444444444444444ull;
    const ParkDisk::Options opt{dir.string(), 0, 0, true};

    // ---- boot 1 (identity 1): three spills; A and B identical (all chunks deduped), C a longer, different
    // conversation (its own chunk set, sharing the identical head bytes)
    SavedConversation origA = make_image(1, 1000), origB = make_image(1, 1000), origC = make_image(5, 1200);
    {
        ParkDisk pd;
        pd.init(opt, ident(M1, C1), ok_limits());
        g_pd = &pd;
        check(pd.enabled(), "boot1: tier enabled");
        pd.spill(SavedConversation(origA), -1);
        wait_for([&] { return list_ses(dir).size() == 1; }, "boot1: A published");
        const size_t chunks_after_A = list_chunks(dir).size();
        pd.spill(SavedConversation(origB), -1);
        wait_for([&] { return list_ses(dir).size() == 2; }, "boot1: B published");
        check(list_chunks(dir).size() == chunks_after_A, "boot1: B deduped: its identical bytes added no chunks");
        pd.spill(SavedConversation(origC), -1);
        wait_for([&] { return list_chunks(dir).size() > chunks_after_A; }, "boot1: C added its own chunks");
        check(pd.entries() == 3, "boot1: three entries");
        // in-boot restore reads the v2 files back through the same read_v2 the boot walk uses
        {
            const auto m = pd.best(origA.live.ids, {}, false);
            check(m.tokens == (int64_t) origA.live.ids.size(), "boot1: A matches in-boot");
            std::string error;
            const auto img = pd.load(m, error);
            check(img.has_value(), "boot1: A loads back in-boot");
            if (img) check(same(*img, origA), "boot1: in-boot load is byte-identical");
        }
        g_pd = nullptr;
    }   // ~ParkDisk drains and joins: this is the restart
    // immediate restart (identity unchanged): the boot walk must re-index all three straight off disk
    {
        ParkDisk pd;
        pd.init(opt, ident(M1, C1), ok_limits());
        check(pd.entries() == 3, "boot1b: an immediate restart re-indexes all three files");
    }
    const std::vector<std::string> ses1 = list_ses(dir);
    const std::vector<std::string> chunks1 = list_chunks(dir);
    check(ses1.size() == 3, "boot1: three .ses files");
    check(chunks1.size() >= 2, "boot1: chunk files on disk (content-defined cuts may land sparse on synthetic bytes)");

    // plant a garbage chunk and a half-written .tmp: a COMPLETED walk deletes these, an unfinished walk must not
    const fs::path planted = dir / "chunks" / ("0123456789abcdef0123456789abcdef.bin");
    const fs::path planted_tmp = dir / "park-000000099.ses.tmp";
    { std::ofstream(planted) << "junk"; std::ofstream(planted_tmp) << "junk"; }

    // ---- boot 2 (identity 2 — the stale-binary restart): all three files are FOREIGN
    {
        ParkDisk pd;
        pd.init(opt, ident(M2, C2), ok_limits());
        g_pd = &pd;
        check(pd.entries() == 0, "boot2: foreign files are not indexed");
        check(list_ses(dir).size() == 3, "M-C5: foreign files are QUARANTINED, not deleted");
        check(list_chunks(dir).size() == chunks1.size(), "M-C5: foreign chunk bytes stay; only the planted orphan left");
        check(!fs::exists(planted), "boot2: a completed walk still sweeps the planted orphan");
        check(!fs::exists(planted_tmp), "boot2: the planted .tmp left");
        // a new spill during a foreign boot must not land on a kept file's serial
        SavedConversation origD = make_image(7, 900);
        pd.spill(std::move(origD), -1);
        wait_for([&] { return pd.best(std::vector<int32_t>{}, {}, false).tokens == 0 && list_ses(dir).size() == 4; },
                 "boot2: the foreign boot spilled its own file");
        check(fs::exists(dir / "park-000000004.ses"),
              "M-C5: the kept serials stay out of the new-spill range (the new file is park-000000004)");
        g_pd = nullptr;
    }
    const std::vector<std::string> chunks2 = list_chunks(dir);   // chunks1 + D's own

    // ---- boot 3 (identity 1 again): the quarantined family re-indexes and RESTORES byte-identically
    {
        ParkDisk pd;
        pd.init(opt, ident(M1, C1), ok_limits());
        check(pd.entries() == 3, "M-C5: the matching boot re-indexes the quarantined family");
        const auto m = pd.best(origA.live.ids, {}, false);
        check(m.tokens == (int64_t) origA.live.ids.size(), "boot3: the restored family matches the prompt");
        std::string error;
        const auto img = pd.load(m, error);
        check(img.has_value(), "M-C5: cross-boot RESTORE reads the quarantined-then-restored family back");
        if (img) check(same(*img, origA), "boot3: the restored image is byte-identical");
        const std::vector<std::string> s3 = list_ses(dir);
        check(s3.size() == 4, "boot3: the still-foreign file (D) is quarantined too: four files on disk");
        check(list_chunks(dir).size() == chunks2.size(), "M-C5: boot 3 swept nothing the kept files name");
    }

    // ---- boot 4: CORRUPTED SHARED CHUNK - containment, one family not the store (CHANGES 152): A and B
    // share one chunk; corrupting it refuses BOTH of them (invalid: unlinked), while C (its own chunks) is
    // indexed, restores byte-identically, and keeps its bytes. The dead chunk goes unreferenced and is swept.
    {
        const std::string shared = first_chunk_of(dir / ses1[0]);
        check(!shared.empty(), "boot4: A names a chunk file");
        flip_byte(dir / "chunks" / shared);
        ParkDisk pd;
        pd.init(opt, ident(M1, C1), ok_limits());
        check(pd.entries() == 1, "boot4: the corrupted chunk costs its family (A, B), not the store");
        check(list_ses(dir).size() == 2, "boot4: A and B's .ses unlinked; C stays, D stays quarantined");
        const auto chunkset = list_chunks(dir);
        check(std::find(chunkset.begin(), chunkset.end(), shared) == chunkset.end(),
              "boot4: the dead chunk, referenced by nothing, was swept");
        const auto m = pd.best(origC.live.ids, {}, false);
        check(m.tokens == (int64_t) origC.live.ids.size(), "boot4: C still matches");
        std::string error;
        const auto img = pd.load(m, error);
        check(img && same(*img, origC), "boot4: C restores byte-identically next to the containment");
    }

    // ---- boot 5: a corrupt v2 HEADER fails closed on that file alone; the quarantined file keeps its bytes
    corrupt_v2_header(dir / "park-000000003.ses");
    {
        ParkDisk pd;
        pd.init(opt, ident(M1, C1), ok_limits());
        check(pd.entries() == 0, "boot5: the header-corrupt file unlinks only itself");
        check(list_ses(dir).size() == 1 && fs::exists(dir / "park-000000004.ses"),
              "boot5: the quarantined foreign file is still kept");
    }

    // ---- boot 6: FOREIGN GEOMETRY for the quarantined set: still a quarantine, still no deletion
    {
        ParkDisk pd;
        pd.init(opt, ident(M1, C1), foreign_geometry_limits());
        check(pd.entries() == 0, "boot6: geometry-foreign files are not indexed");
        check(list_ses(dir).size() == 1, "M-C5: geometry-foreign files are QUARANTINED, not deleted");
    }

    // ---- boot 7: the walk CANNOT finish (limits dead): unvisited files keep their bytes, so an orphan planted
    // while they sat unindexed must survive — an unfinished walk may never sweep
    { std::ofstream(planted) << "junk"; }
    {
        ParkDisk pd;
        pd.init(opt, ident(M1, C1), dead_limits());
        check(pd.entries() == 0, "boot7: the walk stopped where limits failed");
        check(fs::exists(planted), "M-C5: an unfinished walk SKIPS the chunk sweep (keys of unvisited files are unknown)");
    }
    {
        ParkDisk pd;   // boot 8: healthy again: the walk completes, the planted orphan finally goes
        pd.init(opt, ident(M1, C1), ok_limits());
        check(pd.entries() == 0, "boot8: the store re-indexed normally after the dead-limits boot");
        check(!fs::exists(planted), "boot8: the completed walk swept the planted orphan");
        check(list_ses(dir).size() == 1, "boot8: no file was lost across the whole sequence");
    }


    // ---- live-shape arm (live-N repro): thousands of salted blocks, non-uniform scale buffers,
    // scalar regions > 1 MiB; spill, restart, and the boot validator must ACCEPT the files the
    // v3 writer wrote (the live N pass refused 2 of 3 with "table and file size disagree").
    const fs::path dir2 = fs::temp_directory_path() / "mc5_boot_test_store_live";
    fs::remove_all(dir2);
    fs::create_directories(dir2);
    SavedConversation origL = make_live_image(21, 200000), origM = make_live_image(22, 200000);
    {
        ParkDisk pd;
        pd.init(ParkDisk::Options{dir2.string(), 0, 0, true}, ident(M1, C1), ok_limits());
        g_pd = &pd;
        pd.spill(SavedConversation(origL), -1);
        wait_for([&] { return list_ses(dir2).size() == 1; }, "live: L published");
        pd.spill(SavedConversation(origM), -1);
        wait_for([&] { return list_ses(dir2).size() == 2; }, "live: M published");
        std::printf("live arm: %zu chunks\n", list_chunks(dir2).size());
        g_pd = nullptr;
    }
    {
        ParkDisk pd;
        pd.init(ParkDisk::Options{dir2.string(), 0, 0, true}, ident(M1, C1), ok_limits());
        check(pd.entries() == 2, "live: restart re-indexes both v3 files (live-N reject repro)");
        std::string error;
        const auto m = pd.best(origL.live.ids, {}, false);
        check(m.tokens == (int64_t) origL.live.ids.size(), "live: L matches after restart");
        const auto img = pd.load(m, error);
        check(img && same(*img, origL), "live: byte-identical restore after restart");
        if (!img) std::printf("live arm load error: %s\n", error.c_str());
        g_pd = nullptr;
    }
    fs::remove_all(dir2);


    // ---- v1 arm (no chunk tier): foreign identity quarantines the plain file too
    const fs::path dir1 = fs::temp_directory_path() / "mc5_boot_test_store_v1";
    fs::remove_all(dir1);
    fs::create_directories(dir1);
    {
        ParkDisk pd;
        pd.init(ParkDisk::Options{dir1.string(), 0, 0, false}, ident(M1, C1), ok_limits());
        g_pd = &pd;
        SavedConversation e = make_image(9, 1000);
        pd.spill(std::move(e), -1);
        wait_for([&] { return list_ses(dir1).size() == 1; }, "v1: spill published");
        g_pd = nullptr;
    }
    {
        ParkDisk pd;
        pd.init(ParkDisk::Options{dir1.string(), 0, 0, false}, ident(M2, C2), ok_limits());
        check(pd.entries() == 0, "v1: foreign files are not indexed");
        check(list_ses(dir1).size() == 1, "M-C5: a foreign v1 file is QUARANTINED, not deleted");
    }

    // ---- v3 lineage arm (S5c discovery, 2026-11-20): a RAM CoW view child spills through the
    // lineage path, but container v3 writes NO kind==2 view entries: the child's blocks salt and
    // digest IDENTICALLY to the parent's (prefix-ladder salt), so the store dedups them to zero new
    // chunk files. Restoring the child re-reads only those shared chunks and is byte-identical.
    // (Pre-v3 the child would reassemble its parent through view_image; that machinery is inert
    // since 0ccaab99 - this arm pins the fact.)
    const fs::path dirv = fs::temp_directory_path() / "mc5_boot_test_store_views";
    fs::remove_all(dirv);
    fs::create_directories(dirv);
    {
        ParkDisk pd;
        pd.init(ParkDisk::Options{dirv.string(), 0, 0, true}, ident(M1, C1), ok_limits());
        g_pd = &pd;
        SavedConversation P = make_image(21, 1000);
        const SavedConversation Pcopy(P);
        SavedConversation C = make_image(22, 1000);
        C.live.ids.push_back(2000);   // a distinct child conversation over the shared head bytes
        for (size_t i = 0; i < C.kv.size(); ++i) {
            const size_t n = P.kv[i].k.size();
            check(C.kv[i].k.adopt_shared(P.kv[i].k, n), "v3lin: adopt_shared parent k prefix");
            C.kv[i].k.resize_for_overwrite(n);   // fully shared: the whole k part views the parent
        }
        const SavedConversation Ccopy(C);
        const std::vector<int32_t> cids = C.live.ids;
        size_t nsp = 0;
        C.kv[0].k.head_spans([&](const uint8_t* q, size_t l) { ++nsp; });
        check(nsp > 0, "v3lin: child buffer has borrowed head spans pre-spill");
        // The borrowed()/view contract: spans stay views only while the DONOR image is alive (prod
        // pins view_donor holders in the RAM cache; the test mirrors that by keeping P alive here).
        // The parent's park RECORD spills from an independent copy so P itself is never moved-from.
        pd.spill(SavedConversation(Pcopy), 101);
        wait_for([&] { return list_ses(dirv).size() == 1; }, "v3lin: parent published");
        const size_t chunks_after_P = list_chunks(dirv).size();
        C.view_donor = 101;
        pd.spill(std::move(C), -1);
        wait_for([&] { return list_ses(dirv).size() == 2; }, "v3lin: child published");
        check(pd.entries() == 2, "v3lin: two entries");
        check(list_chunks(dirv).size() == chunks_after_P,
              "v3lin: v3 dedups the child's shared blocks to zero new chunks");
        {
            const auto m = pd.best(cids, {}, false);
            check(m.tokens == (int64_t) cids.size(), "v3lin: child matches");
            std::string error;
            const auto img = pd.load(m, error);
            check(img.has_value(), "v3lin: child restores reading only the shared chunks");
            if (img) check(same(*img, Ccopy), "v3lin: restore is byte-identical");
        }
        // ---- S5c: salt-to-donor delta serve on live-shape geometry (chunked 256 KiB KV blocks;
        // make_image's KV is too small to blob). (a) A donor that IS the spilled image's RAM twin:
        // every salted KV block must serve from RAM and the restore stays byte-identical. (b) The
        // same donor with one corrupted byte in a covered block: the salt still matches (ids are
        // untouched) but the re-hash does not - the read retries on the store, the file is never
        // blamed, and the restored bytes are exactly the ones that were written.
        {
            SavedConversation L = make_live_image(27, 200000);
            const SavedConversation Lcopy(L);
            pd.spill(SavedConversation(L), -1);
            wait_for([&] { return list_ses(dirv).size() == 3; }, "s5c: L published");
            const std::vector<const SavedConversation*> dlist{ &L };
            const uint64_t db0 = pd.delta_blocks(), de0 = pd.delta_events();
            const uint64_t rb0 = pd.delta_bytes(), dr0 = pd.disk_read_bytes();
            const auto mL = pd.best(L.live.ids, {}, false);
            std::string errA;
            const auto imgA = pd.load(mL, errA, std::getenv("S5C_OFF") ? nullptr : &dlist);
check(imgA.has_value() && same(*imgA, Lcopy), "s5c: RAM-twin restore is byte-identical");
            // 6 layers x (k+v) x 48 COMPLETE 4096-cell blocks = 576; measured 582 (tails beyond
            // the donor end and inline tails take other paths) - the assertion holds the floor.
            check(pd.delta_blocks() - db0 >= 2 * L.kv.size() * 48, "s5c: every complete KV block served from RAM");
            check(pd.delta_bytes() - rb0 >= (pd.delta_blocks() - db0) * 262144, "s5c: every served block carried its full 256 KiB");   // cumulative-safe: bytes scale with blocks
            L.kv[3].k.visit((size_t) 10 * 4096 * 64 + 8, 1, [](uint8_t* q, size_t, size_t) { *q ^= 0xFF; return true; });
            const auto imgB = pd.load(mL, errA, &dlist);
            check(imgB.has_value() && same(*imgB, Lcopy), "s5c: a divergent donor retries from the store, byte-identical");
            check(pd.delta_events() - de0 == 1, "s5c: divergence recorded exactly once");
        }
        g_pd = nullptr;
    }
    fs::remove_all(dirv);

    std::printf("park_disk_boot_test: %d checks passed\n", checks);
    fs::remove_all(dir);
    fs::remove_all(dir1);
    return 0;
}
