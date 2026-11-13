import sys

def edit(path, pairs):
    with open(path) as f:
        s = f.read()
    for old, new in pairs:
        if s.count(old) != 1:
            raise SystemExit(f"ANCHOR {path} count={s.count(old)}: {old[:80]!r}")
        s = s.replace(old, new)
    with open(path, "w") as f:
        f.write(s)
    print("OK", path)

# 1) conversation_cache.hpp: spill hook + evict_oldest hands the victim over before its RAM goes
edit("include/strata/core/conversation_cache.hpp", [
    ('#include <array>',
     '#include <array>\n#include <functional>'),
    ('''    ConversationCache(size_t budget, size_t slots, int64_t min_tokens = 0)
        : budget_(budget), slots_(slots), min_tokens_(min_tokens) {}''',
     '''    ConversationCache(size_t budget, size_t slots, int64_t min_tokens = 0)
        : budget_(budget), slots_(slots), min_tokens_(min_tokens) {}
    // M-C3a: optional disk tier (conversation_park_disk.hpp). When set, an image evict_oldest() throws away is
    // handed over IN FULL before its RAM goes; the tier decides what it keeps. drop_superseded() and clear()
    // do not spill: superseded copies are stale by construction and clear() drops poisoned state.
    std::function<void(SavedConversation&&)> spill_hook;'''),
    ('''            if (entries_[i].pinned() || (guarded_ != 0 && serials_[i] == guarded_)) continue;
            bytes_ -= entries_[i].bytes();
            erase(i);
            ++evictions_;
            return true;''',
     '''            if (entries_[i].pinned() || (guarded_ != 0 && serials_[i] == guarded_)) continue;
            SavedConversation victim = std::move(entries_[i]);   // M-C3a: the tier may take it before it goes
            bytes_ -= victim.bytes();
            erase(i);
            ++evictions_;
            if (spill_hook) spill_hook(std::move(victim));
            return true;'''),
])

# 2) generate.cpp: include + tier init + admission restore + skip-restore keeps the file
edit("src/program/generate.cpp", [
    ('#include "strata/core/conversation_memory.hpp"',
     '#include "strata/core/conversation_memory.hpp"\n#include "strata/core/conversation_park_disk.hpp"'),
    ('''            id.model = *model_fp;
            id.config = *config_fp;
            return true;
        };
        // What a park captures:''',
     '''            id.model = *model_fp;
            id.config = *config_fp;
            return true;
        };
        // M-C3a: an optional disk tier under the parked cache (STRATA_PARK_DISK=DIR). LRU eviction spills the
        // victim to an identity-bound STRSES file; an admission whose whole usable prefix lives on disk reads it
        // back and restores it through the normal conversation-snapshot path. Read limits (and the RAM preflight
        // inside them) are this engine's, exactly as the /slots disk-session path builds them below.
        strata::core::ParkDisk park_disk;
        if (const auto pd_opt = strata::core::ParkDisk::from_env()) {
            park_disk.init(*pd_opt,
                [&](strata::core::SessionFileIdentity& id, std::string& e) { return session_identity(id, e, [] {}); },
                [&](strata::core::SessionReadLimits& lim, std::string& e) {
                    const uint64_t floor = (uint64_t) o.conversation_cache_min_free_mib << 20;
                    lim.admit = [floor](uint64_t need, std::string& why) {
                        const auto mem = strata::core::sample_host_memory();
                        const std::optional<uint64_t> avail = mem ? std::optional<uint64_t>(mem->available) : std::nullopt;
                        if (strata::core::conversation_memory_admit(avail, need, floor)) return true;
                        why = "not enough free RAM to read the parked image back (" + std::to_string(need >> 20) +
                              " MiB plus a floor of " + std::to_string((long long) (floor >> 20)) + " MiB)";
                        return false;
                    };
                    return strata::core::conversation_session_read_limits(
                        lim, ss, g, mtp.kv_state(), (uint64_t) o.max_context,
                        (uint64_t) std::max(o.prompt_cache, 1), e);
                });
            if (park_disk.enabled() && conversations.enabled())
                conversations.spill_hook = [&](strata::core::SavedConversation&& img) {
                    park_disk.spill(std::move(img));
                };
        }
        // What a park captures:'''),
    ('''            if (parked.tokens > std::max(resume, slot_tokens) && !borrow)
                incoming.emplace(conversations.take(parked.index));
            if (incoming || borrow) slot_source = -1;''',
     '''            if (parked.tokens > std::max(resume, slot_tokens) && !borrow)
                incoming.emplace(conversations.take(parked.index));
            // M-C3a: no usable RAM match at all (batch-slot resume and live state always win): a spilled parked
            // conversation comes back from the disk tier as a full take - no borrowing from disk in v1.
            bool from_disk = false;
            if (!incoming && !borrow && parked.tokens <= std::max(resume, slot_tokens) && park_disk.enabled()) {
                const auto d = park_disk.best(ids, req_imgs, want_cvec);
                if (d.tokens > std::max(resume, slot_tokens)) {
                    std::string derr;
                    if (auto img = park_disk.load(d, derr)) {
                        incoming.emplace(std::move(*img));
                        from_disk = true;
                    } else if (!derr.empty())
                        std::fprintf(stderr, "strata serve: park disk: load refused (%s); rereading the prompt\\n",
                                     derr.c_str());
                }
            }
            if (incoming || borrow) slot_source = -1;'''),
    ('''                    auto retry = std::move(*incoming);
                    incoming.reset();
                    if (!conversations.put(std::move(retry)))
                        std::fprintf(stderr, "strata serve: conversation cache: could not return the skipped snapshot\\n");''',
     '''                    auto retry = std::move(*incoming);
                    incoming.reset();
                    if (from_disk)
                        std::fprintf(stderr, "strata serve: conversation cache: disk image not restored; the file "
                                             "stays for a later turn\\n");
                    else if (!conversations.put(std::move(retry)))
                        std::fprintf(stderr, "strata serve: conversation cache: could not return the skipped snapshot\\n");'''),
])

# 3) build: the new TU links into strata_engine
edit("cmake/strata_gpu_engine.cmake", [
    ('src/core/arch_defaults.cpp src/core/nv_defaults.cpp src/core/vmm.cpp src/core/conversation_file.cpp)',
     'src/core/arch_defaults.cpp src/core/nv_defaults.cpp src/core/vmm.cpp src/core/conversation_file.cpp\n    src/core/conversation_park_disk.cpp)'),
])
print("PATCHES_DONE")
