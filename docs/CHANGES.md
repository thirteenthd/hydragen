# HydraGen change log



Chronological record of every measurement → decision → commit in the program.

Newest first; section numbers are stable, so §N always refers to the same entry.







## §176 — README restructured: measured block leads under the banner; one consolidated Install section (2026-11-17)

Layout pass, no content changes to any number:

- The measured-benchmark block (was "The headline") now sits directly under
  the badge line with no section header of its own — first thing a visitor
  reads is the before/after table and what it means in practice.
- All install text (requirements, try-it commands, prebuilt/SmartScreen/
  glibc notes, dashboard line) merged with the old "Install notes"
  (first-run behavior, model blobs, living beside upstream Strata) into one
  `## Install` section. One duplicate prerequisite sentence folded out; one
  stale sentence ("A source build is the only option" — predates the
  prebuilt-release flow) corrected to state both routes.
- Table-of-contents counters fixed to 181 (they had drifted at 170/167).

Counter: 181.

## §179 — S0 memory-tier levers A/B on the reference machine: mmap file tier loses to the resident arena when the pack fits RAM (2026-11-18)

First rung of the concurrency milestone ladder run on the 12 GB reference machine (RTX 3080 Ti, 60 GiB RAM, IQ3_XXS canonical stack, batch 2). Three arms, one boot swap each, bench N=1/2/4/8 (agg tok/s):

- A0 baseline (resident expert arena): 66.8 / 61.2 / 60.4 / 53.9
- A1 `--mmap-experts` (experts served from the GGUF shards via page cache, no experts.bin): 55.3 / 34.6 / 44.7 / 46.2
- A2 A1 + `STRATA_IO_PF_STAGE=1 STRATA_IO_PF_SCHED=1` (prefetch staging + 149 window byte-budget scheduler): 43.9 / 39.2 / 45.7 / 44.0

Mechanism read from the engine's own file-tier telemetry: the A2 decode windows saw the page cache hold 100% of the decode's expert reads — 94.5 MB from storage (63 major faults) against 35.7 GB of expert reads — so the mmap arms were NOT I/O-bound on this host; the deficit is the per-read access path plus the prefetch scheduler closing its windows (1,387 closed, budget 0 MB/window: nothing to steal when the cache already holds everything). The baseline engine restored after the ladder; canonical state unchanged.

Decisions: (a) the resident arena stays the default everywhere the expert set fits host RAM — the zero-copy file tier is expected to pay only for packs beyond RAM (the IQ3_S-on-64-GiB configuration), which is where S2/S5 will re-test it; (b) the sm_86 decode-flag table verified ON by default in this lineage (STRATA_SM86_PERF is only an off-switch), so there was no table lever to enable; (c) S0 as a free win is CLOSED-NEGATIVE on the reference machine — the ladder's value moves to S1/S2.


## §178 — Installers: IQ3_S model-tier option; repo hygiene (mc3_patches.eml removed) (2026-11-17)

- install.ps1 gains `-Model IQ3_XXS|IQ3_S` and install.sh `--model IQ3_XXS|IQ3_S`; the local
  serve config is now wired from the matching canonical config. New configs/strata-iq3s-m25-0213-ctx131k.json
  is the canonical cow stack applied to the IQ3_S artifacts (same flags and context; IQ3_S pack and
  shards; the learned-profile key follows the blob name automatically). engine setup already accepted
  --model IQ3_S — the installers were the IQ3_XXS-only gap.
- docs/MODELS.md documents the IQ3_S artifact sizes (shard 1 54,817,524,224 B, pinned in
  engine/setup.py; the 28,800,138,432 B PLE shard shared across tiers; tier needs 64 GB RAM, ~15% slower).
- mc3_patches.eml (an exported patch-mail from earlier batch work) removed from the repo root —
  build mail is a working artifact, not repo content.

## §177 — Park-disk v2 proven on Windows: spill, restore, token-identical resume across a restart (2026-11-17)

Engine change (canonical lineage, then mirrored here): three files — `src/core/conversation_park_disk.cpp`, `include/strata/core/conversation_park_disk.hpp`, `CMakeLists.txt`.

- The disk tier's own file primitives (`pd_*`) now build on both platforms. POSIX passes them through as before; the Windows branch builds the same operations on the CRT low-level I/O: reparse points refused before the open (the `O_NOFOLLOW` equivalent), link count and size read through `GetFileInformationByHandle` so the single-linked-regular-file rule is a real check on NTFS, publish via `MoveFileEx(REPLACE_EXISTING)`, and `errno` kept live on every failure so the error lines read the same on both sides.
- Chunk files are created requesting write mode on both platforms. Without it Windows creates them read-only, and the boot-4 corruption-flip test can never flip — the CPU gate failed exactly there.
- Freshness fix: `ParkDisk::best()` drains finished async spills at its top, the contract `drain_finished()`'s declaration states for every serve-thread call. Without it a same-boot follow-up could not find the conversation it had just spilled: the chunks were on disk and `best()` reported zero tokens.
- End-to-end measured on the Windows host (locally built CUDA engine, IQ3_S model, a 768 MiB / 4-slot conversation window sized to force eviction):
  - same boot: the eviction spilled 24,527 tokens to 44 chunk files (1.5 s); the follow-up read them back (405 ms) and reused 24,415 tokens — the follow-up cost 5.3 s against 12.3–12.6 s for the same prompt prefilled fresh.
  - across a process restart: the store index rebuilt in 430 ms, the restore reused 24,415 tokens again.
  - the follow-up's answer through the restored path is token-identical to the answer generated live.
- Engine floor measured alongside: two identical clean disk-off boots diverged their follow-up answers at token 29 — fresh prefill is not bit-reproducible across CUDA boots. The park-cycle identity law therefore compares against the floor first: both sides must be restored from the park store to demand identical answers; a fresh-prefill pair that fails the floor voids the identity claim.
- Linux re-verified on the same sources: 6/6 conversation + park-disk CPU tests pass with the dual-platform translation.

Counter: 182.

## §175 — concurrency wording held to the host-vs-decode line (2026-11-17)

A skeptic pass over the §174 headline caught a real seam: "eight
conversations at a time" implies eight-way simultaneous generation. The
measured serving envelope is: **eight ~100k-deep sessions hosted live, at
most two generating tokens at any instant** (`--batch 2`); the rest live in
the conversation cache and queue — the C8 warm-turn TTFT p50 of 21.2 s *is*
that queue, measured. "8 concurrent" only survives on the concurrent-
*sessions* reading, so the unattackable version of the actual claim went in
instead — concurrency without residency:

- H1 → "massive MoE models, a live fleet of deep chats, on consumer 12 GB
  GPUs."
- Subhead → "Eight 100,000-token conversations live at once. Two answering
  at a time. All of them interactive. ..."
- Badges: "concurrent 100k-token chats: 8" → "hosted 100k-deep chats: 8"
  plus a new "batched decode: 2-up" badge — the split is stated on the face
  of the page.
- GitHub About → "...host up to 8 live ~100k-token agent sessions on one
  12 GB RTX card (2-up batched decode)..."
- The performance table already said "queue-bound at this depth"; no numbers
  changed anywhere. Counter: 180.

## §174 — headline rebalance: sell the measured envelope, not the old ambition (2026-11-17)

The headline had led with "high-concurrency" ever since the program's first
weeks, when 32–64 simultaneous streams were the ambition. The box has always
ended up serving 4–8 concurrent 100k-token-deep chats on one 12 GB card
(`--batch 2`, eight conversation-cache slots), and every body number in the
repo — badges, "Measured expectations" — already said so. Only the marquee
still sold the dream. The promise now matches the product:

- README H1: "high-concurrency MoE inference on 12 GB GPUs" → "massive MoE
  models, eight conversations at a time, on consumer 12 GB GPUs"; the subhead
  gains "Simultaneously." — the concurrency claim is now a number you can
  check, not an adjective.
- GitHub About: led with "Run massive MoE models on consumer GPUs —"; the
  rest of that line was already measured (8 concurrent ~100k sessions, 2x
  wall / 2x warm-TTFT / 25x fewer lost prefixes vs stock).
- `docs/FINAL_REPORT.md`: left intact — its body already states the mission
  as "real decode concurrency (4+ simultaneous streams)"; the program title
  there is a historical marker, not a live promise.

No code, no config, no flag changes. Counter: 179.

## §173 tokenizer-missing startup error made actionable; installers pre-warn on an unbuilt pack (2026-11-16)

A fresh Windows install that ran the installer but not the model-setup step hit
`server.py: error: the model's tokenizer is missing (...vocab.json); run setup
again` with no hint of what "setup" meant or where the tokenizer comes from.
The tokenizer folder is exported by the pack build itself (after the index
pass, before the expert copy starts), so an unbuilt pack is a missing
tokenizer — by design, but the error text didn't say so.

Three fixes:

- `serve/server.py`: the tokenizer-missing error now derives the model dir from
  the config's own tokenizer path and prints the exact repair — the resumable
  `SETUP` rebuild command, plus a direct `strata_tokenizer.py --gguf <shard-1>
  --out <pack-dir>` one-liner for the case where the pack is complete but its
  tokenizer folder isn't (older packs predate the export step, and re-running
  setup skips a completed pack).  Exercised live: it exits at arg-parse time,
  before any minutes of model loading.
- `install.ps1` / `install.sh`: after writing the config, both check
  `<model-dir>/packs/iq3_xxs/tokenizer/vocab.json` and print a NOTE line that
  serve will refuse to start until the model step runs.
- `docs/MODELS.md`: one sentence tying the tokenizer folder to the pack build
  and pointing at the now-self-explaining error.

Same-commit repair: the §172 entry itself had shipped with read-tool line
prefixes (`1|`, `2|`...) accidentally embedded in its first 60 lines during
insertion; stripped.  Counter: 178.

## §172 — integrity oddities closed; expert-profile learns and persists across boots; the second box container rebuilt on the HG engine (2026-11-16)

Three long-standing integrity oddities were walked back to root cause with the
forensics retained on-box (store snapshots + per-era logs):

- `park-000000003.ses` `session blocks: payload checksum mismatch` (x3) and
  `park-000000004.ses` `chunk file is missing or truncated` (x14) +
  `not a single-linked regular file` (x1): all signatures live in the pre-M-C5 /
  disk-V2-default-bug window. Pre-M-C5 boot sweeps were single-pass, so chunk
  files still referenced by living entries could be unlinked under them; on the
  v1 single-file format the same damage surfaces as a payload-checksum mismatch.
  The retained snapshots settle the mutation question: park-3's bytes are
  byte-identical between the run23-era backup and the junk-store backup — a
  published session file was never rewritten after publish; only its referenced
  chunks were eaten. The run53c line `verify-fail parent=4 keeps=13` was the
  walk re-reading that old scar (parent already gone), and the 13 kept children
  are the published-parent guard's content-chunk fallback doing its job.
  Since the fix chain (two-phase reference-protected boot walk, published-parent
  guard, read_v2 fix, V2 default fix): three consecutive clean boots including
  the boot after a hard power cut — store verified 91/91, refused 0,
  quarantined 0.
- M-C7's marker was verified against git history (`df09ca77`); the milestone
  decoder row was corrected from a vague "banked defensive fix" to the actual
  mechanism: a view entry may only name a parent that is already disk-published.
- The run57 `park COW anchor` marker under a `PARK_COW=0` env reading was a
  probe artifact: the marker print is strictly inside the `PARK_COW=1` gate in
  every era of the source, and the probe read a different (newer) PID during a
  back-to-back restart race while the previous boot's engine still owned the
  GPU. The append-only shared log interleaves both boots and produced the
  apparent contradiction.

The expert-routing profile now learns on the machine's own traffic and survives
restarts. The shipped profile is a boot seed; the VRAM tier re-ranks experts
from live accesses at runtime. The installers now wire `--expert-profile-save`
so that ranking is persisted (every `expert_profile_save_min`, default 10 min)
and the next boot re-seeds from it instead of the generic upstream ranking. The
save file is keyed by the model blob name, so switching model artifacts can
never load a foreign ranking — an explicitly-given profile that fails its
layer/expert-count check aborts startup by design upstream, and the artifact
key makes that path unreachable. Coalesce defaults were left untouched: they
are read once at serve import (no runtime retune) and their win zone is
bursts of short requests, not deep sessions.

The second reference box moved onto the rebuilt serving image: current engine
chain, fork-only env names (`HG_*`), and the learned-profile loop wired through
its persistent data mount (`--expert-profile-save` with the model-keyed file
verified on the engine cmdline). Boot rebuild indexed 69 accepted entries
(159.1 GiB) and refused 43 files stamped by the previous image's engine
(`park v2: bound to another model or configuration`) — those went to
quarantine-in-place, which keeps the files and shields the chunk keys they
name, the designed cross-version data-safety behavior. That behavior also
retires an older bookkeeping oddity on that box: `refused=43` alongside a
large retained `chunks/` count was quarantine-in-place, not a counter desync.

One public claim corrected by the same reading: the tier budget is a save-side
gate (`drop_oldest` fires in the spill-admission path); a lower cap does not
delete files at boot. §164's "boot performs the pruning pass automatically"
was wrong — on the reference host the 159 -> 64 GiB reduction happened as
save-side drops during that sweep, and the second box's idle 160 GiB tier
simply waited for its next spill, exactly as the code says.

## §171 prebuilt CUDA binaries via GitHub Releases; Windows park-disk compile guards (2026-11-16)

The installers now install a prebuilt engine first and compile from source only
as a fallback. Release `prebuilt-1` carries `hydragen-linux-cuda-x86_64.tar.gz`
(fat sm_86/89/120 binary + bundled CUDA runtime libs) and
`hydragen-windows-cuda-x86_64.zip` (fat sm_86/89/120 exe + cublas/cublasLt
DLLs), both listed in a SHA256SUMS manifest; provenance (commit, CUDA version,
arch list, glibc floor) ships inside each archive. The Linux asset is built
from a fresh public clone on the reference host; the Windows asset on a stock
Windows 11 + CUDA 13.5 + VS BuildTools host with the installer's exact flags.

- `install.sh` / `install.ps1`: fetch the release asset, verify SHA-256,
  install under `runtime/`; any download, hash, or glibc-floor failure falls
  back to the source build automatically. `--build-from-source` /
  `-BuildFromSource` skips the download path entirely. The Linux config writer
  adds `runtime/lib` to `LD_LIBRARY_PATH` when the prebuilt engine is used; the
  Windows installer runs `Unblock-File` on everything it extracts.
- README + installer headers carry the SmartScreen caveat (manual browser
  downloads of the unsigned zip may warn; More info -> Run anyway) and the
  source-build fallback message.
- Engine: two missing `_WIN32` preprocessor guards in the park-disk v2 tier
  (chunks-dir mkdir, orphan-chunk readdir sweep — dead at runtime on Windows
  because v2 self-disables there, but they still had to compile) broke Windows
  source builds since the v2 tier landed. Both sites now guard exactly like the
  neighbouring blocks. Verified: the first Windows CUDA fat binary of the fork
  chain links and runs from its packaged directory (cublas DLLs colocated).

## §165 the levers ship as defaults: tier cap + admit ordering on every serving surface (2026-11-16)

Per §164's measured reading, both levers became defaults on all three serving
surfaces (all overridable by env):

- Reference-host config generator: park-disk budget default 163840 -> 65536 MiB,
  `STRATA_ADMIT_SHORT_FIRST` default 0 -> 8192 (`MAX_WAIT_S` stays 30). Boot
  verified: the live serve process's own `/proc/<pid>/environ` shows
  65536 / 8192 / 30, and `/v1` answers with context 131072.
- Public installers: freshly generated `hydragen-local.json` now carries
  `STRATA_PARK_DISK_MIB=65536` and `STRATA_ADMIT_SHORT_FIRST=8192`; the harmful
  163840 default is gone from the repo. `docs/HOW_IT_WORKS.md` carries the same
  guidance in the stack paragraph.
- Second box: container config env patched and the container restarted - the
  engine child's `/proc/<pid>/environ` shows 65536 / 8192, `/v1/models` back to
  HTTP 200. Its 160 GiB tier is still on disk 15 min after boot (unlike the
  §164 pruning pass); the budget bites as save-side drops bring it down.

`bash -n` and the PowerShell AST parser both clean; the written env block was
key-checked against a rewritten `hydragen-local.json`.

## §170 milestone decoder doc: docs/MILESTONES.md — every marker defined with intended outcome and progression (2026-11-16)

Milestone codes (M-C10c, cow-m3b, forkb4, b3a, KD4, run56, c4/c8) were used
across every document with no single page decoding them. Added
docs/MILESTONES.md: the full inventory in four eras — the substrate ladder
(M1→M11 + the dense-IQ search), the per-step / speculative-decode / KV
funding era (M12→M25, including the Strata C++ worker adoption that became
this product's substrate), the Strata-fork behavior campaigns, and the KV
copy-on-write campaign (M-C0→M-C10c with every sub-arm's intent, verdict and
notarization pointer) — plus a glossary of the supporting conventions
(§NNN, runNN, trNN, KD2-4, c4/c8, b2a/b3a/b3b, the cow-mX branch names,
forkb4, CN, HTIER, quant classes).

The entries were mined from the server-side change log's 174 section headers
and cross-checked against the milestone commit messages in the vendored
engine chain; no new claims - every row points at an existing notarized
entry. README docs map gained the link, and UPSTREAM_DELTA now resolves its
markers through the new page.

## §169 fork knobs renamed to HG_*: fork-only env names read HG_ first, STRATA_ spelling falls back (2026-11-16)

Per §168's inventory, the thirteen env names introduced by this campaign
(park-disk tier family: PARK_COW, PARK_DISK, _MIB, _V2, _MIN_FREE_MIB; deep
resume; the CoW arena / slots / fork / join / shared / permute experiments)
are fork-only, so renaming them costs nothing and gains legibility. The engine
now reads the HG_ spelling first and falls back to the legacy STRATA_
spelling - pre-rename configs and containers keep working untouched. Upstream-
derived knobs (ADMIT_*, the coalesce family, PARK_TRACE / PARK_ASYNC /
PARK_VRAM_MIB, CRASH_BT) stay plain STRATA_ reads; the identifier diff against
the vendored base confirmed zero overlap either way.

Mechanics: new header strata/core/hg_env.hpp (HG_-first getenv + u64
wrapper); read sites converted in generate.cpp (9), conversation_park_disk.cpp
(2, covering the pd_env_u64 wrapper), layer.cpp (1). The installers write HG_
names into hydragen-local.json; HOW_IT_WORKS, UPSTREAM_DELTA and the README
coexistence note state the naming rule.

Verified on the reference host: merge gates park_disk_boot_test (38 checks)
and conversation_cache_test (4842 checks) green on the renamed tree; a boot
with only the legacy STRATA_ spelling confirmed the fallback, then the HG_
boot went canonical - serve process and engine child each carry the five
HG_ names, and /metrics reports park_disk enabled, budget_mib 65536, 40
entries / 68.5 GiB store, 0 refused. One restart race surfaced and was
re-run clean: the second boot refused with "too little free VRAM" while the
first boot's engine still held the GPU - restart-race law, not a rename bug.

The second box still runs its pre-rename image on legacy spellings via the
fallback, by design; it picks the HG_ names up with its next image rebuild.

## §168 upstream-delta inventory: every v0.2.13 difference classified; lever knobs found to be upstream features (2026-11-16)

The repo carried the delta as history (vendored base merge, 16 individually
named chain commits, one tip sync) and as narrative (CHANGES), but no
consolidated inventory. Added docs/UPSTREAM_DELTA.md: net-new subsystems
(park-disk tier, parked-session CoW dedup, deep checkpoint resume -
identifier-count evidence: STRATA_PARK_DISK / ParkDisk / STRATA_PARK_COW /
RESUME_DEEPEST / CowPeer all 0 hits in the vendored v0.2.13 tree),
re-engineered internals (slot arenas + slot_base indirection, eviction
handoff, store boot rebuild, dashboard arms), shipped-but-off failed
experiments (join-at-fetch CowPeer registry, per-lineage shared region), and
upstream features this campaign measured and enabled.

Finding worth stating plainly: the §164 lever knobs
(STRATA_ADMIT_SHORT_FIRST / STRATA_ADMIT_MAX_WAIT_S, --prompt-cache-tail,
COALESCE_TOKENS, STRATA_CRASH_BT) exist in upstream v0.2.13 (18/18/17/2/14
identifier hits at the vendored base, unchanged across our chain). The
campaign A/B-quantified them, sized them, and ships them on by default; it
did not author them.

Also corrected the README engine line: the M-C3b->M-C10c chain lands as one
vendored-engine sync snapshot (single-parent commit), so granular attribution
covers the early chain and serve/web arms; the per-arm ledger for everything
else lives in this change log.

## §167 README truncation post-mortem: 85 bottom-half lines lost in the §163 edit, restored from git (2026-11-16)

Caught while adding the §166 auto-sizing blurb: the README stood at 39/41
lines, not ~160. The §163 notarization commit (0c5be807b) had accidentally deleted
README lines 38-122 - everything after the first "## The headline" sentence:
the before/after headline table, the "what that means in practice" bullets,
"Why this exists", the docs map, the model-blobs blurb, the fork-coexistence
section, and the [^run10] footnote. The intended §163 README edit was only
the counter bump.

Restored the intact 161-line version from the last clean commit (7beb2eba4, already
scrubbed, no leak exposure), re-applied the later legit edits (counter bumps,
the §166 try-it sentence), and verified the restored sections: table, docs
map, coexistence note, footnote. Counter now 172.

Law adopted: before any notarization commit, check `git diff --stat` - a docs
commit whose README hunk deletes more than the intended edit is malformed,
not notarization.

## §166 installer environment auto-sizing: GPU identity, RAM-sized tier budget, disk advisory (2026-11-16)

`install.sh` and `install.ps1` now detect the machine before writing configs:

- **GPU** via `nvidia-smi` (largest GPU by `memory.total`): echoes name / VRAM /
  compute capability; WARNING under 12000 MiB; WARNING for compute capability
  outside 8.x / 9.x / 1x.x (the build targets sm_80+); NOTE when `nvidia-smi`
  is absent - sizing is skipped, install does not fail.
- **Park-tier budget** = min(total RAM, 131072 MiB), floor 16384 - §164's law
  encoded: a tier budget sized to the RAM class, because a pinned larger cap
  measurably decays steady-state throughput. WARNING below 49152 MiB RAM
  (documented floor is 64 GB).
- **Disk advisory**: NOTE when the model drive's free space is under
  budget + 174080 MiB (model blobs ~160 GiB); advisory, not a gate.

The detected budget flows into `hydragen-local.json`. Probes against real
hardware: RTX 3080 Ti host with 62168 MiB RAM -> 62168 budget; RTX 3060 box
with 64204 MiB -> 64204; Windows host, RTX 3090, 130966 MiB -> 130966
(no clamping hit below 128 GiB). Parser checks clean on both script families.

## §164 park-cap + admit-ordering A/B on the reference host: 1.89x self-decay measured, levers recover it (2026-11-16)

The second-box run2 finding was reproduced on the reference host and then A/B'd. Baseline A0:
stock config (160 GiB tier cap, no admit ordering) on a box whose tier had drifted to its
159 GiB pin - harness C4 wall 316.9 s (run10, smaller tier) -> 598.8 s, a 1.89x decay with
the deep-re-prefill count up 1 -> 5 of 20. Same machine, same traffic: tier state is the
variable. Levers A1 (64 GiB cap + STRATA_ADMIT_SHORT_FIRST=8192, MAX_WAIT default 30 s):

- C4 wall 598.8 -> 371.0 s (-38%, recovering run10's 316.9 s); worst inter-token gap
  45.1 -> 6.1 s.
- C8 warm-turn TTFT p50 25.0 -> 13.5 s (halved), p95 254 -> 227 s; C4 p95 282 -> 104 s.
- Retention under the cap: deep re-prefills 3 -> 2 of 48 at C8 (improved), 5 -> 6 at C4
  (noise-level). Needle QC green all arms.
- Cost as designed: cold TTFT p50 at C8 163 -> 211 s - short turns take the control line;
  a long prompt counts as short after 30 s, so nothing starves.
- The budget gate is save-side: the first spills after the cap change drop older
  entries until the active set fits (tier 159 -> 64 GiB during that sweep); a lower
  cap does not delete files at boot (corrected in §172 after reading the code);
  RAM 57/60 GiB used through the sweep, guard pauses 0, min memavail 1.5 GiB.

Attribution: cap drives the stall/wall recovery (page-cache pressure from tier + pack
streams on 60 GiB hosts); admit ordering drives the warm-TTFT tails. Generator now passes
STRATA_PARK_DISK_MIB / STRATA_ADMIT_SHORT_FIRST / STRATA_ADMIT_MAX_WAIT_S through;
canonical default choice pending owner decision. Sweep files: levers/{a0,a1}/ under the
bench dir; runner ms01_levers.sh, log levers_run.log on the reference machine.

## §163 PCIe-flag correction by load test + suite rerun on the second box (2026-11-16)

§162 flagged the 3060 negotiating PCIe gen1 (gen4 capable). Direct diagnostics on the
box settle it: during a 1500-token generation nvidia-smi samples show the link at
**gen4 x16, SM 1920-1957 MHz, 99-100% util** for the whole burst; only the idle
samples read gen1/210 MHz. The engine metric sampler snapshots at idle, hence the
flag. The degraded-link hypothesis is withdrawn; no physical fix was needed.

Revised attribution of the 1.9-2.4x wall gap: ~1.7x VRAM bandwidth (360 vs 608 GB/s),
plus host-RAM working-set pressure at the 64 GB floor - engine RSS + 40 GiB arena +
a 170 GiB park tier competing for one page-cache pool (min_memavail 8.2 GiB through
the sweep; worst 43 s inter-token gap at C8 is consistent with page-cache thrash
under expert-stream re-reads, not link).

Suite rerun (run2, same four arms, warm box this time): measures run-to-run variance
and steady-state tier behavior; results appended to this entry when complete.

## §182 — installer UX: prebuilt download step announced, repo-clone requirement stated (2026-11-19)

`install.ps1` now prints what the prebuilt-engine download IS before it starts
(the slow step: ~100 MB zip from GitHub Releases; PowerShell's own progress label
reads "Writing web request ..." while bytes land on disk, which reads like an
upload), and reports size + elapsed seconds when done. The usage header now states
explicitly that the installer runs from a clone of this repo: the prebuilt zip
carries only the engine binary + CUDA DLLs; the serve layer, configs, expert-routing
profile and SETUP.bat live in the checkout. Message-only change; behavior
(hash verify, fallback to source build) unchanged.

## §181 — engine/ synced to upstream Strata 0.1.60; answer-stream delta and re-notarized battery constants (2026-11-18)

`engine/` is now upstream 0.1.60 (`38f63bc5`) with the fork overlay reapplied: 49 fork
files re-applied, 18 real three-way merges, and the DeepSeek-V4.1 family deletion this
fork carried is adopted upstream-side (0.1.60 removed it too, so that overlay patch
retires). Serve/, tests and packaging were re-merged against the fork's Nov-6 vendoring
point rather than the 0.1.42 engine base, which avoids resurrecting stale vendored
content.

Two fork regressions were caught before landing, both from files adopted wholesale:
the installed-artifact rename (`OUTPUT_NAME "hydragen"`, restored in
`cmake/strata_gpu_engine.cmake`) and the park-disk source registration, which since
M-C3 had been injected at build time by `mc3_patch.py` and is now registered in-tree in
the same cmake file. Fork-wire greps (`hg_getenv` family, ParkDisk/CoW wiring,
`is_view_donor`/`guarded_`, `RESUME_DEEPEST`) all pass on the merged tree.

Build and tests on modelserver01 (CUDA 13.1, arch 86-real, `STRATA_GEOMETRY=flash_next`):
the build is green and ctest reports 322/352. The 22 GPU-parity failures reproduce
byte-identically on the previous 0.1.41-era build on the same host (pre-existing, with
the live serving arm on the card), and 10 failures are new 0.1.60 tests (nvfp4/q36
families) whose skip behavior on consumer cards was not investigated.

Answer stream, measured cold-boot with the standard 200 s settle protocol, four
configurations each side. QC0-QC6 are byte-identical across every 0.1.60 variant
(default, `STRATA_KV_HOST_DMA=0`, `-DSTRATA_MMQ=ON`) and reproduce across repeated cold
boots of the previous build; QC7 (the longest prompt) forks one token-pattern deep:
the previous stream writes `**Answer:**\n\nThe FASTA record starts with…` and 0.1.60
writes `The FASTA record starts with…`. MMQ on/off and the KV host-DMA default
(0.1.60 turns it on for CUDA; it only stages host copies, it cannot move logits) were
tested as candidates and neither restores the old wording; the fork does not carry a
knob that reproduces 0.1.41 prefill numerics. The fork therefore accepts 0.1.60's
stream as the new baseline - this is the same drift class upstream's own quality suite
documents between releases (top-1 agreement 97.7-98.2%) - and re-notarizes the battery
constants on the canonical arm (modelserver01 s180 config, tail skip 0 per §180):

```
QC0 To calculate $17 \times 23$, we can use the standard multiplication method or break it down into simpler
QC1 The sky appears blue because molecules in the Earth's atmosphere scatter sunlight more effectively at shorter
QC2 Fast cache holds the heat,\nRAM waits in silent rows,\nData flows like streams.
QC3 Le serveur démarre rapidement.
QC4 Here are several implementations of the Fibonacci function in Python, ranging from simple to optimized:\n\n###
QC5 To determine which fraction is larger, $\frac{9}{10}$ or $\frac{10}{11}$, we can use a few different
QC6 97, 101, 103
QC7 The FASTA record starts with the character **`>`** (greater-than sign), and GTIN stands for **Global Trade Ite
```

Acceptance additionally ran the C4 concurrency harness on the same arm: 24/24 requests,
needle 4/4, 2.43 M prompt tokens, memory-guard pauses 0, crash lines 0, park-disk store
at 118 sessions / 4,096 chunks, engine alive after the run.

The production Windows arm (RTX 3090) keeps the 0.1.41-era binary until an explicit
switch; switching it is a deliberate decision, since it moves the served answers onto
this section's stream. Prior §179 and earlier numbers measured the 0.1.41-era stream.

## §180 — Route tail skip disabled on all arms (`STRATA_ROUTE_TAIL_SKIP=0`)

Upstream 0.1.42 made `STRATA_ROUTE_TAIL_SKIP=7` the CUDA-serve default [1]: a *missed* expert that
every token of a verify window routes at rank 7-9 (the smallest top-10 weights) is neither copied
nor computed; its contribution is zero, without renormalisation. Upstream measures +10-20% decode
on 12-16 GB cards, and states the answers differ slightly (KL 0.003-0.006 against 0.1.41
behavior, top-1 agreement 97.7-98.2%, same-engine floor 0.0006-0.0007).

Found on this fork 2026-11-18: the production arm's engine log reported 1,104,355 skipped experts
since start - the mechanism was ON on the serving answer stream, while every exactness and battery
gate in this repo runs with `STRATA_ROUTE_TAIL_SKIP=0` (the identity-gate flag set). The greedy
stream hashes, KL figures and battery results notarized here therefore measured a different answer
path than production served. Two fork-specific side effects beyond the quality question: a skipped
expert is never copied in, so it never enters the arena or the adaptive-LRU usage stats (the async
tier under-sees rank-7+ tail experts); and the draft-acceptance cost of skipping (which upstream
flags as net-negative near a full cache) was never measured against this fork's
`--spec 4 --coupled-draft` configuration.

Decision: off on all arms - reported decode speed describes the stream the answers actually come
from. The upstream +10-20% decode gain is given up. Pinned in the production runtime env, both
reference configs shipped here (`configs/strata-iq3s-m25-0213-ctx131k.json`,
`configs/strata-iq3xxs-m25-0213-ctx131k.json`) and all four modelserver01 experimental run configs.
(The live modelserver01 arm runs `--batch 2`, under which upstream already excludes the mechanism;
pinning the env makes the exclusion explicit rather than incidental.) Turning it back on is a
config change (`=7`); doing so would first take an A/B of quality-per-token-cost on this fork's
model and quant, which has never been run.

[1] `engine/docs/DETAILS.md`: the `STRATA_ROUTE_TAIL_SKIP (7)` settings row and the 0.1.42
    measurements table.

## §183 — S5a: unbuffered (O_DIRECT) park-restore blob reads, Linux, env-gated (2026-11-19)

Park-disk restores previously read blob bytes through the page cache, so every
restored conversation displaced expert-pack pages — a named contributor to the §181
decode stalls (worst 43 s inter-token gap at C8). This adds a transport-only
alternative gated off by default:

- `STRATA_PARK_DIRECT=1` (default off = existing buffered path, byte-identical):
  blob opens use `O_DIRECT` (EINVAL -> buffered fallback for tmpfs or filesystems
  without it); reads go through per-thread 4096-aligned bounce buffers (2 MiB),
  fetching the aligned span in up to 4 parallel `pread`s
  (`STRATA_PARK_DIRECT_THREADS=0..8`, default 4). Exact-byte copy out; short-read
  and EOF counts are identical to `pd_pread`, so every hash/parse fail-closed check
  sees the same bytes as before. Design ported from the DirectFile read-parts
  pattern in the MIT-licensed basalt fork (jesdga95/basalt, read at 63cb68ad) as
  reference; implementation native to this tree.
- Windows keeps the buffered path (stub at the platform boundary; the
  `NO_BUFFERING`/`OVERLAPPED` variant is a follow-up). Linux/Windows call sites are
  platform-identical; the Windows build compiles the stubs.
- Measured on modelserver01 (RTX 3080 Ti 12 GB, 64 GB DDR4, engine fork at §181 +
  this patch): `m25_parkladder100k` N=4 (4 x ~99k-token siblings sharing a 66.5k
  head) with an engine restart between the seed and restore phases, so every
  restore must read the disk blob: restore recall 12/12 under `STRATA_PARK_DIRECT=1`
  across two runs; restore wall times identical between direct and buffered arms
  to 0.1 s (55.5/25.8/26.2/25.9 s) — latency neutral; the engine's `read_bytes`
  (block-device reads; page-cache hits do not count) rose +779 MB across the restore
  phase on the direct arm, and system `Cached` fell across the phase instead of
  climbing — restore bytes stay out of the page cache, which is the point.
- Harness fix in the same change: `bench/m25_conc8_sweep.py` top-level
  `decode_p50` printed `pct(decs, 1)` — the 1st percentile, not the median.
  Values emitted under that label before this fix are p1 (comparisons stayed valid
  because both arms carried the same bug); corrected to `pct(decs, 50)`.

## §184 — S4 v1: /slots exposes per-slot idleness and held conversations (2026-11-19)

A multi-agent client (several Hermes sessions sharing one engine) needs to know
BEFORE sending what the batch slots hold and how long they have been idle -
the pre-refusal compaction question basalt answers with an elastic KV ceiling.
This is the serve-layer half of that, engine-agnostic:

- `GET /slots` per-slot entries gain three fields: `state` (idle / reading /
  decoding), `held_tokens` (the conversation the slot keeps for its next turn)
  and `idle_s` (seconds since the slot was last released; null: never used since
  the engine started). Existing fields unchanged; purely additive.
- Mechanism: a `slot_t_free` timestamp stamped at every busy->free transition in
  the batch manager, surfaced through the existing `slots_view()`, which
  `/slots` and the `/metrics` live view already shared.

Scoped-read finding (why basalt's idle-KV timer was NOT ported): this engine
already evicts cached slots LRU at admission (`pick_slot`), parks the held
conversation BEFORE the overwrite when it can (`park_slot`, all guards), and
restores from the park tier at admission (`park_disk.best` -> load -> resume).
Idle VRAM cost under `--batch` + `--kv-resident` streaming is just each idle
slot's resident KV window; recycling those windows (attach/detach with graph
re-capture) is recorded as the follow-up - it is also the ownership layer S2b
(shared read-only prefix blocks + CoW) needs. Verified on modelserver01: two
concurrent requests landed on separate slots; `/slots` reported
`held_tokens` 317/318 and `idle_s` 0.1/0.0 on the two released slots, null on
the two never used.

## §185 — Park-disk bar zero state: empty store draws the grey track, not full green (2026-11-20)

With the "free space only" budget the Park disk tier progress bar was hardcoded
to 100% width, so an empty store rendered full green - reading like a full
drive. That mode has no denominator, so the bar now fills only when something
IS parked (`bytes > 0 ? 100% : 0%`); an empty store shows the empty track.
Budget-denominated bars keep the used/cap ratio unchanged. Recorded as the
dashboard zero-state law in docs/laws/dashboard-ui-laws.md.

## §186 — Park-disk container v3 (BCS): content-addressed chunks + salted lineage keys; two boot-refuse bugfixes (2026-11-20)

The v2 park store re-wrote whole-conversation files and could not share prefix
state. v3 (BCS: block/content/chunk/salt container) stores KV blocks as
content-addressed chunk files and records per-entry lineage salts, so sibling
sessions with a shared prefix spill as deltas. Notarized live on the 3060 Ti
benchmark node (96,949-token unique park + shared-head trio, RAM-pressure
admission eviction):

- spill: `96596 tokens, 2196.1 MiB` (full parent) then two siblings at
  `1.4 MiB` and `1.2 MiB` - dedup deltas, store `7880 chunks, 4.1 GiB`;
- restart: `index 3 files, 6.2 GiB, 0 refused, 0 quarantined, rebuilt in 6222 ms`;
- restore from the chunk store: `97012 tokens, 1.4 MiB read in 984 ms` and
  `96596 tokens, 2196.1 MiB read in 1087 ms`.

Two defects found while the writer was live and fixed in the same change, each
of which had made the boot walk refuse every v3 file it wrote:

1. the payload counter in the streaming walk counted only the scalar region -
   KV block bytes were never added, so the header's payload field disagreed
   with the table sum (`table and file size disagree (entries 12612573 vs
   payload 9181, ...)`; the refuse verb now names all four compared terms);
2. the read-side pull loop opened a chunk file for INLINE blocks (`kind == 0`,
   bytes live in the `.ses` body): `errno 2` -> `a chunk file is missing or
   truncated`. Blob open + per-block re-hash now apply to chunk entries only;
   inline integrity rides on the payload hash, as in v2.

`park_disk_boot_test` gains a live-shape arm: a 200k-cell full spill plus
shared-head children written by the real writer, re-indexed in-boot and after
restart, restored byte-identical (~1200 chunks). 41 checks, exit 0.

Ops note: park files are bound to a config fingerprint; changing any fingerprint
input (e.g. the conversation-cache budget) quarantines the existing store as
designed - the files survive on disk and re-serve when the config returns.

## §187 — Reads yield by UNREAD tokens, not raw prompt length (#656b, 2026-11-20)

#656's give-way rule compares a waiting request's raw prompt length (×2)
against the active reader's. A continuation of a 128,979-token conversation
must read only its new turn (~25 tokens this test; ~500 in the production
case), but it measures as "long": it never triggers BYIELD, and
STRATA_ADMIT_SHORT_FIRST mis-ranks it the same way. Continuations queue
behind cold full reads for the whole read.

`_unread(e)`: prompt length minus the longest leading-token match against
`slot_held` plus `_recent_reads` (the last `2 × batch` admitted prompts; a
conversation's prefix outlives its slot - `slot_held` is cleared the moment
a new admission lands there, which an earlier cut of this change measured
at `best=0` on every call). Batch admissions now carry the prompt token
list in the wait entry regardless of STRATA_COALESCE_PREFILL; solo
admissions push it to the mirror. BYIELD and the admission classifier use
`_unread`; `STRATA_YIELD_UNREAD=0` restores the raw-length rule exactly.
The estimate can only shorten an entry, never lengthen it, and falls back
to the raw length when no prompt is carried or no prefix matches.

Live on the 3060 Ti benchmark node (2 slots): cold A1 (`128,959` tokens)
completes; cold B (`93,956`) starts; 8 s later A2 (A1's conversation,
`128,979` tokens, ~25 unread):

- raw rule: A2 wall `36.5 s`, completes after B (`40.9 s`);
- unread rule: A2 wall `12.3 s`, completes while B is still reading; B
  `44.9 s` (+10%, resumed mid-read at `cached 23,808`);
- `STRATA_YIELD_UNREAD=0` reproduces `36.5 s` (escape hatch verified).

A2's usage figures are unchanged (`cached 114,688` of `128,979`): the
change moves admission order only, never what is read or reused.

## §188 — Wait-age-gated yield: comparable cold reads interleave (STRATA_YIELD_AFTER_S, default off; 2026-11-20)

An audit of the claim "the engine holds the control lines across all ten
prompt chunks of a batch read" found the engine side already mitigated:
`batch_read_piece` (include/strata/program/batch_read.hpp) reads prompts in
pieces, and at every piece boundary the engine drains its stdin queue -
taking BSTOP, BYIELD (`give_way`, which keeps the part read in the slot for
reuse) - before the next piece (generate.cpp ~19166-19190). What actually
holds is serve's own ctl mutex, and BYIELD pierces it at the next boundary.

What remained true: `#656`'s rule (`unread x 2 <= this read's length`) never
fires for two comparable COLD reads, so cold requests serialize at full
read latency each. Measured on the 3060 Ti arm, two 93,956-token cold reads
8 s apart: first `39.6 s`, second `64.4 s` wall (25 s dead wait).

`STRATA_YIELD_AFTER_S=<seconds>` (0/unset: off): a waiter aged past the gate
earns a BYIELD whatever its length. Measured with the gate at 10 s: the two
reads ping-ponged once each - C1 yielded at its 84,480-token mark and
resumed with `cached 84,480` (no re-read), C2 at 67,584, resumed `cached
67,584`; both progressed concurrently. Cost: whole-batch completion ~5 %
slower (piece-transition tax) - hence default off; it is a fairness dial,
not a throughput one. `YIELDS_MAX` (2) caps each request's yields; the
#656b unread rule and the gate share that cap.

## §189 — S5c audit: the v2 view-entry mechanism is inert; the lineage path is pinned by test (2026-11-20)

S5c's delta-restore step assumed the v2 park format's explicit view entries
(`kind == 2` table rows pointing at a parent's chunks). Building donor
adoption onto it surfaced that the pull lambda has no `kind == 2` branch at
all: the view writer was added in `12cbf447` and deleted by this fork's own
§186 (`0ccaab99`) - `git log -S "ent[0] = 2"` proves both ends. Since §186 a
RAM shared-store child spills through the lineage path, and because v3
sharing is structural (salted content keys), the child adds **zero new chunk
files**: identical parents means identical keys. The donor plumbing built
against the dead path was reverted, not shipped.

`park_disk_boot_test` gains a `v3lin` arm pinning that truth: a view child's
spill re-verifies its kept spans against the file parent, the store dedups
to zero new chunks, and restore comes back byte-identical. This is the
first direct test coverage of RAM CoW view-child serialization. The suite
is now 51 checks (was 41). The stub for parent-image lookup returns false
with a comment saying why: no v3 writer emits those entries, so the dead
path must not grow back.

What S5c actually means in v3 terms: at pull time every table entry carries
its salt `(ph, L, part, c0)`, so a salt-to-donor index can skip store reads
entirely when a RAM sibling already holds those bytes - the feed's per-chunk
re-hash validates the donor bytes for free. That is the corrected target;
design recorded, no code shipped here.

## §190 — Restore-side pinned staging: implemented, measured, rejected (2026-11-20)

The restore path reads park files into pageable heap and copied to the GPU
with plain `cudaMemcpyAsync`, so it was hypothesized that the driver's
pageable bounce cost real time - the spill side already stages through a
pinned double buffer (`ParkAsync`) and the asymmetry looked like an oversight.
A restore pin-ring (two pinned 4 MiB pieces, DMA queued on the private copy
stream, FIFO slot bookkeeping, `STRATA_RESTORE_PINNED` kill switch) was built
and measured on the 3080 Ti arm against the same scenario with tracing on
both paths - the 96,949-token unique-head conversation, ~915 MiB of part
buffers, pinned staging vs the pageable path with identical call sites:

- pageable parts: `13,460-13,720 MiB/s` (~1 ms per 16 MiB part);
- pin-ring parts: `9,060-9,580 MiB/s` (~2 ms per part).

The pageable path already saturates the PCIe window on this host - modern
drivers pipeline their own pinned bounce - and the ring's extra CPU-to-pinned
memcpy makes it net ~30 % slower. The hypothesis is rejected on Linux; the
ring was reverted, not shipped. (The older WDDM-pageable-penalty folklore was
never re-measured here and stays untested on the Windows prod box; if a prod
restore ever profile H2D-bound rather than read-bound, that is when it gets
re-examined.) The real lever for bytes that never need to travel stays S5c's
pull-time salt-to-donor reuse.

## §191 — Crash fix: cache guards nest; the park anchor guard no longer clobbers the borrow donor guard (2026-11-20)

Prod (3090, `HG_PARK_COW=1`, `--batch 4`) died three times with
`restoring a borrowed conversation prefix: gone` and `exit code 1` -
2026-11-18 14:00, 18:23, and 2026-11-20 13:19 - each pair of serve errors
being one engine death hitting two concurrent requests, each crash log
showing `park COW anchor mounted` followed by `evictions=N` immediately
before the failure.

The chain: the borrow path guards the donor (`conversations.guard
(parked.serial)` - parking the outgoing conversation must not evict it),
then `park_current` runs, and inside `park_source` M-C10c's `ParkAnchorGuard`
guards its CoW anchor. The cache's guard was a single scalar slot, and the
anchor comment asserted "parking and restoring never interleave in this
loop" - the borrow path is exactly the case where they do. The inner guard
overwrote the outer one, `make_room` found the donor unguarded and evicted
it under admission pressure, and the restore's `find(parked.serial)` came
back null - fatal by contract ("validated above and guarded while parking").

Fix: the guard is now a small stack. `guard()` pushes, `unguard()` pops
its own entry, and every eviction, drop, and refusal predicate consults
`is_guarded(serial)`, so a donor stays held under any number of inner
anchor guards. `conversation_cache_test` gains a nesting arm that parks
three entries, guards two under a second guard, and pins both survival
and selective release (the 2026-11-20 crash is the scenario). No behavior
change for single-guard callers: the pre-existing seven guard arms pass
unchanged. Binary note: the prod exe predates this fix; a config-only
workaround (`HG_PARK_COW=0`) was rejected on purpose - it changes the
park-disk config binding and would quarantine the store.

## §192 — prebuilt-2 Windows port: the S5a direct-read ternary needed a third stub (2026-11-20)

Building the current main as the Windows prod engine (CUDA 13.5, MSVC, fat
arch 86-real/89-real/120, the installer's exact flags) failed on one
identifier: `pd_pread_direct`. The S5a unbuffered path is Linux-only by
design and the Windows side stubbed `pd_open_ro_direct` and
`pd_fd_is_direct` - but the pull lambda's three-way read
(inline / direct / buffered) references `pd_pread_direct` too, and a branch
that can never execute must still compile. The Windows block now stubs all
three; the buffered path remains the Windows transport (the NO_BUFFERING
variant stays the documented follow-up). Linux is byte-identical to the
stubbed line (same 51-check boot test), and the Windows tree built clean to
`strata 0.1.60` - this is the prebuilt-2 binary line replacing prebuilt-1
(`fe70cb5`, 0.1.59).

## §193 — prebuilt-2 live in prod: 0.1.60-fork on the 3090 with an honest VRAM pin (2026-11-20)

The production engine is now the §181-§192 line (commit `7e9f12cc`, `strata
0.1.60`, fat sm_86+sm_89+sm_120+PTX). The old binary and its park store are
kept beside it: `runtime/hydragen-0.1.59-fe70cb5.bak.exe` and
`park-disk-pre-prebuilt2-backup/` (19 entries) - the identity era changed
with the engine version, so the store re-derives from traffic.

The config fix rides the same restart: `--expert-cache 12288` was a silent
over-request that clamped to 6839 slots (12.98 GiB pinned, 720 MiB idle
free, mid-request allocation warnings as low as 17 MiB). The pin is now
stated honestly as `--expert-cache 6144` with `--vram-reserve-mib 2816`:
boot accepted it with **no clamp line**, 6453 experts resident on the
device, park-disk ON with a clean index (0 refused, 0 quarantined), and the
4-point needle battery ran `4/4` on engine 0.1.60 - needles recovered at
depths 2 500/6 000/12 000/22 000 words (prompt 18 995/46 993/96 994/186 994
tokens) with `overflow_fields={}`.

## §194 — S5c-v3: delta restore — a guarded RAM sibling serves the blocks it already holds (2026-11-20)

The v3 salt names every chunked block exactly: (prefix-ladder hash, layer, part, first cell).
A restore already has a sibling in RAM (the cache donor the lineage spilled from, §186). If the
sibling's own ladder — recomputed from its ids, hashes only, no KV bytes touched — agrees with an
entry's salt, its buffer serves the block: a memcpy instead of a store read. Integrity is NOT
relaxed: every RAM-served block is still re-hashed against the entry key, exactly like a disk
block. A donor whose bytes disagree (same ids, non-identical prefill) is a RETRY event, not an
error: the partial assembly is discarded, the read restarts from the store, the file is never
blamed, and the event is counted (`delta_events`, want: ~0).

Hook: the deep-resume call site hoists one donor via `ConversationCache::common_best` and holds
the PARK GUARD across the load (guards nest, §191). `STRATA_PARK_DELTA=0` disables the path
(default ON). `read_v2(..., donors, ReadStats*)` and `load(Match, err, donors)` carry the donor
list; the restore line now prints `delta: X MiB from RAM / Y MiB from store`.

Writer fix this exposed: the v3 writer flushed the FIRST payload block before any salt was set,
so the scalar head carried an all-zero salt - which is not self-distinguishing (the KV head block
salt is also all-zero: phv[0]=0, layer 0, part 0, cell 0). The probe read it as a real KV fact and
served geometry bytes from a KV buffer; the re-hash caught it (fail-closed, as designed), and every
sibling restore then silently paid a full store retry. The writer now seeds the scalar grid's
first salt before the walk. Other blocks' salts - and their chunk keys - are byte-identical to
before, so no park era is invalidated by this fix.

Gate (`park_disk_boot_test`, ms01, ship-byte): new s5c arm - a 200,000-cell live-shape image
spilled, then restored against its RAM twin. Byte-identical restore; 582 blocks (155.8 MiB)
served from RAM, 3.4 MiB from store; the assertion floor is every COMPLETE 4096-cell KV block
(6 layers x k+v x 48 = 576). A divergent donor (one byte flipped inside a salt-covered block)
restores byte-identical from the store with exactly one `delta_events` bump. Suite: **56 checks
passed** (was 51); targeted ctest 26/27 - the one failure is `expert_cache_segmented_test`
opening 0.09 GiB of expert cache while :8181 holds ms01's VRAM (known per-node constraint,
unrelated to park).

A/B (same file, same box, ms01 boot-test): warm 200k-cell restore 55-57 ms store-only vs
52 ms delta-served; cold delta 102 ms vs cold store 158 ms. Bytes moved from the store fell
~98% (159.2 MiB -> 3.4 MiB for a twin restore). Transport speed was already at the pageable
ceiling (§190) - this is the fewer-bytes lever, and it composes with everything upstream.

Files: `conversation_park_disk.{hpp,cpp}` (ReadStats, donor index in read_v2, probe + retry,
writer salt seed, counters), `generate.cpp` (call-site donor hoist + guard + `STRATA_PARK_DELTA`),
`park_disk_boot_test.cpp` (s5c arm). Ship md5 (local == ms01): park_disk.cpp `1c0f1993…`,
hpp `4b0c1bb4…`, generate.cpp `6ef2054a…`, boot_test.cpp `234e0e5e…`.

## 195. S5d: the park manifest carries a per-conversation expert-hit histogram; a deep-resume pre-stages those experts

The §190 law: fewer bytes fetched and fewer cold misses, not faster transport. S5c cut the
BYTES of a restore. S5d cuts the MISS: a parked conversation now remembers which experts it
was actually routing, and a resume stages them into the expert cache before the first window
runs, instead of letting the early windows pay a cold PCIe miss per expert they touch.

Recording (serve loop, free path): `ExpertDispatch` gains `hist_ptrs`/`hist_n`; the pool
dispatch bumps a saturating u8 cell at `[layer*n_expert + expert]` for every routed id
(`ids` is token-major: `k` ids per token, so row = `i / k`). The batch window site points
`hist_ptrs[t]` at each live slot's `BSlot.exp_hist` (lazy-zeroed, dropped by `sl = BSlot{}` at
takeover); every single-conversation window — the pipelined decode loop, the serial loops, the
prompt rounds — calls `s5d_point`, which points all rows at one file-scope `live_hist`, the
live conversation's own array, cleared at each request's decode start. (The park machinery and
the token loops are sibling blocks of one function, so this state sits at file scope; one serve
process, one model — advisory either way.) `STRATA_PARK_HIST=0` disables it (default ON).

Manifest: `hist_encode` trims the flat array to the hottest 8192 nonzero cells
(`S5DHv1` magic + u16 layers + u16 experts + u32 pairs + `(u16 idx, u8 count)` x pairs —
23 bytes for a three-cell test conversation). The blob rides the `.ses` file right after the
inline body; the v2/v3 header already hashed only `[0,80)`, so bytes 88/96 carry its
`(offset, len)` UNHASHED, and the payload hash never covers it. Pre-S5d files hold zeros
there - "no histogram" - so no park era is invalidated. A malformed section (offset not at
`kPd3Body + inline_bytes`, length > 16 MiB, short read, geometry disagreeing with decode)
degrades to "no histogram", never to a failed load: a wrong hint must not cost a restore.

Prewarm (deep-resume): after a successful restore, `hist_decode` sorts cells hot-to-cold and
stages up to 512 of them (max 32 per layer, `STRATA_PARK_PREWARM_MIB` default 256 MiB total)
through the SAME `admit`/`fill_slot`/`note_async_read` trio the window admission path uses,
on the same serve thread. Only FREE slots are claimed (`slot_of == kNotResident` and
`admit != kNotResident`): prewarm never evicts a live expert, and a full cache just logs
`refused`. A fill failure stops the prewarm only — the restore is untouched.
`STRATA_PARK_PREWARM=0` disables it.

Gate (`park_disk_boot_test`, ms01, ship-byte): new s5d arm — a 23-byte three-cell histogram
spilled with a 60k-token conversation, restored byte-identically, blob round-tripped through
the manifest; then ONE byte inside the section corrupted: the restore is still
byte-identical and `hist_decode` refuses the blob (the hint is dropped, `herr` empty).
Suite: **62 checks passed** (was 56); targeted ctest park/session/conversation/file/buffer
**20/20**.

Serve-path gaps the boot gate could not see: two A/B runs recorded NOTHING. First: the park
that real traffic fires is `park_current` parking the `live` mirror, not the batch-slot park.
Second (the one that mattered): serve's prompt prefill — 96,995 of a probe request's 97,059
tokens, nearly all the routing — runs through the pipelined decode loop, and a short decode
(`max_tokens=64`) is never admitted into a batch slot at all; pointing only batch windows
recorded the 64-token floor. A park-time log line (`park hist: <cells> expert cells, <bytes>`)
turned it measurable: with `s5d_point` at every window-reset site, ms01's ladder parked
96,995 tokens carrying **3,267 nonzero expert cells (9,815 bytes of 24,576 counts)** — the
conversation's hot set, recorded from real dispatch. Forcing a DISK spill for the restore A/B
failed on ms01, not on S5d: every park that lands there is a shared-head conversation with a
mounted COW anchor, so it is not evictable (`evicted 0, 1 still parked`), and over-budget
replacement drops the old park in `make_room` without a spill write — budgets 1536 (admission
refuses: `snapshot 2201 MiB exceeds available budget`), 4096 and 2500 produced no `.ses`. The
spill path itself is the gate's, exercised at 62 checks. So the serve-level prewarm staging
has never run under traffic yet: it rides the next store that carries histograms (prebuilt-3),
where the counters will show it. ms01 stays the recording node; the park-scheduling shape
(evictable unique parks via save-reclaim) is a measurement-harness problem, not an engine one.

Files: `conversation_cache.hpp` (`SavedConversation.s5d_hist`),
`conversation_park_disk.{hpp,cpp}` (`hist_encode`/`hist_decode`, manifest section write +
tolerant read), `expert_source.{hpp,cpp}` (dispatch recording), `generate.cpp`
(`BSlot.exp_hist`, window pointing, park encode, prewarm, kill switches),
`park_disk_boot_test.cpp` (s5d arm). Ship md5 (local == ms01): conversation_cache.hpp
`6ece8211…`, conversation_park_disk.hpp `8d70e718…`, expert_source.hpp `b7358482…`,
conversation_park_disk.cpp `e9970078…`, expert_source.cpp `b4c29d25…`,
park_disk_boot_test.cpp `a5aef446…`, generate.cpp `19c2428e…`.


## 196. S4-v2 idle-detach: an idle conversation leaves its slot — park spill and the S5d histogram chain proven end-to-end on live traffic

The S4 design note reframed the ceiling: what caps concurrent sessions is slot
ownership, not bytes. This lands the attach/detach seam at v1 and retires the
design's open questions empirically.

Seam: `BSlot` gains `t_idle`, stamped at every busy→free transition (both read-admission
paths stamp too). `detach_slot(b)` = guarded park (`park_slot` now returns whether the
conversation is provably safe in the parked tier) + checkpoints returned to `ckpt_pool`
+ slot reset. The slot's resident KV window (`bslot_ss`, boot-sized; the verify graphs
capture its pointers) stays owned by the slot — **v1 detaches conversations, not
bytes**. A sweep runs at the three busy→free batch completion sites and never on an
arrival path (TTFT law). One knob: `STRATA_SLOT_DETACH_MS`, default 0 = byte-identical
control. This is the single ownership point where S2b's shared-head refcount will sit.

Measured on ms01 (RTX 3080 Ti, `--batch 4 --kv-resident 8192`, four concurrent waves
of unique-head ~36k-token conversations; concurrency is a prerequisite for this whole
test surface — with sequential traffic the run logged no batch-completion lines at
all, so slot machinery never ran):

- Detach fires under traffic: `slot 1 detached idle (46394 tokens to the parked tier)`
  — 5 detaches in the ON arm; correctness `restore_qc_fail=0` through
  detach+spill+restore in every arm (prime-list recall checks).
- **Disk spill end-to-end**: 5 `.ses` spilled (`park disk: spilled 45963 tokens, 0.6
  MiB in 3051 ms (3 entries, 3.4 GiB on disk)`), and every spilled manifest carries an
  S5d histogram (`hdr88: off=862784 len=17222` — ≈5.7k hot cells for a 46k-token
  conversation). §195's record→manifest→spill→read chain now has no unproven link.
- Restore economics: 46k-token twins restore in **19–21 s** (parked tier, RAM + spill
  mix) vs **55–105 s** for the same-size cold prefill.
- A/B (off vs 15000 ms): wall-neutral at this shape — W2 wave average 87.0 s off vs
  90.3 s on; the sweep parks during sibling-stream decode, and prefill dominates a
  4-slot 12 GB arena. v1's win is on the admission path (an arrival no longer pays
  park-before-overwrite: ~1–3 s per arrival at 46k, more at 200k) and the ownership
  layer — not wall time at this shape.
- Guards hold by design: an agent-root conversation whose parked copy holds checkpoints
  the slot does not is never detached (#4420), and a slot without per-slot draft K/V
  never parks without `--batch-mtp`.

S5d prewarm, same traffic: the hook runs on every disk restore —
`park prewarm: 0 experts staged, 18–23 refused (no slot free)` — and stages nothing,
because the adaptive tier holds 100% of expert-cache slots on 12 GB nodes even with a
single idle stream. The non-destructive law (claim only free slots) makes prewarm
inert where the cache is saturated; it can pay only where headroom exists (fresh boot
before the tier fills, post-quarantine stores, `--expert-cache` below the working
set). Preempting the coldest non-pinned slot instead would stage reliably but steals
from whoever holds the slot — a policy decision, deferred.

Gates at ship bytes (`generate.cpp` md5 `f9edc0ddf4f6…`, local↔ms01 identical): boot
test 62 checks; targeted ctest park/session/conversation/file/buffer 20/20 — note the
four `conversation_snapshot*` tests need an idle GPU (they abort with `CUDA: out of
memory` while a serve holds the card; same constraint as the segmented-cache test).
Files: `engine/src/program/generate.cpp`.


## 197. S5d-v2 prewarm through the adaptive tier's own picker — wired, proven, and bounded by the same wall as 190

195 shipped the histogram; the ms01 churn (196) showed the free-slot prewarm claim
stages nothing on a saturated cache: `park prewarm: 0 experts staged, 18-23 refused
(no slot free)`. The fix is not a second eviction path. The adaptive tier already
picks its swaps from decayed per-expert `usage` counts (candidates at usage>=2.0,
coldest victims, `adapt_min_gain`, `adapt_swaps` budget, pinned/peer/helper
protected). Prewarm v2 seeds the park's hot set straight into those counts: at a
disk restore, up to 512 hot indices get `usage = max(current, 64 + count)`, and the
tier's own measured policy swaps them in on its next round. The tier's protection of
pinned experts is inherited, not re-implemented. Kill switch:
`STRATA_PARK_PREWARM_BOOST=0` (the free-slot claim is untouched).

Measured on ms01 (same churn as 196):

- The hook acts: `park prewarm: 0 experts staged, 22 refused (no slot free),
  25-380 usage-boosted` per restore; the tier swaps through them (1,020 -> 1,347
  experts swapped in across the run, 85-216 ms/round).
- The payoff is noise-level here, and the reason is now on the record: decode enters
  the resumed conversation at **97.0-97.6% cache hit**; only 1.8% of routed experts
  cross PCIe at all, so the miss cost prewarm can remove is bounded under a
  quarter-second per window on this box — while the restore's 19-21 s wall sits in
  restore setup and window queue behind sibling streams, not in expert residency.
  `park disk: spilled ... (5 entries, 4.5 GiB on disk)` writes only ~0.6-0.8 MiB per
  46k-token park because COW views hold the KV bytes.
- No regression: restore walls 18.9-20.9 s vs 18.8-21.2 s before, `restore_qc_fail=0`.

So S5d's chain (record -> manifest -> spill -> read -> tier hint) is complete and
inert-by-arithmetic on 12 GB nodes exactly where 190's "only lever = fewer bytes"
already said the expert path is thin; it pays where a cache has headroom and misses
actually dominate — which on this program's fleet means the 24 GB production box.
Gates at ship bytes (`generate.cpp` md5 `699dca53…`, local↔ms01 identical): boot 62
checks, targeted ctest 20/20 (idle GPU). Files: `engine/src/program/generate.cpp`.


## 198. S2b re-test: the shared-head stack is farm-negative again at batch 4 under current constraints — the VMM sketch dies

The S4 design note defined S2b as a re-test, not new development: the cow-m1 K/V
sharing stack (arena + admission fork + joins-at-fetch + lineage shared region, shipped
opt-in since 2026-11-12) was verdicted "farm-negative as pinned sharing" at batch 0.
The re-test runs it against today's shape: current build (S4-v2 + S5d-v2 ship bytes),
`--batch 4 --batch-mtp`, detach off both arms, identical config, C4 harness with
battery + needle, arm A off / arm B on (`STRATA_COW_ARENA=1` 30720 slots,
`STRATA_COW_SHARED=32768`, `STRATA_COW_FORK=1`).

| band | A (stack off) | B (stack on) |
|---|---|---|
| wall | 245.5 s | 781.5 s |
| agg decode | 13.4 tok/s | 5.7 tok/s |
| ttft p50 / p95 / max | 4.2 / 90.1 / 112.6 s | 14.4 / 506.5 / 568.3 s |
| window chunk / n | 8.35 s / 7 | 8.46 s / 10 |
| needle | 4/4 | 4/4 |

The deciding lines are the stack's own telemetry: `the admission fork points at 0 of
128 prefix pages per K/V layer`, `joins-at-fetch served so far: 0, lineage shared
fetches: 0` — under this traffic the stack shared nothing, so arm B paid only its
boot cost (12 QSA layers x 63,488 slots = 3.00 GiB INT8 streamed) and machinery
overhead, and cold TTFT went 6-10x worse. Correctness held (needle 4/4 both arms);
benefit was nil; the 2026-11-12 verdict stands under the new constraints.

Per the design note's rule this retires the VMM attach/detach sketch: shared-head is
a capacity question, and capacity moves at 196's detach seam — not at a sharing layer
that measures zero dedup on real conversation traffic. No engine changes; docs-only
entry. Probe: ms01 `s2b_probe2.sh`, bands `c4_s2bA.log` / `c4_s2bB.log`. Note for
future batch-8 runs on 12 GB boxes: arm A's guard logged `min_memavail_gib 1.9` with
a 6,144 MiB park budget at `--kv-resident 8192` — RAM, not VRAM, is where batch 8
will bind first.

## 199. S7-3: the pooled indexer-key table can be stored FP16 — +175 resident experts at batch 2, quality held

The ladder's last capacity lever: the per-conversation indexer-key pool is
`pooled_rows x idx_dim x 4 B x 12 QSA layers` — 201.5 MiB per 131k slot, the largest
per-slot VRAM block after the KV pools. `STRATA_IDX_POOL_F16=1` halves it; fmt=0 stays
the default and is byte-identical to every earlier build.

Numerics first, per the design note: upstream's dormant probe (`STRATA_IDX_FP16_CHECK`,
rounds the pool values and re-selects with the unmodified FP32 kernels — exactly what an
FP16 store reads back) ran on ms01 against the canonical config: `FP16 indexer keys:
82048 of 89544 selections identical; where the selection is sparse, 99.9864% of cells
shared (64932 queries)`; the full quality battery passed 7/7 with rounding on. The ~8%
differing selections are block-boundary score flips on sparse selections; outputs held.

Implementation: one value, read once in `qsa_real_shapes()`, so allocation, reservation
math, boot memset, QC-hash width, the snapshot/park format flag **+32** (a pool taken in
one format never restores into the other) and every kernel site agree by construction.
`pool_put` / `pool_get` / `pool4` are the only sanctioned accessors: round once at store,
expand exactly on read, scorer accumulation order untouched. HIP/Metal builds are forced
to the FP32 pool. The checkpoint spare-row rewind rounds host-side through a new
`qsa_pool_round_copy`.

The live pool writer is NOT qsa.cu: `layer_verify_compatible()` hard-requires the native
QSA indexer, and every boot's verify graph appends through
`native_qsa_indexer_append_steps` (verify.cpp, three sites). The first A/B round died
exactly there — refusing fmt=1 inside the native indexer's geometry THROW validators made
every request fail with `verify layer 3: native QSA indexer requires fixed geometry...`.
The fix converts the native indexer instead: `append_body`/`append`/`append_layers`/
`append_first`/`append_blocks` templated on the pooled element type, `pool_put` the only
write, host wrappers dispatch on the flag, span checks in real bytes. Law recorded:
geometry guards are validators, not eligibility filters — a feature refusal belongs where
the caller picks the path.

Verdict on ms01 (ctx131k, `--batch 2`, canonical config, serve bytes = this commit):

| gate / band | fmt=0 | fmt=1 |
|---|---|---|
| boot test / ctest subset | 62/62 + 20/20 | 62/62 + 20/20 |
| boot sizing input | 5.09 GiB free | 5.37 GiB free |
| expert cache | 2548 slots / 4.16 GiB | 2723 slots / 4.44 GiB |
| quality battery | 7/7 | 7/7 |
| C4 harness needle | 4/4 | 4/4 |
| C4 wall / agg decode | 502.1 s / 6.7 tok/s | 575.2 s / 7.1 tok/s |
| ttft p50 / p95 | 51.6 / 107.5 s | 50.0 / 113.9 s |
| guard pauses / min MemAvailable | 0 / 7.6 GiB | 0 / 8.0 GiB |

+175 expert slots (+6.9%) land for free at batch 2; the saving doubles with every added
conversation slot (~101 MiB/slot at 131k), which is where batch 4 and the 24 GB prod
card collect it. The C4 walls are not directly comparable: the probe-blessed ~8%
selection flips change completion lengths (3,348 vs 4,103 completion tokens) and which
park waves run, and a second fmt=0 run on the same bytes wall-ed 579.5 s against 502.1 s.
Per-token decode was slightly better on fmt=1 (7.1 vs 6.7 tok/s aggregate); no quality
or guard regressions. Watch-item: one `max_stall_gap` of 134.3 s appeared in the fmt=1
harness (6.9 s in fmt=0) — single occurrence, park/restore-interaction class, not yet
characterized.

Probe: ms01 `s73_ab.sh` round 5, bands `c4_s73a.log` / `c4_s73b.log`, battery tags
`s73a`/`s73b`, boot lines in `strata-m25-iq3xxs-lan-cow.log` under the `S73AB 16:`
window. A driver law learned the same day: a `flock` guard's fd leaks into every child —
the round-1 daemonized serve held the A/B lock for hours as an invisible holder behind
serve's inherited fd; child launches must close it (`200>&-`).

Addendum (same day): both watch-items resolved from the round-5 logs and a targeted re-run.
The 134 s stall is the slot-give-way preemption class at 102k prompts, not a park/restore
interaction: one request decoded at ~2.7 tok/s while 69k-94k-token prompt re-reads kept
overwriting the other slot (`skip parking ... 8192 MiB floor` refusals and `slot not
parked ... without --batch-mtp` are its log signature). The class is symmetric - the fmt=0
arm's window holds the same shape at 119 s and 143 s with expert hits down to 46%; fmt=1's
longer completions only raised its exposure. The `qsa_prompt_attn_batch (v3)` illegal-access
watch-item retires UNREPRODUCED: the probe-env recipe (probe + battery + C4, card near
full) re-ran on this commit's bytes - battery clean, the probe verdict identical to the
byte-for-byte quoted numbers (82048 of 89544, 99.9864% cells shared), zero illegal-access
lines, and the engine stayed healthy through the full 40-minute probe-env C4 (the harness
timed out: the probe's per-prompt rounding costs real time; the engine itself was intact).


## 200. S4-v2 follow-up KV-GROW: a detached slot hands its whole session arena back, and the re-carved slot at admission holds a bigger resident window

`STRATA_SLOT_KV_GROW=1` (needs `STRATA_SLOT_DETACH_MS>0`, one stage, no
`--batch-mtp`/`STRATA_PIPED_MTP`/`STRATA_ROPE_TABLE=1`): the §196 idle detach now
destroys the batch-window graphs that baked the slot's pools
(`Verifier::drop_batch_graphs`, new: destroy-then-free order, device quiet at both
call sites), `cudaFree`s the slot's session arena, and the next admission re-carves
it — doubled resident window first (kv-resident 20480 -> 40960), falling back to the
boot size when the budget check (one ordinary session reserved for every other freed
slot plus a 64 MiB band) does not pass. With the env unset the path is inert: off by
default, byte-identical.

Mechanism proven end-to-end on the node (ms01, `parallel: 4`, detach@15000 ms,
measurement config): `slot 1 detached idle (31856 tokens to the parked tier)` ->
`session arena returned (0.58 GiB back)` -> on re-admission
`slot 1's session re-carved (kv-resident 40960, 0.83 GiB)`. At the shipped
`--vram-reserve-mib 512` the budget check declines the doubled candidate and the
fallback carve takes boot size - the safety band working as designed. The grown
carve was funded by raising the reserve to 1536 MiB: expert cache drops to 914
slots and S-wave prefill walls go 47-49 s -> 89-109 s. That is the honest economy
on a 12 GB card: the grown window competes with the expert cache and must be paid
for; the bytes for the grown carve are 0.83 GiB vs 0.58 boot.

Payoff at bench shape is parity, not speed-up: 800-token decode on a 32k
conversation measured 5.1 s on the grown 40960 slot vs 4.8 s on the boot-window
slot (11.6k streamed tokens cost ~6 ms per decode token at `--pcie-frac 0.5` -
nothing to win at this size). The stall class this aims (0.56 s/token at 102k,
§199 addendum) needs a window that covers the prompt, not one that merely doubles;
what KV-GROW delivers today is optionality with measured cost: idle slots return
their memory, and the next carve sizes to the budget the moment it exists.

No-regression evidence (all on these bytes): the two-arm A/B came green both ways
(battery, C4 harness, 8x96k churn - churn walls 46.5-46.7 s both arms,
restore_qc_fail=0), boot park-disk test 62/62, ctest park/session/conversation/
file/buffer subset rc=0. The rebuild from these sources reproduced the build5
binary byte-for-byte (579bdd02...), so build6/build7's §199 control comparison
(`b6ec8844...`) applies to the source pair directly.

Watch-item, blame-isolated: an intermittent engine-death class (`the engine
reported an error:` with an empty message, engine down, clients get 503) hits BOTH
these bytes and the §199 control bytes under the measurement config - it is NOT
introduced here. Same family as the `qsa_prompt_attn_batch (v3)` watch-item retired
unreproduced in the §199 addendum, now with a repro family: two mid-size bench
runs in one boot where the second restores parked-tier conversations (crashes at
the next admission ~every time), once even on a cold first run. Suspected fuel:
stale cross-boot park-disk session files (a 2.67 GiB park restored at every boot);
carried forward with that offload as the test lever.

Null-test laws learned the expensive way (all three walls were benches, not code):
slot machinery never sees serial solo arrivals (main-session path, no caching);
the detach sweep only wins when donors sit in slots the traffic never touches
(`parallel >= 4` geometry); and the parked tier's own budget (`--conversation-
cache-mib`) plus the RAM admission floor (`--conversation-cache-min-free-mib`,
8192 on this box is a B=8-era leftover that refuses every detach park at 2.5 GiB
per park) both gate parking before the sweep ever runs.


## 201. Fan-out measurement: shared-head concurrency on shipped bytes - the sharing already works, and where the ceiling actually is

Shape (ms01, 3080 Ti, shipped canonical config: `--batch 4 --kv-resident 20480`,
131k context, conversation cache 16 GiB / min-free 2 GiB, tail skip off): one
115,687-token shared head ("runbook"), then C worker conversations that start
with the head and add ~3k unique tokens. Forced 2,300-token decodes (repetition
task) for the long-occupancy arms; 200-token answers for the admission arms.
Five arms: fana C=4 short decode, fanb C=8 short decode, fand/fanf C=4 forced
decode (differing only in whether the bare head also sits in the conversation
cache), fane C=2 warm steady-state, fang C=2 cold solo-then-solo. No engine
code involved: these bytes are §200 build8, `STRATA_SLOT_DETACH_MS`/`GROW`
unset (inert), engine-error 0 and cancelled 0 across the clean arms.

Admission sharing already works without any new layer. Each worker's first read
line: `108288 of 115687 tokens` reused for the head at ~1,000-1,500 tok/s
(~5-7 s), and the worker parks carry 97% COW-anchored bytes
(`reused_kv_bytes` against 5.0 GiB park bodies). At C=8 on 4 slots with short
decodes: all 8 workers land in two waves, walls 15.6-65.7 s, `no room beside`
0, `skip parking` 0, `evicted` 0. The vLLM/SGLang-style prefix effect the
milestone list was chasing is already delivered by the fork's COW prefix reuse
at this shape - no forkb4-style block machinery is needed.

Capacity law (measured, fand vs fanf): with long decodes, cached conversations
must fit the slots. fand cached the bare head as its own conversation: 5
conversations vs 4 slots, every admission gave way to another worker's
conversation, walls 86-87 s behind retry loops (`0 generated ... (cancelled)`
+ `1 generated in ~16 ms` summaries). fanf is the same traffic with the head
living only inside worker 0's conversation: 4 = 4 slots, walls 74.6-78.4 s,
all length-capped, `cancel=False` on every done line. The §199 watch-item's
give-way preemption signature was caught live here for the first time, and it
is geometry overshoot, not a bug. Client/farm rule: never keep the bare head
cached once workers hold it as prefix.

Decode ceiling: aggregate is flat at ~100-120 tok/s from two streams up, and
the tax is expert-cache dilution, not KV streaming. One stream at the same
115k context: `done: 2300 tokens in 75 s (142.0 tok/s), expert cache 96.1%
hit`; a second solo stream `160.5 tok/s, 96.6% hit`. Three streams together:
33.4/37.1/41.8 tok/s, hit 63-76% (~20% of routed experts streaming over PCIe
per step vs 2.8% solo). Two streams warm measured ~50 tok/s each. The shared
head does not mean shared decode experts: prefill experts are shared, but
each stream's generated tokens route its decode steps to different experts,
so the union of working sets exceeds the arena and the LRU thrashes
super-linearly. On a 12 GB card this is a hard ceiling that added VRAM
headroom does not lift by capacity alone - it is why the expert cache sizing
and per-slot windows compete, and why the same measurement on a 24 GB card
(arena 6,662 slots per the §S2 prod rule) is the open question, not the
local one.

Bench arithmetic kept honest: qc flags read the final marker line, which the
2,300-token cap truncates on the repetition arms (repetition of 320 lines is
~1,920+ tokens before the marker) - the false `qc=False` on fand/fanf/fane/
fang rows is the harness, not generation quality; all four arms hit the token
cap cleanly.


## 202. Naming the silent batch-window death: the crash-class forensics protocol, what it caught, and what it refused to show

Class. The ms01 cumulative serve log carries 14 deaths of one signature: the
supervisor reports `the engine reported an error:` with an EMPTY message, the
engine exits 1, and clients get 503/400 mid-request. Ten of the eleven scenes
end on the same last engine line, `park COW anchor mounted for N of M tokens`,
with N/M ranging 8/459 to 31604/31614 to 9/120595 - size-independent. The class
is byte-agnostic: §199 control bytes die it on the same geometry (blame
isolation by hash-matched rebuild), and emptying the park-disk files before the
boot does not prevent it, so stale cross-boot park state is not the fuel. The
deaths cluster while the host's MemAvailable sits near or below the 2 GiB RAM
admission floor with the parked tier fully loaded.

The quoted last line is an artifact. The engine's own empty stderr line
(`strata serve: ` + newline, printed by the fail-fast just before exiting)
shifts the supervisor's last-log-line quote past the park line. The park-COW
mount is where the scene stands, not where the crime happened.

The channel. The death path is the prompt-read fail-fast (#224 gate): after a
batch window fails with `batch_fatal` set, `err` is empty: it prints the empty
ERR, `cudaPeekAtLastError()` is `cudaSuccess` (the #224b named-sticky line has
never appeared in any death scene), and the engine returns 1.

What the instruments showed. Eleven cold-path labels at the suspected silent
sites (`Verifier::run_slot_rows`: pre_launch, stage_batch, chain end,
sample_rows; plus the window/commit/prompt-read ERR sites): the class died with
them live and NONE fired. An at-exit probe (execinfo backtrace + sticky naming,
registered at namespace scope so any `exit()` prints the stack) went live across
four boots and twelve bench pairs: it never printed - those boots never died.
Tally: control bytes died on 5 boots, clean on 1; the labels-only build died
once (labels silent); every boot carrying the at-exit probe - labels or not -
survived the repro geometry (0 deaths in 4 boots). One control boot with
`STRATA_PF_STEP_SYNC=1` reproduced twice back to back; re-runs of that shape
then stayed clean. The class flees observation: state-dependent race on the
park/restore path, not deterministic logic. (`CUDA_LAUNCH_BLOCKING=1` is
self-poisoning on this engine - issue #1341: the verify windows cannot run with
blocking launches, and the 60 s no-progress watchdog SIGABRTs the boot; the
supported narrowing switch is `STRATA_PF_STEP_SYNC=1`.)

The change (shipped; the root cause stays open). Three failure-path-only
backfills convert every silent window death into a named one: the #224
prompt-read fail-fast backfills `err` with `a batch window died without a
reason (no CUDA fault sticky)` before printing, and the two batch-window ERR
sites print `a batch window failed without a reason` when `err` is empty. No
hot-path cost: each edit sits on a path that was already printing ERR and
unwinding. The #224 sticky `_Exit` branch is untouched, so a true CUDA-context
fault still names itself and leaves at once. The next occurrence in the field
logs a named line instead of an empty one, and the supervisor restarts on a
named event. These bytes ran the geometry that killed the pre-§202 build with
zero deaths on ms01 - stated without softening: the fix is verified by
construction and by green boots, not by re-triggering a named death locally.

Next-occurrence protocol. If a named line appears anywhere: the two site
strings separate the channels (prompt-read fail-fast vs batch-window loop), and
the scene (park state, admission refusals, MemAvailable) discriminates among the
surviving hypotheses. Labels never fire, so per-site instrumentation is retired
for this class: any future arm starts from the at-exit probe.

## 203. The prod arm: window recycling runs end-to-end on the 24 GB box; the decode ceiling did not move

prebuilt-3: built on the prod host from 884f351b (§194-§202) with
packaging/build_cuda_windows.py, CUDA 13.5, archs 86;89;120, Ninja,
`hydragen.exe` 138,326,016 bytes, md5 615a359136ad9b7bf5b5582703756864,
`--version` strata 0.1.60. The patch is proven in the binary by its literals
(`died without a reason` and `session re-carved` present; `DIAG at-exit`
absent - the §202 probes stay out of shipped builds).

A packaging bug surfaced and is fixed in this change: the staged check demanded
`strata.exe`, which this fork's engine target does not produce (the fork builds
`hydragen.exe`, no `strata_kq.dll`), so a green `[158/158]` link still aborted
with `the build made no strata.exe`. Staging now accepts whichever engine exe
the build made. prebuilt-3 itself was staged by hand as usual:
`runtime/hydragen-pb3.exe` beside the live exe, and `hydragen-local.json`
diffed by exactly three lines - the `exe` path, `STRATA_SLOT_DETACH_MS=15000`,
`STRATA_SLOT_KV_GROW=1` - with engine args byte-identical. Rollback is
`hydragen-local.json.pre-pb3.bak` plus one service restart (host's hand).

Gate first, measurements second: needle battery 4/4 after the restart
(depths 18,995 -> 186,994 tokens), same as the prebuilt-2 baseline taken
minutes before. Zero deaths across the whole arm; the engine pid never changed
and §202's backfill never had to print.

The §200 cycle fired in production, verbatim:

    conversation cache: slot 3 detached idle (117923 tokens to the parked tier)
    conversation cache: slot 3 detached idle: session arena returned (0.55 GiB back)
    conversation cache: slot 3's session re-carved (kv-resident 16384, 0.65 GiB)

The re-carve took the doubled window (boot carve 8192 -> 16384) with no
fallback to the boot size: 24 GB funds it. The re-admission is what window
recycling is for:

    prompt 115705 tokens = 115698 reused + 7 read in 874 ms

seven tokens re-read of 115,705. The same arm caught the give-way retry once
more - `0 generated ... (cancelled)` after `16384 of 115705 read in 7408 ms`,
then the successful pass above - geometry, not defect (§199).

Fan-out at the §201 shape (a ~115.7k-token head across 4 workers, 2,300-token
forced decode) on the 24 GB card:

    pass 1 cold: walls 139.7/158.9/160.8/156.7 s, aggregate 59.9 tok/s
    pass 2 warm: walls 42.3/113.7/139.9/137.9 s, aggregate 107.8 tok/s
    C=2:         cold 175.1 s then warm 66.2 s,  aggregate 47.9 tok/s

Per-stream decode at C=4 sits at 14.3-20.2 tok/s with decode-window expert
hits at 78.8-87.0% against 92.4-99.6% idle - the same dilution signature §201
measured on 12 GB, where the comparable fanf walls were 74.6-78.4 s
(~30 tok/s a stream). Doubling the VRAM did not lift the aggregate band
(~100-120 tok/s flat): the ceiling is the per-stream expert working set, not
free bytes on the card. What the 24 GB box bought is fundability of the grown
carve and admission economics - 0.55 GiB returned by each idle slot, and a
115k conversation restored in 0.9 s against a 65.0 s cold re-prefill
(`16384 reused + 99321 read in 64973 ms`, 1,528.7 tok/s) - not decode speed.

Not claimed: the 102k `0.56 s/token` class (§199 addendum) remains unmeasured;
the fan-out shape tops at 115.7k prompts. The disk park tier refused its index
at boot - `park v2: bound to another model or configuration` for 16 entries,
quarantined - which is the fingerprint law doing its expected job after an
engine change; the RAM park tier (6144 MiB) carried the arm (25 parks,
5 restores, 30 of 43 requests reused). Re-arming the disk tier against the new
fingerprint is recorded, not taken.

prod stays on prebuilt-3 with `STRATA_SLOT_DETACH_MS=15000` and
`STRATA_SLOT_KV_GROW=1`.

## 204. The S8 arm: the CoW stack fires on a shared head for the first time, and a warm fan-out decode escapes the dilution law

The fan-out bench (§201, `m25_fanout3.py`: one ~90k shared head, C=4 workers
with unique ~3k tails) ran for the first time with the cow-m1 stack armed —
`STRATA_COW_ARENA=1`, `COW_ARENA_SLOTS`/`COW_SLOT_CELLS`, `COW_FORK=1`, and
the one knob that had never been switched on before, `COW_JOIN_HH=8`
(join-at-fetch OFF by default, which is why every §198/W1r log read
`joins: 0`). ms01, engine `f60d86e67d54fc142440dabbb894c273` (§202 bytes;
the stack ships inside them), config copy `...-s180-s8.json`, restart via
`s8_restart.sh`. Staging caught that the live process's
`--kv-resident/--batch` came from a boot-era config that no longer listed
them; the arm states them explicitly.

Arm 2, the W1r-proven small shape (`--kv-resident 4096`, arena
9216 x 12, SLOT_CELLS 4096):

    cow: K/V arenas: 12 QSA layers x 9216 slots (0 shared region) = 0.44 GiB (INT8 streamed)
    strata generate: expert cache 2355 slots, 3.77 GiB of VRAM

0.44 GiB of arena replaces the private 20480-cell windows of the fanf basis
boot: the expert cache grows from **1375 to 2355 slots** before a request
runs. Fan pass 1: fork/join telemetry non-zero for the first time in the
stack's existence:

    cow: the admission fork points at 1023 of 28927 prefix pages per K/V layer (upto 115705 cells)
    cow: joins-at-fetch served so far: 0 -> 9 -> 24
    strata serve: conversation cache: borrowed 28672 tokens from a parked conversation,
      which stays parked, in 162.3 ms; parked=7 ... borrowed=2

(1023 pages x 4 = 4092 cells = the whole 4096-cell window: the fork covers
the worker's entire resident window; the head's rest streams.) Walls: W0
solo cold 63.0 s, workers 79.9-81.1 s — level with the fanf basis
(74.6-78.4 s; the forced 2,300-token decode dominates). Needle battery
**3/3 at every depth ms01's 131,072 context admits** (18,995/46,993/96,994;
the fourth depth returns the context-cap 400, as it must), before and after
the fan passes.

Arm 3, the prod mirror (`--kv-resident 8192`, SLOT_CELLS 8192, arena 18432,
plus `STRATA_SLOT_DETACH_MS=15000` + `STRATA_SLOT_KV_GROW=1`): arena
`12 x 18432 = 0.87 GiB`, expert cache **2070 slots**; the detach/grow
posture coexists — no geometry refusal, no `could not be re-carved`, arena
sessions produced no re-carve lines at all. Forks track the window: the
fresh head conversation forks **2047 pages** (= the 8192 window). Cold pass
68.0 / 75.9-77.0 s (basis-familiar); joins 0 -> 80 across the two passes.

The payoff line, arm 3's warm pass, a worker completing the full 2,300-token
forced decode:

    [strata] done: 2300 tokens in 21 s (139.6 tok/s) ..., expert cache 97.2% hit (+1.8% of the routed experts over PCIe)
    [strata] done: 2300 tokens in 24 s (101.1 tok/s) ..., expert cache 97.6% hit (+1.6% ...)

Fan-window decode hit rates: cold 74.8/80.1 %, warm **97.2/97.6 %** — the
§201 dilution signature (63-76 % under streams) does not apply to a warm
shared-head fan-out once the head's experts stop being paid for N times.
§203's law survives as written for COLD traffic (the cold pass walls are
basis-level) and breaks for WARM fan-out exactly where S8 aimed.

Not claimed: prod has not run the cow arm (§203's prod posture stays); the
ms01 head parks measured 4.37 GB, over prod's 6144 MiB park budget, so the
prod arm needs the budget raised before borrow can fund the head; join
totals are modest (80 across two passes) and `lineage shared fetches: 0`
stays zero — the cow-m1e shared region is not armed; arm-3 fan-4's
7.9-21.3 s walls for qc=True workers are the warm shortcut (direct answer,
not the 2,300-token repetition path) and are excluded from wall evidence;
needle depth 4 on ms01 is unreachable at 131k, its 4/4 baseline remains
prod's (section 203).

Next: the prod arm — cow env + park-budget raise onto `hydragen-local.json`
(128 GiB host RAM funds it), restart on the host's hand.

## 205. The prod cow arm: the mechanism runs on the 24 GB box, the shape regresses, and prod rolls back

The arm was staged exactly as §204 proposes — `STRATA_COW_ARENA=1`,
`ARENA_SLOTS=18432`, `SLOT_CELLS=8192`, `FORK=1`, `JOIN_HH=8`, and
`conversation-cache-mib 6144 -> 16384` on `hydragen-local.json`; binary
unchanged (`hydragen-pb3.exe`); host restart. Boot carried every expected
signal: `cow: K/V arenas: 12 QSA layers x 18432 slots (0 shared region) =
0.87 GiB (INT8 streamed)`, the park-era quarantine lines (`index refuse ...
bound to another model or configuration`, files kept), needle battery
**4/4 at full depth** (the 186,994-token prompt answered). Mechanism live:
`the admission fork points at 1424 of 28927 prefix pages` on the shared
fan head, `85/515 of 36811` on a 147k-token conversation, `joins-at-fetch
served so far: 131`.

It is still the wrong shape for this node, on two grounds:

1. Capacity sign flips. On 12 GB the arena replaced private windows and
   grew the expert cache (§204: 1375 -> 2355); on the 24 GB box it is
   simply an extra pool: expert cache **6453 -> 6235 slots** (-218). And
   `ARENA_SLOTS=18432` under-funds the live demand of batch 4 x
   `SLOT_CELLS=8192` = 32768 — the 2026-11-19 sizing law ("arena slots
   must cover all session windows") applies at 24 GB too; copying the
   ms01 arm-3 numbers was my error, disclosed here.
2. The cow window bypasses §200's window-growth answer. Every cow boot
   printed **zero** `re-carved (kv-resident 16384 ...)` lines; the §203
   boot printed them for exactly the big conversations that now ride the
   arena. The `OVERFLOW (too few resident cells)` KV-streaming flag is
   NOT cow-caused (77 such lines sit inside the pre-cow §203 boot — it is
   the `--kv-resident 8192` at 200k-context era), but re-carve was how
   prod had been paying for 115k-token conversations, and the arm
   silently removed that mitigation.

Measured cost, cold fan pass (same §201/§204 shape): walls
161.8/189.6/199.7/207.9 s, **aggregate 48.9 tok/s** vs §203's cold basis
139.7-160.8 s / 59.9 — a -18 % cold penalty before the warm pass even
started. The warm pass never got data: mid-window the engine died
silently — last line `park COW anchor mounted for 112177 of 115705
tokens`, no error text, no `session is up` until the supervisor's boot
marker; the §202 scene naming did not print either (the exit path skips
the handler — the unnamed crashw class, §202's open root cause; cow
blame is NOT established, but the death sits inside the arm's fan
window).

Rollback staged: config restored to the pre-arm file (md5-identical to
`hydragen-local.json.pre-cow.bak`; the arm config is kept as
`hydragen-local.json.cowarm-attempt.json`); restart on the host's hand.
The 16384 MiB park budget survives in the kept file only. ms01 stays on
arm 3 — harmless on the bench node and the §204 warm-fan payoff stays
real at the 12 GB shape.

S8-v2 gate before any prod re-arm, both engine-side: (1) arena sizing
enforcement (`ARENA_SLOTS >= batch x SLOT_CELLS` + headroom), and
(2) arena-backed sessions must honor KV_GROW re-carve, or the arena must
size its window the way §200 does. Prod posture in the meantime: the
§203 shape, unchanged.

## 206. S8-v2: the arena allocator reclaims in any order, the sizing law is enforced at boot, and the detach/re-carve loop is measured green on the bench node

*Amendment to §205 first:* the "under-funding arithmetic" in that section
(`18432 slots vs 25165` etc.) conflated two units. A slot is a page of four
cells, so the boot clamp in cells is `(1 + 2 x batch) x max(SLOT_CELLS,
kv-resident)` — at the ms01 shape (batch 4, 8192 cells) demand is 18432
slots/layer, exactly the arena shipped. The §205 regression stands, but its
funding argument is retired.

What landed engine-side (the two §206 gates from §205, plus the detach
reachability fix):

- Arena allocator: the bump-pointer could only reclaim the newest range, so
  any out-of-order free fragmented the arena forever. Replaced with a
  first-fit free list (`g_cow_free`) with sorted insert and coalescing, plus
  a zombie list (`g_cow_zombie`) for ranges pinned by refcount>1 at release
  — retried on every later release. `cow_range_clean` refuses to zero a
  pinned range. `qsa_set_cow_arena()` now clears the range maps, so a fresh
  arena never inherits a stale map.
- Sizing enforced at boot: `ARENA_SLOTS >= (1 + 2 x batch) x
  max(STRATA_COW_SLOT_CELLS, kv-resident) / 4`, with a printed raise when
  the configured arena is short (the §205 prod shape printed exactly that
  clamp line; ms01's 18432 met demand exactly and printed nothing).
- KV_GROW interaction fixed: `ensure_slot_session` raises the cow window per
  candidate (`qsa_set_cow_slot_cells(cand)`) before re-init and restores the
  boot window after; a failed re-init releases the half-built session.
- Detach reachability: the sweep fires only at the three batch-window
  completion sites, and a solo request never enters the batch arm at all
  (serve sends plain `GEN`). Measured law: the traffic shape that fires the
  sweep is a **concurrent fan with one long-tail stream** — windows decode
  slots in lockstep, so an all-short fan stamps every slot fresh at the same
  instant and nothing ever ages past `SLOT_DETACH_MS`. §203's lone prod
  detach was staggered completions, which this explains without any
  different mechanism.

Measured on ms01 (bench node, engine `961dc4905995d2b1ad2411340ba3c56d`,
cow env on, `SLOT_DETACH_MS=15000`, `SLOT_KV_GROW=1`, `TAIL_SKIP=0`), fan
of four 14k-token conversations, `max_tokens` 32/32/32/2600:

    strata serve: conversation cache: slot 1 detached idle (43539 tokens to the parked tier)
    strata serve: conversation cache: slot 1 detached idle: session arena returned (0.34 GiB back)
    strata serve: conversation cache: slot 2 detached idle (43540 tokens to the parked tier)
    strata serve: conversation cache: slot 2 detached idle: session arena returned (0.34 GiB back)
    strata serve: conversation cache: slot 1's session re-carved (kv-resident 16384, 0.34 GiB)
    strata serve: conversation cache: slot 2's session re-carved (kv-resident 16384, 0.34 GiB)

First detach + arena handback + re-carve observed anywhere (every §203/§204/§205
boot printed `detached 0`). The re-carve carries the raised resident window
(16384), which is the §200 mitigation surviving inside the arena — the exact
capability §205 found missing on prod. First-fit reclaim served the next
admissions with zero `no room` / `arenas hold` refusals across the whole log.

Prod posture unchanged: the §203 shape stays live; a cow re-arm needs a
Windows build of this tree first, and the sizing clamp above is what will
either fit the arena on 24 GB or say so at boot instead of bleeding expert
cache slots silently.

## 207 — Sync repair: the vision build's models/ sources had never left the §181 worktree

A clean-clone vision build fails at CMake generate: the tracked
`third_party/strata-llama/tools/mtmd/CMakeLists.txt` names 45 `models/`
sources, but `.gitignore` line 6 (`models/`, meant for local model dirs)
swallowed that upstream directory out of the §181 sync commit. They
survived untracked in the original sync worktree — so every build to date
passed while the public repo silently could not. Restored from the
`C:/DevOps/hydragen-sync180` clone (all 45 md5-verified), added a
negation rule for that exact path, and committed the sources (8,400
lines). No engine behavior change; this is what unblocks the pb4
(068a7cf6) Windows build.

## 208 - the models/ ignore swallowed a second tree: src/models/

The 207 fix found one casualty of the root .gitignore `models/` rule; the
pb4 rebuild found the second. Upstream llama keeps its per-model builders
in `third_party/strata-llama/src/models/` (155 .cpp + models.h, pulled in
by `file(GLOB)` at src/CMakeLists.txt:9), and `llama-model.cpp:23` includes
`models/models.h` directly. Both facts invisible to git: check-ignore maps
the path to the same line 6.

Symptom on pb4 attempt 2: the fresh vision configure succeeded (207
sources back), then ninja died at `llama-model.cpp(23): fatal error
C1083: Cannot open include file: 'models/models.h'` - and worse, the GLOB
had silently contributed zero model sources to every prior full-tree
configure on a clean worktree, which would have link-failed even with the
header present. Old worktrees kept building only because the files
survived untracked on disk.

Restored all 156 files from the sync clone (C:/DevOps/hydragen-sync180,
the only surviving copy on either machine - modelserver01's clone lacks
them too; its build-s180 is another stale-worktree survivor), added the
scoped negation, committed. 157 files, +41,344 lines. Vision dirs must be
fully reconfigured, not ninja-resumed: the GLOB ran empty at their
configure time.

## 209 - second benchmark node (bazzite.chung.lan) + battery numbers; park-restore crash loop

Benchmark node roster is now two 12 GB cards. bazzite.chung.lan runs the
engine in a podman quadlet (`hydragen-server`, host port 8080, RTX 3060
12288 MiB); its bench scripts were written for ms01's `:8181` and needed
only a URL rewrite. Numbers below are same-battery, per-node - not
directly comparable across builds (bazzite runs the 181-era container
build; ms01 runs 961dc490).

ms01 mech arm (961dc490, ctx 131072, B=4, detach 15000 ms, tail-skip 0):
single 7,828-token probe 3.8 s; four-way 45k fan (cold) 41.7-54.4 s per
stream; tail-shape fan 46.9-47.5 s (ttft 44.7-45.7); after the detach
fired, a re-carve probe on a parked conversation answered in 0.4 s. The
same fan warm: 1.8-5.0 s (~10x). Staggered fan 5k-20k: 5.3-8.2 s.
Concurrency=2 (128-token streams): aggregate 56.5 tok/s cold, 78.3 tok/s
warm; per-stream decode 34.2-45.1 tok/s.

bazzite (RTX 3060): quality battery 8/8; park ladder cold prime 88.1 s,
warm primes 28.2 s, restores 0.4-24.5 s, restore_qc_fail=0; fan ladder
C1-C4 aggregate 15.0 -> 38.8 tok/s (2.6x on 4 streams), per-stream
decode 42.5-48.8 with one suffix-draft winner near 97-99 tok/s per
round.

Incident on ms01 (worth a fix): the engine died once mid-battery (cause
unnamed), and every supervisor restart then crash-looped at boot - the
log repeated `the engine exited (code 1) after it reported: restoring a
conversation checkpoint failed - please report` while RAM was idle at 4/60
GiB used. Renaming the park-disk store out of the way cleared the loop
and the engine booted clean in ~10 s. A single unreadable checkpoint
entry should be skipped (or quarantined), not fatal at restore.

Supervisor gap on bazzite: the container stayed `Up` for 3 days with no
engine child and an empty `/v1/models` - nothing on the host asked for
the model, and unlike ms01's supervisor there was no lazy (re)load on
first request. `podman restart` restored service (~80 s to loaded).

## 210 - mech-arm concurrency scaling curve (ms01, C1-C8)

First proper scaling curve on the mech arm (961dc490, ctx 131072, B=4,
detach 15000 ms, tail-skip 0), farm shape identical to bazzite's §209
ladder: every stream sends the same 22.5k-token conversation (25x
runbook preamble + 900 incident-note lines), 800-token decode, QC =
prime list 211/223/887 present - all levels passed.

Cold ladder (fresh conversation per level), aggregate tok/s:
C1 28.2 (per-stream decode 70.9) | C2 49.9 (1.77x, decode 54) |
C4 70.4 (2.50x, decode 34-38) | C8 98.0 (3.48x, decode 34-39).
Warm C8 (same conversation, primed by the ladder): **142.4 tok/s**
(5.05x over cold C1, 1.45x over cold C8) with per-stream decode
unchanged at ~36-39.

- Aggregate keeps climbing to C8; per-stream decode halves by C2 then
  holds ~35 tok/s all the way to C8 - streams queue and share rather
  than starve each other.
- MemAvailable dropped 4 -> 2 GiB at C4 and stayed there; no refused
  requests, no restarts, no restore failures - park/detach absorbed it.
- draft_n_accepted=0 reported at every stream of every level; open
  question whether the mech arm reports that counter at all (the same
  streams' decode rates look draft-driven).
- Context: 24 GB prod fanouts ran 47.9 tok/s @C2 / 107.8 @C4 (§206
  era); a 12 GB mech-arm box now sits at 98 @C8 - within 10% of the
  24 GB box's C4 ceiling on 4x the stream count.


## 211 — dyn-conc P2 stage A: the CoW arenas become VMM-backed, boot to a mapped floor, and admission gates on mapped capacity

The arena campaign so far reserved every window slot at boot — the arenas
were a fixed tax on the expert cache, and §209's incident class (refusals,
crash-loops at the arena ceiling) had no elasticity to absorb it. Stage A
(ms01 branch `dynconc-arena`, commit `71f52ff2`) rebuilds the K/V arenas on
the CUDA VMM API: the whole window region is *reserved* (pointers never move,
so graphs and kernels keep them) while only a *floor* is *mapped*.

Lineage first, because this chain does not sit on §206's tree: the ms01 arm
tree re-cut to upstream 0.1.60 with the fork overlay (`ccdc8923`, 2026-11-18),
then carried the S8-v2 line over (`29f10bf4`, 2026-11-23) — slot detach and
KV_GROW gating, park-hist, checkpoint pool, rope-table guards — and added
`STRATA_COW_ARENA_CONC`, which sizes the arena for expected resident streams
instead of `--batch`: measured same-shape C1, arena 0.87 → 0.29 GiB, engine
self-clamped batch 4 → 2 at boot, resident experts 2709 → 3515 (+29.7%).
Stage A lands on top of that shape.

Mechanism:
- Boot line `cow: K/V arenas: 12 QSA layers x N slots ... reserved, all
  mapped (INT8 streamed, VMM)`; the admission gate is now mapped slots, not
  the reservation. Unmapped reservation costs no physical VRAM.
- `STRATA_COW_ARENA_FLOOR` boots a smaller mapped window (floor ≥ the first
  admission's slots). Freed chunks return to the driver and the auto-sized
  expert cache spends them: floor 18432 → 6144 on the bench node gave
  2709 → 3430 resident experts (+26%), and the engine boot-clamped `--batch`
  from 4 to 2 by refusing slot-session init at the mapped-capacity gate —
  the clamp is graceful, the server serves on the slots that fit.
- Elastic API (`qsa_cow_arena_need/grow/shrink/mapped`) quiesces like
  `--kv-grow`; shrink refuses while any range cursor or refcount names a
  chunk past the target.

Cost law (measured, closed-negative to fight): every VMM region pads to
2 MiB chunk alignment — `cuMemGetAllocationGranularity` MIN == RECOMMENDED ==
2 MiB on the 3080 Ti — so all-mapped VMM arenas cost ~160 MB vs cudaMalloc
(2709 → 2611 experts). Packing tricks cannot claw it back. The elasticity,
not the padding, is the win: cudaMalloc arenas could not grow at all.

## 212 — dyn-conc P2 stage B: windows follow demand, detach parks chunks in a bank, and a reservation becomes a claim the admission can spend

Three commits (`51c0ad3c`, `0f467b4c`, `3665a350`, all 2026-11-23) close the
elasticity loop, inert unless `STRATA_COW_ARENA_ELASTIC=1`:

- A slot whose window sits above the floor boots **unattached** (the verifier
  skips an uncarved state instead of exiting) — exactly the state an idle
  detach leaves. Admission grows the window for itself; the detach sweep
  trims to the highest range still owned and **parks the freed chunks in a
  bank** (`STRATA_COW_ARENA_BANK_MIB` caps it) instead of handing them to the
  driver, so the next grow takes them before touching VRAM. Measured cycle:
  `arena trimmed to 6144 of 18432 window slots (96 chunks parked, 0 given
  back)`, VRAM 11525 → 10861 MiB; the next wave's two grows drew the bank's
  48 chunks first; every stream served, floor never crossed.
- Trim shrinks only from the top of the arena (a stack); releasing a low
  range while an attached slot holds the top correctly reports nothing spare
  above the target.

Four laws earned the hard way, each one a measured death:
- **Deferred must be priced.** Anything sized after the expert cache must be
  added to its reserve (`qsa_cow_arena_note_deferred`); unpriced, every C4
  re-attach failed at `the slot's session could not be re-carved` and the
  stream ended at one token.
- **The growth line guards only a nonzero ask.** Bank chunks are already
  allocated; mapping them back costs no VRAM. Refusing free grows while the
  card already sat under the 512 MiB line manufactured a death signature
  (ask 0 MiB, 505 free vs the line → decline → capacity gate → 3 streams
  dead in the window).
- **The doubled re-carve window and elastic growth spend the same budget.**
  Under elastic, `STRATA_SLOT_KV_GROW` takes the single-size candidate;
  doubling pushed the map to 14336/18432 and the card to 145 MiB free.
- **A reservation is a claim, not only a subtraction.** The deferred
  session's reservation shrank the cache sizing, but nothing let the later
  admission *spend* those bytes — the same GiB was charged twice and
  reservations went idle while admissions starved. Fix (`3665a350`): the
  budget is drawn on attach success and in the grow, given back on detach,
  and only the part of an ask the reservation does *not* cover meets the
  free-VRAM line.

Stage-B A/B (bench node, ctx 131072, batch 4, cells 8192, floor 6144, seven
phases with unique tags so every phase admits a fresh conversation): eager
control kept 2774-slot cache and all streams alive but rode the card to
7 MiB free at peak; elastic kept 2679 slots (~3.5% cache), grew the map
6144 → 10240, all streams alive, zero `#2065` warnings, and detach returned
0.34 GiB per idle slot. With the cache left on auto (2849 slots) the second
deferred session cannot pay — C4 dies — which is the capacity line at this
context, not a bug.

## 213 — the floor is a free-VRAM dial: FLOOR=4096 promoted on the bench mech, and the host-RAM cliff that elastic + floor removed

`STRATA_COW_ARENA_FLOOR` is not a safety margin — it is a dial on how much
VRAM the expert cache gets at boot, because the cache is sized exactly once:
all-committed boot → 2849-slot cache; floor 6144 → 3360 slots / 5.47 GiB;
floor 4096 → 3628 slots / 5.90 GiB (boot prints the free figure and the slot
count every time — confirm the flip by reading `expert cache N slots`, never
by trusting the env edit). Step the dial downward and re-probe each step.

Matched warm-profile A/B, minutes apart (probe6 = three 2-up and three 4-up
waves): the 6144 arm reproduced its old sums (4-up waves summed 31–41);
4096 nearly doubled them (52–72; 2-up 29.5 → 52.5). Composition: +269 expert
slots, −1.6 GiB boot `RssShmem`, +1.5 GiB `MemAvailable` under load.
201-cycle soak at 4096: 601 streams, 0 dead, 0 errors, p50 18.0 tok/s,
`MemAvailable` min 8680 MiB (at 6144: p50 14.7, min 7170, four streams
under 5).

The pre-elastic baseline soak on the same harness had the cliff: 18 cycles,
45 streams, 2 errors, **three engine deaths** (exit −9, OOM killer),
`MemAvailable` min 386 MiB, p50 11.8. The promoted config soaked 30 cycles /
76 streams / 0 errors / min 7283 MiB / one engine pid throughout. The
smaller boot commit is worth ~4–5 GiB of real headroom at this traffic
shape; the engine's RSS is a ~40 GiB anonymous arena that does not climb —
the few GiB of churn on top is what used to tip the killer.

**Confound law (cost a fake attribution here):** the traffic-learned expert
profile (`--expert-profile-save`) persists across restarts. Any knob
re-measure must re-run the *baseline* arm on the same warm profile before
attributing a delta to the knob — a jump that survives a same-profile
baseline is real; one that doesn't was profile warmth. All §212–§213 numbers
above are matched-warm, same-boot where claimable.

Posture: the bench mech config (`:8181`) now runs `ELASTIC=1 FLOOR=4096` on
binary `3665a350` (sha1 `02afaf20`) — a config-env flip, no rebuild; revert
is `ELASTIC=0` plus restart.

## 214 — the mech tuning grid: batch is the survivorship dial, fraction is nearly irrelevant

Nine-cell sweep (batch 4/8/12 × expert-cache fraction 35/50/65, bench node,
full seven-leg battery, 9/9 cells RC=0). Headlines:
- **Batch decides who survives; fraction barely moves anything** (all
  f-variants inside noise). b4 is fastest (farm 73, worst-leg 55 tok/s, hit
  95–97%) but cannot hold the farm-C4-warm or nat-C4 legs — 2 of 4 streams
  died in *every* b4 cell: 4 sessions for 4 streams plus the prefill burst
  (the prompt path borrows ~3.8 GiB of CUDA0 slots) is an exact-fit pool
  overrun, and it dies silent (`make_room`, zero log lines).
- **b8 is the only zero-casualty profile** at C=4 × 25k (farm-C4-warm 64–65
  with all four streams, nat-C4 27.4 all four, hit 90–92.5%). b12 survives
  like b8 at ~15% lower decode and worse hit (80.9–85.8%) — and the engine
  clamps `--batch` to `--conversation-cache-slots` anyway (b12 boots as
  eight slot sessions), so the b12 cells ≡ b8.
- Mechanism (verified from boot logs): `--batch N` allocates N VRAM slot
  sessions (0.34 GiB each on this model) — an admission pool per live
  stream, not a decode-speed knob — and the sessions eat the expert-cache
  budget verbatim (b4 leaves 5.55 GiB for the cache, b8 leaves 3.39). The
  hit-rate and speed gradients track exactly the VRAM handed to sessions.
- Decision: **b8_f50 is the mech re-arm shape** (farm within 12% of b4 at
  the only survivable batch); applied to the bench mech config.

## 215 — COW_SHARED re-armed under the elastic floor: the v5 catastrophe is fixed and the verdict still says OFF

The M-C1e shared-region verdict was earned pre-batching; both grids below
re-test it as a *verdict-retest* on the 12 GB node, disclosure per the
provenance law.

v5 (pre-elastic, `COW_SHARED=32768`, negative and banked on the ms01 arm
repo): paired cells show the shared region is a **boot-time reservation
that taxes the expert cache at every concurrency level** — b8: 502 experts
/ 0.83 GiB cache, hit 72.9%, +17–22% PCIe, 1 GiB available; farm-C4 64–65 →
43.3. b4 −13%, b8 −38%. The C1 "regression" never needed a runtime
explanation: the arena is sized before the cache gets its leftover, and the
engine prints the trade per boot (audit line: `expert cache auto: ... -> N
slots`, second witness `the slot sessions take X GiB ... that the expert
cache would otherwise hold`).

v6 (elastic floor 4096, four cells, farm-C4-warm legs / nat C2 / worst C2 /
nat C4 / expert hit, tok/s):

```
b4_f50_e      67.0  75.8  36.5  56.7  39.1   97.6%
b4_f50_e_sh   59.9  72.5  33.6  51.9  32.9   94.2%
b8_f50_e      67.5  81.0  36.3  57.0  38.8   97.6%
b8_f50_e_sh   62.0  70.9  33.5  50.4  36.9   96.3%
```

The elastic floor removed the v5 catastrophe (b8_sh farm-C4 43.3 → 62.0,
available 1 → 9 GiB) — but the shared arm is still uniformly ~8% negative
at healthy headroom. Leg caveat (learned from the v7 rerun): in **every**
b8+elastic cell, v6 and v7 alike, the farm-C4-warm and nat-C4 legs lost
2 of 4 streams mid-leg (the engine died and restarted — `avail` jumped
9 → 59 GiB between legs); the shared-arm deltas above are read from the
surviving streams and the clean legs. The death family is batch-elastic
battery shape, not shared-arm-specific, and predates stage C — open item
(closed by §217).
Root cause of the shared delta is structural: the 32768-slot shared region
is boot-committed and mapped (3.19 GiB) and **not floor-gated** — it walks
past the elasticity and takes the cache's boot gift (3628 → 2623 slots).
That is M-C1e's birth law restated at the batch era: you cannot buy
resident capacity with resident capacity; every region slot costs an
expert slot 1:1.

**Verdict: `STRATA_COW_SHARED` stays OFF on 12 GB.** Opt-in mechanism stays
banked for roomier substrates; a re-arm would have to floor-gate the shared
region itself, and nothing measured here asks for that yet.

## 216 — P2 stage C: the expert cache can now fund an arena grow (implemented; the b8 grid came back null — unreachable there)

With stages A+B, the mech shape's ceiling stopped being VRAM and became the
window-region cap `STRATA_COW_ARENA_SLOTS`: the stage-B boot reservation
already covers the 18432-slot cap entirely, so grows never hit the price
line — and above the cap, nothing was left to pay. Stage C (patch on
`3665a350`, tree md5-matched on the arm node) makes the expert cache a
funding source:

- `STRATA_COW_CACHE_FUND=1` boots the expert cache VMM-backed
  (`ExpertCache::set_vmm(true)` before the cache opens) — a segmented cache
  can give slots away in chunk-sized pieces. The standing price of being
  fundable: the same boot sizes the cache at 1758 slots / 2.87 GiB against
  3628 un-fundable (VMM slot granularity plus the unattached-sessions
  reservation; the A/B grid measures the full trade).
- Economics order at admission: **reservation first** (stage B's claim is
  already-paid money), then the funder pays **only the uncovered part**,
  then the free-VRAM line sees what remains. A cache slot is expensive —
  its expert pays a PCIe miss until a retake refills it — so it is never
  spent while the reservation still covers.
- The funder is the prompt-lending quiesce dance aimed at the arena:
  quiesce → rebuild the owner/rank/cold tables on *every* call (the cache's
  own admission moves experts between dances) → walk slots down from the
  prompt's loan edge, never below `STRATA_COW_FUND_MIN_PCT` (default 50) →
  a hotter expert moves into the coldest loan slot or drops to the CPU tier
  → wholly-free chunks unmap and park in the bank (same one-bank rule as
  detach trim, same cap). The reverse leg runs after every detach trim:
  bank chunks map back into the cache's range and refill hottest-first from
  the learned profile.
- Silent paths made loud, because the v5/v6 mystery (grow block present,
  zero grows, zero complaints) was only unmaskable retroactively: every
  grow failure now prints (`arena grow attempt failed: ...`), and a
  one-shot line names the first reason the fund declined to pay.

Smoke (b8 shape, floor 4096, cap 18432, fund on): boot registers
`cache-funded arena admission on`; 12 concurrent ~4k-token streams all
served 200 (10.1–16.9 s each); **12 `arena grown to` lines — the mapped
window climbed 4096 → 16384+**, the first grows ever observed in this
family (v5/v6 grew zero, silently); 16 transient `no room` refusals all
cleared on later admissions — with grows keeping pace the capacity gate is
backpressure, not death. The funder itself stayed dormant (0 calls, 0
complaints), correctly: the reservation covers a cap that size.

The end-to-end grid (three cells on the b8 shape: control cap 18432 /
cap 34816 fund-off / cap 34816 fund-on) came back **null, and the why is
a law of its own**: the §206 boot clamp printed `the arenas hold 18432
slots/layer, live demand is 34816 (main + 8 slot windows + one doubled
round): raising the arena` — at batch 8 the config cannot set
`STRATA_COW_ARENA_SLOTS` below `(1 + 2 x batch) x max(cells, kv-resident)/4`,
so the control cell booted at the same 34816 as the raised cells, and the
control-vs-cap pair was dead on arrival. All three cells then agreed within
1% on every battery column and all three booted the cache at 1758 slots /
2.87 GiB: at b8 it is the **stage-B reservation** (3106 MiB of unattached
slot windows charged to the cache sizing) that sets the boot gift — against
the 3628-slot gift of the batch-4 mech — and that reservation covers the
whole clamped window region. The battery legs (slot sessions pre-sized at
8192 cells, concurrency <= batch) therefore never reach the funder, and
even the 12-concurrent probe's twelve grows were paid from the
reservation's draw with the funder still dormant.

Stage-C verdict as measured: **the fund path is currently unreachable on
the b8 mech shape** — clamp plus reservation fully fund the window region —
so stage C ships opt-in and dormant, honestly untested end-to-end. A real
A/B needs a shape where demand outruns the reservation: a batch-4 mech
(smaller reservation, raisable cap) or a concurrency-above-batch probe
arm. Both are queued behind the open b8-elastic death-family item (the
mid-leg 2/4-stream deaths above; closed by §217).

Landing note: the whole `dynconc-arena` chain (`29f10bf4` → `3665a350` +
the stage-C working patch) is snapshotted onto this branch's tree — the
five touched engine files are byte-identical to the ms01 binary that ran
the smokes and the v7 grid (`layer.cpp` md5 `d964a48c…`, `generate.cpp`
`94d81616…`, `layer.hpp` `a8749ebf…`; earlier md5 notes predating the
fix-round re-scp are superseded by these). The serve/dashboard/packaging
surfaces deliberately keep trunk's newer versions — the arm's base
predates §184/§185/#656/§203 and taking the arm copies would have
reverted them. `main` awaits ctest + a soak on the merged tree.

## 217 — M-D2: the b8-elastic battery death family — a silent CoW-view park write; the guard closes it

§215's leg caveat and §216's "open death family" are now closed, root cause
found and the fix measured on the same battery that reproduced it.

The signature (identical in every b8-elastic cell, v6 and v7 alike): the
wrapper logged `the engine reported an error:` with nothing after the colon,
then `the engine stopped unexpectedly (exit code 1)`; the engine's last log
line was always `park COW anchor mounted for N of M tokens`. Two deaths per
cell, auto-healed by the serve wrapper mid-battery — which is what zeroed
the farm-C4-warm and nat-C4 legs' streams and jumped `avail` from 9 to
58 GiB.

The trace: park failure exits serve at the one site that prints
`ERR %s` + `return 1`, and the message was empty — so the failing callee
was one of the two sites in the whole park→save→`conversation_kv_save`
chain that return false **without setting the error string**:
`ConversationBuffer::visit_impl` — the range guard and the
`never write through another owner's bytes` borrowed-segment guard
(`conversation_buffer.hpp`).

The mechanism: farm conversations share a 19.8 k-token prompt prefix, so
their parks take the M-C2 CoW route — `adopt_shared` mounts read-only
*views* of the donor image's bytes (the `park COW anchor mounted` line).
`--conversation-cache-park-gain` defaults to 0, so the warm leg re-parks the
same conversation, and `take_reuse_for` hands back its **own previous
image buffers — still holding those borrowed view segments**. The reuse
branch rewrites from the page-rounded kept prefix; when that range lands
under the old view head, `visit()` aborts with no message, the park fails,
serve exits 1, the wrapper restarts it. The geometry is rare — one
anchor-park plus a re-park diverged back under the view head — which is
why a 12-stream smoke never hit it but the battery did twice per cell. It
predates P2-C (it is M-C2-era code), exactly as §215 observed.

The fix (3 files, §202 law — silent paths go loud):
- `conversation_buffer.hpp`: `has_borrowed(offset, count)` — check before
  visiting for a write.
- `conversation_snapshot.cpp` `conversation_kv_save`: when the rewrite
  range hits a view, drop this part's reuse and copy it whole, fresh, with
  a one-shot notice (`the reused K/V part holds CoW views under this
  park's rewrite range; copying the part fresh`).
- `generate.cpp`: `if (!saved)` names the silent case
  (`conversation park: the K/V save aborted without a message`) instead of
  exiting with an empty `ERR`.

Measured same-boot against the reproducing battery (b8 grid config,
`STRATA_PARK_TRACE=1`, tree `3665a350` + P2-C + M-D2, ms01 3080 Ti):
**before**: 3 boots / 2 deaths, farm-C4-warm `decode=[72.6,72.5,0,0]`,
nat-C4 `[73.0,0,0,24.9]`. **after**: `boots=1`, `deaths=0`,
`cowguard=1 silenterr=0` — the notice fired exactly once (the rare
geometry, caught) — legs whole: farm C1 42.6, C2 69.9, C4 69.3,
C4-warm `[72.5,72.5,73.2,72.2]`, nat C2 50.6 s (no restart stall inside),
nat C4 `[73.2,73.7,45.2,45.5]`; `avail` steady 8–9 GiB throughout. Surviving-leg
decodes from §215 reproduce within noise, so the death family was the only
confounder in those legs.

Landing: patch on the canonical `m-d-conc` tree and on the ms01 arm tree
(`conversation_buffer.hpp`, `conversation_snapshot.cpp`, `generate.cpp`;
originals `.bak-md2`); `build-s180` rebuilt; the live b4 mech restarted on
it (:8181, one completion smoke, clean). This unblocks the §216 fund A/B:
the queue-vs-fund arm can finally measure without mid-leg deaths (verdict at §218: the funder is structurally unreachable on 12 GB geometries; the serve queue is the sole absorber).
### §218. Fund-vs-queue reachability audit: the five-grid null chain (v8-v12), and the rung-preset finding (2026-11-23)

The §216 fund A/B ran to five arms on ms01 (all batch-4 mech-config, same
battery, `STRATA_PAR_COUNTERS=1`, minutes apart; every arm `boots=1, deaths=0,
silenterr=0` - the §217 guard fired 1-2x per battery and kept all seven legs whole
in every arm):

- v8 (rung 8192, SLOT_CELLS 16384, FLOOR 4096): both arms `grown=0 funded=0`;
  the pair identical within noise (farm C4-warm agg 78.7 vs 78.6; both boot at
  cache 2586 slots).
- v9: the rung-16384 edit no-oped - the grid script regenerates cell JSONs from
  the mech base at run time and clobbered the hand-written config; v9 re-measured
  v8 and matched to the decimal. Ops law: edit the script's embedded generator,
  and verify by the ENGINE BOOT LINES, never by the JSON on disk.
- v10 (rung 16384, FLOOR 1024): boots print `arena floor: 4096` - the floor
  clamps up to the first-admission window (layer.cpp:1377, "never starve the
  first admission"). `grown=0` again.
- v11 (rung 4096, FLOOR 1024 sticks, FUND=1): windows are 4096 cells against
  19-21k-token streams (everything past the rung lives in the CPU/park tiers);
  still `grown=0 funded=0`, all legs whole.
- v12 (v12 = v11 with FUND off): the clean pair - cache boots IDENTICAL at
  2970 slots / 4.84 GiB (the VMM boot tax is zero at this sizing, unlike the
  b8 shape's 3628->1758), last hit-rate legs 98.9% vs 98.6%, decode within
  noise (farm C4-warm agg 79.7 vs 80.5; worst C2 55.5 vs 55.9; nat C4 tail
  54.9 vs 54.8).

Verdict: the stage-C funder is STRUCTURALLY UNREACHABLE on 12 GB mech geometries.
Three walls, each verified from source and boot lines: (a) arena demand is pinned
by the `--kv-resident` rung - long conversations park past the window instead of
growing it, so windows never exceed their reservation; (b) `STRATA_COW_ARENA_FLOOR`
clamps up to the first window, so a small floor cannot buy grow pressure; (c) the
stage-B unattached reservation (2.06-2.16 GiB at b4) exceeds the whole arena
region (1.74 GiB at 36864 slots x 12 layers) - boot pre-funds every batch
window. The transient `no room` + the serve-side queue (§165) is therefore the
only absorber in this hardware class: §216's economics resolves to the queue
half BY CONSTRUCTION, not by measurement of a contested grow. The funder stays
opt-in for geometries where windows outgrow their reservation (no-rung windows,
or attach-prefetch beyond batch - the 12-stream b8 smoke's 12 grows all paid
from reservation draw, 0 funded).

Rung-preset finding (feeds the P2-D calibration stage): rung 4096 BEAT rung 8192
on every leg while handing the expert cache +384 slots (2970 vs 2586): farm
C4-warm agg 80.5 vs 78.7, worst C2 agg 55.9 vs 52.4, nat C4 tail 54.8 vs 53.0,
hit-rate legs >=98.6%. For 21k-token working sets, halving the resident window
costs nothing and returns ~1 GiB to the cache.

Artifacts: ms01 `grid-results-v{8,9,10,11,12}-*`, `logs/v{8,9,10,11,12}-b4_*.log`,
scripts `~/s8_tune_grid_v{8..12}.sh`.
### §219. P2-C.5 reservation-collateral: the b8 cache unlock (v13 pair, 2026-11-24)

The Gemini critique's prescription, implemented: with `STRATA_COW_CACHE_FUND=1`,
the deferred slot sessions' arena-window bytes are no longer subtracted from the
expert cache at boot - the cache's fundable slice is their collateral - while the
per-session buffer part stays priced in (plain cudaMalloc at attach; nothing lends
against that). One file: the cache-pricing ledger (generate.cpp ~8552):
`cow_reserve = deferred x session_bytes + (fund ? 0 : deferred_window_bytes)`, and
the arena grow budget goes to zero with the fund on, so any grow pays the funder
before the price line instead of a boot claim. FUND off: stage B byte-for-byte.

v13, same-boot pair, b8 mech shape (FLOOR 4096, SLOTS 34816, PAR_COUNTERS):
- Boot, fund arm: `reservation collateral: 672 MiB of deferred windows is NOT held
  out of the cache`; reservation 3106 -> 2434 MiB; cache **2153 slots / 3.53 GiB**
  vs control **1758 / 2.87 GiB**. The deferred window is 96 MiB/session at this
  shape (demand 6144 = main + one slot + one doubled round, all priced).
- Battery (7 legs, minutes apart): the fund arm won EVERY leg. farm C1 48.8 vs
  42.6; farm C4 71.5 vs 69.2; farm C4-warm agg **76.7 vs 72.6**; nat C2 **52.5 vs
  45.0**; worst C2 **53.3 vs 45.2**; nat C4 agg 62.1 vs 56.1, tail **53.0/52.4 vs
  45.7/45.5 (+15-16%)**. Hit rate: 94.4% vs 91.6% (PCIe 3.2% vs 5.1%) on the
  heaviest leg.
- Runtime counters identical in both arms: `grown=0 declines=0 refused=0 funded=0
  gave=0 back=0 cowguard=1 re-carved=0 boots=1`. The funder never had to pay:
  session buffers arrive via the admission cudaMalloc (priced, fits), and CoW
  claims stayed inside the mapped floor. The whole gain is the boot unlock, at
  zero measured runtime risk - but note the corollary: with the budget at zero,
  an uncollateralizable grow would now go straight to the funder and the price
  line (no boot claim in front), so hysteresis care is owed before any shape
  where the dance actually fires. The give/take dance remains unexercised.
- §214 survivorship is untouched (same 8-slot ceiling, same attach-on-demand);
  §218's structural verdict stands for the reachable shapes. This is the
  b8-survivorship + b4-class-cache synthesis the concurrency goal asked for:
  the 101 MiB indexer pool and the doubled-round window claim are the remaining
  priced-in residue (P2-C.6 candidate).

Artifacts: ms01 grid-results-v13-b8_{fund,ctrl}-1123-23*.log, logs/v13-b8_*.log,
~/s8_tune_grid_v13.sh; engine: generate.cpp collateral hunks (.bak-p2c5 backup).

ctest gate on the merged arm tree (green, 2026-11-24 00:25): 340/352 at -j4; the
serial re-run of the 12 fails shows 10 are -j4 contention flakes (snapshot
families, prefill_fused_iq_*, gr_upmix all pass solo). Only two true fails, both
pre-existing in the ctest190 baseline: ple_parity (wants ../../Q2_0/*.gguf, an
asset path absent in the arm tree) and cuda_spill_guard (known). Ops law: the
ctest gate is full-suite -j4 then re-run-failures serially; never conclude a
regression from a -j4 parity/snapshot fail alone.

### §220. KV-window degradation bisected to the dyn-conc arena VMM conversion (2026-11-24)

The endurance-battery content degradation (W3 restores returning 3-6-token
refusals; repetition collapse at temp 0) was bisected on ms01 to the dyn-conc
stage-A/B arena conversion, not to M-D2, P2-C.5, the learned profile, spec
decode, or the park store:

- stage A = 11/21 source state (arena arrays = plain cudaMalloc): W3 4/4 green.
- stage B = +4c39aa5 (arena window arrays live on one VmmRange per arena:
  CowArena::vr + off[11] chunk-aligned offsets; boot maps the floor, windows
  attach at admission): W3 4/4 red, restore walls normal, expert hit 93-98%,
  no engine errors, counters clean.
- arena0 = same B binary, STRATA_COW_ARENA=0: 8/8 GREEN - the first perfect
  W1 wave (4/4) ever recorded on the m25_churnpar v2 battery, which had been
  67/68 red since its first use. W1's chronic red was the same bug.

Same battery (mtime Nov 21 02:52, proven unchanged), same seeded profile,
same cold park store, stock mech config, one variable per arm. The old
11/21 green logs predate the v2 battery and were never comparable to it.

Consequences, held to what is measured:
- Every dyn-conc-era grid number (v7-v13, §216-§219) was collected while
  every session's KV windows were degraded; cross-arm deltas are suspect and
  §219's promotion gate stays blocked until the arena defect is fixed and
  the pair re-run.
- The fix hunt starts at the conversion's invariants: chunk-aligned off[11]
  addressing, zero-fill on fresh VmmChunk maps, and whether grow at admission
  can shift array offsets under live ranges. Mechanism NOT yet identified.
- Local Windows engine (C:\hydragen, Nov-20 install) predates the conversion
  and is not in the affected family.

Control probes (same boot, single variable, seeds varied per arm): stage A
(11/21 binary) FAILS the 4-up parallel probe too (0/4 clean at ~13k, 2/4
refusals at ~1.6k) while single-stream sweeps are clean at every size. Two
families share the arena feature set: Family 1 = fresh+parallel corruption
(present at stage A and B, absent arena-off; the battery's chronic W1-red),
Family 2 = restore corruption (era-D only; W3 flips stage-A green -> stage-B
red; the VMM conversion). Arena-off is clean on every probe tested, hence
arena0's first-ever 8/8. Env-knob bisection of Family 1 (COW_FORK/PARK_COW/
COW_JOIN_HH) follows on the stage-A binary.

Method laws banked this session: (1) a battery whose mtime postdates the
"green baseline" logs is not a baseline - check versions before bisecting;
(2) .bak revert chains must be path-verified (src/core vs src/program) or
the && chain silently reverts nothing while still booting the old binary;
(3) overlapping arms on one host contaminate each other - one experiment,
one queue.

### §221. Bisection audit: the carrier is the arena window machinery; the fork claim was a false positive (2026-11-24)

*(Superseded by §222: the dead-path audit below is itself wrong.)*

Corrected verdict table (C.5 tree 3eff8fd, cold store, m25_churnpar v2 battery; and the
4-up parallel probe at mt 900 with per-arm seeds):

- arena0 (COW_ARENA=0): 8/8 green, restore_qc_fail=0. The one true green axis.
- c5fork0 (COW_FORK=0): 8/8 green - VOID as evidence: hg_getenv("STRATA_COW_FORK") is
  read only inside direct_slot_copy (generate.cpp:15646), which requires
  STRATA_SLOT_COPY_DIRECT, unset on mech; layer.cpp never reads COW_FORK. On the mech
  shape the admission-fork path is dead code, so fork0 == stock. (An earlier green on
  the stage-A binary likewise exercised no fork site.)
- join0 (COW_JOIN_HH=0): 0/8 red. Join-at-fetch exonerated.
- detach0 (SLOT_DETACH_MS=0): W3 green at 4.2-4.5 s walls = shallow warm-slot restores,
  path class changed, not a fix.
- parkcow0 (PARK_COW=0): 0/8 red. Park COW anchor mounts exonerated.

Green requires COW_ARENA=0. Every knob inside arena-on stays red, and the same
fresh+parallel corruption holds at stage A (pre-era-D), so the carrier is the S8-v2-era
arena window machinery itself (per-layer slot arenas, first-fit release/reclaim with
range splitting, bump cursor, zombie guard; era-D adds floor mapping + elastic grow
with chunk release/re-map), not one sub-feature.

Leading hypothesis (static analysis, unconfirmed): conversation images and checkpoints
carry pool slot indices; when reclaim/re-map moves a session's window base between
park and restore (era-D floor+grow makes bases unstable; serial single-session runs
reclaim the same base, concurrent ones do not), readers that resolve stale indices
follow tables into other sessions' ranges. Fresh-session parallel corruption may be
range-handling on the claim/release path (splits/zombie guard). Instrumentation
(table-vs-window content compare, opt-in env gate) is the next step; knobs are
exhausted.

Consequences: §219 promotion stays blocked; era-D numbers stay suspect; the earlier
"Windows :8080 fork-off mitigation" claim is also void (same dead-path reasoning).

### §222. The §221 audit was itself wrong: the CoW fork IS on the live path, and it corrupts through four verified bugs (2026-11-24)

The §221 reversal rested on `cow_fork_on` being read only inside `direct_slot_copy`. It is a
`static` defined at generate.cpp:15645-15648 and consumed at THREE sites: :15680 (inside
`direct_slot_copy`: that transport is indeed dead without `STRATA_SLOT_COPY_DIRECT`), :15814
inside `copy_to_slot`'s legacy branch (:15796-15827, the branch the mech shape takes), and
:15890 inside `copy_from_slot`. The fork fires on every batch admission and every slot->main
continuation. The per-admission print at :15822 ("cow: the admission fork points at %lld of
%lld prefix pages") is the proof of liveness; §204-era engine logs carry it ("points at
1023 of 28927 prefix pages"). Consequences: c5fork0 8/8 green and the fork0 probe 8/8 are
VALID evidence; arena0's green is the same axis (qsa_cow_fork returns a silent no-op when
`slot_use` is null, layer.cpp:898-899, so arena-off disables forking too); the allfloor arm
(COW_ARENA_FLOOR=18432, static full map: 0/8 red, restore_qc_fail=4) kills the elastic
grow/shrink-churn hypothesis.

Mechanism, verified by code reading (an external deep-read corroborated all four sites):

1. Starved-window blinding (primary, both families): every request runs `read_windows` and
   token 1 on `ss` (slot_base 0) before admission; the fork pins ~all of `ss`'s own slots
   (use 1->2) for the batch slots; resolve never takes a pinned victim (`cow_hold`,
   kv_stream.cu:202); the next conversation admitted/restored on `ss` overflows resolve
   (`placed < need` -> page -1 + ctl[3], kv_stream.cu:262, whose comment claims it "never
   happens"), and decode attention silently masks -1 blocks to -FLT_MAX
   (qsa_decode_attn.cu:206-209). The conversation answers blind to ~99% of its KV, zero CUDA
   errors. Explains the size threshold (900-line arm clean vs 1300 red), W1 and W3 together,
   and the era-D flip (window shrank so one fork starves `ss`).
2. `qsa_state_zero` (pre-fix layer.cpp:1679-1699): calls `kv_stream_reset` WITHOUT
   `qsa_cow_release` (leaks pins, leaves stale conv/floor registry stamps: pin leak matches
   the endurance soak's growing restore_qc_fail) and memsets its own slot range
   unconditionally, wiping bytes still pinned by a live batch slot: the next fresh request's
   `session_zero(ss)` corrupts W1 sessions mid-decode, which then park corrupted state.
3. Lineage floor gating: `cow_lower_floor` ran only when `g_cow_shared > 0` (layer.cpp:931),
   and resolve read the floor only when `lin_all != nullptr` (kv_stream.cu:98) - with
   `STRATA_COW_SHARED=0` the floor stayed infinite, so joins-at-fetch could name a sibling's
   divergent tail. Latent (join0 red: joins were not the carrier) but real.
4. Rewind overwrite (unfixed, flagged): resume>0 from an in-memory checkpoint without
   `refresh_residency` leaves positive table entries over fork-pinned slots, and neither
   `kv_append` (kernels.cu:2562) nor `kv_append_q8` (kv_q8.cu:64) checks `slot_use`: a new
   branch writes in-place through a shared page. Not shown active in the battery shapes.

Fix (7cb7f1e): `qsa_state_zero` releases fork pins before resetting and zeroes its mapped
range in runs, skipping slots with `slot_use > 1` (no-pin ranges zero identically to before);
`qsa_cow_fork` keeps the source a headroom of unpinned slots (`STRATA_COW_FORK_HEADROOM`,
default 512, clamped to n_slots/2; refused pages stay cold and re-fetch from the host image);
the floor is lowered unconditionally at fork and read whenever the registry exists.
Test: fix222 arm (stock mech, fork ON) on ms01 - verdict pending at writing.

Consequences: the fork/arena story of §220 stands; §221's knob verdicts (join0, parkcow0,
detach0 path-class) stand; the Windows :8080 exposure note of §221 is RESTORED (the fork is
live there too) though the fix, once proven, supersedes any mitigation. §219 promotion
stays blocked until fix222 is green and grids are re-run.

### §223. The §222 fix is not sufficient: arena-on still degrades on the fixed binary — 1/4 twice, vs 8/8 arena-off (2026-11-24)

The §222 test verdict is in, on a rebuilt ms01 mech binary (`build-s180/hydragen`, mtime
Nov-24 07:11, carries `STRATA_COW_FORK_HEADROOM`, i.e. built from 7cb7f1e). The discriminator
is the 4-up parallel probe (`probe222x.py parallel`: 13k-token prompts, 900-token arms, four
concurrent seeded conversations, qc = both numeric marks echoed and `finish=stop`):

1. Stock mech (`COW_ARENA=1 COW_FORK=1 JOIN_HH=8`), two rounds, seed bases 8300 and 8310:
   **1/4 and 1/4**. The failing tails are degeneration/refusal — `3.14`, `no meaningful
   answer to extract from this string of characters`, `27 3 1`, `34 12`, `...clarify the
   goal`, `3.1415926535` — every arm `finish=stop`: corruption, not truncation. One arm
   survives in both rounds.
2. Same binary, same boot ladder, `COW_ARENA=0` (`strata-green1.json`), two rounds, seed
   bases 8200 and 8210: **8/8**.
3. The restore-QC arms stay green on arena-on (`fix222b`, `fix222b2`: `restore_qc_fail=0`):
   §222's four bugs genuinely healed on the restore path. The surviving corruption carrier
   under the probe shape is elsewhere in the arena-on path and needs concurrent forks
   (serial W-probes were clean pre-fix too).

So §222's gate reads: restore-QC green, degradation red, and degradation is the gate that
counts. The §220 law ("everything arena-on red") survives the fix; the only green axis
remains `COW_ARENA=0`.

Live reproduction of the same corruption class on the Windows :8080 desktop service:
pre-fix binary (`hydragen-pb4.exe` built Nov-22 09:57, before 7cb7f1e), `COW_ARENA=1
COW_FORK=1 JOIN_HH=8`, interactive sessions to ~126k-token depth — the service log shows
4-token degenerate generations, `KV streaming: OVERFLOW (too few resident cells)` at >90 %,
`park COW anchor mounted for 3 of 125830 tokens`, `reused_kv_bytes` ~1.9 GB. §221's exposure
note (restored by §222) is confirmed: the desktop service has been serving on the corrupt
path. Mitigation staged in config (`COW_ARENA=0`, backup `hydragen-local.json.bak-arenaoff`),
restart pending.

Next engineering target: the carrier sits inside the arena claim/release path under
concurrent fork pinning; the `STRATA_COW_DEBUG` knob (c172460: CLAIM/RELEASE/ZOMBIE/GROW/
SHRINK per-layer events) is the intended observation under the 4-up shape. §219 promotion
stays blocked; the known-green production axis is `COW_ARENA=0`.

## §224 — conversation-identity arms: fingerprint and tab-injective serials, both 1/4 (2026-11-24)

Two source arms against the §222-fixed tree, each synced to `~/hydragen-sync180`,
`ninja` in `build-s180`, gated by the 4-up degradation probe (`probe222x.py parallel`,
four 13k-token prompts with two marker numbers, qc = both markers + `finish_reason=stop`):

1. **§224 fingerprint identity** — `qsa_cow_fork(…, const std::vector<int32_t>* ids)`:
   the lineage serial is chosen by an FNV-1a fingerprint of the conversation's token ids
   (first `upto` ids, length mixed in) against a `ids-fingerprint → conv` table, so two
   admissions of the same conversation adopt the same conv and different conversations
   never share one. Both call sites pass ids (`copy_to_slot`: `&ids`; `copy_from_slot`:
   `at ? &at->ids : &bs[b].ids`). Verdict: **1/4 and 1/4** (seeds 8300/8310) — the
   failure tails are the same degeneration/refusal class as §223 (`e i a e i c y r`
   character soup, number recitation, one refusal legibly answering a different task).
2. **§225 tab-injective conv serials** — host-side hint-tab ownership: `g_cow_lin_owner`
   maps each of the `kCowLinTabs` hint tabs to the live conv holding it; `cow_conv_new()`
   hands out a serial injective over live tabs (LRU victim tab reset via
   `cow_lin_reset(victim)` *before* the fresh holder claims it), and re-entering an
   already-owned conversation no longer wipes its tab. Kernel untouched. Verdict:
   **1/4 and 1/4** (seeds 8300/8310). Control on the same §225 boot: `parallel`→**serial**
   probe seed 8500: **4/4** — and after 40 admissions the census still prints
   `0 of 3193 prefix pages` allowed with neighbor admissions at 512/1023/29.

Read-out: identity is *not* the carrier. The hint tabs are keyed `conv % kCowLinTabs`;
§224 made conv honest per conversation, §225 made it injective per tab, and both still
degrade 3-in-4. Both patches stay in the tree (they close real aliasing holes and the
kernel's protocol is untouched); reverting them would only re-open holes the §222 note
records.

## §225 — the serial-vs-parallel asymmetry points at the join mechanics, not identity (2026-11-24)

Arm ladder on the §225 binary/config (`strata-mechdbg.json`, park-disk serve shape,
port 8181, arena 18432 slots, floor 4096):

| arm | probe | verdict |
|---|---|---|
| stock mechdbg (fork1 hh8 shared0) | parallel ×2 | 1/4, 1/4 |
| serial control (same boot) | serial | 4/4 (0 joins served) |
| `COW_SHARED=0 COW_JOIN_HH=0` (JOINOFF) | parallel ×2 | 1/4, 0/4 |
| `COW_FORK=0` (rest as JOINOFF) | parallel ×2 | **4/4, 4/4** |

Corroborated earlier on the pre-§224 binary: NOFLOOR 4/4, FORK0FULL 4/4×2,
`COW_ARENA=0` 8/8×2, `JOIN_HH=0` alone 1/4, `SHARED=1` 1/4, `SHARED=1+JOIN_HH=0` 2/4.

The green family is exactly the arms that kill **either** the fork splice **or** the
join paths; the red family is exactly "a join path is live". Code read closes the loop
the JOINOFF arm left open: `join_ctr` bumps at exactly two sites in
`kv_stream.cu` (:149 hint adoption, :179 sibling adoption); the sibling path consumes
the per-entry `hhead` budget (gate at :172-176) but the **hint-adoption path at :149 is
gated by neither `hhead` nor the shared region** — only `lin != nullptr` (which needs a
registry conv ≥ 0 and a mapped hint table, both true with `SHARED=0`, because
`qsa_set_cow_shared(0)` maps 0 region slots but leaves `lin_all` wired) and `b < fl`
(the fork floor). That is why `JOIN_HH=0` alone scored 1/4 and why the JOINOFF config
arm still served 15,508 joins: **config knobs never disabled hint adoption.** The
`b < fl` gate also explains NOFLOOR and REKEY going green: both remove the gate, which
kills adoption.

Note on log hygiene: the FORK=0 rerun re-used the JOINOFF runner's output files
(`probe-226.log`, `probe-226b.log`), overwriting the JOINOFF probe logs after their
verdicts (1/4, 0/4) had been recorded above — verdicts preserved here, logs not.

The fork floor law says only blocks below `fl` are byte-identical across the lineage,
and adoption pins with the prev-gated M-C1d protocol plus a revalidate against
reclaim-move; the resolve path fences its re-point (`slot_use` 1→0 CAS after detaching
the entry, §220) — but the **append/write path names slots without any `slot_use`
check**: `kv_append_q8_kernel` writes `kv_k/kv_v[page …]` for any `table[pos/page_size]
≥ 0`, and `conversation_kv_restore` writes named slots directly. A window that joined
a sibling's slot below the floor is only safe while the holder never rewrites that
slot; the fork-join kernel's pin (`cow_fork_join_kernel` +1 per spliced foreign entry)
protects against *eviction* (§222 bug #3) but nothing marks the *host-side rewrites*
in `conversation_kv_save/restore` — which run on every `copy_to_slot`/`copy_from_slot`
while other windows hold pins. Arm in flight at entry-time (§227): fork ON + hint
adoption disabled at the source (`if (false && L >= 0)` at :141) — if that goes green
while fork splice stays live, the adoption race is the carrier and the fix is a
pin-checked rewrite, not a knob.

## §234 — VMM tail contract: hint tabs stay unmapped/null when the shared region is off; arm-227 kill reverted (ms01, 2026-11-24)

§223's red axis (arena ON 1/4 vs arena OFF 8/8 on the same fixed binary) has its root
cause in commit `4c39aa5`, the VMM conversion of the CoW arena. The pre-VMM contract,
stated verbatim in the wiring comment at `layer.cpp:1675`, was: `lin/sblk/sst/srf/sctr`
are null/zero when `g_cow_shared == 0`. The conversion broke it three ways: the hint-tab
tail became ordinary always-allocated chunks, `a.lin = (int32_t*)(pb + a.off[6])` etc.
were assigned unconditionally, and the `0xFF` memset stayed gated on `g_cow_shared > 0`
— so with sharing off (the mechdbg config: `12 QSA layers x 18432 slots (0 shared
region)`) the lineage hint table was **non-null, zero-filled garbage**. The resolve-time
wiring (`kv_stream.cu:103`) gated only on `m.lin_all != nullptr` — no `n_shared` term —
so every cold miss below the floor read `L = 0` and adopted **slot 0's bytes** as the
missed block: the masked-KV signature, end to end. The arm-227 kill (`if (false &&
L >= 0)`) was never a cure, only a mask: it disabled adoption, while `lin[b] == 0`
simultaneously defeated *miss registration* (the `-1 → -2` claim CAS never fires on
zero), which is why its runs went green measuring a dead path.

Fix (both belts, applied):
- `layer.cpp`: when `g_cow_shared == 0` the arena's total ends at `off[6]` — the whole
  hint-table tail (`lin, sblk, sst, srf, sctr` + fetch counter) is never mapped and all
  five pointers stay null. The comment's contract is now true by construction, and
  ~24 MiB of unused VMM per arena is no longer reserved (user's estimate; not
  separately measured).
- `kv_stream.cu:103`: `lin` is wired only when `m.n_shared > 0 && m.lin_all != nullptr`;
  the floor (`fl`) stays wired whenever `m.reg != nullptr` (cow-m1d join-at-fetch is
  unaffected).
- `kv_stream.cu:141`: the arm-227 kill is reverted — `if (L >= 0)` restores hint
  adoption as live code on the `STRATA_COW_SHARED > 0` path; it is unreachable when
  `g_cow_shared == 0` because the resolve gate above now holds.

Measurement (arm-234, ms01, `~/hydragen-sync180` rebuilt clean from HEAD + this fix;
mechdbg config, arena ON, fork at default): probe222x parallel seeds 9140/9150 →
**4/4 + 4/4**, zero corruption markers in `base_mech.log`. The §223 run of the same
fixed binary before this fix: 1/4 + 1/4; §224/225 identity arms: 1/4 each. The arena
axis is closed by the tail contract, not by a knob.

Retraction note: the §230–§233 measurements (386K violations, the `owner_says=-1`
ring rows, and "FORK=0 is NOT green") are void — that binary carried the uncommitted
arm-228 release-decrement bug (`slot_use` decaying to 0 with no re-seed ⇒
`atomicCAS(slot_use,1,0)` fails forever ⇒ stranded slots, `page_table[b] = -1`,
cold-miss storm). None of it is committed; the diagnostics are archived off-tree.

Next: the shared-region path (`n_shared > 0`, adoption live under the restored
`L >= 0`) has not yet been exercised end-to-end — every mechdbg config here is
`0 shared region`. And the Windows service still runs the pre-fix pb4 binary with the
ARENA=0 mitigation staged; the restart is pending user availability.

## §235 — adoption probe: hint-path fetches flow, but the shared region corrupts while it rolls (ms01, 2026-11-24)

The §234 open item — "the shared-region path has not yet been exercised end-to-end" —
is now exercised, and it does not pass. Config: ms01 mechdbg, arena ON,
`STRATA_COW_SHARED` toggled; binary = sync180 @ `feed840` (HEAD at probe time).

Adoption itself is proven live at `shared=2048`: boot reads
`12 QSA layers x 20480 slots (2048 shared region)`; telemetry shows joins served
(48→365) and **lineage shared fetches flowing (4286, region saturated)**; admission
forks point at real floors (1536/488/1032/841 of ~3190 prefix pages per K/V layer).

Failure matrix (probe222x, same prompt contents per seed):
- `shared=2048`: parallel seeds 9170 and 9172 → **3/4 both** (P3 degenerates:
  repetition loop / token-scramble, finish=length). serial 9171/9173 → 4/4.
- `shared=0` control: parallel 9170 + 9172 → **8/8** (identical contents).
- `shared=8192` (region ≥ the 4-lineage working set, no mid-generation roll):
  parallel 9170 + 9172 → **8/8**; fetches 14566, joins 114.

Zero corruption/wrong-naming markers in the boot itself (the 29 grep hits in
`strata-mechdbg.log` are §230-era boot residue — the log accumulates across boots;
retraction hygiene from the §234 retraction note applies).

Finding: the failure is **deterministic per content-class when the region rolls**
(two different seeds, same victim, serial immune), not sampling noise, and it is
not §234's hint-tab axis (that tab hygiene held: fetches flow, windows clean).
The mechanism the matrix points at: the region sweep's "PIN LAW holds by
construction" claim (region slots victimable only while `slot_use == 0`) does not
hold for **adopted** entries — `miss_sh_slot` commits a window-table entry naming
an arena-global shared slot, but the tail slots' eviction clock is the region's
own sweep (srf ref-bits), not the `slot_use` pins; between commit and gather a
sweep pass can re-take the slot under another lineage. `sst` epoch stamping is
conservative at *resolve* time only — it cannot guard an already-committed entry.
Capacity (8192 clean) is a load-dependent margin, not a fix.

Stance: `STRATA_COW_SHARED` stays **0** everywhere (ms01 mechdbg restored to 0 and
restarted; Windows prod unaffected). Opening the shared path requires the tail-slot
pin fix first (extend the `slot_use`/live-entry accounting over the shared tail, or
gate the sweep on live-entry count); then re-run the 2048 arm before any region
policy is set. counter 188.

## §236 — P2-D1 arms: batch-8 lives on 12 GB, the governor and the admit floor both hold their laws (ms01, 2026-11-24)

Arms (ms01 3080 Ti, :8181, harness `~/hydragen-experimental/p2d1_arms.sh`,
evidence `bench_A0..A3.json` + `logs/p2d1/`): A0 = build-s180 baseline
(batch-4 mechcfg), A1 = p2d1 `a417b8d` identical args, A2 = p2d1 `--batch 8`,
A3 = A2 + `--min-stream-tps 30`, A4 = p2d1 batch-4 +
`--conversation-cache-min-free-mib 20000`.

Parity (A0 vs A1): throughput identical to noise — N=1 **65.1 = 65.1**,
N=2 59.0/59.6, N=4 63.3/61.7. Battery: 5/8 QC lines byte-identical; QC0/2/7
diverge only at greedy fork points ("as we grow"→"as we go"). Cross-binary
parity is near-identical, not token-identical; the same-binary token-identity
law (§177) is untouched — no kernel or numeric path changed, the divergences
are float-order forks amplified at decision tokens.

N=8 on the batch-4 config fails on **both** binaries (A0: 400 tok/6 errors,
A1: 400 tok/7 errors): the §235 churn/adopt corruption, unchanged by P2-D1 —
as scoped, `COW_SHARED` stays 0.

A2 — the payload: **`--batch 8` boots and runs on the 12 GB card.** 8-row
boot fits with GPU 11,111 MiB vs 11,883 at batch-4 — the shared RoPE tables
(session_init `share_rope`) bought back ~0.77 GiB; elastic arenas grew to the
`main + 8 slot windows` window (34,816 slots/layer) with a 4,096-slot mapped
floor, the rest of the VRAM left for the expert cache. N=8: **8/8 conversations
complete (2,400 tok)**, 50.5 agg tok/s, 8.6 per-stream; two first attempts hit
`ERR BGEN: slot session unavailable (memory limit)` (the new admit gate) and
the server's slot-retry recovered both when turns reclaimed. Cost, honestly:
N=1 drops 65.1→56.2 (−14%) and hit rate 88–91%→78–82% at N=8 — the row tables
squeeze the expert cache. Batch-8 on 12 GB is reachable and graceful, at a
per-stream tax.

A3 — governor law: `decode governor: on (min_stream_tps=30.0 hit_floor=86.0%
init_n=4 quantum=2000ms)`; it shrinks with evidence (N 4→3→2→1 at hit 71/63/64%)
and grows back only into warm cache (hit 90–96%). With the floor at 30 — above
this host's width-2 per-stream ceiling (~23) — it collapses to width 1–2 and
N=8 aggregate falls 50.5→39.9 (per-stream 6.1 with queue wait; all 2,800 tok
still delivered, 1 error). **Law: `min_stream_tps` above the box's per-stream
ceiling forces width→1; the governor is correct, the knob was mis-set.** Tuning
arm (floor ~15–20) queued for the next battery.

A4 — floor law: the host-RAM admit floor is a **hard policy valve, not a
self-healing one**. Floor 20,000 MiB: N=1 posts the run's best 66.2 (GPU
10,523 MiB), then at load, headroom < floor ⇒ *sustained* refusal (28 refuses;
N=4 loses all slots, retries exhaust, requests error). Server and engine
survive throughout — refusal is graceful but useless when mis-set. A2/A3's
transient refusals+recovery remain the well-behaved regime.

Harness notes (not engine): the first run's `make_cfg` died on a python
scoping bug (`a +=` inside `setv` ⇒ UnboundLocalError) so A2–A4 re-ran clean;
`bench_quick` then crashed on `statistics.mean([])` when A4 N=4 had zero
successes — guard it next battery.

Stance: p2d1 builds stay on ms01 for arms; **ms01 mechdbg remains build-s180**
until the governor-tuning battery and the §235 pin fix land; Windows :8080
swap still open. counter 189.

## §237 — Phase 0 opens: v0.2.21 fetched, the control baseline tagged, gates written (2026-11-25)

The v0.0.1 rebase base is fixed: upstream **v0.2.21** = annotated tag object
`0f8a6b78` → commit `d46f5f268f8ed797e83a5c63a69bffbd7a581cf6`, tree
`ac94d63dca450fd38be5c65bf67337c450566428`, `project(strata VERSION 0.1.67)`,
**1,748** commits past our `38f63bc5` (0.1.60). Neither local clone had the
objects — `hydragen-sync180`'s `upstream` sat at `c77572806` (v0.2.15) — so the
fetch was a Phase 0 step, not a given.

The control baseline is frozen as annotated tag **`control-pre-0221`** at
`654b078` (§236), pushed to `origin` with the 21 commits that were unpushed.
`docs/control-pre-0221.md` notarizes the **runtime** baseline too, because the tag
cannot: the exe serving ms01 `:8181` is `build-s180/hydragen` md5
`e5ada445f1d5f1a3967343c4f9b3b30f` (52,225,184 B, 2026-11-24 18:50:40Z). Its build
tree `~/hydragen-sync180` sits at HEAD `3665a350` — an object that **does not exist
in the public repo** — **with six engine files modified and uncommitted**, and its
`layer.cpp` / `generate.cpp` / `layer.hpp` md5s (`f32b1dc2` / `b1430daa` /
`fad6d751`) match **neither** `4c39aa5` (`d964a48c` / `94d81616` / `a8749ebf`, the
hashes that commit's own message names) **nor** `654b078` (`5ec03355` / `a984d738`
/ `ca12b2c4`). `build-p2d1/hydragen` md5
`6d7a1714c4cd6c83fc98c51b3504fdf7` has no VCS at all. Both builds are
`STRATA_WINDOW_MAX_T=auto` → **8 rows** on CUDA, arch `86-real`, Release. **No ms01
binary today traces to a public commit**, so Phase 1's parity control must be built
from the tag and re-baselined; the ms01 binaries pin *posture*, not identity.

Mech posture recorded verbatim (20 env entries + 27 args). `STRATA_COW_CACHE_FUND`
is **absent** → `g_cow_cache_fund = false` (`generate.cpp:306`): the control runs
**CACHE_FUND=0**, so the locked Stage-C prune is behaviour-preserving here.

`docs/PHASE1_GATES.md` writes the gates before the work: G1 ctest re-notarized
against v0.2.21 (not §181's 322/352), G2 §177 identity against a tag-built binary,
G3 upstream's KV/batch parity suite (§181 never ran it), G4 both traffic shapes
(divergent `bench_quick` **and** agentic fan-out — §236's evidence base is
divergent-only), G5 the §223 4-up probe riding the agentic arm, G6 the
`git ls-files` 7,332/7,403 gate across the 0.2.16 layout move (upstream `docs/`
holds `MODELS.md` and `HOW_IT_WORKS.md` — exact-name collisions with our root
`docs/`), G7 the 8-row vs 16-row arms, G8 the CACHE_FUND unbundle (prune the
funder, keep C.5's boot pricing, leave the grow budget funded from new VRAM),
G9 upstream's `slot save/restore … not available with parallel requests`
(`terminal/serve/server.py:4460`) checked by test, not assumption.

P0.3 separability, settled in code: the join registry (`CowPeer`, `st.map.reg`) is
gated at `layer.cpp:1653/1665` on the **arena** and on `slot_use`, never on
`g_cow_shared`, so `COW_JOIN_HH` survives the `COW_SHARED` prune — matching §204's
`joins 0 → 80` with `(0 shared region)`. Two prune depths; Phase 1 takes depth 1
(host-side knob and allocation, map fields pinned null/0, byte-identical).
**The prune is hygiene, not the §235 fix**: §235 lives in the arena slot pin law
(`slot_use`, `ablk[L] == b`), which it does not touch; the pin fix stays a hard
gate on Phase 4.

P0.2's input is now complete and on disk: the engine-touching stack since §181 is
**30 commits** (`scratch/phase0_engine_commits.txt`, oldest first), and the range
that earlier recon lost is recovered — `0a3d105` (M-D2 §217 `has_borrowed()`
guard), `6270e23` (§219 P2-C.5), `c172460` (`STRATA_COW_DEBUG` arena event log),
`7cb7f1e` (§222 CoW fork fixes + `COW_FORK_HEADROOM` 512). Note `5074249` (§235)
touches **no engine file** — the adoption probe was run with existing knobs, so the
pin fix is still unwritten, not merely unfixed. `4c39aa5` remains the one
non-topical commit to decompose: 7 files, **+740/−32**, bundling five experiments.

Live ms01 state at capture: a stale `p2d1_arms.sh` wrapper alive 9,364 s after its
battery finished (not killed), and 10 GiB `MemAvailable` against a 16 GiB
conversation cache. counter 190.

## §238 — Phase 0 closes: patch stack, touchpoint audit, layout inventory merged — the collision map is narrower than planned (2026-11-25)

P0.2/P0.4/P0.5 merged into `main` as `c69acfd` (3 files, **+233/−78**): new
`docs/PATCH_STACK.md` and `docs/LAYOUT_MIGRATION_0221.md`, and `docs/UPSTREAM_DELTA.md`
rewritten for the `0.1.60 → v0.2.21` audit — the superseded v0.2.13-era text stays
readable at `4bf3603:docs/UPSTREAM_DELTA.md`. Every load-bearing number was
re-measured against the objects rather than trusted from the prose, and all matched.

Fork overlay against upstream `38f63bc5`: 7,344 upstream files vs 7,332 in our
`engine/` → **7,269 byte-identical, 52 modified, 11 fork-only, 23 dropped** (all 23
under `.github/`, from `ebea5fe`). Upstream churn column verified line-for-line:
`generate.cpp` **+3,206/−518**, `serve/server.py` **+2,125/−2,864**, `prefill.cpp`
+953/−210, `verify.cpp` +874/−151, `expert_source.cpp` +601/−84,
`conversation_snapshot.cpp` +477/−0, `qsa_select.cu` +129/−18.

The finding that re-shapes Phase 1's risk map: the files carrying our biggest fork
deltas are the ones upstream barely touched. `layer.cpp` upstream churn is **+9/−0**
against our **+845/−14**; `layer.hpp` +8/−0 against +79/−1; and `session.cpp`,
`session.hpp`, `qsa.cu`, `qsa.hpp` are **byte-identical** across 1,748 upstream
commits (`eb91c380` / `4d633205` / `a635e81e` / `8eb89e2c`). Replay conflict is not
spread across the tree — it concentrates in `generate.cpp` (94 fork hunks landing
into +3,206 upstream lines) and in `terminal/serve/server.py` (+284/−74 landing into
a +2,125/−2,864 rewrite).

Corrections and narrowings: `4c39aa5` engine-only is **+727/−31 across 5 files** —
§237's +740/−32 counted the two docs files bundled with it. `v0.2.21` subtree counts
confirmed at 8,456 total (engine 2,898, docs 2,547, terminal 1,522, app 1,282, tools
165, `.github` 33, root 9). Two knobs narrow instead of replaying:
`STRATA_ADMIT_SHORT_FIRST` already exists upstream
(`v0.2.21:terminal/serve/server.py:450,454,456,1808,1809,1827`), so #656b's fork
delta reduces to the UNREAD-token ranking; upstream `verify.cpp` has **zero** hits
for `max_cells == 0` and `drop_batch_graphs()`, so both fork hunks stay fork-owned.
Two dead lines caught by the inventory: `run_parkv2_cuda_build.bat:7` still launches
`build\cuda-win\strata.exe` (L0 fix to `hydragen.exe`), and `mc3_patch.py` is one of
the 11 fork-only files — dead since §181 moved source registration in-tree, pruned at
L0.

Phase 0 is complete: P0.0–P0.7 each have an artifact on `main`
(`control-pre-0221.md`, `PHASE1_GATES.md`, `PATCH_STACK.md`, `UPSTREAM_DELTA.md`,
`LAYOUT_MIGRATION_0221.md`). Nothing was built, run, or measured; ms01 untouched.
Open before the first Phase 1 arm: the ms01 hygiene debt (stale `p2d1_arms.sh`
wrapper, 10 GiB `MemAvailable` against a 16 GiB conversation cache), and the §235
tail-slot pin fix, which remains unwritten and hard-gates Phase 4. counter 191.
