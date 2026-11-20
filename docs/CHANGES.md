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
