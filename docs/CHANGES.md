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