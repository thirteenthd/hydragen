# M-C3 investigation: disk-tier the parked conversation cache, with dedup

Status: investigation complete; not implemented. Date 2026-11-13, the reference host, cow build + STRATA_PARK_COW=1.

## Why

- Parked sessions are host-RAM only. Budget raised 6144 -> **8192 MiB** (live, battery lan-8192/8192b green, QC6 exact 97/101/103).
- Host RAM is the binding constraint at depth: with the engine up, MemAvailable is 9-10 GiB; the 100k ladder crashed the host once at the 6144 budget. The 8192 budget fits only ~4-6 deduped 100k siblings; LRU eviction then drops images and the harness pays measured 22-36 s re-prefills (Letta run) or 43.7 s cold admission (102,748 tokens).
- Eviction currently loses the image entirely. A spill tier turns "evicted" into "cold but restorable".

## What the engine already has (Strata v0.2.13, in our vendored chain)

- **On-disk session format v1** (`src/core/conversation_file.cpp`, 1256 lines, pure host code — no CUDA): STRSES magic, identity binding (model + resolved config hashes; foreign files refused before parse), corruption hashes, fail-closed read order, mode 0700 dir / 0600 files. Contents: running state + deepest checkpoint + every QSA layer's authoritative K/V + draft-layer K/V — i.e. exactly what a parked conversation holds. docs/DETAILS.md "Session files (disk)": ~1.2 GB per 63K tokens.
- **Slot SAVE/RESTORE API**: `--slot-save-path DIR`, `POST /slots/N?action=save|restore {"filename":...}`; guard knobs `--session-min-free-mib` (disk free to leave, default 4096), `--session-save-reclaim` (reclaim unpinned host caches if SAVE needs RAM), `--memory-limit-mib` (RAM floor honored by parking/save/restore, default 2560).
- **HARD BLOCKER (measured live, 2026-11-13)**: `501 slot save/restore is not available with parallel requests ("parallel" / --batch)`. The upstream tier is the legacy single-stream path; our canonical runs `--batch 2`. Probe with slot_save_path enabled: both save and restore return 501; knob removed from config as inert.
- Upstream files carry **no dedup**: every session file is a full copy.

## Economics measured on the reference host

- Disk: sequential write 2.5 GB/s (dd 2 GiB fdatasync to /home). A deduped 100k sibling image (1.35 GB owned, §111) spills in ~0.6 s; restore read ~1 s. Against ~20 s restore walls and 22-43 s re-prefill/cold-admission costs, the disk hop is noise — ~40-80x cheaper than re-reading the prompt.
- Capacity: /home 17 GiB free (97% used), /mnt/docker 6.4 GiB (98% — expert blobs live there). With the 4 GiB min-free guard, the usable tier today is ~13 GiB = ~10 deduped 100k siblings, vs ~4-6 in 8 GiB RAM. Real deployments need free NVMe; the tier is disk-bound, not bandwidth-bound.

## Design (two milestones, both opt-in, default OFF)

**M-C3a — spill/load wired into the parked ConversationCache (no dedup).**
Hook the parked LRU eviction: instead of freeing the image, write it as a session file (v1 format) keyed by conversation id; on a hit for an evicted conversation, load from file instead of re-prefilling. RAM budget accounting excludes spilled bytes; new disk budget knob reusing --session-min-free-mib. Eviction order: least-recently-parked first; spilled images keep their (small) GDN/PLE/checkpoint metadata index in RAM for identity + fast reject. Restore path already exists in conversation_file.cpp; the new part is the hook + keying + accounting inside the batch-capable cache. Knob `STRATA_PARK_DISK=DIR`.
Gate: battery lan green + park ladder 100k with a deliberately small RAM budget (e.g. 2048) + disk ON: fleet must survive LRU churn, restore 0/N fail, reused bytes exact.

**M-C3b — deduped session files (format v2).**
M-C2's ConversationBuffer segment directory is the ready-made dedup seam: the shared 66.5k head exists in RAM as refcounted 16 MiB segments already verified by memcmp (verify-then-view, §110). v2 file = manifest + segment list, where a shared-head **file is written once** (per-segment hash in the manifest) and siblings store (file, offset, len) views + only their own tail segments. On load: verify each view's segment hash against the manifest before adoption; any mismatch falls back to the sibling's own full copy (cow-m2fix fail-closed policy). Expected on-disk ratio = the §111 RAM ratio: 2.19 GB -> 1.35 GB per 97k sibling (842.8 MB exact prefix reuse).
Gate: byte-exact restore parity vs full-copy files; ladder fleet fits ~2.4x more siblings in the same GiB.

## Risks / notes

- Disk contention: the expert store is also file-backed on the same disks; spill bursts compete with expert refill reads. Mitigate: spill on idle/engine quiescent point; the 2.5 GB/s single-stream number bounds the worst burst at ~1 s per image.
- Files are bound to model+config identity — canonical relaunches with changed args invalidate spilled files (refused at read; treat as miss, never as corruption).
- Hashes are anti-corruption, not authentication: restore trusted files only (upstream's own wording).
- 8192 budget caution: MemAvailable floor ~1 GiB possible with a full RAM tier + full spill bursts; keep --memory-limit-mib (2560 default) armed — it already guards parking/save/restore.
- The slot SAVE API stays as-is (single-stream tools/users keep it); M-C3 does not touch that path.

## Decision point

On this box the win today is modest in capacity (13 GiB tier) but large in tail latency: it deletes the 22-43 s re-prefill cliffs the Letta/Gemini harnesses hit on eviction, and it removes the RAM ceiling that crashed the host once. If the fleet target grows past ~10 deep siblings, disk capacity (not code) becomes the next purchase.
