# HydraGen change log

Chronological record of every measurement → decision → commit in the program.
Newest first; section numbers are stable, so §N always refers to the same entry.



## 137. run12 C8 on KD3+#224b: wall 903.7 — FIRST tier arm to beat the tier-OFF baseline at C8; no crash
- run12 C8 harness: wall 903.7 vs baseline 951.2 (-5%), run9 KD3 1038, run6 full-copy 1284. needle 8/8, 48/48 requests, zero HTTP failures, zero engine death (no #224 sticky fault under C8 churn on the instrumented build).
- cold TTFT p50 122.1 / p95 264 / max 554 = best C8 cold admission ever (baseline 182/403; run6 193/784). warm TTFT p50 37.4. guard floor 2.7 GiB.
- Honest variance note: C8 wall on KD3 tiers at {1038, 904} across two arms; baseline single-sample 951. Claim kept as "tier neutral-to-positive at C8, big cold-TTFT gain", not "beats baseline" as a hard fact.
- Store again at ceiling (31.3 GiB, 2016 chunks); dedup skipped 0.0 on tail spills (small sessions) - big skips were mid-run.
- Crash status: 3 clean C-sweeps since forensics; #224b will name the fault if it returns.

## 136. run11 C4 clean on #224b build: no crash, needle 4/4; b2a adoption DEAD again (ADOPT=0, second clean confirmation)
- run11 C4 harness: wall 523.9 (vs baseline 501.0, run4 M-C3a 472.2 - inside noise band at C4), needle 4/4, zero engine death, guard floor 2.8 GiB.
- No #224 sticky CUDA error fired this sweep: crash is sporadic under C-churn, not every-run; the 2125391e instrumentation will name it when it returns.
- ADOPT=0 second clean C4 run: restore-side RAM-donor adoption is conclusively dead for deep traffic - at restore moment no RAM donor exists (siblings live-on-GPU or spilled). b2a code stays (harmless no-op, will matter for head-pinned designs) but the restore lever MUST be disk-lineage (parent chain + tail-only reads) or pinned heads.
- KD3 spill dedup contributed ~0 at C4 (few sibling re-spills); its payoff is C8 (run9 1038 vs 1284).

## 135. run10 C4 lost to engine exit 1 (same tail line as run8): park-COW path crash, NOT b2a adoption
- Supervisor log: engine exited code 1, last line "conversation cache: park COW anchor mounted ..." at BOTH run8 and run10 crashes. b2a adoption never ran (ADOPT=0 in run10) - restore-side adoption is structurally silent in deep C traffic: at restore moment the RAM tier holds no donor (siblings live-on-GPU or spilled). Adoption starves exactly where needed.
- Correction banked against 130: cross-session same-prefix KV bytes ARE equal (KD3 sibling skips 0.6 GiB + M-C2 "842.8 MB exact" reuse). Zero-dedup causes were boundary misalignment + chunk scale, not byte divergence.
- Restore savings therefore need disk-lineage parents (child file + parent chain, tail-only reads) or pinned head sessions; run10 invalid (engine death mid-sweep; killed-client 400s).

## 134. M-C3b2a landed: restored images re-join host-tier CoW (donor-adopted views on restore)
- cow-m3b / be71fc77, built green. New core fn conversation_kv_share_prefix: per part, segment-walk prefix_equal verify -> adopt_shared views of the RAM-parked donor (common_best match) -> owned tail re-appended; whole pages + complete indexer rows, strict prefix; no-op on divergence. New primitive ConversationBuffer::prefix_equal.
- Mirrors the M-C2 park-anchor rule on the RESTORE side: restored siblings own only tails; reused view bytes count 0 against the parked budget -> attacks both the MemAvailable floor and budget churn at 8-deep.

## 133. run9 C8 on KD3 (spill-side dedup only): wall 1284 -> 1038, over half of run6 regression gone
- Three-arm C8 now: baseline 951 / run6 full-copy 1284 / run9 deduped 1038 (+9% vs baseline, -19% vs run6). Needle 8/8, zero failed requests (summary requests=48), guard mem floor 3.0 GiB (run6 2.7), cold TTFT p50 128.6 s (best of all arms; warm store carried from ladders).
- 25 dedup lines during run9; store saturated the 32 GiB budget again (30.8 GiB deduped bytes + 92 drop lines): retention is now 1.3x deeper in sessions for the same bytes, churn persists at 8-deep -> b2a restore-skip and/or budget sizing remain the levers.
- Sampler park columns read null again (monitor curl; engine log is ground truth) - run9 sampler park fields unusable, counts taken from log.
- Verdict: spill I/O contention was ~half of run6 C8 regression; the rest lives on the restore side (full reads, preflight refusals) -> M-C3b2a is next.

## 132. M-C3b1d: dedup FIRE on real KV (0.6 GiB skipped per sibling spill) — the chunker was mis-scaled, not dead
- Root refinement of 130: equal spans are PER-REGION (~6.6 MiB per k/v region for a 66.5k shared head at int8), not per-image. Mean-4MiB/min-2MiB cuts could not land inside them. Mean 2 MiB + min 512 KiB (mask 21) does.
- Ladder N=6 (l7, KD3, store wiped to zero baseline): dedup lines 0.2/0.6/0.6/0.6 GiB on the big sibling spills; anchor mounted 66510 of 96598 tokens (RAM CoW fires in S-phase after all; earlier none-for-0 readings were other contexts). 12/12 QC green, battery lan-91 green.
- 614 MiB/sibling = 77% of the RAM tier 842.8 MB achievable; the gap is region-boundary straddle loss (~1 MiB x ~128 regions) + the uncuttable leading min-run. Spill writes 1.9 -> 1.3 GiB/sibling (32% less).
- Still standing: restore side reads FULL images (preflight refusals under pressure unchanged) -> b2a (donor-adoption read-skip) remains the next lever. C8 re-sweep with KD3 pending as the economics gate.
- Commit: cow-m3b / 75fc47b2.

## 131. Tier forensics: three real failure modes under pressure (run8 window), each with its fix
- RAM preflight load refusals: under C-sweep pressure (MemAvailable ~2.7 GiB, floor 2560 MiB), reading a 1.9 GiB image back fails the admit check by design -> restore refused -> re-prefill. This is a LARGE hidden cost in run6/run8 deep_warm classes, previously attributed only to I/O. Fix = b2a (donor adoption: read only the tail; the shared prefix costs zero RAM and zero disk).
- Cross-segmentation v2 incompatibility: run7-era ses files carry fixed-16MiB-grid chunk keys; the CDC reader recomputes keys under content-defined boundaries, mismatch -> fail-closed refuse+delete (safety correct, boot churn ugly). Chunker segmentation is part of the format: bump the envelope version on any chunker change (KD2->KD3 pending).
- Identity refusals ("saved with another model geometry") on some cross-boot files: identity = model-file fingerprint + config knobs; suspect cvec_digest or config_fp inputs differing across the supervisor-restart window; self-heals by file deletion; revisit with b2 (identity is doing real work and must not rot casually).
- Park CoW mounts DO fire in harness traffic ("mounted 8276 of 27539", "13 of 8387"): RAM-tier donor views work in mixed-size traffic; the ladder siblings ("none for 0") never share a PARK-time prefix because each was the first of its kind resident when parked.

## 130. PARK DEDUP VERDICT: content-addressed chunking is dead for this workload; lineage/donor views are the path (parkladder N=6 evidence)
- parkladder100k N=6 (shared 66.5k head, 6 siblings ~99k, ladder auth patched for the LAN key):
  ingest 6/6 QC green, restore 6/6 QC green, no engine crash (CDC binary), disk restores fired
  (restored 96-100k tokens, parked 3->0).
- Every sibling spill added ~110 NEW chunks, 0 skipped, under BOTH fixed-grid and CDC chunkers,
  cross-sibling AND cross-generation. The KV bytes for the same prefix tokens are NOT byte-equal:
  separate prefill geometries (chunk boundaries, co-read batching) produce numerically different
  rows, and int8 quantization amplifies that to full-byte divergence. This is exactly why M-C2
  dedups by token-prefix match with memcmp verify, never by content hash (closed in M-C0).
- Consequence: M-C3b1/b1c (content chunking) cannot deliver byte savings on a real engine.
  The chunk machinery is correct (readers/refcounts/fail-closed verified) but dedups nothing here.
  It remains as store hygiene (identical generations DO dedup when content is truly equal, e.g.
  restored-unchanged sessions re-spilled) - measurable, small.
- The fix for the full-copy disk cost is M-C3b2 as designed: restore-time donor-view adoption
  (token-prefix match into a live RAM session, M-C2 matcher) so a restored sibling mounts the
  shared head from the donor and reads only its tail from disk, AND donor-referenced spills
  (a spilled session with a live CoW donor serializes only segments it owns). Restore then
  preserves the RAM CoW accounting (fixes the MemAvailable 4.3 -> 2.7 GiB collapse).
- Park COW anchor: "none for 0 of 96598" when no sibling is resident (all spilled - expected);
  "mounted for 65895 of 102340" when a donor exists (run8). Mechanism works; only it moves bytes.

## 129. M-C3b1 fixed-grid dedup structurally missed; content-defined chunking landed (cow-m3b); run7/run8 invalidated by process accidents
- run7 (fixed-grid V2, 142 spills): 0 dedup skips, 0 saved. Root cause: the payload
  grows internally (ids list, per-layer KV rows), so fixed 16 MiB grid boundaries shift
  on every re-park and no chunk key is ever seen twice. Fixed-grid v2 is retired from
  the benchmark narrative (it is format-correct: readers, refcounts, fail-closed all worked).
- M-C3b1c: content-defined chunking (gear hash, bounded window, check from 2 MiB,
  mean 4 MiB via 22-bit mask, 16 MiB forced ceiling) landed on cow-m3b, built green,
  committed. Battery lan-89/lan-90 green on the CDC binary.
- run7 and run8 are INVALID for wall comparison: run7 had an accidental second sweep
  (redaction-mangled runner auth) and run8 an engine crash mid-sweep (exit 1, last line
  "park COW anchor mounted for 65895 of 102340 tokens") plus supervisor restart picking
  up the freshly built binary. run5/run6 remain the only clean tier pairs.
- Two engine boot scans refused every v2 file present ("a chunk file is missing or
  truncated") and fail-closed deleted ses + chunks (2091 swept). Not yet root-caused;
  the boot-refuse storm is the #1 suspect for any future tier loss and must be
  reproduced in a lab (fresh tier dir) before the next sweep.
- Decisive experiment running: parkladder100k N=6 (4->6 siblings, shared 66.5k head)
  forces cross-sibling disk dedup. If dedup works, siblings 2..N skip ~1 GiB each.

## 128. M-C3b1 landed: deduped block files for the park-disk tier (STRATA_PARK_DISK_V2)
cow-m3b / 10f90c7e, built green (356/356). The tier now writes a STRP/KD2 envelope whose payload is
exactly the STRSES v1 byte stream: session_payload_blocks walks put_payload through host buffers and
hands ParkDisk (offset,ptr,len) blocks; blocks >= 64 KiB are published as content-keyed chunk files
(128-bit key = two SessionHasher seeds over a 16 MiB block) under park-disk/chunks/, everything else
stays inline in the file body; a 32 B-per-entry table (<= 2048 entries, 8 GiB payload cap) records the
order. A sibling of a parked fleet now spills only its unique tail: the shared prefix chunk set is
written once per fleet, refcounted per table occurrence, unlinked at the last holder, orphans swept at
boot. Restore feeds the SAME bounded parser through session_payload_unblocks and RECOMPUTES each
chunk key while its bytes serve it: a mutated or missing chunk fails closed and deletes the ses file.
Any structural refusal falls back to a plain v1 write for the same image, and readers speak v1 and v2
side by side regardless of the write knob, so the existing 26-file fleet stayed indexed untouched
(0 refused). POSIX-only for now; the knob forces v1 on Windows. Default OFF pending the C8 discriminator.
Serve layer: ParkDiskLog parses the v2 activation, dedup and sweep lines into /metrics (v2, chunks,
chunk_bytes, saved_mib); the Park disk tier card gains a "Block store" fact row (lan-87 green with the
knob off = v1 path unchanged on the new binary; lan-88 green with v2 active, restores cross-boot at
0.6-1.2 s). run7 = C8 harness sweep under V2 with chunk-store sampling: the bytes-per-spill verdict.

## 127. run6 C8 at 32 GiB tier: failures gone, but full-copy I/O is net-negative at 8-deep (verdict on the discriminator)

run6 (mc3b/, harness C8, tier budget 32768, everything else = run5): wall 1283.9 s vs baseline 951.2 vs run5 1019.4. Needle 8/8, zero failed requests (run5 at 8 GiB tier: 13/48 failed). Guard min MemAvailable 2.7 GiB (base 4.3, run5 3.0). Tier counters over the run: 118 spills = 140.9 GiB written, restores read back 121.8 GiB, 18 budget drops at the 30-entry/31.9 GiB ceiling; boot scan was 4 files.

Verdicts: (1) run5 contamination confirmed - the 13 zero-output failures were tier-budget churn (files dropped mid-fleet, restore cascade, starvation); a roomy tier makes them vanish. (2) With the failure mode gone, the real M-C3a C8 cost is visible: full-copy spill/restore puts ~263 GiB of disk traffic on the same LV the expert blobs stream from, and at 8-deep the contention costs more than the ~1.1 M tokens of re-prefill the tier saved (+35 percent wall). Cold TTFT p95 783 s. (3) Warm decode still improves at C8 (13.8 to 22.4 tok/s), and C4 remains a clean win (501 to 472 s) - the tier pays off below ~6-deep, loses above it on a shared volume. (4) Spills serialize the whole logical image (2.19 GiB/sibling) even when park CoW owned only 1.35 GiB, and restores never re-form CoW views - so spill volume is 1.6x the owned-bytes floor and every restore re-private-izes the shared head (MemAvailable 2.7 vs 4.3 floor).

Decision: M-C3b promoted from optimization to requirement - segment-view spill files (unique tail + state, ~1.35 GiB not 2.19) and restore that adopts memcmp-verified donor views instead of full takes. Interim tuning is exhausted: retention is not the lever (32 GiB already eliminated drops-as-failures); bytes-per-spill is.
## 126. Disk cleanup (277 GiB freed) + disk tier budget 8 to 32 GiB + C8 rerun under way

User-supplied plan (verified before deletion, not taken on trust): config path scan of the live cow config (only ~/.models_backup/Strata-data, ~/hydragen-experimental, ~/strata-cow referenced), lsof across all 16 candidates (zero open handles), no systemd unit references, strata-0213 local commits checked one by one against strata-cow (9/9 resolve there, zero unique work), HF source checkpoints confirmed present (~/models, ~/hf, /mnt/docker/hf-dl) so every old blob store is re-derivable. Deleted: hg-store (113G), hg-store-iq3s-blobs (47G), hg-store-iqxxs-blobs (40G), hg-store-dense-iq2, hg-store-dense-iq, hg-draft-mtp, q4e_w3, q4e-m1, q4e-mini-bak, vllm, vllm-env, sg-env, hg-install-test, strata-src, strata-0213, strata-packs. /home went 98 percent used to 39 percent (287 GiB free). Note: this removes the SGLang-vintage fallback substrate; regenerable from HF sources but not adopted anywhere since M25 parity, and the public repo never shipped it.

Tier budget raised via lan_cow_switch.sh: STRATA_PARK_DISK_MIB 8192 to 32768 (guard keep-free 4096 unchanged). Boot scan revalidated 4 files 7.2 GiB 0 refused. Battery lan-86 green (QC6 exact 97 101 103; answers 2.3-2.9 s via cross-boot disk restores).

C8 rerun (run6, mc3b/): the discriminator for the restore mechanism question (user framing: restores are not delta-only and inflate RAM ownership). M-C3a is a full-copy tier by design: spill writes whole images, read-back materializes whole images as private takes (disk entries never borrow park CoW), so restores re-private-ize shared prefixes and push the RAM cache toward exactly the LRU churn + MemAvailable cliff seen at C8. Rerun at 32 GiB disk budget isolates that: if failures persist with a roomy tier, the mechanism is restore-side RAM ownership, not disk capacity, and M-C3b segment views (restore = adopt verified donor views + read only unique tail/state) become the fix. run6 samples MemAvailable, GPU MiB and park_disk counters every 15 s to mc3b/run6_resources.log.

## 125. run5 C8 arm contamination: 13/48 tier-ON requests failed outright (correction to 124)

Dashboard recent-requests review (user screenshot): the red Error rows are run5 C8 traffic, not live serving. Client jsonl forensics: mc3/conc8_harness_C8.jsonl has 13 rows with no TTFT and 16 with no prompt accounting; baseline conc8_harness_C8.jsonl has zero. Engine log shows 12 of those completing with 0 or 1 generated tokens after full prefill (4x zero, 8x one; the 13th is the 437 s starve). VRAM-free warnings (391/261 MiB at request start) fire during C8 - but also 17 times under normal battery traffic since restart, so the warning is endemic to the full card, NOT tier-specific; the 0/1-token outcome is what is new vs baseline.

Consequence: C4 tier comparison stands (0 failures both arms). C8 tier row is contaminated - treat wall 1019 vs 951 as unmeasured until a rerun, not as a regression finding. C8 failures correlate with full-take restores: M-C3a disk restores mount images without park CoW donor sharing, so 6+ deep restores at 8-deep churn inflate the resident working set that baseline dedup kept shared.

Actions banked: rerun MC3 C8 with park_disk counter + free monitoring logged per request; sweep v4 must count HTTP-level failures explicitly in the summary instead of surfacing only via needle counts; M-C3b segment-view files become the fix for restore-time sharing. README C8 note corrected in public sync.

## 124. Dashboard disk-tier visibility + first park-disk comparison sweep (cow-m3 + 1b1d4001)

UI: Conversation cache card renamed to RAM; new Park disk tier sibling card (entries, bytes/budget bar, spills/drops/read-backs/fallbacks, last switch); .cache-pair grid puts RAM left / disk right at desktop width, RAM full-width when the tier is off. Serve-side ParkDiskLog parses the engine park-disk stderr lines into /metrics park_disk (engine gained one drop log line, 1b1d4001). Public chain: 5bfdcfde8 + 74abe296a via git am --directory=engine, dashboard commit 38e3150a9/1e0171b5c. Browser-verified live: RAM x=24, disk x=841, same row, 4 entries 7.2/8.0 GB, 8 read-backs.

Battery lan-85 green, QC6 exact 97,101,103 — and for the first time every QC ran 2.3-2.8 s because each QC restored the 104,402-token battery-doc image from disk (1.6 s read) instead of re-prefilling: cross-boot warm battery as a side effect of the tier.

Comparison sweep (conc8_run5.log, mc3/): MC3 C4 vs v3 C4 — wall 472.2 vs 501.0 s; deep_warm 4 vs 5 events, 411k vs 505k prefill tokens; warm decode p50 16.0 vs 8.0 t/s; max stall 91 vs 108 s; needle 4/4. MC3 C8 vs v3 C8 — wall 1019 vs 951 s (worse), one turn-0 request starved 437 s with zero output (needle 7/8 = empty completion, not a wrong answer; restored sessions answered correctly).

Taxonomy finding: engine log shows 6 disk restores of 98-107k tokens (0.8-3.6 s reads) inside the C8 window, but usage.prompt_tokens_details.cached_tokens stays ~0 for disk-restored admissions — the client classifies them deep_warm_reprefill, inflating apparent re-prefill. Real C8 re-prefill was roughly 716k - ~600k restored = ~116k. C8 wall regression hypothesis: 11 spills (~15 GiB writes) + 4 RAM-preflight refusals on the nearly-full /home LV contend with expert-blob streaming during 8-way cold prefill; C4 (smaller tier churn) wins cleanly.

Actions: sweep v4 must diff /metrics park_disk.restores per request to add a deep_warm_disk_restore class; C8 needs spill-churn reduction (M-C3b segment views shrink images ~2x) or tier on a quieter volume. min_memavail 2.9 GiB during C8 (vs 4.0 baseline) — watch the floor.

## 123. M-C3a landed: park-disk tier under the parked cache (cow-m3 / c9d6fc44)

- ParkDisk module (include/strata/core/conversation_park_disk.hpp + src/core/conversation_park_disk.cpp):
  evict_oldest() hands the FULL victim image to the tier before its RAM goes; admission with no usable RAM
  match reads the deepest disk prefix back and restores through the normal conversation-snapshot take path.
  Reuses STRSES v1 (conversation_file.cpp): model+config identity fingerprints, bounded parse, payload hashes,
  RAM preflight, temp+rename publish, folder 0700 files 0600. Knobs: STRATA_PARK_DISK=DIR (default OFF),
  STRATA_PARK_DISK_MIB (budget), STRATA_PARK_DISK_MIN_FREE_MIB (keep-free, default 4096). Pinned prefixes and
  the guarded donor never reach the tier; drop_superseded/clear never spill. Boot scan revalidates every file
  and deletes foreign/corrupt ones (warm boot: index rebuilt, no RAM parked images until a request reaches them).
- Gates on the reference host, canonical cow build + STRATA_PARK_COW=1 + STRATA_PARK_DISK on (8 GiB tier on
  /home, keep-free 4 GiB):
  * battery lan-83 green (QC6 exact 97,101,103) tier ON fresh boot.
  * 6-sibling 102k forked-agent gate (bench_4x102k_shared.py range(6)): 22 LRU evictions spilled
    (1.2-2.1 GiB files in 2.9-3.9 s, ~300-600 MB/s on /home), 5 disk restores read back in 0.87-1.6 s,
    all streams completed 256 tokens, decode 9.8-113 t/s.
  * fail-closed exercised for real: 2 restores REFUSED by the RAM preflight (2.1 GiB read + 2560 MiB floor
    with ~4.6 GiB avail) -> fell back to reading the prompt, no partial state.
  * race safety: no crash under --batch 2 + 6-deep churn; park COW anchor-lost fallback held (cow-m2fix path).
  * boot scan with 4 tier files (7.4 GiB): 0 refused, index rebuilt in 7.0 s; battery lan-84 green after that boot.
- Measured economics: spill write ~3 s, disk read ~1 s, then the usual ~20 s state remount; vs ~44 s cold
  re-prefill at 100k. Break-even is immediate; the real win is harness deep-warm turns (the 505k/712k wasted
  prompt tokens from sweep v3 harness C4/C8).
- Comparison sweep launched (harness C4+C8 arms with tier ON, tags in mc3/, log conc8_run4.log) - the README
  comparison rows the owner asked for come from those numbers.
- Known v1 inefficiencies (M-C3b): full-copy spills (a re-spilled conversation replaces its file by LRU, but
  stale superseded files can sit until the budget evicts them); spill path does not reuse M-C2 segment views;
  disk entries never participate in borrows, only full takes.


## 122. archive refs deleted from public remote

- Owner decision: provenance stays local. Remote deleted: archive/experimental-main, tags experimental-final-20261112, archive/feat-* , v0.4.0 (it pointed at old-main tip 72f825d0 and kept ~150 unscrubbed commits addressable).
- hydragen-engine fork repo: confirmed gone (GraphQL: repository not found).
- Residual: refs/pull/1 (merged PR #1, 15 early commits, README mentions the reference host x3) is not deletable via API; contents are early src-refactor code only. Full removal = GitHub support request or private-repo toggle; noted for owner.
- Local insurance: all deleted refs fetched under refs/tags + refs/local/pr1 in the working clone; the reference host repo retains full history.
## 121. conc8 deep sweep complete; Windows install.ps1 fully verified; public docs privacy scrub

- conc8 sweep (client-side SSE timing, needle QC on every arm): natural C1 TTFT 37.3s decode 100; C2 8.6/stream; C4 8.3; C8 TTFT 219/358. harness C1 TTFT 2.5 decode 74 (park CoW warm reuse); C2 2.5/8.3; C4 41.2/95 decode 1.0; C8 72.6/282.7 decode 4.1. min MemAvailable 4.0 GiB. README gains "Measured expectations": C1 comfortable, C2 hard ceiling, C>=4 deep = dont.
- Windows: install.ps1 (vcvars64 activation, quoted -D args, --target strata) BUILT strata.exe end-to-end on the Windows dev host. Product deployable on Linux AND Windows from a clone.
- Public docs privacy scrub: full product history rewritten (filter-branch tree filter): hostnames->"the reference host", user paths->~, LAN IPs->"the-LAN-address", home domain->"the home LAN domain", agent nick->"the agent"; law file renamed source-build-12g-ampere.md; sweep paths env-parameterized (HG_CFG/HG_OUT). engine/ untouched (upstream blobs). LAW: every server->repo doc sync must pass the scrub mapping first.
- OPEN: archive/experimental-main + experimental-final tags still UNSCRUBBED public refs (old bench internals + 7 drop-in zips) - awaiting owner decision.

## 120. conversation cache raised to 8192 MiB + M-C3 park-disk-tier investigation
- canonical lan-cow boot now --conversation-cache-mib 8192 (was 6144); /metrics budget_mib:8192; battery lan-8192 + lan-8192b green, QC6 exact 97/101/103. MemAvailable with engine up: 9-10 GiB.
- upstream v0.2.13 already ships an on-disk session format (conversation_file.cpp STRSES v1, identity-bound, fail-closed) + slot SAVE/restore API + session guard knobs. MEASURED BLOCKER: save/restore returns 501 under --batch ("not available with parallel requests") - the tier is single-stream legacy, incompatible with our concurrency shape. knob slot_save_path probed and removed as inert.
- disk economics on this box: 2.5 GB/s seq write; 1.35 GB deduped image spills ~0.6 s, restore read ~1 s vs 22-43 s re-prefill/cold-admission. /home 17 GiB free -> ~13 GiB usable tier behind the 4 GiB min-free guard (~10 deduped 100k siblings vs 4-6 in RAM).
- M-C3 design banked in docs/M-C3-park-disk-tier.md: a) spill hook inside the parked ConversationCache eviction path (batch-capable, v1 format, STRATA_PARK_DISK knob), b) v2 deduped files = M-C2 segment views serialized (write shared head once, siblings view+tail, per-segment hashes, fail-closed full-copy fallback). gates: battery + ladder100k with small RAM budget + byte-exact restore parity.

## 119b. Monitor layout: Concurrency States replaces Model State; Speed shows aggregate

Per user spec, matching the pre-experimental HydraGen dashboard: slots-dashboard-card is now the first Monitor card; Model State card removed from index.html (its state-label/detail/bar updates kept null-safe for stale caches; #state-badges loop inert). Speed card = current AGGREGATE decode: server _tok_s() already sums all parallel streams, verified live tok_s=85.7 with two 42 t/s slots; sub-label "Decode now - sum of N streams". Slot tiles now show true per-slot tok_s/generated (slots_view carries them under load; idle slots show held_tokens). Browser-proven under 2 streams: top card Concurrency States 2/2 Active, Speed 79.0t/s sum-of-2. Deployed to live serve/web; pushed to hydragen@product.
## 118b. Concurrency States panel rebuilt to match the pre-experimental HydraGen dashboard

User showed the pre-experimental HydraGen UI (old 3060 host): top Concurrency States card with per-slot live tiles + Queue sub-card. My minimal conc-card was not what was expected. Ported the full original panel from tag experimental-final-20261112 (calcConcurrencyAndQueue + slot-tile grid + queue tile) onto 0.2.13 payload (slots[] state+held_tokens; live.tok_s/prefill_tok_s_mean fallbacks; queued/outside/waiting derived). Branding: dashboard title + header now HydraGen (Strata attribution stays in engine/LICENSE + NOTICE). Deployed to live serve/web, browser-proven during 4 concurrent 1500-token streams: "Generating (2 queued) | 2 / 2 Active", Slot 0 43.2 t/s 378 tokens, Slot 1 41.9 t/s, Queue 2 Queued. Product commit pushed.

## 119. index/monitor HTML stale-cache hole fixed; Gemini bench audit (post-fix picture)

User still saw no concurrency panel: server served patched app.js (no-cache) but the index/monitor HTML routes sent NO Cache-Control - browser kept old DOM, renderConcurrency guard skipped silently. Fixed: Cache-Control: no-cache on both HTML routes (live serve tree + product engine commit, pushed). Proof: fresh browser opens dashboard, card renders "0 / 2" + two slot pills with held tokens; /metrics confirmed (slot 0,1) = (decoding,decoding) under two concurrent streams. Gemini 17-request audit: zero errors/rejects; park dedup engaged (899,407,872 B shared views byte-exact across three 103k siblings, anchors mounting); wild TTFT 0.6-91.7 s + decode 11.8-167 t/s = capacity edge, not a bug: 4 identical 103k prompts sit ~4% past the proven 4x99k 97%-fit point so the parked cache oscillates (evictions 14), and two co-resident 103k decodes thrash the 20,480-cell KV window. Harness guidance: incremental histories (Letta hit 89% reuse) or cap 100k-fleet concurrency at 3 / raise --conversation-cache-mib.


## 118. park COW SIGSEGV fixed: anchor re-resolved after the admission evictions (cow-m2fix / 57bf5844)
- Bug: the live canonical STRATA_PARK_COW=1 arm (strata-iq3xxs-m25-0213-ctx131k-lan-cow.json, :8181) died with exit -11 (SIGSEGV, not OOM; free -m 60.4 GiB after) right after logging `park COW anchor mounted for 70977 of 103057 tokens`. Repro shape: /tmp/bench_4x102k_shared.py (4 concurrent ~102k siblings sharing a ~71k system head + 32k unique tails, parked budget 6144 MiB), Phase 3: agent_2 evicted, cold 103057-token prefill (2616 tok/s, 4 checkpoints), its park mounted a sibling anchor, crash.
- Root cause (static audit; no gdb on the reference host): park_source (generate.cpp ~12236) mounts `anchor_img = conversations.find(m.serial)` — a raw pointer into the parked deque — then `make_room(estimate)` (the estimate is the UNCAPPED full-snapshot size, ~5.9 GB at 103k tokens, so parked bytes near the 6.44 GB budget force sibling evictions) and the physical-RAM admission `evict_oldest()` loop both run before `conversation_snapshot_save`. If either erases the donor, the per-layer `conversation_kv_save(..., &anchor->kv[j], ...)` reads the freed SavedConversation -> SIGSEGV. Post-adoption donor eviction is refcount-safe (shared_ptr segment stores, the M-C2 eviction law); only the PRE-CAPTURE window dangles. The 100k parkladder missed it: the dedup-on fleet fit the budget at 97% so make_room evictions never overlapped a deep mount; a shared-head fleet at the budget cap makes the overlap routine.
- Fix (cow-m2fix, 57bf5844 on cow-m0 over cow-m2 a53973bf): record `anchor_serial = m.serial` at mount; after the admission evictions and immediately before the capture, re-resolve by identity (`find(anchor_serial)` + geometry recheck `!kv.empty() && live.ids >= anchor_tokens`); gone means no anchor - the park takes the full verified copy. New telemetry: `park COW anchor lost: the donor left the cache while admitting this park`. Incremental rebuild 2 targets; patched exe sha256 1be778a3..., pre-fix binary banked as build/strata.pre-parkcow-fix (798e0e02...).
- Verification (canonical :8181, ctx 131k batch 2): bench on the PRE-FIX binary ran 49 anchor mounts + 13 evictions without crashing (the window is timing-dependent: donor must be among that park's eviction victims) - the original crash stood as the pre-evidence. Bench on the PATCHED binary: all 9 legs BENCH_EXIT=0, engine alive, and telemetry caught the exact window twice: `anchor mounted for 70977 of 103057` -> `anchor lost` -> `parked 103057 tokens ... evictions=10/12` completed instead of dying. battery_quality_m25.py tag cow-fixed-1113: QC0-QC7 green (QC6 97/101/103 exact, QC7 FASTA '>' + GTIN intact), QC_DONE.
- Pitfall banked: the `anchor mounted` line as the last log line = this crash's signature; `anchor lost` = the window fired and was handled (same reading discipline as the R-PHASE `anchor none` - never a dedup failure). Bench copy kept at ~/hydragen-experimental/bench_4x102k_shared.py (/tmp is wiped by any hard reboot).


## 117. measured: 4 concurrent unique 102k streams on the canonical cow arm (STRATA_PARK_COW=1)
- Shape: m25_depth_b4.py (hydragen-experimental/): 4 simultaneous streaming legs, fully-unique ~102.2k-token prompts (needle recall at ~50% depth + cross-stream contamination probe), 400-token answers, live canonical :8181 (ctx 131k batch 2), ran alongside live agent traffic. Runner logs logs/depthb4_13_1443 (94.6k mis-calib leg) + logs/depthb4fix_13_1448 (headline leg).
- Headline leg: depth 102,172-102,257 tok/leg, cold reused=0 all 4. Admission strictly serialized: TTFT 69.7/116.8/164.7/211.7 s (~48 s waves, plus live-traffic interleaving); cold prefill at depth 2158-2680 tok/s. Co-resident decode 8.5-9.2 tok/s/stream (batch windows avg ~1.8 rows; prefill-ring loan dominates placement mid-run: routed VRAM 17-35%, PCIe up to 35%, CPU up to 48%; batch-window MTP accept 67-100%). KV streaming is NOT the depth wall: 67.7-68.7% of block reads hit the 20480-cell VRAM window at 102k depth, ~55 MiB/round from RAM. Total wall 237.4 s; QC 4/4, contam 0.
- Park/CoW at depth: parks 210-436 ms per 102k leg; parked image ~2.15 GB -> the 6144 MiB budget LRU-churns under a unique 4-fleet (parked count oscillates 2-3, evictions +16 across the run); reused_kv_bytes anchor-only (202,752 B = 16 aligned tokens) — unique fleet has no dedup opportunity, as expected. Host MemAvailable floor 4 GiB.
- Contrast observed same run (live traffic): a warm 104k session restored "104105 reused + 1 read" -> 91.5 tok/s decode, 283/291 drafts, KV streaming 95.3% hit. At 100k+ depth the warm checkpoint path carries concurrency; a cold unique fleet pays serialized prefill waves + 6.5-9 tok/s co-resident decode.
- Pitfalls banked: prompt-length calibration is tag-width sensitive (6-digit probe seed inflated tok/line by ~2 -> 7% short of target; calibrate with matching digit width or override nlines from measured L). urllib.request.Request(url, dict) sends the dict as the POST body — pollers need Request(url, None, headers).

## 116. install.sh - source-build installer, verified end-to-end from the public clone

Product need: people who clone hydragen must be able to build. Upstream engine ships setup.sh but that is the release-binary runtime path - useless for a vendored fork (release binaries lack the CoW chain). Wrote install.sh at repo root: CUDA autodetect (/usr/local/cuda* highest with nvcc; proven box recipe = /usr/local/cuda-13.1, cmake+ninja from pip into .venv), upstream cuda-compile release flags, default arch sm_86 (3060 Ti/3080 Ti), HYDRAGEN_CUDA_ARCH override, builds all non-test targets, generates hydragen-local.json (exe+log rewritten). .gitattributes forces LF for .sh/.py (Windows checkout CRLF would break the shebang). TEST: fresh clone of product on the reference host (~/hg-install-test), bash install.sh: CUDA detected 13.1, engine/ built (379 targets), build/strata --help exit 0, local config exe path correct. README: new Install section (blobs + serve + knobs truth: arenas/fork always on, PARK_COW recommended, join/shared off) + changelog counts 120.

## 115. canonical switched to cow build + STRATA_PARK_COW=1 (product default ON)

User approved after Letta-harness bench audit: three 100k agent turns paid 22-36 s re-prefills (LRU eviction, reused=22272 anchor) - the known COW-off parked-budget cost from 111. Canonical LAN engine now runs ~/strata-cow/build/strata with STRATA_PARK_COW=1 (forkb4 chain + parked dedup ON in serving). Config: strata-iq3xxs-m25-0213-ctx131k-lan-cow.json (exe swapped, log lan-cow, key unchanged, 600). Switch script: ~/hydragen-experimental/lan_cow_switch.sh (script-file pkill law respected). Battery lan-cow QC0-QC7 green, QC6 exact 97,101,103. expert_slots 2456 -> 2451 (noise; canonical win is host-tier parked dedup, not slot gain at b2). Env verified on strata-host proc. Battery patched to send BATT_KEY bearer header (bak at .py.bak).


## 114. dashboard concurrency panel ported to the product engine

User: default Strata UI shows no concurrency. Found the old HydraGen dashboard in the archived history (tag experimental-final-20261112, serve/web/app.js: calcConcurrencyAndQueue, per-slot state cards + queue sub-card, vLLM/TRT/Strata era). New engine's /metrics already carries everything (engine.batch_slots, live.slots[state,held_tokens], running, queued, outside_slots, waiting, admitting). Ported the panel fresh onto 0.2.13 Monitor (renderConcurrency + conc-card in index/app.js/app.css), not a copy-paste (old app.js is a different lineage). Deployed to ~/hydragen-experimental/strata/serve/web by scp; static files serve from disk - no engine restart needed; verified GET /web/app.js carries renderConcurrency and / carries conc-card. Repo commit on product: serve/web concurrency panel.


## 113. engine consolidation: hydragen-engine fork merged into hydragen product branch (subtree-vendored Strata v0.2.13 baseline + the six cow commits re-applied via git am under engine/), submodule dropped, README/NOTICE updated to vendored-attribution wording (upstream commit recorded in the vendoring merge), hydragen-engine repo deleted


## 112. the reference host hard crash during §111 notarization (host-RAM pressure from the 100k ladder; reboot cleared all pinned pools), /tmp wiped - runners rebuilt as product versions; canonical ctx-131k reboot HEALTH:200 battery green QC6 exact 97,101,103; product cut published: orphan branch product on thirteenthd/hydragen (README results table, NOTICE + engine attribution to Niko1221/Strata MIT, engine submodule -> thirteenthd/hydragen-engine@hydragen-cow = v0.2.13 + 6 CoW commits, configs, gate suite, 116-section changelog + FINAL_REPORT + laws); default branch flipped; old history frozen as tag experimental-final-20261112 + archive/experimental-main; full-history secret scan clean

## 111. 100k-depth park-dedup validation: anchors mount at exactly the shared head (66,510 tok), 2.19 GB parked image -> 1.35 GB owned (-38.5%, reused_kv_bytes exact to the byte); a 4x99k fleet exceeds the 6.4 GiB parked budget WITHOUT dedup and fits at 97% WITH it; host avail floor 2 -> 4 GiB; both arms restore 0/4 fail; final benchmark report banked (restored verbatim after the crash lost the un-fsynced append)

## 110. M-C2 host-tier parked-session dedup LANDS: verify-then-view on the ConversationBuffer segment directory — sibling parks pay owned-tail bytes only (395 MB vs 577 MB per 22.3k sibling); battery+farm green, ladder restore 0/6 fail; commit a53973bf

The last lever of the CoW campaign is the only one that never touches VRAM: parked conversations
are host-RAM ConversationBuffer images (16 MiB segments), and a fleet of sibling subagent
conversations sharing a system-prompt head duplicates that head's K/V once per sibling.

Mechanism (`STRATA_PARK_COW=1`, default 0 = today exactly):
- Donor search is a **longest-common-token-prefix scan** over parked entries — `ConversationCache::best()`
  is whole-conversation-prefix by design and never matches siblings.
- Adopt is **verify-then-view**: `memcmp` the session pool bytes against the donor's segments over the
  shared range BEFORE any mutation; mismatch leaves the buffer untouched and the plain copy runs.
  Honors §103 (heads prefilled under different chunk plans are not byte-identical).
- Views hold the donor's `store` shared_ptr with `cap = 0`: they **bill 0 bytes** — the cache budget
  charges shared bytes once, to the donor. Write visits into borrowed segments fail closed.
  Restore/verify/disk-save read through `visit()` and needed zero changes.
- Views flatten to the original donor store (view-of-view adopts the underlying store) — multi-generation
  sibling chains share one copy.

Pitfall banked (two SIGSEGVs): the **indexer `pooled` pool is device-resident** (and `kv_mode 0` pools
can be too) — CPU `memcmp` on them segfaults; small anchors hid it because pooled rows = at/idx_block = 0.
Adopt is now gated `!on_device(src[i])`; device-resident parts fall back to the copy path.

Gate (parkc4 arm, batch 2, ctx 131k):
- battery QC_DONE green; farm C1-C4 all `qc_primes=True` (C1 162.4, C4 agg 86.8) — zero serving regression.
- park-ladder (6 siblings sharing a 14,360-token head, 22,348 tokens each):
  | metric | COW off | park CoW |
  |---|---|---|
  | per-sibling parked image | 577.4 MB | **395.5 MB owned** (reused_kv_bytes = 181,969,920 = 14,360 tok x 12,672 B — exact) |
  | park wall | 164-192 ms | **58-61 ms** (3x less copy work) |
  | ladder host-RAM delta (6 parks) | ~3.4 GB naive | **646 MiB** |
  | restore after eviction churn | — | **restore_qc_fail = 0/6** |
- 31.5% parked-byte saving per sibling at 64% head share; the prize scales with head share:
  a real agent fleet (one 20k head, hundreds of sessions) converges to head-once + tails.

Economics: unlike M-C1d/e (VRAM: pinned names stole working set; region slots competed 1:1 with
experts), host-tier views have zero VRAM interaction — pure host-RAM and park-CPU win.
Eviction law holds: donor eviction frees only its own bytes; views keep the store alive by refcount,
so shared bytes are never double-freed and never double-billed.

The CoW campaign is now complete across all three tiers:
M-C1c forkb4 (VRAM admission fork, banked champion §107), M-C1d/e (VRAM fetch/region sharing,
closed negative §108/§109), M-C2 (host parked dedup, LANDED). All opt-in, canonical default unchanged.

## 109. M-C1e per-lineage shared region: race-clean and floor-guarded, but farm-negative at expert parity — the cold-head tax is resident capacity, not duplicated fetch traffic; the CoW campaign closes at M-C1c/forkb4

Built the lineage shared region (commit `93f218da`, branch `cow-m0`): arena tail region
`[arena_slots, arena_slots+STRATA_COW_SHARED)` owned by no session window, per-lineage hint
tables (`lin_all`, 16 rolling tabs per arena, block -> region slot, -2 fetching), region-local
clock arrays (`sblk/sst/srf`), a region sweep inside `resolve_kernel` whose candidate predicate
is `slot_use == 0` — the PIN LAW boundary by construction: no live table names the slot, so no
gather can be in flight when it is claimed (`CAS(use, 0, -2)`). Adoption is the M-C1d prev-gated
add + revalidate protocol; publish order is refcount, entry, fence, hint, so a joiner that sees
the hint always finds a pinnable refcount. All three gates battery-green with QC6 exact.

New laws banked the hard way:

- FLOOR LAW: fork siblings generate diverging tails, so blocks at or above the fork floor
  (`P = upto / 4`, complete pages) are NOT byte-identical across the lineage. Sharing them (join,
  splice, region adoption) adopts another session's divergent bytes = corruption. The patch adds
  a per-registry-entry `floor` lowered by every fork and guards EVERY sharing path with it. This
  also closes a latent M-C1d hazard: its join/splice had no floor check and was only saved by the
  farm's uniform fork shape.
- SPLICE LAW (tightened): the admission splice now drops every foreign entry to cold; a table may
  only name slots the holder or the fork-join kernel counted a ref for. Siblings re-adopt region
  pages at resolve in one step. Closes the un-pinned foreign-name eviction race.
- WINDOW FLOOR: a streamed slot window must be >= the resolve block width (1024 slots, i.e.
  STRATA_COW_SLOT_CELLS >= 4096); the engine refuses boot below that (`512 slots is fewer than
  the resolve block (1024)`).

Gate table (farm5 = 22.5k-cell sessions, 432-tok prime answers):

| arm | slots (private+region) | expert slots | C1 | C4 decode | C4 wall | farm QC | battery |
|---|---|---|---|---|---|---|---|
| forkb4 (champion, region off) | 9216+0 | 2136 | 151.2 | 47.1-49.9 | 18.5 s | all true | green |
| e1 region 4608 | 9216+4608 | 1969 | 141.0 | 39.0-42.0 | 21.7 s | all true | green |
| e2 region 5632 | 6656+5632 | 2026 | - | sibling window 512 < 1024: engine refused farm leg (503) | - | - | battery green, farm refused |
| e3 region 3072 at parity | 6144+3072 | 2126 | 147.1 | 40.0-43.1 | 20.3 s | all true | green |

Telemetry: e1 served 4,237 region fetches and ~17k adoptions; e3 3,392 fetches / 14k adoptions —
the mechanism works exactly as designed (one fetch per lineage-page, siblings adopt, zero
corruption, QC prime checks all true, battery 8/8 QC6 exact `97, 101, 103`).

Economics verdict: at expert parity (e3) the region still LOSES to forkb4 (C4 40-43 vs 47-50,
wall 20.3 vs 18.5 s), and paying VRAM for the region (e1) loses more. The engine's own counters
are the smoking gun: 97.7% of block reads hit VRAM and only 16.8 MiB came from host RAM across
the whole farm — duplicated cold-head FETCH TRAFFIC was never the cost. The cost is resident
CAPACITY: forkb4's admission fork already converts prefix duplication into window coverage, and
every region slot competes 1:1 with an expert slot for the same VRAM. You cannot buy capacity
with capacity. Sharing beats duplication only when the shared set fits the region outright; on
12 GB it does not fit alongside windows and experts.

Campaign close: the CoW KV campaign lands at M-C1c — admission fork (forkb4): +473 expert slots
(1663->2136), C4 decode 42.5-45.1 -> 47.1-49.9, C4 wall 23.9 -> 18.5 s, race-clean, battery-green
at every step. M-C1d (pinned join-at-fetch) and M-C1e (lineage shared region) stay committed
opt-in with defaults OFF (`STRATA_COW_JOIN_HH=0`, `STRATA_COW_SHARED=0`), their mechanisms
proven race-clean for any future substrate with real VRAM headroom (bigger cards, or host-tier
parked-session CoW where host RAM dedup is the actual prize — no VRAM competition there).
Canonical restored; `battery_quality_m25` 8/8.
## 108. M-C1d join-at-fetch: the mechanism is proven race-clean, but pinned sharing is net-negative on small windows — the real prize needs a lineage shared region (M-C1e)

Goal: stop duplicate host->VRAM fetches of the same cold prefix block across forked batch siblings (forkb4 farm left ~4600 cold prefix pages thrashing their own 1024-slot windows each).

Mechanism landed on `cow-m0` (`~/strata-cow`):
- `CowPeer` registry (device array, per streamed arena-backed state): `conv` (fork lineage, single 4 B mutation law), `hhead` (join headroom), `ord/base/n_slots/n_blocks`, `pt` pointer.
- resolve step 3 cold miss: scan same-ordinal same-lineage peers; if peer table names block b at global slot s: join = headroom gate -> unconditional `atomicAdd(slot_use)` with prev>=1 gate -> revalidate peer entry -> point own table at s; stale joins roll back. ANY number of joiners per page.
- `fork_join_kernel` + `cow_release_kernel` moved fork/refcount RMW on-device (no host RMW racing device joins).
- telemetry: `jctr` per arena -> "cow: joins-at-fetch served so far: N" in engine log.
- env: `STRATA_COW_JOIN_HH=<n>` seeds per-range join headroom (0 = joins OFF = exact forkb4 behavior).

Gate table (farm = m25_farm5, batch4, arena 9216, slot-cells 4096):

| arm | joins | C1 | C2 | C3 | C4 decode | C4 wall | farm QC | battery |
|---|---|---|---|---|---|---|---|---|
| forkb4 (M-C1bc, joins off) | 0 | 151.2 | 88.7/81.7 | 64.7/58.6/60.7 | 47.1-49.9 | 18.5 s | all true | green (QC6 97,101,103) |
| forkb4d (uncapped joins) | 10,855 | 151.6 | 69.9/75.6 | 71.4/67.1/17.8 | 59.0/38.1/19.0/59.9 | 23.2 s | C2+ false (streams stalled->truncated) | green |
| forkb4d3 (hh = n_slots/2) | 8,615 | 151.6 | 77.1/71.4 | 53.4/51.9/56.4 | 40.7-43.1 | 19.9 s | all true | green |
| forkb4d_hh256 | ~4.5k | 152.1 | 84.6/61.2 | 58.3/67.2/65.1 | 32.7/36.5/38.2/36.6 | 22.9 s | all true | green |

Laws banked:
1. JOIN IS MECHANICALLY SOUND: joins-at-fetch served 8.6-10.9k shared cold pages across siblings with zero corruption — battery QC6 exact and farm prime-QC green even when joins stall streams (damage is scheduling, never bytes).
2. PIN LAW (why joins lose): a joined page is un-evictable by its owner (clock-sweep-hold is load-bearing: a peer table entry naming a slot means a live gather can read it cross-stream; evicting any named slot is a corruption race, not just policy). Every join therefore steals working set from the fetcher's window. With sibling windows = 1024 slots, sharing damage scales monotonically with the cap: 512 -> 41-43 tok/s, uncapped -> stream death. Joining into small windows is net-negative; the mechanism only pays into ranges with big slack.
3. HEADROOM KNOB: `STRATA_COW_JOIN_HH` default 0 => joins off => canonical stays exactly forkb4. hhead refunds are illegal mid-epoch (fork pins never spent headroom; over-refund decays the cap) — budget only refreshes at reset (8 B conv+hhead store).
4. The cold-prefix arithmetic still holds (22.5k prefix = 4600 pages = 233 MB across 12 layers), but the share must live OUTSIDE any session window: M-C1e = per-lineage SHARED REGION in the arena (dedicated slots, fetched once, page tables point into it, eviction legal only when refcount == 0 i.e. no live namer -> no cross-stream gather can be in flight -> race-free by construction). Sizing fits today's arena: 4608 shared + 2048 main + 3x1024 siblings ~= 9216 slots. That converts the farm's 4x duplicate cold-head streaming into one pass without pinning anyone's window.

Next: commit `cow-m1d` (knob default 0), canonical restore, then M-C1e shared-region design gate.
## 107. M-C1b/M-C1c live: slot_use refcounts + admission fork land at full parity — batch-4 farm sharing is REAL (CoW end-to-end on the canonical streaming KV)

The share-at-fork CoW primitive is now live on the Strata C++ worker's canonical mode-1
streaming KV (kv_stream), with zero GPU-kernel changes to read/write paths:

### Mechanism (all env-gated; canonical path untouched when off)
- `KvStreamMap.slot_use`: one refcount array per QSA-ordinal arena (device).
  Clock-sweep eviction (resolve_kernel step 2) skips any slot with `use > 1`
  (`cow_hold`), bounded sweep leaves the miss unresolved on host read -> graceful stall-free.
- Step-1 residency marking range-checks: entries naming another session's arena range
  (forked pages) are looked through, never stamped/ref'd locally; the owner's clock governs.
- `qsa_cow_fork(from, to, upto)`: host-side table splice. Reads the source's live page table
  (D2H), drops entries outside the source range (aliased/cold -> -1), bumps `use[]` over the
  source range (batched H2D), splices `to.page_table[0..P)` (H2D). Zero new kernels.
- FLOOR law: `P = upto/4` — only COMPLETE pages are forked. The partial tail page must stay
  cold: two sessions appending different tokens into one shared page would trample each other.
- Release law: `qsa_cow_release` runs in `refresh_residency` BEFORE `kv_stream_reset`:
  a session lets go of foreign-range entries it held when its own map resets. Owners never
  release other holders' refs; holders release their own. Parked sessions keep pinning shared
  prefix pages (use>1) — that's a feature: resume keeps the prefix resident.
- Range law: per-ordinal cumulative `g_cow_next[]`, NOT `ctr * p.slots` — with
  `STRATA_COW_SLOT_CELLS` the first (main) session keeps the full window and every later
  session gets a smaller one; ranges must stay disjoint under shrink.
- Base-0 law: `slot_use` is baked for every arena state — session 0's range legitimately
  starts at slot_base 0; keying the bake on `cow_sb != 0` silently no-ops every main-sourced
  fork (the farm shape's main case).
- Observability: engine stderr lands in the config `log` file
  (~/hydragen-experimental/logs/strata-m25-iq3xxs.log), NOT the runner nohup log
  (server.py: stderr=self.log, stdout=PIPE).

### Gates
- fork2 (b2, arena 15360, fork on, pre base-0 fix): battery QC green, farm green —
  fork wiring fires but shares 0 pages (the base-0 bug, caught by the log:
  "points at 0 of 5860").
- fork3 (b2, post base-0 fix): battery green, farm C1 163.3 / C2 85.5+91.0 /
  C3 84.5+186.8+80.6 / C4 [81.5, 81.8, 85.7, 91.2] agg 87.7 (parity with fork2 89.0 —
  b2 windows fit everything alone, sharing is free-but-uneventful there);
  fork telemetry: 529-2569 pages per admission (resident-only fork degrades gracefully:
  main's window holds the prompt TAIL, the cold head streams from host per session).
- expert cache held at 2451 across the arena boot — private-pool carve fully reclaimable,
  pricing exact (b2 arena 15360 = 0.73 GiB printed at boot).

### b4 economics (farm shape = 4 concurrent sessions of one 22.5k-token conversation, ladder5 answers)
| arm | expert slots | C1 spec | C2 | C3 | C4 co-resident | C4 wall |
|---|---|---|---|---|---|---|
| plainb4 (no CoW)      | 1663  | 128.3 | 75.1/80.2       | 52.4/54.5/57.7       | 42.5/43.5/44.5/45.1 | 23.9 s |
| forkb4 (CoW fork)     | 2136  | 151.2 | 88.7/81.7       | 64.7/58.6/60.7       | 49.9/48.9/47.1/47.7 | 18.5 s |
| b4k8 (§105 sim floor) | 2109  | 166.6 | —               | —                    | 14.6/10.8/23.5 (unique) | 52.9 s |
| base5 (§100, b2)      | 2456  | 172.0 | —               | —                    | 29.4/28.8/29.1 (unique) | 50.4 s |

forkb4 boot: arena 9216 slots = 0.44 GiB printed, expert 2136 (3.50 GiB); fork telemetry
1233-1249 pages per b4 admission (slot windows 4096 cells = 1024 slots each).
QC battery green on every arm (QC6 97,101,103 exact; farm qc_primes True on all streams).

### Law banked
- CoW share-at-fork is live on the canonical streaming KV: batch 4 recovers +473 expert
  slots (1663->2136, above the b4k8 2109 floor) AND beats plain b4 at every concurrency
  level (C4 agg 72.2->93.2, wall 23.9->18.5 s). Quality untouched.
- Remaining C4 poison is the COLD HEAD: ~4600 prefix pages outside main's resident window
  thrash each slot's 1024-slot window independently (x4 duplicated fetch churn).
  Next lever (M-C1d): join-at-fetch — when several fork-siblings miss the same cold block,
  fetch once, point all their tables at it (use>1 pins it; sweep guard already honors that).
  The entire 22.5k prefix is only ~297 MB across 12 layers; the budget exists — it is the
  per-session window policy that refuses to share fetched cold pages.
- Slot-window shrink (STRATA_COW_SLOT_CELLS) is the funding knob: main keeps the full
  --kv-resident window, later sessions get the small one; ranges disjoint via per-ordinal
  cumulative counter.
## 106. M-C1a/M-C1a2 global K/V slot arenas land at full parity: per-layer arenas + map.slot_base are graph- and battery-safe, and the private-pool carve is fully reclaimable

The kv_stream CoW blueprint's allocation half is now real code in `~/strata-cow`
(cow-m1a commit + m1a2 patch).

### M-C1a (plumbing, double-counted)
- `KvStreamMap.slot_base`: local slot j names arena page `slot_base + j`; set once per state at
  init, graph-baked. Device resolve: page table stores GLOBAL names (`sl + slot_base`), copy dst
  offsets by `sl + m.slot_base`, stamp/ref stay LOCAL arrays, and out-of-range entries are already
  skipped for stamp/ref (that skip IS the future CoW eviction-safety primitive).
- One arena per QSA ordinal: 4 cudaMallocs (k/v codes int8, k/v scales uint16), lazily allocated
  on the first streamed state. **n_qsa_layers = 12 on this model** (not 16 as earlier assumed).
- Env: `STRATA_COW_ARENA=1` + `STRATA_COW_ARENA_SLOTS=<slots/layer>` (config env dict, as always).
- Gate run (arena 25600, double-counted): expert cache fell to 1707 experts / 2.81 GiB —
  exactly arena bytes (25600 x 4224 B x 12 = 1.27 GiB ~= -749 expert slots) — the allocation is
  real and lands BEFORE expert-cache auto sizing; battery stayed green.

### M-C1a2 (pricing exclusion + range-correct consumers)
- `qsa_state_bytes` excludes the int8 streamed pool for arena states; `qsa_state_init` skips the
  four private takes; `QsaState.cow_codes/cow_scale` carry element offsets of the state's range;
  `conversation_snapshot pools()` returns range-offset device pointers; `qsa_state_zero` zeroes
  only the state's range (the M-C1a zc/zs expressions were missing *head_dim on codes — fixed by
  the stride fields; harmless pre-fix because fresh sessions never read past their cursor).
- Parity gate vs base5 (arena 15360 = 3 sessions exactly; boot never printed "no room", so
  canonical serve creates exactly 3 streamed mode-1 sessions):

| metric | base5 canonical | m1a2 arena |
|---|---|---|
| expert cache | 2456 slots | **2451 experts / 4.02 GiB** |
| C1 spec | 172.0 | 164.2 (accepted 324, boot variance band) |
| C4 co-resident | 29.4/28.8/29.1 | **29.0/28.9/29.1** |
| C4 wall | 50.4 s | **50.4 s** |
| battery | QC_DONE | QC_DONE, QC6 97/101/103 exact |

### Laws banked
- Per-layer arenas + slot_base indirection need NO CUDA-graph changes: tables are baked per
  session and the resolve kernel offsets are constants once captured.
- Session count on canonical serve = 3 streamed states (1 main + 2 batch slots at --batch 2).
- Pages >= a page-aligned shared prefix are never written after sharing: appends only touch the
  newest page, and fork boundaries are block-aligned (checkpoint reuse is 4-token aligned).
  **The farm shape therefore needs no append copy-on-write at all** — only complete, immutable
  pages are ever forked.
- Next (M-C1b): per-layer device `slot_use[]` + clock-sweep skip of `use>1`, admission fork as a
  host-side D2H/H2D page-table splice + batched use bumps, and the session-reset release hook
  (decrement out-of-range entries before kv_stream_reset memsets the map).

Files: ~/strata-cow/src/core/layer.cpp, src/kernels/cuda/kv_stream.cu, src/core/session.cpp,
src/core/conversation_snapshot.cpp, include/strata/*; runner /tmp/m25cowrun3.sh; log
~/hydragen-experimental/logs/cow-m1a2.log.

## 105. b4k8 economics bound (batch 4 + --kv-resident 8192): the KV window is first-order; CoW prize and target metric are now concrete

ladder5 (432-tok answers, cold 22.5k prompts, ctx 131k):
```
arm                              expert slots   C1 spec   C4 co-resident   C4 wall
base5  batch2 kv20480            2456           172.0     29.4/28.8/29.1   50.4 s
b4     batch4 kv20480            1663           142.5     17.9/10.4/9.1    ~66 s
b4k8   batch4 kv8192             2109           166.6     14.6/10.8/23.5   52.9 s
```
b4k8 boot: "--batch-mtp: 4 slot drafters (50 MiB each); 4.91 GiB free"; auto 1595 -> cache 2109 slots 3.46 GiB; prompt chunk auto clamped to 5888 tokens / 512-slot ring at batch 4.
Laws:
1. Halving the per-session resident KV window at batch 4 returns +446 expert slots (1663->2109) and C1 142.5->166.6. The KV-resident window is a first-order, reclaimable VRAM cost; the arithmetic CoW claims (per-session window bytes) is real, not speculative.
2. But window shrink taxes long sessions in proportion to streamed fraction: 22.5k stream vs 8192-cell window = ~64% attention reads from pinned RAM; C4 cold co-resident stays poisoned (10-23 tok/s). b4k8 = the NO-CoW lower bound: same VRAM reclaimed, but paid in per-session window quality.
3. CoW strictly dominates b4k8 for the farm shape: shared 20480-cell prefix window (bytes freed WITHOUT shrinking any session's window) + N private tails. M-C1 success metric is now concrete: batch4 boot >= 2456 expert slots (as if one session paid the window) and C4 co-resident >= 29 tok/s (base5 unpoisoned rate).
4. Implementation shape re-confirmed (cow-m0/m1 findings): mode-1 streaming map allocates resident slots ON-DEVICE (kv_stream map, slot_ref refcounts exist for eviction). True cross-session sharing needs a device-side global frame free-list + atomic pop in kv_stream promote/evict (2-3 device fns) + host arena wiring. PERM validation (§104) proved read/write paths are table-indirect safe; the remaining risk is purely allocator promotion.
5. Stale-arm law extended: a config FILENAME does not identify a running arm; read the engine's live cmdline (--batch/--kv-resident via /proc/PID/cmdline) before trusting :8181. First b4k8 attempt booted base under the b4k8 filename (runner regenerates config from base when EXTRA is empty).
## 104. cow-m1 PERM validation: page_table indirection is kernel-safe — CoW needs zero kernel changes

cow worktree (clean v0.2.13 + one patch): STRATA_COW_PERMUTE=1 swaps adjacent pages in the mode-0 identity table at allocation (core/layer.cpp ~968). All QSA pools boot permuted; serve battery (battery_quality_m25) PASSes vs identity control on the same binary: QC6 97,101,103 exact both arms, QC4/QC5/QC7 match.
Law: every hot kernel path (kv_append_q8/q4, kv_gather_step, qsa selection attn, verify, snapshot) resolves logical->physical through st.page_table with NO identity short-circuit anywhere in the host call path (mode-0 just fills identity; kernels always receive the real table). Therefore ref-counted cross-session page sharing = host-side allocator + admission plumbing ONLY; GPU code is untouched.
Remaining M-C1 surgery map:
- copy machinery: generate.cpp copy_to_slot/copy_to_main/copy_session_between (~14373), direct_slot_copy (14372, device2device by state table), conversation_kv_copy (core/conversation_copy.hpp, shim over 14390), conversation KV exchange buffers (536).
- fork fast-path insertion point: conversation_kv_copy callers at admission: instead of copying pages [0, upto), re-point to.page_table entries at from physical pages + refcount; copy only tail pages.
- allocator: add per-arena frame refcounts; free = deref; last-ref frees. Appends must allocate private frames (allocate-on-write for the boundary page).
Next gate after implementation: battery + ladder farm bench (N forked sessions sharing a 22k prefix), boot accounting must show expert slots near canonical with batch 4.

## 103. M-C0 determinism spike: cross-run KV is NOT bit-identical; CoW must be share-at-fork (engine's own token-prefix reuse already vouches for it)

Native tooling found (use this, don't re-patch): STRATA_STATE_HASH=1 (config env dict only; server.py does NOT pass shell env to the engine) fingerprints every finished request: gdn/ple/tail/pooled/kv/mtp/stale/dead FNV-64 over [0,L), page-aware (page_size, per-head ranges). Hybrid-model serve loop + fingerprint live in src/program/generate.cpp (~19544-19650); the q36_engine.cu serve printfs are a DIFFERENT standalone-serve variant that never runs under serve/server.py.
Spike (cow worktree clean v0.2.13, same 9952-token prompt as first request after boot, max_tokens=1, plans auto:16384 / 4096 / 8192):
```
plan          read tok/s   kv=              gdn=             ple=
auto:16384    2337         a4df868dd62ecf7b  a1f9eacffa5264cb 0f8dfbf41b70f4cb
4096          1763         b89f178bcecdbb39  bf94853f861d4ac3 64db0c02e2dc8600
8192          2378         c5a066692a7adb9d  12876e0195779855 862fa5b0a84e0308
```
All differ (kv AND gdn AND ple: chunk plans change hidden states, not just attention K/V; adaptive-tier residency changes mid-read add another source of run-to-run variance).
Laws:
1. Content-addressed KV dedup (hash-match independently prefilled prefixes) is IMPOSSIBLE on this engine. Dead end, closed.
2. CoW design is therefore share-at-fork: the FIRST session to prefill a token prefix owns the pages; later sessions with the same token prefix REFERENCE them (refcount + private append pages). No new numerical bet: serve already reuses prefix bytes across sessions by token key (22528-of-22535 reused; checkpoint restores; QC gates green program-wide). CoW extends the existing trust to VRAM page ownership, saving the ~0.36 GiB copy per forked session.
3. Fork discipline consequence: sessions that arrive cold-parallel (no first-mover) each prefill their own copy (different numerics); sharing only helps sessions forked after a prefix exists. For the agent-farm shape (shared system/repo prompt pinned at session 0) that is exactly the dominant case.
4. M-C1 target files updated: generate.cpp serve/admission + core QSA KV pools (page_size already exists in the fingerprinting loop = paged layout already in use: (pg*n_head_kv+hd)*page_size cells - refcounting is the addition, not paging).
5. Serve-env law: engine env ONLY via config "env" dict; a runner injecting env must write the config json, never export before nohup.

## 102. Expert-profile re-rank: CLOSED-NEUTRAL (learned top-2456 ties shipped, even in-sample)

Toolchain found (good to know exists): tools/make_profile.py (STRP v1, 48x512=24576 ranked pairs; --reorder = traces decide top), serve-side --expert-profile-save PATH (#477: saves residents+routing counts on QUIT / every N min), --dump-routing trace format.
Learn loop: canon + --expert-profile-save, drove ladder5 + 3x6-turn code-review sessions + QA burst + coding prompts + battery; QUIT saved data/hg-learned.bin (196,632 B = full 24576-pair re-rank, no coverage loss).
A/B (ladder5, ctx 131k):
```
arm                     C1 spec   C2      C3      C4 wall   battery QC
shipped profile         174.1     24.1    37.2    50.1 s    8/8
learned profile         175.1     24.3    37.3    50.1 s    8/8
base5 earlier           172.0     22.9    35.9    50.4 s    -
```
Laws:
1. On this engine the expert-profile only sets the BOOT PRE-LOAD; the adaptive tier (adapt-every 1, swaps 400, decay 0.97) converges to the conversation's hot set within the first windows, so profile re-rank is a statistical no-op in steady state AND at first-request (shipped top already overlaps our corpus's hot set).
2. Expert hot-set residency was hypothesized as "the remaining scarce axis" — measured NEUTRAL. The remaining walls are pure budget (slots vs experts, §100) and rows (kVerify=8, §95). CoW KV pages + kVerify widen stand as the only engine levers with real headroom.
3. --expert-profile-save is kept as a restart-warmup affordance, not an adoption.

## 101. --no-prefill-borrow arms: poison immunity is REAL, paid in permanent residency (anti-CoW lesson)

```
arm (ctx 131k, ladder5 432-tok answers)      expert slots   C1 spec   cold co-resident   C4 wall
base5 (borrow default)                        2456           172.0     12.0 (poisoned)    50.4 s
npb4 (--no-prefill-borrow --prefill 4096)     736  (1.21GiB)  83.5      23.2-26.6 stable   69.0 s
npb  (--no-prefill-borrow, auto:16384)        verify-window instantiate OOM -> all requests 400, engine inert
```

Laws:
1. --no-prefill-borrow carves the prompt chunk buffers + ring PERMANENTLY. At --prefill 4096 the carve costs 1720 expert slots (2456->736). It does buy what it claims: co-resident decode during cold 22.5k reads holds 23-26 tok/s instead of cratering to 12 (zero cache interference). C1 spec decode collapses 172->83.5 from 736-slot residency (7%).
2. Bare --no-prefill-borrow with auto:16384 on 12 GB = engine survives but verify windows cannot instantiate ("batch instantiate: out of memory", every request HTTP 400). Never combine no-borrow with big chunks on 12 GB.
3. Verdict: not adoptable on 12 GB (permanent tax > transient poison), but it PROVES the poison mechanism is cache interference, and it is the anti-CoW contrast: npb removes interference by separating budgets and pays experts; CoW removes interference by deleting KV duplication and can create budget. If/when CoW KV pages land, per-slot KV cost drops ~0.36->~0.05 GiB for shared-prefix sessions and this whole trade flips positive.

## 100. Wide-batch arm (b4) CLOSED + long-decode ladder5 baseline + PD-split infeasible — the 12 GB one-budget law

ladder5 (long decode: 432 tokens/stream, primes 200-900, cold 22.5k prompts, ctx 131k):
```
arm                          C1              C2               C3                    C4
base5 (batch 2 spec4 bmtp)   172.0 spec      [99.4, 28.9]     [185.2, 29.0, 29.2]   [29.4, 28.8, 29.1, 98.1]
b4 (batch 4 spec4 bmtp)      142.5 spec      [102.8, 19.4]    [18.3, 85.3, 10.8]    [17.9, 10.4, 78.0, 9.1]
b4x (batch 4 spec 0)         IMPOSSIBLE: native IQ packs require --spec T>=2 at boot (engine exit 2)
```
Boot forensics: batch 2 -> expert cache 2456 slots; batch 4 -> 1663 slots (-32%). 4 slot sessions + 4 batch-mtp drafters (50 MiB each) ate 1.27 GiB of expert budget BEFORE any token decoded. b4 co-resident streams: 9-19 tok/s (vs 29 at batch 2). C1 solo also fell 172->142.5 from the squeeze.

Laws banked:
1. THE 12 GB ONE-BUDGET LAW: expert cache, batch slots, prefill ring, KV-resident window, drafters, output head all draw from one VRAM budget. Every measured attempt to trade between them (LEND_PCT, chunk, batch width, int8 KV, KV streaming) moves the wall, never lowers it. §95/§97/§99/§100 together = closed.
2. Wide continuous batching does NOT beat "2 users + drafts" on 12 GB IQ3_XXS: slots cost experts at ~0.45 GiB each; kVerify widening (8->16) is moot until slots get cheap (they'd need KV + drafter state near-zero).
3. --split-serve PREFILL,DECODE (serve-level PD disaggregation, docs/SPLIT_SERVE.md) is TWO-CARD only (two engine loads). the reference host has one 12 GB card -> infeasible here; it is the right primitive on a 2-card box.
4. Long answers amortize MTP much better: spec decode 172-185 tok/s sustained on 432-token answers (accepted 321-324 of ~330) vs ~90-130 on short answers. Program best C1: 185.2.
5. Prefill poison is residency arithmetic, not policy: the engine already lends TAIL (coldest) slots first ("list the pairs from the highest slot down, as the borrowing takes them"), refills after the read, and yields by STRATA_BATCH_YIELD_CHUNKS=8. A 22.5k-token stream's routing footprint simply does not fit in the ~12% residency that survives its own cold read. Product mitigation = checkpoint reuse makes it first-turn-only (admission already serializes cold reads at ~2400 tok/s = 9.5 s per 22.5k).
6. Next lever (pre-agreed): expert-profile re-rank from our corpora. Economics now clear: it raises warm hit AND shrinks the decode-time footprint that must survive cold reads (compact hot sets decode 29 tok/s at batch2; the squeeze sensitivity in b4 shows footprint-vs-residency is the live axis).

## 99. Prefill-isolation config sweep CLOSED — the poison is structural (ctx 131k, cold 22.5k sessions, ladder4 fresh-prompt-per-stream)

```
arm                                   C1     C2     C3     C4 wall   co-resident cold decode
base (--prefill auto:16384)           9.3    18.9   28.6   38.0 s   11.5-12.0 tok/s
--prefill 8192                        9.2    18.6   28.1   37.3 s   12.1-12.6
--prefill 4096                       12.3    24.8   37.2   49.7 s   10.0-17.7
STRATA_COALESCE_PREFILL=1             9.4    18.9   28.8   38.3 s   7.4-11.9
STRATA_PREFILL_LEND_PCT=50           24.1    48.5   72.5   96.6 s   18-19 (prefill collapsed 2450->243-973)
```

Laws banked:
1. No config knob isolates decode from cold-prefill poisoning. Chunk size, yield granularity (implicit via chunk), serve-level coalesce, and lend-fraction are all flat or negative. p8192 is free-but-neutral; p4096 buys ~+40% co-resident speed at −30% wall.
2. Prefill THROUGHPUT is ring-coupled: the prefill ring is the MMQ expert-staging rate limiter; shrinking it (LEND_PCT=50 -> ~1205 slots) multiplies streaming waves and collapses read rate 2.5-5x, stretching the poisoned period over the whole read. Poison duration ~= prompt_tokens / prefill_rate, and prefill_rate collapses with staging.
3. Engine default lend on this card = 90% of expert-cache slots (STRATA_PREFILL_LEND_PCT default; kAutoLendPct pinned_share>=0.9 ? 90 : 85; g_lend_keep floor 128 slots).
4. The fix is ENGINE POLICY, not config: (a) pin top-N hot decode slots against prefill-ring eviction while KEEPING ring size, (b) refill-then-decode ordering, or (c) the serve-level PD-disaggregation primitive spotted in usage: server.py --split-serve PREFILL,DECODE (unexplored).
5. Swap law: serve holds :8181 for ~5-15 s after TERM (graceful drain). Kill loop must WAIT for curl-refused before relaunch, else the next arm silently probes the old server or dies on bind.
6. kVerify=8 is `static constexpr int kVerify = 8` (q36_engine.cu:1165) — widening is one constant + buffer sizing, pending after config-level batch arms price the row ceiling.

## 98. Long-context concurrency arm (ctx 131k, 22.5k-token sessions, 0.1.59 MTP bmtp tuned)

Design: 4 sessions, ~22535 tokens each (shared ~1.1k-token runbook head + unique 900-line body), 512-token greedy answers, C1->C4 on the live 131k stack. NOTE: repeated levels re-ran the same prompts (cache-hit confound); cold-vs-warm separated via engine telemetry, not request layout.

Ladder: probe 22535 tok prefill 9.2s = 2451 tok/s. C1 57.6 agg (cached, short answer). C2 9.0 (one cached stream 123.3 + one cold-prefill stream 11.7). C3 13.6 [68.4, 39.9, 139.1]. C4 16.5 [12.2, 42.9, 122.7, 48.4].

Findings banked:
1. LONG-CONTEXT DECODE IS CHEAP. Cached sessions decode at 22.5k ctx at 122-152 tok/s (spec'd, drafts accepted 95.7-97.8%) and 40-68 tok/s in batch windows — at or above short-context levels. KV streaming: 74-97% of block reads hit VRAM (20480-cell window vs 22.5k ctx; only 45-100 MiB/period from pinned RAM). The "long context = slow decode" fear is unfunded on this engine.
2. PREFIX REUSE IS ALIVE AT SCALE: "prompt 22535 = 22528 reused + 7 read in 240 ms"; "restored 22528 tokens (checkpoint) in 58.3 ms"; conversation cache reports reused_kv_bytes=343,863,168 — KV BYTES move with the conversation, not recomputed. At this scale Strata delivers RadixAttention's compute half; only the storage half (shared ref-counted blocks) remains SGLang-only.
3. PARKING MATHS: parked 22535-token conversation = ~394-437 MB (int8 KV + snapshot overhead); cache bytes grew 1.4->4.4 GiB with 5-8 parked sessions, byte-cap evictions fired gracefully (evictions=6). 6 GiB conv cache ~= 14 sessions at 22k, ~2 at 128k — a dial (--conversation-cache-mib), not a cliff.
4. THE REAL LONG-CONTEXT COST IS ADMISSION: a cold 22.5k prefill inside batch windows poisons co-resident streams: windows stretched to 53.5-56.4 ms (vs 25-28 warm), routed VRAM hits fell to 21% (CPU experts 45-47%), and co-running streams dropped to 11.7-42.9 tok/s. Chunked prefill exists ("prompt read in 2 parts, slots decoding 1273 ms between them") but the interleaving budget is naive.
5. Prefill scales: 2408-2812 tok/s at 22.5k (vs ~700 at 1.3k) — the prefill ring batch-amortizes; 128k-token cold prefill ~= 45-55 s.
6. Verdict on the RadixAttention repricing question: NOT repriced upward for this architecture. What to fix instead: prefill/decode window scheduling (isolate cold prefills from decode windows), and conv-cache byte budget for 64-128k sessions.

## 97. Context-size curve on 0.1.59 (MTP bmtp tuned, IQ3_XXS, batch 2) — KV streaming discovered

```
ctx       expert slots  C1      C2      C3      C4
8192      2392          106.9   88.1    96.6    90.5
32768     ~2410         89.3    77.2    84.9    80.7
131072    1856          90.6    74.7    95.9    80.2
```

Facts banked:
1. 128k ctx boots HEALTHY on 12 GB: engine enables KV STREAMING automatically: 20480 of 131072 cells per QSA layer stay in VRAM, the rest of K/V lives in 1.55 GiB pinned host RAM. GPU KV cost does NOT scale linearly with max-context on this path (earlier 0.56 GiB/slot linear estimate applies to <=32k full-reserve only).
2. Slots at 131k: 1856 (-23% vs 32k) — the KV window + pinned pool is funded partly from expert cache; ladder still only -5 to -8% vs 32k.
3. 8192 gave the program best single-stream: 106.9-109.1 tok/s; the win at short ctx is attention/decode-path cost, not just slots (8k had ~equal slots to 32k but +14%).
4. expert-cache auto slots are internally capped (~2400 on this card regardless of ctx when free grows: 8k had 6.14 GiB free -> 2392 not ~3100); more free VRAM does not buy slots 1:1 — auto-sizing + profile/ring constraints.
5. UNTESTED DIMENSION: the ladder is short-prompt (~0.5k). The real agent-harness cost of 128k lands in KV window thrash: sessions beyond 20480 cells/page out K/V reads over PCIe per decode token; parking many 128k conversations in the 6 GiB conversation cache overflows (~2.2 GiB each). Need a long-context concurrency arm (>=20k-token prompts, C2-C4) to measure it.

## 96. Strata 0.1.59 (v0.2.13) built on 12 GB + DFlash2 arm — parity confirmed, engine self-warns

Built v0.2.13 engine on the reference host: git worktree ~/strata-0213, cmake -G Ninja -DSTRATA_ENABLE_CUDA=ON (default OFF; without it there is NO 'strata' target), ninja+cmake from ~/hydragen-experimental/.venv-build/bin, -DCMAKE_CUDA_ARCHITECTURES=86-real -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc, target strata. -> strata 0.1.59, BUILD_OK.

DFlash2 packs fetched from v0.2.13 assets (sha256 OK), kept at ~/.models_backup/Strata-data/dflash2/: q4_0-bf16sel (498 MB) and q4_0-selq4 (324 MB; selector codebooks Q4_0 = 174 MiB less VRAM, same drafts/text). Wire-up: --draft dflash2 --draft-model PACK + keep --mtp DIR (draft_vocab only; MTP layer itself NOT loaded) + --batch-mtp enables STRATA_BATCH_DF2 slot drafters (85 MiB private K/V ring each).

Decode-bound ladder (512 tok, fresh sessions, temp 0), IQ3_XXS, batch 2:

```
arm                            C1     C2     C3     C4
0.1.57 bmtp                    92.5   74.0   80.9   90.9
0.1.57 bmtp tuned              89.6   75.8   85.8   80.9
0.1.59 bmtp tuned MTP          89.3   77.2   84.9   80.7   (no regression; fixes #5096/#4844/#5531 free)
0.1.59 bmtp tuned DFlash2      93.6   69.2   81.7   90.2   (best C1 seen; C2 dips)
```

DFlash2 facts on 12 GB:
- Engine prints on boot: "--draft dflash2: on a 12 GB NVIDIA card the MTP drafter was faster in most measurements (code prompts at a window of 6 gained 3%); continuing as asked".
- drafter = 591 MiB VRAM Q4_0 + 178 MiB draft head; replaces MTP layer (839 MiB) -> expert cache 2410 -> 2598 slots (+188), decode hit-rate 88-89%.
- Batch windows: "DFlash2 slots: N rows deferred (no spare window row)" — batch drafts only ride SPARE rows; the kVerify=8 row budget is the same hard wall DFlash2 cannot widen.
- Fresh-session penalty: "context rows of the window are not the drafter's to give... drafts poorly for the first rounds (answers the same)" — every ladder stream is a new conversation, which is DFlash2's worst case; steady logged sessions would draft better.
- "MTP proposals accepted 688 of 333 (206.6%)" telemetry lines in df2 runs: counter mislabels; do not read as >100%.

VERDICT: DFlash2 is REAL on 12 GB with 0.2.13 (user's read was right) but PARITY with MTP on our mixed prose/code ladder, exactly as Strata's own per-card measurements say (Auto keeps MTP on 12 GB; DFlash2 default only on cards with all experts resident / specific quants). Not adopted; MTP stays. 0.1.59 adopted for the free engine fixes. Configs: /tmp/strata-m25-{mtp,df2}0213.json. The one card's decode aggregate stays ~= 1x single-stream specced: row-budget-limited, confirmed independently by DFlash2 deferral telemetry.

## 95. Concurrency decode anatomy on the q36 C++ worker — --batch-mtp law, window-row ceiling, DFlash2 scope

Decode-bound ladder (512-token answers, fresh A/B prompts, temperature 0) on IQ3_XXS C++ worker, the reference host, single 12 GB card:

```
arm                 C1      C2      C3      C4
no --batch-mtp      91.1    73.0    80.0    73.9   (batch windows carry ZERO MTP proposals; solo keeps spec)
--batch-mtp         92.5    74.0    80.9    90.9   (C4 +23%)
--batch-mtp tuned*  89.6    75.8    85.8    80.9
IQ2_XS smoke (93)   ~90     72.6    100.2   87.6   (3343 slots = 33% residency)
```

* tuned = --pcie-frac 0.5, --adapt-swaps 400, --vram-reserve-mib 512.

Laws banked:

1. `--batch-mtp` is REQUIRED for spec in batched decode. Without it, only solo (one-row-window) streams get MTP proposals; batch-decoded streams decode specless. C2 aggregate (2 specless ~38) < C1 (one spec'd 93) — the "C2 worse than C1" signature is spec attribution, NOT PCIe contention.
2. serve-layer per-request `draft_n_accepted` under-reports on batch windows (shows 0 while the engine accepted). Trust the engine telemetry line `strata batch: MTP proposals accepted X of Y` — observed 78.4-84.0% acceptance with bmtp.
3. Window-row ceiling: `kVerify = 8` (compile-time), batch windows of up to 2 sequences for layers [0,48); bmtp adds slot proposals into the same 8 rows (room = kVerify - S, greedy slots in turn). Aggregate decode is capped by window rate: 36-43 ms per 8-row window -> ~25 windows/s -> ~80-97 tok/s aggregate ceiling regardless of C. C4 ~= C1 with bmtp; never meaningfully >1x on IQ3_XXS.
4. Window time anatomy (C4): CPU experts 14-29 ms + CUDA wait 12-20 ms + PCIe 1-4 ms. PCIe is NOT the wall (2.5-11% of routed entries ride PCIe). Tuning (pcie-frac/adapt-swaps) moves entries from CPU-spill to PCIe-staged but the SUM stays flat: the true variable is expert residency. IQ3_XXS = 2410/10240 (23.5%); IQ2_XS = 3343/10240 (33%) is the only ladder that exceeded 1x solo (C3 100.2).
5. DFlash2 scope: implemented only in the split engine (`strata-split-N --draft dflash2 --draft-model PACK`, two cards) and vendored llamacpp dflash.cpp. The q36 single-GPU engine has ZERO dflash references (grep). Documented economics: +5-14% greedy code/JSON, slower on sampled text, agent turns, and at 4 users; Auto never selects it. Consistent with SGLang-side §84/§85 DFlash2 closure. Not a lever for this substrate.
6. Double-server trap: a killed-but-not-dead old serve keeps port 8181 and answers HEALTH_200 while the new config never serves. After any swap verify the live ENGINE cmdline actually carries the flags (`tr '\0' '\n' < /proc/$(pgrep -x strata)/cmdline`), and kill serve + engine by name in a script FILE.
7. Concurrency value on one 12 GB card is prefill overlap + conversation parking (8 slots), not decode aggregate. Decode aggregate scales past 1x only via residency (IQ2 tier), profile rerank (next lever), or the documented two-card split-engine path.

Artifacts: /tmp/m25_ladder2.py, /tmp/strata-m25-tuned.json, /tmp/m25clean.sh, /tmp/m25tuned.sh (the reference host). Configs: ~/hydragen-experimental/strata-iq3xxs-m25.json (bmtp now included).

## 94. M25 Phase 2: IQ3_XXS C++ worker parity run — ADOPTED as funding architecture; prefill 13x, decode 2-2.5x, QC 8/8 (the agent, 2026-11-12)

- Artifacts: IQ3_XXS shards downloaded ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF (47,039,860,096 + 28,800,138,432 = byte-exact to fingerprint), 11 min @ ~72 MB/s, to ~/.models_backup/Strata-data/models/IQ3_XXS/IQ3_XXS/. Existing backup pack packs/iq3_xxs reused as-is (index-only; shards resolved beside --native).
- Config: /tmp/strata-m25-iq3xxs.json = same 12 GB line as iq2 (vram-reserve 768, kv int8, batch 2, conv-cache 6144/8, spec4 coupled mtp, max-context 32768, adapt 160/round async).
- Boot accounting: arena 39.97 GiB host file-cache @3.22 GiB/s; expert cache auto -> 2410 slots / 3.95 GiB (pre-filled from profile 24.3 GB/s); batch slots 0.56 GiB x2; prefill ring borrows 2168 slots; 790 MiB free with everything.
- Gates (m25_gates.py, thinking off, greedy):
  - QC 8/8; QC6 exact "97, 101, 103"; QC7 exact; per-QC 0.4-1.5 s (vs SGLang 2.6-15.3 s).
  - TTFT fresh-seed 6000-char (1300 tok): 3.0/2.5/2.4 s = 684-732 tok/s prefill — reproduces the §80 anchor band 681-883 from open components; vs SGLang+A2 32.1 s = ~13x.
  - Ladder: C1 decode 87 tok/s (MTP burst), C2 62.3 agg (32.6/37.1), C3 76.3, C4 74.2 (~37-47/stream). vs CN24+A2 single 20-22 and cap4 agg 36-45.
- §80 anchor replay note: anchor C2 agg 46 vs today 62 — anchor ran 131k-context budgets (fewer expert slots); 32k ctx + int8 KV + profile prefill is faster at equal quant.
- LAW: pkill/pgrep -f patterns self-match when the SAME ssh command line also contains the literal target string (launcher half) — bracket trick insufficient; put kill+launch in a script FILE so targets never appear in any live cmdline.
- Decision: C++ worker (build/strata + serve) ADOPTED for the high-concurrency massive-MoE campaign on 12 GB. SGLang CN24+A2 stays as fallback line; island/A2 work superseded for serving but A2 laws port conceptually to any Python-front substrate.
- Open follow-ups: 128k-ctx kv-resident stress (deferred by user gate), IQ3_S upgrade path (needs /home headroom: delete IQ2 backup first, user call), expert-profile re-ranking from OUR traffic corpora vs shipped profile, /home at 20 GB free = disk hygiene needed before any further model pulls.

## 93. M25 Phase 1: Strata C++ execution worker booted on 12 GB — funding architecture VALIDATED (IQ2_XS smoke) (the agent, 2026-11-12)

- Direction (user-ordered): stop hunting VRAM inside PyTorch; adopt Strata's static C++ worker (build/strata + serve/server.py) to escape mem_fraction_static entirely.
- Assembled on the reference host from local artifacts only:
  - engine: ~/hydragen-experimental/strata/build/strata (95 MB, Nov 6 build of vendored tree)
  - front: serve/server.py --engine strata --config (OpenAI /v1/chat/completions, tokenizer from pack, port 8181)
  - pack: tools/iq_pack.py on ~/.models_backup IQ2_XS gguf -> ~/strata-packs/iq2_xs (index-only; experts stay in the GGUF, indexed by native_experts.txt)
  - IQ packs REQUIRE --spec T>=2 (MTP always on): used backup mtp/rt, --spec 4 --coupled-draft --spec-min-p 0.45
- Boot accounting (12 GB, vram-reserve 768, kv int8, batch 2, conversation-cache 6144MiB/8 slots):
  - expert cache auto -> 3343 slots = 4.50 GiB resident GPU experts (pre-filled from profile at 23.9 GB/s)
  - batch: 2 slot sessions 0.56 GiB each; prefill ring borrows 2949 slots; MTP draft 839 MiB (675 experts)
  - 799 MiB free with EVERYTHING loaded. No PyTorch hold at all.
  - expert arena = 33 GiB host file-cache (buffered, cudaHostRegister ok, THP skipped)
- Concurrency ladder (greedy, thinking off, 128 tok): WARM decode 40.1 tok/s (MTP accepted 3/3); C2 agg 72.6 (41.2/45.9 per-stream); C3 agg 100.2 (one stream 130.5 MTP burst); C4 agg 87.6 (~48-50 each). True 2-seq co-decode windows verified ("batch windows of up to 2 sequences, layers [0,48)").
- Contrast with SGLang substrate: 3343 resident experts vs CN=24; MTP spec ALIVE here (draft experts resident) vs dead-EAGLE there; conversation-cache parking vs KV-pool eviction.
- CAVEATS: IQ2_XS is a boot vehicle, far below Q5_K/quality floor; §80 anchor (agg 46 at c2) was IQ3_XXS on a loaded system — not directly comparable. Prefill rate on this boot unmeasured beyond 28-token prompts (81.8 tok/s = ramp-dominated).
- Next (M25.2 decision point): parity quant choice — IQ3_XXS (matches §80 anchor, 76 GB download from ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) vs IQ3_S (matches our canonical quality tier). /home has 92G free: either fits alone, both together do not.
- Config kept: /tmp/strata-m25-iq2.json (server up on :8181 while M25 active; SGLang canonical parked).
## 92. M24 Phase A2: expert-affinity extend staging (HG_EXT_AFF) — SHIPPED OPT-IN; prefill island ~2.3x, staging ops 10.4x cut (the agent, 2026-11-12)

- Island: pyhook/hydragen_moe_native_graph.py, opt-in HG_EXT_AFF=1 (default 0, slice path untouched). Knobs: HG_AFF_TOK (batch token cap, 256), HG_AFF_PAIR (pair cap, 512), HG_AFF_EX (expert cap, _CACHE_N). Launcher launch_phase2_m24a2.sh = canonical + HG_EXT_AFF=1. Snapshot: .a2-working (135-line diff vs .bak-a2prof).
- Mechanism: sort window (token,slot) pairs by expert, batch into applies (experts<=CN, tokens<=TOK, pairs<=PAIR), gather xg via index_select, one fresh xf/out per apply, combine via res.index_add_. Same pair compute as slice path, but each expert staged once per window.
- Measured (T=1024 extend window, prof arm): staging ops M 4416 -> ~420 (10.4x), window apply 1187 -> ~517 ms (prof tax included).
- TTFT 6000-char greedy: A2 lean 32.1/34.6/34.7 vs same-day canonical lean 68.9/44.6/39.9; steady-tail 32.1 vs 39.9 = -20%; cold-first-seed 32.1 vs 68.9 = -53% (seed-order/wire-cache effect).
- Quality: battery_quality 8/8 QC_DONE, no QC5/6/7 echo degeneration; LONG 1262-token prime oracle EXACT "211, 223, 227, 229, 233, 239, 241, 251, 257". Fairness warm 39.3/40.5->44.6 (canonical band).
- Decode untouched: T<=TCHUNK keeps §86 slice path; graphs unaffected.
- LAWS banked (bug hunt, this section):
  1. native_expert_grouped entry semantics: out row = ent_dst (GLOBAL row of out, not per-token slot; per-token j must be remapped tokpos*K + j). Writing bare j aliases every token into rows 0..K-1 -> coherent-but-empty or garbage outputs, downstream qsa_indexer illegal access.
  2. cap_groups/cap_entries must equal ACTUAL nul/E of the call (grp_start has nul+1 entries; larger cap = OOB group bounds).
  3. Any scratch/xq8 growth that swaps st buffers must be pre-sized OUTSIDE the apply loop (freed block while prior _ng still in flight = corruption); keep-alive store in st avoids stream-allocator races.
  4. DBG arms that re-run the slice path after affinity DOUBLE-COUNT res by construction (rel=0.5/rel2=1.5/rel4=3.5 signature = identical results, not half-results).
  5. _METAB arena: values must be python ints for torch.tensor mixes; np.int64 conversion works but dump hid the real issue — trust coverage counters (aff_E vs slc_E) over max-diff forensics.
- Not adopted canonical yet: soak + co-edit serialization pending; A2 stays opt-in. §93 (M25) next: C++ execution worker (libhydragen_core.so / bin/strata) to escape PyTorch mem_fraction_static hold and fund CN=32/48 properly.
## 91. M24 Phase A1b: native fused GDN decode wired live — mechanism PASS, economics NEUTRAL (the agent, 2026-11-12)

Artifacts (the agent-owned; co-edit only with CHANGES entry):
- bridge/hydragen_bridge_gdn.cu v2: added hg_gdn_decode_fused — pool-native fused decode kernel,
  grid(B*HV, V/4), one warp per v-column, i-dim 32-strided coalesced; fuses q/k l2norm+scale,
  softplus/-exp(A_log) gating, sigmoid beta, group pairing i_h=hv/(HV//HK), in-place state IO,
  dead-slot zero path; templated pool x act x slot-idx (fp32/bf16 x fp32/bf16 x int32/int64).
  Zero permute: operates directly on SGLang pool (slots,HV,V,K).
- test_gdn_fused_parity.py: PASS all 8 dtype combos vs live Triton packed_decode, 8 chained steps:
  fp32/fp32 out 5.96e-08 state 7.15e-07; bf16 worst out 9.77e-04 state 1.56e-02 (bf16 quantum).
- pyhook/hydragen_gdn_native.py (HG_GDN_NATIVE=1, sitecustomize-gated): wraps
  TritonGDNKernel.packed_decode; guards ReplaySSM/dtype/shape/stride -> Triton fallback;
  derives HK from qkv_dim assuming K==V (holds: DK=DV=128); launches on current stream
  (graph-capture-safe); counters log native hits / fallback reason.
- launch_phase2_m24g1.sh (= cn24c + HG_GDN_NATIVE=1), launch_phase2_m24g1cp.sh (+ prof).
- sitecustomize.py: appended one gated import block (§91 comment). Canonical launch_phase2_m11d5cn24c.sh unchanged.

Build/ABI lessons: nvcc wants __forceinline__ (not MSVC __forceinline); templates must live
OUTSIDE extern "C"; void*->typed pointer needs explicit casts inside a non-template launcher
helper (<<<>>> will not implicitly convert void*); macro bodies with if/else need do{...}while(0);
shfl_down reduction full-sum lives in lane 0 only — broadcast with __shfl_sync before use
(caught by parity: state err 1.86 -> 7e-7).

Live boot (m24g1 lean): H_200; "[gdn-native] installed" x4 procs; native hits=64 triton=0
through bs1+bs4 breakable-graph capture; 693 MiB free (canonical ~679); no exceptions.

Quality (m24g1): battery_quality 8/8; QC6 direct exact primes 97,101,103,107,109; QC7 correct;
QC2 haiku / QC4 fib / QC5 fraction clean. QC0/QC1 think-preambles = known canonical boot drift.

Fairness (m24g1 lean): 41.13/41.28 then 45.53/47.39 warm. Do NOT compare across boot epochs:
paired profiling same-epoch below is authoritative.

Paired profiling A/B (m24g1cp vs cn24cp, identical probe sequence SS x3 + fairness):
- island apply T=1: native 1.19 vs canon 1.22 ms/layer (island untouched, as designed)
- step fwd bs4 deep-warm last20: native 63.5 vs canon 60.0 ms -> NEUTRAL (drift band)
- single-stream SS3: 20.5 vs 22.1 tok/s -> NEUTRAL
- fairness prof: 36.16/42.63 vs 42.77/37.76 -> NEUTRAL overlap

LAW (decode-step anatomy, cn24 canonical): bs4 decode step ~60 ms ≈ 48 layers x ~1.2 ms island
apply ≈ 58 ms; ALL non-MoE torch/triton (incl. Triton GDN packed decode + conv + QSA FA + dense)
shares ~3-6 ms total. GDN fusion ceiling is therefore ~1-2 ms/step (2-3%), below boot drift.
Closes GDN/QSA/HC fusion-for-decode as NOT-ECONOMIC on the staging-bound substrate (same death
as M19/§79b/§85: decode is island-staging-bound; attack island apply h2d/cs/preadd or prefill,
not the ~5% non-MoE remainder). Hook + kernel stay on disk, opt-in HG_GDN_NATIVE=1, verified
capture-safe and quality-neutral.

Next per §89 plan: A2 prefill expert-staging ring (extend windows: island apply 16.8 s/1024-tok
chunk, h2d 10.2 + cs 5.2 + preadd 3.6 — that IS the prefill gap).

Canonical restored: launch_phase2_m11d5cn24c.sh relaunched (PID in logs/sg-real.log).
## 90. M24 Phase A1: libstrata_gdn_kernels.so build + GDN/QSA parity — PASS (the agent, 2026-11-12)

Ownership (as declared in §89): bridge/hydragen_bridge_gdn.cu, build_gdn_kernels.sh,
libstrata_gdn_kernels.so, test_gdn_parity.py, test_gdn_triton_parity2.py — the agent.

### Inventory (main-engine single-card kernels)
- src/kernels/cuda/gdn.cu          — reference GDN layer ops (self-contained, 248 L).
- src/kernels/cuda/native_gdn.cu   — pinned llama.cpp-derived fused GDN step (warp-shuffle).
- src/kernels/cuda/native_qsa.cu   — QSA rms_norm_weighted / rms_norm_grouped / gate_apply.
- include/strata/kernels/{gdn,native_gdn,native_qsa,native_qsa_indexer}.hpp — contracts.
- Vendored third_party/ggml NOT needed by these three TUs (only gdn/native_gdn/native_qsa
  includes + strata/core/geometry.hpp) — build is clean without ggml-cuda.

### Build
- build_gdn_kernels.sh: /usr/local/cuda/bin/nvcc -O3 -use_fast_math -arch=sm_86 -shared -fPIC
  gdn.cu native_gdn.cu native_qsa.cu bridge/hydragen_bridge_gdn.cu -> libstrata_gdn_kernels.so
- Status: COMPILED first try (1,175,328 B ELF .so). Exports hg_* flat C ABI (9 entry points).
- Flat-ABI shim required because GdnShapes is passed by const-ref (not ctypes-friendly).

### ABI contracts pinned empirically (these were NOT fully documented in headers)
1. strata ref gdn_step state layout is (S_i, h_v, S_j): k-dim OUTERMOST, head middle,
   v-dim fastest — NOT per-head (h,k,v). native_gdn_step same.
2. gate argument is LOG-space (kernel computes expf(gate) internally).
3. beta argument is pre-sigmoid (caller gates; gdn_beta_gate is the gating op).
4. l2norm of q,k is an UPSTREAM preprocess for BOTH kernels (native_gdn.cu contains none;
   the "fused l2" in split gdn_core.cuh is the multi-card engine only).
5. Readout scale: ref path = caller pre-scales q (scale*K); native path = 1/sqrt(S) internal
   (native_gdn.cu:118).
6. Head pairing: strata indexes k/q rows by h % h_k (CYCLIC). SGLang pairs hv->h by GROUP
   (hv // (HV//HK)). BRIDGE ADAPTER: pre-expand q/k rows to (HV, DK) with group mapping and
   present h_k = HV so the modulo becomes identity.
7. Native API requires a NON-NULL cuda stream (torch default stream handle 0 throws
   std::invalid_argument); use a dedicated torch.cuda.Stream().
8. qsa rms_norm_weighted/grouped: row-major rows of n_cols with gamma[n_cols]; gate_apply:
   attn (n_head, head_dim), q_full (n_head, 2*head_dim) rows = [q|gate], head-major.
9. gdn_conv_step conv_state is (C, d_conv-1) channel-major, oldest-first per channel row;
   kernel[0] taps OLDEST, new input appended at last tap.

### Parity results
- Unit parity (test_gdn_parity.py vs independent torch reference, 12 checks):
  WORST 1.192e-06 PASS (step ref/native state+out, l2, beta, conv+shift+append, outnorm,
  qsa_rms, qsa_gate).
- Triton parity (test_gdn_triton_parity2.py vs LIVE SGLang kernel
  sglang.kernels.ops.attention.fla.fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode,
  production convention use_qk_l2norm_in_kernel=True, scale=DK^-0.5, HK=16 HV=48 DK=DV=128,
  8 chained decode steps, in-place state pool both sides):
    native=0 (gdn_step):      max out 1.192e-07, max state 4.768e-07
    native=1 (native_gdn_step): max out 5.960e-08, max state 4.768e-07
  PASS < 5e-3 (ULP-level). No drift over 8 steps.
- SGLang decode path located: gdn_backend.py packed_decode -> fla.fused_recurrent packed
  kernel (state pool (slots,HV,V,K) in-place; packed qkv row [q|k|v]).
- Canonical server untouched during A1: H_200 after tests, 11524 MiB used.

### Next (A1b)
pyhook/hydragen_gdn_native.py wiring with HG_GDN_NATIVE opt-in per §89 plan; the adapter
must do (K,HV,V) state permute + group-pairing expansion + l2 preprocess. QSA/HC wiring is
separate (qsa decode/prompt attn kernels not yet exported in the shim — only norm/gate ops
are, since qsa attention itself in SGLang is a Triton FA variant, not the strata QSA path).
## 89. M24-KICKOFF: native kernel borrowing from Strata (Antigravity proposal) — ACCEPTED WITH CORRECTIONS

Proposal reviewed from Antigravity. Files real: strata/src/kernels/cuda/ has fused_gdn,
native_gdn(.preprocess), native_qsa(+indexer,+score), qsa_(decode|prompt)_attn, native_rope,
native_router, native_gr_*, moe prompt kernels; strata/src/split/src/ has gdn_core/qsa/hc/
moe_mma_iq/prefill .cu(h). third_party/ggml is vendored (vecdotq headers present).

CORRECTIONS to the proposal's VRAM audit (live cn24c boot log 07:28, authoritative):
- Load weight end: "mem usage=11.26 GB", avail 11.45 -> 0.19. NOT "~2.9 GB resident".
- Mamba cache: conv 0.04 + ssm 0.90 = 0.94 GB (claim 1.44). Graph bs[1,4] capture: 0.36 GB
  (claim ~1.25). Post-capture avail 0.69 GB (the real headroom, matches §82/§88).
- There is NO "~3.85 GB allocator slack". Observed reserved-UNALLOCATED spikes were
  620-863 MB at OOM moments (§82/§87/§88), not a permanent 3.85 GB pool. Do NOT budget
  work around freeing 3.85 GB; the wall is the 11.26 GB weight region + 0.69 GB real headroom.
- split/ is a SEPARATE multi-card EXECUTABLE (CMake: strata-split-N, one per card count,
  line-protocol driven, anonymous-namespace kernels hardcoded to session layouts, needs
  llama.cpp ggml-cuda headers). "compile split .cuh into a .so" is not a drop-in.
  The MAIN engine kernels (src/kernels/cuda/native_*.cu, include/strata/kernels/*.hpp) are
  free functions in namespace strata::kernels with raw-pointer signatures
  (native_gdn_step(float* state, q,k,v,...), native_qsa_* etc.) — THAT is the borrow source,
  same as §60 iq_kernels precedent.

Phase-0 measurement (cn24cp arm, HG_NATIVE_PROF+HG_STEP_PROF):
- [hg-step] decode fwd avg 71 ms (prof-inflated; lean 50 ms), extend fwd 127 ms avg.
- Decode island apply avg 1.35 ms/layer (cs .72 h2d .4 ng .04) = ~27-31 ms of a ~50 ms lean
  step (per §77/§81 laws). => NON-MoE (GDN/QSA/HC/PLE/router torch+triton) = ~20-25 ms/step,
  i.e. ~40-50% of decode. Fused-kernel borrow ceiling ~= decode 50 -> 35 ms = +40%.
- Prefill remains island-staging dominated (16.8 s/chunk anatomy §83) -> the 15x gap is
  expert STAGING PIPELINE quality, not GDN/QSA torch ops.

ADOPTED SEQUENCE (supersedes proposal ordering):
A1. GDN/QSA/HC native bridge for DECODE: hydragen_bridge_gdn.cu + libstrata_gdn_kernels.so +
    opt-in pyhook/hydragen_gdn_native.py (HG_GDN_NATIVE=1/QSA/HC flags separate), static
    ping-pong scratch INSIDE the bridge (Strata design, safe outside capture). Target:
    decode step -30-40%.
A2. Prefill expert-staging ring port (pf_common/prefill.cu design: batched 64/128-row
    per-expert prompt jobs, pinned ring, async): attacks the 15x prefill gap AND may revive
    HG_EXT_WIN (window buffers live in the C++ ring, not torch's allocator — §87 poison is
    graph-capture + torch pool, not the C++ arena).
A3. Only then revisit CN=32 funding via scratch savings.
A4. Spec decode re-open stays behind A1+A2 (laws §79b/§85: per-row cost must flatten).

Co-edit declaration (file ownership for §89, all NEW names):
pyhook/hydragen_gdn_native.py, bridge/hydragen_bridge_gdn.cu, build_gdn_kernels.sh,
launch_phase2_m24g*.sh, logs/nohup_m24*.out, references m24-*.md.
Antigravity: do NOT touch these; stay off pyhook/hydragen_moe_native_graph.py (canonical
.bak-extwin restored) while A1 runs.

## 88. M23 int8 KV vs CN=32 funding face-off — CLOSED: CN=24 stays canonical; int8 KV validated on canonical

Premise: §82 CN=32 holds the decode record (50.39/44.53) but died unfunded (+221 MB island).
Hypothesis: int8 KV (HG_KV_INT8=1, §74/§75) frees the funding.

Funding reality discovered: on the ctx2048/4096-token canonical the KV pool is TINY —
"KV Cache is allocated. #tokens: 4096, K 0.02 GB, V 0.02 GB". Pool is cap-bound, not
budget-bound; int8 KV frees only ~20 MB there. int8 KV funding matters on the ctx8/65536-pool
arm (long sessions, §76), not the short-context canonical.

Arms (launch_phase2_m11d5cn24i.sh / cn32i.sh / cn28i.sh):
- CN24+int8: HEALTH_OK, 635 MB free, QC 8/8 (QC0 exact), fairness cold 9.86/12.87,
  warm 35.71/43.35 — decode NEUTRAL-TO-POSITIVE vs fp8 CN24. Validated as opt-in on canonical.
- CN32+int8: died in graph INIT both boots: breakable-graph segment instantiate hit
  `CUDA error: out of memory` inside capture (hydragen_kv_int8.py:114 frame); the
  `markCaptureEnd ... no captures in progress` ASSERT is the secondary cascade, not the cause.
  §82's "transient assert + clean retry" did NOT repeat — CN32 is a hard funding wall at 0.95.
- CN28+int8: boots (315 MB free), QC 8/8, fairness warm 33.05/41.57 (NO gain vs CN24i in noise
  band), then DIED on first 6k-char extend probe: Scheduler exception, 108 MB free.
  Confirms §82 law: island slot-cache growth must stay < ~1/3 runtime headroom.

Verdict:
1. CN=24 remains canonical on 12 GB. CN=32 record decode re-opens only on >=16 GB cards or a
   funded KV/weight rebalance — M14 dense floor (Q5_K) blocks the weight route (§72).
2. HG_KV_INT8=1 is validated neutral-to-positive on canonical decode+QC; canonical launcher
   unchanged (fp8), cn24i kept as the int8 opt-in arm; use int8 KV for long-session ctx arms.
3. Law: markCaptureEnd INTERNAL ASSERT at graph init = capture-body CUDA OOM cascade; look for
   `CUDA error: out of memory` at instantiate frames below it, not the assert itself.

Canonical launch_phase2_m11d5cn24c.sh restored, HEALTH_OK, 679 MB free.
## 87. M22 rung3b HG_EXT_WIN window enlargement — CLOSED NOT-FEASIBLE on 12 GB canonical

Goal: enlarge extend/prefill apply window (default _TCHUNK=128 -> opt-in HG_EXT_WIN up to 1024)
so each unique expert stages once per large window instead of re-staging per 128-token block.

Implementation trail (island hydragen_moe_native_graph.py):
- min-persist replay (`_PB` shared arenas, window-level zero/copy): ILLEGAL ACCESS at init.
  persistent reuse of xf/out buffers ACROSS _ng calls is poison (mechanism not pinned;
  pointer offsets and arena sizes verified in-bounds). Empirical law: fresh allocator blocks
  per slice are safe and effectively free via torch block reuse.
- fresh out per slice + window loop (`.bak-working` -> later `bak-window-exp`): boots and gates
  green on canonical: QC 8/8, fairness 16.44/38.57, control 6k probes 45.2/44.3/40.0 s TTFT.
- WIN=512 canonical: CUDA OOM (50 MB window buffers, 768 MB reserved-UNALLOCATED fragmentation).
- bucketed _rows + WIN=384: seed0 TTFT 54.7 s vs control 45.2 s (WORSE), then fragmentation
  OOM at _short_conv (20 MB, 826 MB reserved-unallocated). Slice overhead (13-17 slices per
  384-window vs 6 per 128) ate the staging-bytes saved before memory could fund it.

Laws:
1. Window buffers scale as win*K*H*4 (384->37 MB, 1024->100 MB per slice); on 12 GB at
   mem_fraction 0.95 this fragments the runtime pool to death (20 MB conv1d failures while
   ~800 MB sits reserved-unallocated). Do not retry window>256 without KV-pool shrink.
2. Economics already negative at 6k prompt: more slices = more Python/apply overhead per
   chunk; staging-byte reduction from one-stage-per-expert did NOT win at WIN=384
   (54.7 vs 45.2 s TTFT). Re-opens only on boxes with >= 2 GB runtime headroom AND long
   prefill where h2d bytes dominate; the M-churn anatomy (4190 ops vs ~1400 unique) is
   real but is better attacked via CN funding, not window size.
3. ctx8/q4e-fp8 65536-pool arm with pure §86 island runs 60k cold probes fine
   (seed0 TTFT 737 s at HTIER256, no OOM); window-patch island family died there.
   When a crash has NO Python traceback and null-byte log flood: read the lines above
   "Sleeping 5 seconds before crash diagnostics" in sg-real.log — the real class
   (OOM/illegal) is usually 3-8 lines above it.
4. probe_cold_g.py TARGET_CHARS now env-overridable (TARGET_CHARS=6000 for ctx2048 pairs).

State restored: island = PURE §86 (.bak-extwin), canonical launch_phase2_m11d5cn24c.sh