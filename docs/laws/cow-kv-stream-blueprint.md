# kv_stream CoW implementation blueprint (grounded v0.2.13, ready-to-build)

## The streaming design (from include/strata/kernels/kv_stream.hpp — read it before patching)
- Streamed QSA layer: AUTHORITATIVE K/V in pinned device-mapped HOST memory (identity layout [block][kv_head][page_size][head_dim]); VRAM n_slots pages = pure cache; page_table = residency map block->slot or -1.
- page_size = 4 cells = 1 indexer block = ALL kv heads, K+V, 4,224 B int8. 20480-cell window = 5120 slots/layer.
- kv_stream_resolve runs ON DEVICE inside captured CUDA graphs: (1) mark resident + claim misses (-1 -> -2 into miss list), (2) CLOCK sweep picks victims (never a slot this call uses), re-points page table, (3) copy host->slot over PCIe. Writers append to host always, slot only if resident.
- KvStreamMap arrays: page_table, slot_block, slot_stamp, slot_ref (clock bit), ctl (epoch/hand/misses/overflow/u64 counters), miss_block, miss_slot. Fixed addresses baked into graphs.
- MTP drafter ring: static table block % n_slots, no resolve.

## CoW surgery (M-C1) — exact shape
1. Pool arena: session.cpp:112 carves per-session pools from session arena. Make K/V pool arrays ONE GLOBAL ARENA PER LAYER ORDINAL (cache budget sized, not per-session window); st.k_q/v_q/scales = arena base + layer stride. Per-session map arrays stay per-session.
2. Device eviction guard: add per-layer device int32 slot_use[] (global). Admission fork (host): for shared-prefix blocks, to.page_table[block] = from slot, atomicAdd host-side staging or via small kernel; bump slot_use. Clock sweep in kv_stream.cu resolve: skip candidate slots with slot_use != owner-only (treat >1 or other-session-pinned as second-chance-immune). Decrement on session free/evict-host.
3. Writers: append to host always (unchanged); write slot only if resident — for SHARED slots, a session writing to a shared block must break sharing (copy-on-write the block to a private slot first). Only boundary blocks get written, so CoW break is rare; unshared prefix interior is append-immune by construction (prefixes only grow at the tail).
4. Graph capture: maps are baked at fixed addresses — global arena bases are also fixed (allocate before any capture). slot_use lives in the same arena.
5. Gates: kv_stream_parity.cpp / kv_window_parity.cpp extended with 2-session shared-map case; then battery_quality_m25 + battery_fairness + ladder5 farm (shared 22k prefix) vs §105 bar: batch4 boot >=2456 expert slots, C4 co-resident >=29 tok/s.

## Prize math (v0.2.13 int8)
- 4,224 B/slot; 5120 slots/layer/session at kv-resident 20480; ~16 full-attn layers => ~346 MB/session VRAM cache.
- Farm (shared 22k prefix + short tails): global pool ~8k slots vs 20k private => ~840 MB returned => +~500-790 expert slots => batch 4 funds; host side M-C2: snapshots already dedupe by reuse, refcount on host identity copy optional.
- b4k8 (§105) is the no-CoW lower bound: same bytes via window shrink, but pays streamed-attention tax (C4 10-23 tok/s). CoW must beat it at full windows.

## Sequencing
M-C1a: global arena, no sharing (pure plumbing) — battery + ladder parity proves allocation/graph safety.
M-C1b: fork + slot_use guard in parity harness ONLY (device tests).
M-C1c: fork at admission in generate.cpp (conversation_kv_copy / direct_slot_copy sites ~14372-14484), farm bench vs bar.
Do NOT attempt device resolve changes without the parity harness extended first.
