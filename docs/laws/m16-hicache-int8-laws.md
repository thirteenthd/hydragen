# M16 HiCache + int8 KV scale-transfer laws (§75, CLOSED PASS)

## Ship design
- int8 KV (HG_KV_INT8=1) composes with HiCache (--enable-hierarchical-cache --hicache-size N)
  once fp32 scale rows ride the host transfer. Shim-only, no fork edits.
- Wrap MHATokenToKVPoolHost.backup_from_device_all_layer and load_to_device_per_layer
  (sglang/srt/mem_cache/pool_host/mha.py; base class is the one instantiated for the MHA KV pool).
- Host mirror: pinned torch.zeros(host_pool.size, H) fp32 lists per layer, lazily allocated
  on first backup; backup copies scale rows D2H per layer, load copies H2D per layer_id
  (map layer_id - device_pool.start_layer to the _hg_sk list, same convention as pool shim).

## Index-device law
- HiCache physical transfer path passes CUDA-resident host_indices AND device_indices.
- Normalize: host_indices.cpu() before indexing pinned mirror; device_indices.to(pool device)
  before indexing device scale tables. Error signature:
  "indices should be either on cpu or on the same device as the indexed tensor (cpu)".

## Measured (ctx131072, int8 GPU pool 40960, hicache-size 8)
- Host tier: 205,376 tokens / 2.52 GB (int8 payload halves bytes/token; bf16 was 149,248 / 3.67 GB).
- Funding law holds: 2x27k cold then re-hit 1.3s cached=27328.
- Addressable KV: 40,960 GPU + 205,376 host = ~246k tokens.
- Correctness oracle probe_m16_correct.py: 64-token greedy continuation on host-restored
  26k-token KV token-identical to cold reference (MATCH PASS). This is the definitive
  scale-transfer gate.
- QC6 note: 91-prefix slips recur on some boots across ALL arms (first-list-element drift);
  16-token direct window 97,101,103 and "Is 91 prime? -> not prime" are the stable reads.

## Launchers
- launch_phase2_m11d5ctx8h.sh = int8 + HiCache long-context arm.
- Proof logs in server log: "[hg-kvint8] write path ACTIVE", "host BACKUP scales ride",
  "host LOAD scales ride" (one-shot each).

## Remaining frontier
- Pool ~78k rung on int8 GPU tier unmeasured (0.74 G free at 40960 in ctx128 arm).
- 4x simultaneously ACTIVE 128k streams still unfunded; host tier funds parked sessions.
