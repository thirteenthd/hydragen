# M9 loader-skip utility (boot-time shard skipping)

Shim: `~/hydragen-experimental/pyhook/hydragen_loader_skip.py` (v2 semantics).
Opt-in `HG_SKIP_LOAD=1`; launch script runs `--check <model_dir>` preflight first and
only exports HG_SKIP_LOAD when the check stamps OK (`LOADSKIP_FAILED_FULL` fallback = full boot).

## Gate design (learned the hard way)
- Gate on CHECKPOINT identity, never store bytes: sha256(config.json)+sha256(index.json)+
  64KiB head/mid/tail samples of 3 shards; store file-set + PLE size sanity only.
- hg-store bytes are byte-UNSTABLE across boots (native loader rewrites shared
  `torch.from_file(...)` mappings in place; requant ULP churn). Store hashing cannot work.
- Manifest: `~/hg-store/LOADER_SKIP_MANIFEST.json`; stamp: `LOADER_SKIP_STAMP`
  (workers trust fresh stamp <=20 min; preflight avoids racing the loader's store refresh).
- Store glob must be `**/*.bin` (manifest/stamp live in the same dir and self-tripped the file-set check).
- `HG_SKIP_KINDS=exp,ple` selects which tensor classes to skip (bisect knob);
  file-level skip in `_prepare_weights` (pure-class shards), name-level filter in `_get_weights_iterator`.
- Drop counts: full skip 79/131 shards (130.2 GB); exp-only 48 (80.6 GB); ple-only 31 (49.6 GB).

## THE SEED LAW (root cause of skip-boot garbage)
`PagedExpertsMoEMethod.create_moe_runner` (sglang srt/layers/moe/paged_experts/method.py,
~line 165) seeds the whole host store: `host.copy_(layer.<param>.data[:1].cpu().expand_as(host))`.
When expert tensors are skipped that slice is uninitialized garbage and the broadcast
CLOBBERS the verified hg-store through the shared mmap — skipping actively destroys the
bytes it relies on (symptom: degenerate prompt-echo output 'the capital of France is the
capital of'). Fix in shim: HostExpertStore.__init__ stashes `.host` to `_hg_host` and sets
`host = {}` (seed loop iterates zero times); create_moe_runner wrapper restores the dict
afterwards. Version-fragile: wraps native method; re-verify after sglang upgrades.
PLE tables have NO such seed — model file only copies rows as they load.

## Correctness gate
Skip-boot speed is NOT proof. Gate = battery_fairness.py (must return ' Paris' / 'Berlin' /
'Rome'); a raw curl oracle with the chat-template prompt returns im_end on the healthy
model — do NOT use it as the gate. Full boot repairs a clobbered store (loader rewrites
every row); after repair, re-stamp, then skip-boot.

## Boot-time physics
- Full boot 1,014-1,400 s; skip boot 118 s warm; expert-skip-only ~575 s cold.
- Cost is dominated by PLE shard re-read (49.6 GB); page-cache warm/cold swings ~10x
  (3.5 s/it warm vs 16-18 s/it cold; 60 GB RAM cannot hold the 130 GB file set).

## Ops pitfalls (this campaign)
- `pkill -f "sglang::"` self-matches the ssh wrapper command line; use bracket trick
  `sglan[g]::`. Same for launch[_]server (already banked).
- `kill <wrapper>` does NOT kill inner launch_server children; kill PIDs explicitly and
  verify `nvidia-smi` 9 MiB before relaunch.
- Never grep the watch pattern inside the watcher loop (notification spam + false re-fires);
  one-shot greps with exit, and add a lead-in sleep to skip stale log content.
- Backgrounded `cmd1 && cmd2 &` chains can survive ssh detach and run late (timestamp
  files rewritten minutes later); split stamp + launch into separate ssh calls.
