# Phase 2 loader-skip boot utility (M9, CHANGES §41)

Boot: full 1014 s -> skip 164 s cold / 118 s warm. Drops 79/131 shards (130.2 GB:
48 pure-expert + 31 pure-PLE files). `pyhook/hydragen_loader_skip.py` v2.

## Gate law
- NEVER fingerprint hg-store bytes: native loader rewrites shared `torch.from_file`
  mappings every boot; bytes are ULP-unstable. Gate on CHECKPOINT identity instead:
  sha256(config.json)+index.json+64KiB head/mid/tail samples of 3 shards.
- Preflight `--check <ckpt_dir>` stamps; workers trust stamp <=20 min. Launch script
  exports HG_SKIP_LOAD=1 only if check passes.
- `HG_SKIP_KINDS=exp,ple` bisect knob.

## Arming skip for a NEW checkpoint dir (rebuilt/spliced ckpt)
- Subcommands are `--make <dir>` (writes fingerprint manifest) then `--check <dir>`
  (compares + stamps). There is no `--stamp` flag; both subcommands read
  HG_PAGED_STORE_DIR and SGLANG_QWEN4_PLE_FILE_DIR from env — export the same
  values the launch script uses or they die with KeyError.
- Manifest `LOADER_SKIP_MANIFEST.json` lives in HG_PAGED_STORE_DIR and keys ONE
  checkpoint at a time: `--make` for a new dir un-trusts the old one.
- Trusting a new checkpoint is valid only when the skipped classes (exp, ple)
  are byte-identical to the store's source — e.g. expert/ngram shards hardlinked
  from the old checkpoint. Verify hardlinks (inode match) before making.

## The seed trap (root cause, fixed)
`paged_experts/method.py create_moe_runner` seeds the whole host store via
`host.copy_(layer.<param>.data[:1].cpu().expand_as(host))`. With skipped shards the
param is uninitialized -> garbage broadcast CLOBBERS hg-store through the shared mmap.
Fix: HostExpertStore.__init__ parks `host` dict -> seed loop no-ops; create_moe_runner
wrapper restores it after. Shim-wrapped => re-verify on every sglang upgrade.

## Canonical re-anchor after store clobbering
full boot (battery PASS ' Paris'/' Berlin') -> `--check` stamp -> skip boots.
battery_fairness.py is the correctness oracle gate for any boot-path change.

## Boot physics
PLE re-read (~49.6 GB / 31 files) dominates non-skip boot; cold ~16 s/it vs warm
~3.5 s/it; 60 GB RAM cannot hold 130 GB file set -> expect cold reloads.
