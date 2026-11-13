# M25 Strata C++ worker bootstrap laws (§93)

Goal: escape PyTorch mem_fraction_static; fund high-concurrency MoE via static C++ worker.

Bring-up recipe (validated on the reference host 12 GB):
1. Engine: ~/hydragen-experimental/strata/build/strata; front: serve/server.py --engine strata --config <json> --port 8181 (config schema = prod hydragen config: exe/args/cwd/tokenizer/log/port/env/parallel).
2. Pack per quant: tools/iq_pack.py --gguf shard1 --out DIR (index-only, experts stay in GGUF; tokenizer auto-exported).
3. IQ packs REQUIRE --spec T>=2 + --mtp RT dir: MTP draft is architecturally mandatory for native packs; draft costs ~839 MiB VRAM resident.
4. 12 GB boot budget (IQ2_XS): expert-cache auto -> 3343 slots/4.5 GiB; batch 2 slots 0.56 GiB each; prefill ring borrows 2949 slots; 799 MiB free. No torch hold. Contrast island CN=24.
5. Expert arena = full expert bytes as host file cache (33 GiB IQ2), cudaHostRegister ok; THP auto-skipped when free 2MiB blocks < arena.
6. QC battery: no /v1/completions on serve — use /v1/chat/completions chat variant (battery_quality_m25.py), enable_thinking False.
7. TTFT probe law: serve has prompt/conversation caching — REPEATED same prompt = cache hit (prompt_n=7 diff). Cold TTFT needs fresh-seed unique prompts.

Measured IQ2_XS (boot vehicle, NOT quality tier): decode 40-50/stream, c2 agg 72.6, c3 100.2, c4 87.6; prefill 818 tok/s (1303 tok, TTFT 2.4 s) = ~20x SGLang+A2; QC battery 8/8 clean.

Gates for parity (IQ3_XXS, §80 anchor quant + matches canonical routed quant): QC battery + fresh-seed 6000-char TTFT + c1-c4 ladder.

## §94 IQ3_XXS parity RESULT — ADOPTED

- 2410 slots / 3.95 GiB resident (arena 39.97 GiB @3.22 GiB/s); 790 MiB free with everything.
- QC 8/8 exact; TTFT 2.4-3.0 s @684-732 tok/s (=§80 anchor band, 13x SGLang+A2); C1 87 (MTP burst), C2 62.3, C3 76.3, C4 74.2 agg.
- Anchor replay runs FASTER at 32k ctx than the 131k-ctx prod anchor line (expert slots scale inversely with ctx budget: drop max-context to fund residency).
- Config kept at ~/hydragen-experimental/strata-iq3xxs-m25.json; stack on :8181 is now the serving experiment; SGLang parked as fallback.
- pkill/pgrep -f self-match law: kill+launch must live in a script FILE, never the same ssh -c line.
- /home hygiene: 20 GB free after IQ3_XXS pull; IQ2_XS backup (64 GB) deletion is a user-only call before further pulls.

## §95 concurrency decode laws
- `--batch-mtp` required or batch-decoded streams run specless; C2<C1 is spec attribution, not PCIe.
- Per-request `draft_n_accepted` unreliable on batch windows; trust engine line `strata batch: MTP proposals accepted X of Y` (78-84% seen).
- Window ceiling kVerify=8 rows; batch windows up to 2 seq (layers [0,48)); aggregate decode cap ~80-97 tok/s on IQ3_XXS single card; C4 90.9 with bmtp vs 73.9 without.
- Residency is the scaling variable: IQ3_XXS 2410/10240 (23.5%) never >1x solo; IQ2_XS 3343 (33%) hit C3 100.2.
- PCIe never the wall (1-4 ms of 36-43 ms windows; 2.5-11% entries). Tuning (pcie-frac 0.5, adapt-swaps 400) trades CPU-spill (23.7→13.3%) for GPU-wait (+7 ms) — sum flat.
- DFlash2 exists only in split engine (two cards, `--draft dflash2`) + vendored llamacpp; q36 single-GPU engine has zero dflash; doc economics +5-14% greedy code/JSON, negative sampled/agent/4-user.
- Double-server trap: old serve holding :8181 fakes HEALTH_200; after swap verify `tr '\0' '\n' < /proc/$(pgrep -x strata)/cmdline` carries the flags.

## §96 0.2.13/0.1.59 build + DFlash2 verdict
- Build 0.2.13: cmake -G Ninja -DSTRATA_ENABLE_CUDA=ON (default OFF = no strata target); cmake+ninja in ~/hydragen-experimental/.venv-build/bin (ssh PATH misses them); 86-real + cuda-13.1 nvcc; git worktree ~/strata-0213.
- DFlash2: --draft dflash2 --draft-model PACK (+--mtp DIR for draft_vocab; MTP layer unloaded); packs in ~/.models_backup/Strata-data/dflash2 (selq4 = 174 MiB less VRAM); --batch-mtp => STRATA_BATCH_DF2 slot drafters.
- DF2 on 12 GB (supersedes the split-engine-only note above; true for 0.1.57 only): parity with MTP on mixed ladder (C1 95.2 best, C4 90.2); frees 839 vs 591+178 MiB = +188 expert slots; batch drafts ride SPARE rows only (rows deferred telemetry); fresh-session cold-draft penalty; engine self-warns MTP faster on 12 GB. Not adopted.
- Canonical :8181 stack now 0.1.59 bmtp tuned, config ~/hydragen-experimental/strata-iq3xxs-m25-0213.json; no regression vs 0.1.57.
