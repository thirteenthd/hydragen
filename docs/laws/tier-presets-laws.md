# Tier presets (hot-expert seed presets) laws

- Presets = per-layer top-K hot-expert lists (JSON {layer_id: [expert_ids]})
  loaded at boot via HG_TIER_SEED, deferred until graph capture completes.
- Calibration: run representative sessions with HG_ROUTE_DUMP (88B records:
  i32 layer, i32 k, 10 ids, 10 weights), rank experts per layer by
  Σ(weight × tokens), take top-K ≤ HTIER (256/layer on 12GB/60GB hosts,
  ~17 GB pinned; presets are NOT stackable — one per boot).
- Ship interface: tier-seeds/PRESET_<domain>.json + launcher
  HYDRAGEN_TIER_PRESET=<domain> -> HG_TIER_SEED path.
- DECISIVE MEASUREMENT (d5 substrate, coder prompts): matching preset cuts
  cold prefill ~2x (17.3s vs 32.1s; 5.9 vs 13.3s); unrelated-domain prompt
  ties. Cross-domain preset is not penalized below its own floor.
- CRITICAL ORACLE LAW: preset effects are INVISIBLE on floor-bound
  batteries (warm prefill, NOPRE cold-concurrency are staging-bandwidth
  floors). Use mid-length COLD novel-domain probes as the preset oracle.
  This law resolved the earlier "seeding gains don't reproduce" mystery.
- Distinct expert sets across workload domains overlap ~50%; coder trace
  covers ~55% of mixed traffic — per-domain presets are justified, a
  single universal preset is not.
