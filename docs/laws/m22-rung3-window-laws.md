# M22 rung3b: HG_EXT_WIN window-enlargement laws (CHANGES §87, CLOSED NOT-FEASIBLE on 12 GB)

Goal was one-stage-per-expert via bigger extend windows. Verdict: not fundable and not faster on 12 GB canonical.

## Hard laws
1. Persistent buffer REUSE across `_ng` calls (shared `_PB` xf/out arenas, window-level zero/copy)
   = CUDA `illegal memory access` (native_expert_grouped/gu) even in eager `--disable-cuda-graph`
   + `CUDA_LAUNCH_BLOCKING=1` arms. Pointer offsets and metas-arena sizes verified in-bounds;
   mechanism never pinned. NEVER retry persistent xf/out reuse; fresh per-slice torch allocations
   are safe and effectively free via allocator block reuse.
2. Window buffer size law: `out_bytes = window_tokens * K * H * 4` (K=10, H=2560):
   W128=13MB, W256=25MB, W384=37MB, W512=50MB, W1024=100MB per slice.
   On 12 GB at mem_fraction_static 0.95 this fragments the runtime pool to death:
   OOM on a 20 MB conv1d while ~768-826 MB sits reserved-UNALLOCATED. Do not retry win>256
   without shrinking --max-total-tokens first. `expandable_segments:True` stays banned (§82 law).
3. Economics were negative even before OOM: WIN=384 first 6k-probe TTFT 54.7 s vs control
   (WIN=0, same island) 45.2/44.3/40.0 s. Slices per chunk rise faster than staging bytes fall:
   ul@384~200 experts = 13 slices/window vs ul@128~90 = 6 slices/window. Python per-apply
   overhead ate the win. Window re-opens only with >=2 GB runtime headroom; the 4190-ops-vs-1400
   unique staging churn is better attacked via CN funding.
4. Silent-death reading law: when a crash has NO Python traceback and the log shows a null-byte
   flood, the real exception class (OOM / illegal / Scheduler hit) is 3-8 lines ABOVE
   `Sleeping 5 seconds before crash diagnostics` in logs/sg-real.log. grep that marker, then
   sed the ~100 lines above it; the full traceback may be 95+ lines long (layer_boundary frames).
5. Bucketing `_rows` to 128-row multiples does NOT defragment this workload (mixed 37/25MB +
   decode 0.4MB + scr regrow `empty_cache()` still wedges the pool).
6. `HG_EXT_WIN` knob (default 0 = inert) + windowed apply code live in
   `pyhook/hydragen_moe_native_graph.py.bak-window-exp` and `.bak-working` for big-VRAM boxes.
   Canonical island = pure §86 = `.bak-extwin`. If any canonical boot dies oddly, restore
   `.bak-extwin` FIRST before blaming launchers.
7. probe_cold_g.py TARGET_CHARS is env-overridable; use TARGET_CHARS=6000 for ctx2048 canonical
   cold pairs, 60000 for ctx32k. ctx8/q4e 60k probes: pure §86 island survives (TTFT ~737 s at
   HTIER256); window-patch island family dies on them — island-vs-launcher discriminator confirmed.
