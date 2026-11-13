# HydraGen dashboard laws

- Monitor layout law (user spec 2026-11-13): Concurrency States card is the FIRST
  card and replaces Model State entirely (state-card absent from index.html; JS
  state-label/detail/bar updates are null-guarded, #state-badges loop inert).
  Speed card = AGGREGATE: server _tok_s() already sums all parallel streams
  (verified tok_s=85.7 for two ~42 t/s slots); sub-label "sum of N streams".
  slots_view() carries per-slot tok_s/generated under load, held_tokens when idle.
- The Monitor's Concurrency States panel must be the ORIGINAL full panel — per-slot
  tiles (state badge, decode/prefill t/s, progress bar, generated/held tokens) +
  Queue sub-card — lifted from tag experimental-final-20261112 (serve/web
  app.js/index.html/app.css: calcConcurrencyAndQueue + slot-tile grid). A compact
  summary strip was built and REJECTED; user confirmed the tiles are what he wants.
- Branding: title + header brand = HydraGen (not Strata). Strata attribution lives
  in engine/LICENSE + NOTICE only.
- Data mapping onto 0.2.13 /metrics: live.slots[] {state, held_tokens}; per-slot
  tok_s absent -> fall back to live.tok_s (decode) / live.prefill_tok_s_mean
  (prefill); queued derived via calcConcurrencyAndQueue from outside_slots/waiting/
  queued; engine capacity engine.batch_slots (canonical currently 2).
- Live serve tree: the reference host ~/hydragen-experimental/strata/serve/web — static
  from disk, scp suffices, NO engine restart for web changes (server.py changes DO
  need restart via ~/hydragen-experimental/lan_cow_switch.sh).
- server.py HTML routes must send Cache-Control: no-cache (patched; without it the
  browser keeps a stale index.html and new cards silently don't appear even though
  app.js is fresh — guard-based renders skip missing DOM).
- Verify UI work by browser-automation screenshot DURING concurrent streams
  (launch 2+ long max_tokens curl streams on the box first; short 300-400 token
  streams finish before the shot). Never claim a UI fix from grep markers alone.
- LAN dashboard: http://the-LAN-address:8181/ (the reference host), API key in
  strata-iq3xxs-m25-0213-ctx131k-lan-cow.json (chmod 600, vault-grade secret).
- Cache cards: the RAM parked-cache card and the Park disk tier card sit in a
  .cache-pair grid row (RAM left, disk right at desktop width; RAM keeps the
  row alone when the tier is off). The old assumption that the .metrics
  auto-fit grid pairs sibling cards is FALSE - cards there are full width.
- /metrics park_disk comes from serve parsing the engine's stderr park-disk
  log lines (ParkDiskLog in serve/server.py); a new engine log verb needs a
  matching regex there, and serve must be restarted for parser changes.
