# M17 int8 GPU pool bump laws (§76, CLOSED PASS)

## Sizing law
- ctx32768 int8 arm: requested --max-total-tokens 78848 -> pool auto-sized 70720
  (mem_fraction_static 0.95 is the wall, not KV bytes; 0.38 G free, GPU 11784 MiB).
- int8 GPU tier ceiling at this substrate/ctx: ~70.7k tokens. Higher ctx arms (ctx131072)
  keep pool 40960 + graph prefill buffers, less headroom.

## Funding law (replaces M15's 2x24k)
- 3 concurrent distinct sessions x17007 ptok: TOTAL wall=1440.9, identical per-leg walls,
  zero evictions (51k co-resident) — probe_fund3.py.
- pair at 23.8k co-resident (47.6k), third 23.8k leg then coexists sequentially: total 71.3k
  within 70720 via pages+retraction headroom — probe_fund.py pair+sequential.
- Practical: fund N sessions ~= pool/N tokens each; 70720 funds 3x~23.5k max, 3x17k comfortably.
- Chunked prefill interleaves concurrent cold prefills: identical per-leg wall time is the
  co-residency signature (a serialized admission would show staggered walls).

## Oracle notes
- probe units: probe_fund3 unit = marker + 10 words ~= 20 tokens; 820 units ~= 17k tokens.
  Oversized units yield HTTP 400 (prompt > ctx), not slow runs.
- grep "token_usage" in server log matches the boot server_args dump — filter by line
  prefix or use decode-log lines instead.
- Launchers: launch_phase2_m11d5ctx8x.sh (bumped pool), probes probe_fund3.py/probe_fund.py.
