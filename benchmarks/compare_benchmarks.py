import json
import glob
import sys
import os

def main():
    f0 = "benchmarks/results/benchmark_baseline_m0_1793725043.json"
    f2 = "benchmarks/results/benchmark_lpt_m2_1793726768.json"
    m3_files = sorted(glob.glob("benchmarks/results/benchmark_post_prefill_adapt_m3_*.json"))
    f3 = m3_files[-1] if m3_files else None
    m4_files = sorted(glob.glob("benchmarks/results/benchmark_hybrid_mailbox_m4_*.json"))
    f4 = m4_files[-1] if m4_files else None

    with open(f0) as f:
        m0 = json.load(f)
    with open(f2) as f:
        m2 = json.load(f)
    m3 = None
    if f3 and os.path.exists(f3):
        with open(f3) as f:
            m3 = json.load(f)
    m4 = None
    if f4 and os.path.exists(f4):
        with open(f4) as f:
            m4 = json.load(f)

    print("=" * 108)
    print(" PIPELINE COMPARISON: BASELINE (M0) vs LPT PREFILL (M2) vs POST-PREFILL (M3) vs HYBRID MAILBOX (M4)")
    print("=" * 108)
    header = f"{'Test Case':<30} | {'Metric':<9} | {'M0 Base':<9} | {'M2 LPT':<9} | {'M3 Adapt':<9} | {'M4 Box':<9} | {'M4 vs M0':<9}"
    print(header)
    print("-" * 108)

    m3_records = m3["records"] if m3 else [None] * len(m0["records"])
    m4_records = m4["records"] if m4 else [None] * len(m0["records"])

    for r0, r2, r3, r4 in zip(m0["records"], m2["records"], m3_records, m4_records):
        test_name = f"{r0['test_type']} ({r0['prompt_tokens']}p)"

        # TTFT
        ttft0 = r0["ttft_client_s"]
        ttft2 = r2["ttft_client_s"]
        ttft3 = r3["ttft_client_s"] if r3 else 0.0
        ttft4 = r4["ttft_client_s"] if r4 else 0.0
        ttft_diff = ((ttft4 - ttft0) / ttft0) * 100 if ttft0 > 0 and r4 else 0.0
        print(f"{test_name:<30} | {'TTFT':<9} | {ttft0:6.3f}s   | {ttft2:6.3f}s   | {ttft3:6.3f}s   | {ttft4:6.3f}s   | {ttft_diff:+6.2f}%")

        # Decode speed
        dec0 = r0["decode_client_tok_s"]
        dec2 = r2["decode_client_tok_s"]
        dec3 = r3["decode_client_tok_s"] if r3 else 0.0
        dec4 = r4["decode_client_tok_s"] if r4 else 0.0
        dec_diff = ((dec4 - dec0) / dec0) * 100 if dec0 > 0 and r4 else 0.0
        print(f"{'':<30} | {'Decode':<9} | {dec0:6.1f} t/s | {dec2:6.1f} t/s | {dec3:6.1f} t/s | {dec4:6.1f} t/s | {dec_diff:+6.2f}%")

        # VRAM hit rate
        hit0 = r0.get("vram_hit_rate", 0.0) * 100
        hit2 = r2.get("vram_hit_rate", 0.0) * 100
        hit3 = (r3.get("vram_hit_rate", 0.0) * 100) if r3 else 0.0
        hit4 = (r4.get("vram_hit_rate", 0.0) * 100) if r4 else 0.0
        hit_diff = hit4 - hit0
        print(f"{'':<30} | {'VRAM Hit':<9} | {hit0:5.1f}%    | {hit2:5.1f}%    | {hit3:5.1f}%    | {hit4:5.1f}%    | {hit_diff:+5.1f}% pts")
        print("-" * 108)

if __name__ == "__main__":
    main()
