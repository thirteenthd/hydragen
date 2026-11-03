import json
import glob
import sys
import os

def main():
    f0 = "benchmarks/results/benchmark_baseline_m0_1793725043.json"
    f2 = "benchmarks/results/benchmark_lpt_m2_1793726768.json"
    m3_files = sorted(glob.glob("benchmarks/results/benchmark_post_prefill_adapt_m3_*.json"))
    f3 = m3_files[-1] if m3_files else None

    with open(f0) as f:
        m0 = json.load(f)
    with open(f2) as f:
        m2 = json.load(f)
    m3 = None
    if f3 and os.path.exists(f3):
        with open(f3) as f:
            m3 = json.load(f)

    print("=" * 96)
    print(" PIPELINE COMPARISON: BASELINE (M0) vs LPT ATTENTION (M2) vs POST-PREFILL ADAPT (M3)")
    print("=" * 96)
    header = f"{'Test Case':<32} | {'Metric':<10} | {'M0 Base':<10} | {'M2 LPT':<10} | {'M3 Adapt':<10} | {'M3 vs M0':<10}"
    print(header)
    print("-" * 96)

    m3_records = m3["records"] if m3 else [None] * len(m0["records"])

    for r0, r2, r3 in zip(m0["records"], m2["records"], m3_records):
        test_name = f"{r0['test_type']} ({r0['prompt_tokens']}p)"

        # TTFT
        ttft0 = r0["ttft_client_s"]
        ttft2 = r2["ttft_client_s"]
        ttft3 = r3["ttft_client_s"] if r3 else 0.0
        ttft_diff = ((ttft3 - ttft0) / ttft0) * 100 if ttft0 > 0 and r3 else 0.0
        print(f"{test_name:<32} | {'TTFT':<10} | {ttft0:7.3f}s   | {ttft2:7.3f}s   | {ttft3:7.3f}s   | {ttft_diff:+7.2f}%")

        # Decode speed
        dec0 = r0["decode_client_tok_s"]
        dec2 = r2["decode_client_tok_s"]
        dec3 = r3["decode_client_tok_s"] if r3 else 0.0
        dec_diff = ((dec3 - dec0) / dec0) * 100 if dec0 > 0 and r3 else 0.0
        print(f"{'':<32} | {'Decode':<10} | {dec0:7.1f} t/s | {dec2:7.1f} t/s | {dec3:7.1f} t/s | {dec_diff:+7.2f}%")

        # VRAM hit rate
        hit0 = r0.get("vram_hit_rate", 0.0) * 100
        hit2 = r2.get("vram_hit_rate", 0.0) * 100
        hit3 = (r3.get("vram_hit_rate", 0.0) * 100) if r3 else 0.0
        hit_diff = hit3 - hit0
        print(f"{'':<32} | {'VRAM Hit':<10} | {hit0:6.1f}%    | {hit2:6.1f}%    | {hit3:6.1f}%    | {hit_diff:+6.1f}% pts")
        print("-" * 96)

if __name__ == "__main__":
    main()
