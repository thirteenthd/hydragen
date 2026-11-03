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
    m5_files = sorted(glob.glob("benchmarks/results/benchmark_mtp_precondition_m5_*.json"))
    f5 = m5_files[-1] if m5_files else None

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
    m5 = None
    if f5 and os.path.exists(f5):
        with open(f5) as f:
            m5 = json.load(f)

    print("=" * 120)
    print(" PIPELINE COMPARISON: BASELINE (M0) vs LPT (M2) vs ADAPT (M3) vs MAILBOX (M4) vs MTP PRECOND (M5)")
    print("=" * 120)
    header = f"{'Test Case':<28} | {'Metric':<9} | {'M0 Base':<8} | {'M2 LPT':<8} | {'M4 Box':<8} | {'M5 MTP':<8} | {'M5 vs M0':<9}"
    print(header)
    print("-" * 120)

    m3_records = m3["records"] if m3 else [None] * len(m0["records"])
    m4_records = m4["records"] if m4 else [None] * len(m0["records"])
    m5_records = m5["records"] if m5 else [None] * len(m0["records"])

    for r0, r2, r4, r5 in zip(m0["records"], m2["records"], m4_records, m5_records):
        test_name = f"{r0['test_type']} ({r0['prompt_tokens']}p)"

        # TTFT
        ttft0 = r0["ttft_client_s"]
        ttft2 = r2["ttft_client_s"]
        ttft4 = r4["ttft_client_s"] if r4 else 0.0
        ttft5 = r5["ttft_client_s"] if r5 else 0.0
        ttft_diff = ((ttft5 - ttft0) / ttft0) * 100 if ttft0 > 0 and r5 else 0.0
        print(f"{test_name:<28} | {'TTFT':<9} | {ttft0:5.2f}s   | {ttft2:5.2f}s   | {ttft4:5.2f}s   | {ttft5:5.2f}s   | {ttft_diff:+6.2f}%")

        # Decode speed
        dec0 = r0["decode_client_tok_s"]
        dec2 = r2["decode_client_tok_s"]
        dec4 = r4["decode_client_tok_s"] if r4 else 0.0
        dec5 = r5["decode_client_tok_s"] if r5 else 0.0
        dec_diff = ((dec5 - dec0) / dec0) * 100 if dec0 > 0 and r5 else 0.0
        print(f"{'':<28} | {'Decode':<9} | {dec0:5.1f} t/s | {dec2:5.1f} t/s | {dec4:5.1f} t/s | {dec5:5.1f} t/s | {dec_diff:+6.2f}%")

        # MTP acceptance rate
        acc0 = r0.get("mtp_acceptance_rate", 0.0) * 100
        acc2 = r2.get("mtp_acceptance_rate", 0.0) * 100
        acc4 = (r4.get("mtp_acceptance_rate", 0.0) * 100) if r4 else 0.0
        acc5 = (r5.get("mtp_acceptance_rate", 0.0) * 100) if r5 else 0.0
        acc_diff = acc5 - acc0
        print(f"{'':<28} | {'MTP Acc':<9} | {acc0:5.1f}%   | {acc2:5.1f}%   | {acc4:5.1f}%   | {acc5:5.1f}%   | {acc_diff:+5.1f}% pts")

        # VRAM hit rate
        hit0 = r0.get("vram_hit_rate", 0.0) * 100
        hit2 = r2.get("vram_hit_rate", 0.0) * 100
        hit4 = (r4.get("vram_hit_rate", 0.0) * 100) if r4 else 0.0
        hit5 = (r5.get("vram_hit_rate", 0.0) * 100) if r5 else 0.0
        hit_diff = hit5 - hit0
        print(f"{'':<28} | {'VRAM Hit':<9} | {hit0:5.1f}%   | {hit2:5.1f}%   | {hit4:5.1f}%   | {hit5:5.1f}%   | {hit_diff:+5.1f}% pts")
        print("-" * 120)

if __name__ == "__main__":
    main()
