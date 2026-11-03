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
    m7_files = sorted(glob.glob("benchmarks/results/benchmark_clamped_branchless_kv_m7_*.json"))
    f7 = m7_files[-1] if m7_files else None
    m6_files = sorted(glob.glob("benchmarks/results/benchmark_cp_async_pipeline_m6_*.json"))
    f6 = m6_files[-1] if m6_files else None

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
    m7 = None
    if f7 and os.path.exists(f7):
        with open(f7) as f:
            m7 = json.load(f)
    m6 = None
    if f6 and os.path.exists(f6):
        with open(f6) as f:
            m6 = json.load(f)

    print("=" * 144)
    print(" PIPELINE COMPARISON: BASELINE (M0) vs MTP PRECOND (M5) vs CLAMPED KV (M7) vs CP.ASYNC (M6)")
    print("=" * 144)
    header = f"{'Test Case':<28} | {'Metric':<9} | {'M0 Base':<8} | {'M5 MTP':<8} | {'M7 Clamp':<8} | {'M6 Async':<8} | {'M6 vs M0':<9} | {'M6 vs M7':<9}"
    print(header)
    print("-" * 144)

    m5_records = m5["records"] if m5 else [None] * len(m0["records"])
    m7_records = m7["records"] if m7 else [None] * len(m0["records"])
    m6_records = m6["records"] if m6 else [None] * len(m0["records"])

    for r0, r5, r7, r6 in zip(m0["records"], m5_records, m7_records, m6_records):
        test_name = f"{r0['test_type']} ({r0['prompt_tokens']}p)"

        # TTFT
        ttft0 = r0["ttft_client_s"]
        ttft5 = r5["ttft_client_s"] if r5 else 0.0
        ttft7 = r7["ttft_client_s"] if r7 else 0.0
        ttft6 = r6["ttft_client_s"] if r6 else 0.0
        ttft_diff_m0 = ((ttft6 - ttft0) / ttft0) * 100 if ttft0 > 0 and r6 else 0.0
        ttft_diff_m7 = ((ttft6 - ttft7) / ttft7) * 100 if ttft7 > 0 and r6 else 0.0
        print(f"{test_name:<28} | {'TTFT':<9} | {ttft0:5.2f}s   | {ttft5:5.2f}s   | {ttft7:5.2f}s   | {ttft6:5.2f}s   | {ttft_diff_m0:+6.2f}%   | {ttft_diff_m7:+6.2f}%")

        # Decode speed
        dec0 = r0["decode_client_tok_s"]
        dec5 = r5["decode_client_tok_s"] if r5 else 0.0
        dec7 = r7["decode_client_tok_s"] if r7 else 0.0
        dec6 = r6["decode_client_tok_s"] if r6 else 0.0
        dec_diff_m0 = ((dec6 - dec0) / dec0) * 100 if dec0 > 0 and r6 else 0.0
        dec_diff_m7 = ((dec6 - dec7) / dec7) * 100 if dec7 > 0 and r6 else 0.0
        print(f"{'':<28} | {'Decode':<9} | {dec0:5.1f} t/s | {dec5:5.1f} t/s | {dec7:5.1f} t/s | {dec6:5.1f} t/s | {dec_diff_m0:+6.2f}%   | {dec_diff_m7:+6.2f}%")

        # MTP acceptance rate
        acc0 = r0.get("mtp_acceptance_rate", 0.0) * 100
        acc5 = (r5.get("mtp_acceptance_rate", 0.0) if r5 else 0.0) * 100
        acc7 = (r7.get("mtp_acceptance_rate", 0.0) if r7 else 0.0) * 100
        acc6 = (r6.get("mtp_acceptance_rate", 0.0) if r6 else 0.0) * 100
        acc_diff = acc6 - acc0
        print(f"{'':<28} | {'MTP Acc':<9} | {acc0:5.1f}%   | {acc5:5.1f}%   | {acc7:5.1f}%   | {acc6:5.1f}%   | {acc_diff:+5.1f}% pts |")

        # VRAM hit rate
        hit0 = r0.get("vram_hit_rate", 0.0) * 100
        hit5 = (r5.get("vram_hit_rate", 0.0) if r5 else 0.0) * 100
        hit7 = (r7.get("vram_hit_rate", 0.0) if r7 else 0.0) * 100
        hit6 = (r6.get("vram_hit_rate", 0.0) if r6 else 0.0) * 100
        hit_diff = hit6 - hit0
        print(f"{'':<28} | {'VRAM Hit':<9} | {hit0:5.1f}%   | {hit5:5.1f}%   | {hit7:5.1f}%   | {hit6:5.1f}%   | {hit_diff:+5.1f}% pts |")
        print("-" * 144)

if __name__ == "__main__":
    main()
