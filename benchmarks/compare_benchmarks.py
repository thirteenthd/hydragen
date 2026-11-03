import json
import sys

def main():
    f0 = "benchmarks/results/benchmark_baseline_m0_1793725043.json"
    f2 = "benchmarks/results/benchmark_lpt_m2_1793726768.json"

    with open(f0) as f:
        m0 = json.load(f)
    with open(f2) as f:
        m2 = json.load(f)

    print("=" * 78)
    print(" MILESTONE 1 (BASELINE M0) vs MILESTONE 2 (LPT ATTENTION M2)")
    print("=" * 78)
    for r0, r2 in zip(m0["records"], m2["records"]):
        ttft_diff = ((r2["ttft_client_s"] - r0["ttft_client_s"]) / r0["ttft_client_s"]) * 100
        dec_diff = ((r2["decode_client_tok_s"] - r0["decode_client_tok_s"]) / r0["decode_client_tok_s"]) * 100
        test_name = f"{r0['test_type']} ({r0['prompt_tokens']} tok)"
        print(f"\n{test_name:<35}:")
        print(f"  TTFT   : {r0['ttft_client_s']:6.3f}s -> {r2['ttft_client_s']:6.3f}s  [{ttft_diff:+6.2f}%]")
        print(f"  Decode : {r0['decode_client_tok_s']:6.1f} tok/s -> {r2['decode_client_tok_s']:6.1f} tok/s  [{dec_diff:+6.2f}%]")
        if "vram_hit_rate" in r0:
            print(f"  HitRate: {r0['vram_hit_rate']*100:5.1f}% -> {r2['vram_hit_rate']*100:5.1f}%")

if __name__ == "__main__":
    main()
