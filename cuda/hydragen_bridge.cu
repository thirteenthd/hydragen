#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_moe.hpp"
#include <cstdint>
#include <cstddef>

extern "C" {

void strata_vllm_expert_blob_info(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, uint64_t* out_info) {
    auto L = strata::kernels::native_expert_layout(gu_type, d_type, n_embd, n_ff);
    out_info[0] = (uint64_t)L.gu_row;
    out_info[1] = (uint64_t)L.d_row;
    out_info[2] = (uint64_t)L.up_off;
    out_info[3] = (uint64_t)L.down_off;
    out_info[4] = (uint64_t)L.bytes;
}

uint64_t strata_vllm_expert_scratch_bytes(int64_t cap_entries, int64_t n_ff) {
    return (uint64_t)strata::kernels::native_expert_scratch_bytes(cap_entries, n_ff);
}

void strata_vllm_native_expert_grouped(
    int gu_type, int d_type, int64_t n_embd, int64_t n_ff,
    int64_t n_tokens,
    const unsigned long long* grp_ptr,
    const int32_t* grp_start,
    const int32_t* n_groups,
    const int32_t* ent_dst,
    const int32_t* ent_tok,
    int64_t cap_groups,
    int64_t cap_entries,
    const float* x_f32,
    void* x_q8_1,
    void* scratch,
    float* out,
    void* stream
) {
    strata::kernels::quantize_q8_1_rows(x_f32, n_tokens, n_embd, x_q8_1, stream);
    auto L = strata::kernels::native_expert_layout(gu_type, d_type, n_embd, n_ff);
    strata::kernels::native_expert_grouped(
        L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
        cap_groups, cap_entries, x_q8_1, scratch, out, stream, 0
    );
}

void strata_vllm_moe_combine_multi(
    const float* parts, const float* weights, const float* shared,
    float* output, int64_t n_embd, int64_t k, int n_tok, void* stream
) {
    strata::kernels::native_moe_combine_multi(parts, weights, shared, output, n_embd, k, n_tok, stream);
}

}
