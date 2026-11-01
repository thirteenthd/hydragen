// SPDX-License-Identifier: Apache-2.0
#include "paged_attention_kernel.cuh"
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>

namespace hydragen::paged_attention {

// CUDA block: 1 thread block handles 1 (sequence, head) pair
template<int BLOCK_SIZE = 16, int HEAD_DIM = 128>
__global__ void paged_attention_decode_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const int8_t* __restrict__ k_cache,
    const int8_t* __restrict__ v_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens,
    int32_t num_seqs,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t max_blocks,
    float scale,
    float kv_scale
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    if (seq_idx >= num_seqs || head_idx >= num_heads) return;

    const int context_len = context_lens[seq_idx];
    if (context_len <= 0) return;

    // GQA: Map query head to corresponding KV head
    const int kv_head_group = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / kv_head_group;

    const int tid = threadIdx.x; // 0 .. HEAD_DIM-1
    const int lane = tid;

    // Load query vector element
    float q_val = (lane < HEAD_DIM) ? q[(seq_idx * num_heads + head_idx) * HEAD_DIM + lane] * scale : 0.0f;

    // Running online softmax state
    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    float acc_out = 0.0f;

    __shared__ float s_logits[BLOCK_SIZE];
    __shared__ float s_warp_reduce[32];

    const int num_context_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Iterate over physical blocks in this sequence's page table
    for (int blk_idx = 0; blk_idx < num_context_blocks; ++blk_idx) {
        int physical_block_id = block_tables[seq_idx * max_blocks + blk_idx];
        if (physical_block_id < 0) break;

        int tokens_in_block = min(BLOCK_SIZE, context_len - blk_idx * BLOCK_SIZE);

        // Process each token inside this physical page
        for (int t = 0; t < tokens_in_block; ++t) {
            // Compute dot product Q . K
            size_t k_offset = ((size_t)physical_block_id * num_kv_heads + kv_head_idx) * (BLOCK_SIZE * HEAD_DIM) + (size_t)t * HEAD_DIM + lane;
            float k_val = (lane < HEAD_DIM) ? (float)k_cache[k_offset] * kv_scale : 0.0f;

            float dot = q_val * k_val;

            // Warp reduce dot product
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }

            int warp_id = tid / 32;
            int lane_id = tid % 32;
            if (lane_id == 0) {
                s_warp_reduce[warp_id] = dot;
            }
            __syncthreads();

            if (tid == 0) {
                float total_dot = 0.0f;
                for (int w = 0; w < (HEAD_DIM / 32); ++w) {
                    total_dot += s_warp_reduce[w];
                }
                s_logits[t] = total_dot;
            }
            __syncthreads();

            // Online Softmax update
            float logit = s_logits[t];
            float new_max = fmaxf(max_score, logit);
            float exp_diff = expf(max_score - new_max);
            float exp_new = expf(logit - new_max);

            sum_exp = sum_exp * exp_diff + exp_new;
            acc_out = acc_out * exp_diff;

            // Load Value vector from physical block and accumulate
            size_t v_offset = ((size_t)physical_block_id * num_kv_heads + kv_head_idx) * (BLOCK_SIZE * HEAD_DIM) + (size_t)t * HEAD_DIM + lane;
            float v_val = (lane < HEAD_DIM) ? (float)v_cache[v_offset] * kv_scale : 0.0f;

            acc_out += exp_new * v_val;
            max_score = new_max;
            __syncthreads();
        }
    }

    // Final normalization
    if (lane < HEAD_DIM) {
        float final_val = (sum_exp > 0.0f) ? (acc_out / sum_exp) : 0.0f;
        out[(seq_idx * num_heads + head_idx) * HEAD_DIM + lane] = final_val;
    }
}

void launch_paged_attention_decode_f16(
    float* out,
    const float* q,
    const void* k_cache,
    const void* v_cache,
    const int32_t* block_tables,
    const int32_t* context_lens,
    int32_t num_seqs,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t block_size,
    int32_t max_blocks,
    float scale,
    bool is_int8_kv,
    cudaStream_t stream
) {
    dim3 grid(num_seqs, num_heads);
    dim3 block(head_dim); // 128 threads per block

    float kv_scale = is_int8_kv ? 0.015625f : 1.0f;

    paged_attention_decode_kernel<16, 128><<<grid, block, 0, stream>>>(
        out,
        q,
        reinterpret_cast<const int8_t*>(k_cache),
        reinterpret_cast<const int8_t*>(v_cache),
        block_tables,
        context_lens,
        num_seqs,
        num_heads,
        num_kv_heads,
        max_blocks,
        scale,
        kv_scale
    );
}

} // namespace hydragen::paged_attention
