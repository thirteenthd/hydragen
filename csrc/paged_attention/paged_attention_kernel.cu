// SPDX-License-Identifier: Apache-2.0
// Fast Paged Attention CUDA Kernels with Inline PTX DP4A SIMD for Ampere SM_86+
#include "paged_attention_kernel.cuh"
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>
#include <cstdint>

namespace hydragen::paged_attention {

// Ampere SM_86 hardware-accelerated 4-way 8-bit integer dot product with 32-bit accumulation
__device__ __forceinline__ int32_t ptx_dp4a(int32_t a, int32_t b, int32_t c) {
    int32_t d;
    asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
                 : "=r"(d)
                 : "r"(a), "r"(b), "r"(c));
    return d;
}

// Warp-level reduction across 32 threads
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// CUDA block: 1 thread block handles 1 (sequence, head) pair
// Block size: 128 threads for head_dim=128
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
    float kv_scale,
    bool is_int8_kv
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
    __shared__ float s_warp_reduce[4]; // 128 threads = 4 warps

    const int num_context_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Iterate over physical blocks in this sequence's page table
    for (int blk_idx = 0; blk_idx < num_context_blocks; ++blk_idx) {
        int physical_block_id = block_tables[seq_idx * max_blocks + blk_idx];
        if (physical_block_id < 0) break;

        int tokens_in_block = min(BLOCK_SIZE, context_len - blk_idx * BLOCK_SIZE);

        // Process each token inside this physical page
        for (int t = 0; t < tokens_in_block; ++t) {
            float dot = 0.0f;

            if (is_int8_kv && (tid < 32)) {
                // Specialized Warp 0: Computes entire 128-element dot product with 32 threads using PTX dp4a
                // Each thread handles 4 elements: [tid*4 .. tid*4+3]
                int q4_offset = (seq_idx * num_heads + head_idx) * HEAD_DIM + tid * 4;
                // Quantize local 4 Q values to INT8 on the fly
                int8_t q0 = (int8_t)fminf(fmaxf(q[q4_offset + 0] * scale * 64.0f, -128.0f), 127.0f);
                int8_t q1 = (int8_t)fminf(fmaxf(q[q4_offset + 1] * scale * 64.0f, -128.0f), 127.0f);
                int8_t q2 = (int8_t)fminf(fmaxf(q[q4_offset + 2] * scale * 64.0f, -128.0f), 127.0f);
                int8_t q3 = (int8_t)fminf(fmaxf(q[q4_offset + 3] * scale * 64.0f, -128.0f), 127.0f);

                int32_t q_packed = (static_cast<uint32_t>(static_cast<uint8_t>(q0))) |
                                   (static_cast<uint32_t>(static_cast<uint8_t>(q1)) << 8) |
                                   (static_cast<uint32_t>(static_cast<uint8_t>(q2)) << 16) |
                                   (static_cast<uint32_t>(static_cast<uint8_t>(q3)) << 24);

                size_t k_offset = ((size_t)physical_block_id * num_kv_heads + kv_head_idx) * (BLOCK_SIZE * HEAD_DIM) +
                                  (size_t)t * HEAD_DIM + tid * 4;
                int32_t k_packed = *reinterpret_cast<const int32_t*>(&k_cache[k_offset]);

                // Hardware PTX DP4A instruction
                int32_t dp_int = ptx_dp4a(q_packed, k_packed, 0);
                float partial_dot = (float)dp_int * (kv_scale * 0.015625f);

                // Reduce across the 32 threads of Warp 0
                dot = warp_reduce_sum(partial_dot);
                if (tid == 0) {
                    s_logits[t] = dot;
                }
            } else {
                // Fallback / standard path for FP16 or remaining threads
                size_t k_offset = ((size_t)physical_block_id * num_kv_heads + kv_head_idx) * (BLOCK_SIZE * HEAD_DIM) + (size_t)t * HEAD_DIM + lane;
                float k_val = (lane < HEAD_DIM) ? (float)k_cache[k_offset] * kv_scale : 0.0f;
                dot = q_val * k_val;

                // Warp reduce dot product
                dot = warp_reduce_sum(dot);

                int warp_id = tid / 32;
                int lane_id = tid % 32;
                if (lane_id == 0) {
                    s_warp_reduce[warp_id] = dot;
                }
                __syncthreads();

                if (tid == 0) {
                    float total_dot = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < (HEAD_DIM / 32); ++w) {
                        total_dot += s_warp_reduce[w];
                    }
                    s_logits[t] = total_dot;
                }
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
        kv_scale,
        is_int8_kv
    );
}

// Reverse Query-Tile Attention Scheduling (LPT - Longest Processing Time First)
// Prefill Causal Paged Attention for Ampere SM_86+
template<int BLOCK_SIZE = 16, int HEAD_DIM = 128, int TILE_Q = 4, bool REVERSE_LPT = true>
__global__ void paged_attention_prefill_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const int8_t* __restrict__ k_cache,
    const int8_t* __restrict__ v_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens, // past context tokens before this prefill chunk
    int32_t num_seqs,
    int32_t query_len,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t max_blocks,
    float scale,
    float kv_scale,
    bool is_int8_kv
) {
    const int total_query_tiles = (query_len + TILE_Q - 1) / TILE_Q;
    // Reverse Query-Tile Attention Scheduling (LPT):
    // In causal attention, later query tiles attend to past context plus all prior chunk tokens,
    // making them significantly heavier than earlier query tiles. Inverting tile dispatch ensures
    // that SMs launch the heaviest computational workloads first; short tiles backfill the trailing
    // wave to minimize SM idle time.
    const int tile_idx = REVERSE_LPT ? (total_query_tiles - 1 - blockIdx.x) : blockIdx.x;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.z;

    if (seq_idx >= num_seqs || head_idx >= num_heads || tile_idx < 0) return;

    const int past_len = context_lens ? context_lens[seq_idx] : 0;
    const int kv_head_group = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / kv_head_group;

    const int tid = threadIdx.x;
    const int lane = tid;

    const int q_start = tile_idx * TILE_Q;
    const int q_end = min(q_start + TILE_Q, query_len);

    __shared__ float s_logits[BLOCK_SIZE];
    __shared__ float s_warp_reduce[4]; // 128 threads = 4 warps

    // Process each query token in this tile
    for (int q_tok = q_start; q_tok < q_end; ++q_tok) {
        // Causal boundary: attends to past context + prompt tokens up to and including q_tok
        const int context_len = past_len + q_tok + 1;
        if (context_len <= 0) continue;

        // Load query vector element for (seq_idx, q_tok, head_idx)
        size_t q_offset = ((size_t)seq_idx * query_len + q_tok) * num_heads * HEAD_DIM + (size_t)head_idx * HEAD_DIM + lane;
        float q_val = (lane < HEAD_DIM) ? q[q_offset] * scale : 0.0f;

        // Running online softmax state
        float max_score = -FLT_MAX;
        float sum_exp = 0.0f;
        float acc_out = 0.0f;

        const int num_context_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int blk_idx = 0; blk_idx < num_context_blocks; ++blk_idx) {
            int physical_block_id = block_tables[seq_idx * max_blocks + blk_idx];
            if (physical_block_id < 0) break;

            int tokens_in_block = min(BLOCK_SIZE, context_len - blk_idx * BLOCK_SIZE);

            for (int t = 0; t < tokens_in_block; ++t) {
                float dot = 0.0f;

                if (is_int8_kv && (tid < 32)) {
                    // Warp 0 DP4A PTX instruction
                    int q4_offset = ((size_t)seq_idx * query_len + q_tok) * num_heads * HEAD_DIM + (size_t)head_idx * HEAD_DIM + tid * 4;
                    int8_t q0 = (int8_t)fminf(fmaxf(q[q4_offset + 0] * scale * 64.0f, -128.0f), 127.0f);
                    int8_t q1 = (int8_t)fminf(fmaxf(q[q4_offset + 1] * scale * 64.0f, -128.0f), 127.0f);
                    int8_t q2 = (int8_t)fminf(fmaxf(q[q4_offset + 2] * scale * 64.0f, -128.0f), 127.0f);
                    int8_t q3 = (int8_t)fminf(fmaxf(q[q4_offset + 3] * scale * 64.0f, -128.0f), 127.0f);

                    int32_t q_packed = (static_cast<uint32_t>(static_cast<uint8_t>(q0))) |
                                       (static_cast<uint32_t>(static_cast<uint8_t>(q1)) << 8) |
                                       (static_cast<uint32_t>(static_cast<uint8_t>(q2)) << 16) |
                                       (static_cast<uint32_t>(static_cast<uint8_t>(q3)) << 24);

                    size_t k_offset = ((size_t)physical_block_id * num_kv_heads + kv_head_idx) * (BLOCK_SIZE * HEAD_DIM) +
                                      (size_t)t * HEAD_DIM + tid * 4;
                    int32_t k_packed = *reinterpret_cast<const int32_t*>(&k_cache[k_offset]);

                    int32_t dp_int = ptx_dp4a(q_packed, k_packed, 0);
                    float partial_dot = (float)dp_int * (kv_scale * 0.015625f);
                    dot = warp_reduce_sum(partial_dot);
                    if (tid == 0) {
                        s_logits[t] = dot;
                    }
                } else {
                    size_t k_offset = ((size_t)physical_block_id * num_kv_heads + kv_head_idx) * (BLOCK_SIZE * HEAD_DIM) + (size_t)t * HEAD_DIM + lane;
                    float k_val = (lane < HEAD_DIM) ? (float)k_cache[k_offset] * kv_scale : 0.0f;
                    dot = q_val * k_val;
                    dot = warp_reduce_sum(dot);

                    int warp_id = tid / 32;
                    int lane_id = tid % 32;
                    if (lane_id == 0) {
                        s_warp_reduce[warp_id] = dot;
                    }
                    __syncthreads();

                    if (tid == 0) {
                        float total_dot = 0.0f;
                        #pragma unroll
                        for (int w = 0; w < (HEAD_DIM / 32); ++w) {
                            total_dot += s_warp_reduce[w];
                        }
                        s_logits[t] = total_dot;
                    }
                }
                __syncthreads();

                // Online Softmax update
                float logit = s_logits[t];
                float new_max = fmaxf(max_score, logit);
                float exp_diff = expf(max_score - new_max);
                float exp_new = expf(logit - new_max);

                sum_exp = sum_exp * exp_diff + exp_new;
                acc_out = acc_out * exp_diff;

                size_t v_offset = ((size_t)physical_block_id * num_kv_heads + kv_head_idx) * (BLOCK_SIZE * HEAD_DIM) + (size_t)t * HEAD_DIM + lane;
                float v_val = (lane < HEAD_DIM) ? (float)v_cache[v_offset] * kv_scale : 0.0f;

                acc_out += exp_new * v_val;
                max_score = new_max;
                __syncthreads();
            }
        }

        // Final normalization and store out for this q_tok
        if (lane < HEAD_DIM) {
            float final_val = (sum_exp > 0.0f) ? (acc_out / sum_exp) : 0.0f;
            size_t out_offset = ((size_t)seq_idx * query_len + q_tok) * num_heads * HEAD_DIM + (size_t)head_idx * HEAD_DIM + lane;
            out[out_offset] = final_val;
        }
    }
}

void launch_paged_attention_prefill_f16(
    float* out,
    const float* q,
    const void* k_cache,
    const void* v_cache,
    const int32_t* block_tables,
    const int32_t* context_lens,
    int32_t num_seqs,
    int32_t query_len,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t block_size,
    int32_t max_blocks,
    float scale,
    bool is_int8_kv,
    bool reverse_lpt,
    cudaStream_t stream
) {
    constexpr int TILE_Q = 4;
    const int total_query_tiles = (query_len + TILE_Q - 1) / TILE_Q;
    dim3 grid(total_query_tiles, num_heads, num_seqs);
    dim3 block(head_dim); // 128 threads per block

    float kv_scale = is_int8_kv ? 0.015625f : 1.0f;

    if (reverse_lpt) {
        paged_attention_prefill_kernel<16, 128, TILE_Q, true><<<grid, block, 0, stream>>>(
            out,
            q,
            reinterpret_cast<const int8_t*>(k_cache),
            reinterpret_cast<const int8_t*>(v_cache),
            block_tables,
            context_lens,
            num_seqs,
            query_len,
            num_heads,
            num_kv_heads,
            max_blocks,
            scale,
            kv_scale,
            is_int8_kv
        );
    } else {
        paged_attention_prefill_kernel<16, 128, TILE_Q, false><<<grid, block, 0, stream>>>(
            out,
            q,
            reinterpret_cast<const int8_t*>(k_cache),
            reinterpret_cast<const int8_t*>(v_cache),
            block_tables,
            context_lens,
            num_seqs,
            query_len,
            num_heads,
            num_kv_heads,
            max_blocks,
            scale,
            kv_scale,
            is_int8_kv
        );
    }
}

} // namespace hydragen::paged_attention
