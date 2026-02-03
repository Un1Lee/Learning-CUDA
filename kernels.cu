#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <iostream>

// #include "../tester/utils.h"

template <typename T>
__device__ __forceinline__ T warp_reduce(T val) {
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, delta);
    }
    return val; 
}
template <typename T>
__global__ void trace_kernel(const T* __restrict__ input, T* __restrict__ output, size_t min_dim, size_t rows, size_t cols) {
    // extern __shared__ T smem[];
    extern __shared__ char smem_buffer[];
    // 转换为T类型的指针（编译期处理，无运行时开销）
    T* smem = reinterpret_cast<T*>(smem_buffer);
    size_t tid = threadIdx.x; 
    size_t idx = blockIdx.x * blockDim.x + tid;
    // int warp_id = tid / 32;            // Warp ID（块内第几个Warp）
    // int lane_id = tid % 32;            // Warp内线程ID（0~31）

    // smem[tid] = (idx < min_dim) ? input[idx] : 0;
    // __syncthreads();
    T sum = 0;
    for (size_t i = idx; i < min_dim; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    //     if (tid < s) {
    //         smem[tid] += smem[tid + s];
    //     }
    //     __syncthreads();
    // }
    T warp_sum = warp_reduce(sum); // 每个Warp内规约，得到Warp和
    if (tid % 32 == 0) {
        smem[tid / 32] = warp_sum;
    }
    __syncthreads();


    // if (tid == 0) {
    //     atomicAdd(output, smem[0]);
    // }
    if (tid < 32) { // 块内Warp间规约
        T block_sum = (tid < (blockDim.x + 31) / 32) ? smem[tid] : T(0);
        block_sum = warp_reduce(block_sum);
        if (tid == 0) { 
            atomicAdd(output, block_sum);
        }
    }



}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    if (h_input.empty()) return T(0);
    
    size_t min_dim = std::min(rows, cols);
    std::vector<T> h_diag(min_dim);
    for (size_t i = 0; i < min_dim; ++i) {
        h_diag[i] = h_input[i * cols + i]; // 一次性提取对角线元素，避免kernel取时导致非合并访问慢
    }
    size_t bytes = h_diag.size() * sizeof(T);


    T *d_input;
    T *d_result;
    T h_result = 0;

    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_result, sizeof(T));

    cudaMemcpy(d_input, h_diag.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(T)); 

    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, deviceId); 
    // int threadsPerBlock = prop.maxThreadsPerBlock;
    int threadsPerBlock = 512;
    int blocksPerGrid = std::min((int)((min_dim + threadsPerBlock - 1) / threadsPerBlock), 2048);
    // size_t smem_size = threadsPerBlock * sizeof(T); // 共享内存大小=线程数×类型大小
    size_t smem_size = ((threadsPerBlock + 31) / 32) * sizeof(T); 

    // 每个线程计算一个对角线元素；0-512：找到第0-512个对角线数据，然后各个线程的结果相加
    trace_kernel<<<blocksPerGrid, threadsPerBlock, smem_size>>>(d_input, d_result, min_dim, rows, cols);

    cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_result);

    return h_result;
}




/**
 * @brief CUDA Kernel for Flash Attention (Naive implementation for demonstration)
 * 
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 */
template <typename T>
__global__ void flash_attention_kernel(const T* __restrict__ Q, 
        const T* __restrict__ K, const T* __restrict__ V, T* __restrict__ O,
        int batch_size, int tgt_seq_len, int src_seq_len, int query_heads, 
        int kv_heads, int head_dim, bool is_causal, float scale) {
    
    const int Br = 64; 
    const int Bc = 64; 
    int tx = threadIdx.x; 
    
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int seq_block_idx = blockIdx.x;

    int kv_head_idx = head_idx / (query_heads / kv_heads);
    
    // Stride 计算
    long long q_batch_stride = (long long)query_heads * tgt_seq_len * head_dim;
    long long q_seq_stride = (long long)query_heads * head_dim;
    long long q_head_stride = (long long)head_dim;
    long long q_base_offset = batch_idx * q_batch_stride + head_idx * q_head_stride;

    long long k_batch_stride = (long long)kv_heads * src_seq_len * head_dim;
    long long k_seq_stride = (long long)kv_heads * head_dim;
    long long k_head_stride = (long long)head_dim;
    long long k_base_offset = batch_idx * k_batch_stride + kv_head_idx * k_head_stride;

    extern __shared__ float smem[]; 
    float* s_Q = smem;               
    float* s_K = s_Q + Br * head_dim; 
    float* s_V = s_K + Bc * head_dim; 

    float acc_o[128] = {0.0f};       
    
    double m_prev = -INFINITY;       
    double l_prev = 0.0;             

    int q_len_end = min((seq_block_idx + 1) * Br, tgt_seq_len) - seq_block_idx * Br;
    // 偏移 = batch × (seq_len × num_heads × head_dim) +  // 跨batch的总元素数 已算入 q_base_offset
    //    seq × (num_heads × head_dim) +            // 跨seq的总元素数 
    //    head × head_dim +                         // 跨head的总元素数 已算入 q_base_offset
    //    dim                                       // 跨dim的元素数
    // 加载 Q 
    for (int i = tx; i < Br * head_dim; i += blockDim.x) {
        int r = i / head_dim; 
        int c = i % head_dim; 
        if (r < q_len_end) {
            long long cur_q_offset = q_base_offset + (seq_block_idx * Br + r) * q_seq_stride + c;
            s_Q[r * head_dim + c] = static_cast<float>(Q[cur_q_offset]); 
        } else {
            s_Q[r * head_dim + c] = 0.0f; 
        }
    }
    __syncthreads();

    int num_kv_blocks = (src_seq_len + Bc - 1) / Bc;
    
    for (int j = 0; j < num_kv_blocks; j++) {
        int kv_len_end = min((j + 1) * Bc, src_seq_len) - j * Bc;

        // 加载 K, V
        for (int i = tx; i < Bc * head_dim; i += blockDim.x) {
            int r = i / head_dim;
            int c = i % head_dim;
            if (r < kv_len_end) {
                long long cur_k_offset = k_base_offset + (j * Bc + r) * k_seq_stride + c;
                s_K[r * head_dim + c] = static_cast<float>(K[cur_k_offset]); 
                s_V[r * head_dim + c] = static_cast<float>(V[cur_k_offset]); 
            } else {
                s_K[r * head_dim + c] = 0.0f; 
                s_V[r * head_dim + c] = 0.0f; 
            }
        }
        __syncthreads();

        if (tx < Br && tx < q_len_end) {
            int row_idx_global = seq_block_idx * Br + tx;
            
            double m_curr = -INFINITY; 
            double scores[Bc];

            // A. Q * K^T
            for (int k = 0; k < kv_len_end; k++) {
                float dot = 0.0f; 
                for (int d = 0; d < head_dim; d++) {
                    dot += s_Q[tx * head_dim + d] * s_K[k * head_dim + d]; 
                }
                
                if (is_causal && (j * Bc + k > row_idx_global)) {
                    scores[k] = -INFINITY; 
                } else {
                    scores[k] = (double)dot * (double)scale; 
                }
                m_curr = fmax(m_curr, scores[k]);
            }

            // B. Online Softmax (Double Precision)
            double m_new = fmax(m_prev, m_curr); 
            double l_curr = 0.0;                
            
            for (int k = 0; k < kv_len_end; k++) {
                if (scores[k] > -INFINITY) {
                    scores[k] = exp(scores[k] - m_new);
                    l_curr += scores[k];
                } else {
                    scores[k] = 0.0;
                }
            }

            double scale_prev = exp(m_prev - m_new); 
            double l_new = scale_prev * l_prev + l_curr; 

            // 更新 acc_o (混合精度：acc_o 是 float，但乘数是 double)
            // 这比纯 float * float 精度要高
            for (int d = 0; d < head_dim; d++) {
                double pv = 0.0; // 使用 double 累加当前块的 P*V
                for (int k = 0; k < kv_len_end; k++) {
                    pv += scores[k] * (double)s_V[k * head_dim + d]; 
                }
                acc_o[d] = (float)((double)acc_o[d] * scale_prev + pv); 
            }

            m_prev = m_new;
            l_prev = l_new;
        }
        __syncthreads();
    }

    if (tx < Br && tx < q_len_end) {
        int global_seq_idx = seq_block_idx * Br + tx;
        double inv_l = (l_prev < 1e-10) ? 0.0 : 1.0 / l_prev; 

        for (int d = 0; d < head_dim; d++) {
            long long cur_o_offset = q_base_offset + global_seq_idx * q_seq_stride + d;
            O[cur_o_offset] = static_cast<T>((double)acc_o[d] * inv_l); 
        }
    }
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  
    T *d_q, *d_k, *d_v, *d_o;
    size_t q_size = h_q.size() * sizeof(T);
    size_t k_size = h_k.size() * sizeof(T);
    size_t v_size = h_v.size() * sizeof(T);
    size_t o_size = h_o.size() * sizeof(T);

    cudaMalloc((void**)&d_q, q_size);
    cudaMalloc((void**)&d_k, k_size);
    cudaMalloc((void**)&d_v, v_size);
    cudaMalloc((void**)&d_o, o_size);

    cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), k_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), v_size, cudaMemcpyHostToDevice);

    const int Br = 64;
    const int Bc = 64;
    // 使用 128 个线程，大于 Br(64)，方便做内存加载的并行
    int threads_per_block = 128; 
    
    int grid_x = (target_seq_len + Br - 1) / Br; 
    dim3 blocks_per_grid(grid_x, query_heads, batch_size); 
    
    // Shared Memory: Q[Br][D] + K[Bc][D] + V[Bc][D]
    size_t shared_mem_size = (Br * head_dim + Bc * head_dim + Bc * head_dim) * sizeof(float);
    
    float scale = 1.0f / sqrtf((float)head_dim);

    flash_attention_kernel<T><<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        is_causal, scale
    );
    cudaMemcpy(h_o.data(), d_o, o_size, cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);


// *********************************************************************
// Test / Debug Main Function
// Append this to the end of your .cu file
// *********************************************************************

// #include <iostream>
// #include <random>
// #include <iomanip>

// void getGPUInfo(int deviceId = 0) {
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, deviceId);  // 获取第0块显卡的参数
    
//     std::cout << "=== GPU 硬件参数 ===" << std::endl;
//     std::cout << "显卡名称: " << prop.name << std::endl;
//     std::cout << "SM 数量: " << prop.multiProcessorCount << std::endl;
//     std::cout << "每个 SM 最大线程数: " << prop.maxThreadsPerMultiProcessor << std::endl;
//     std::cout << "每个块最大线程数: " << prop.maxThreadsPerBlock << std::endl;
//     std::cout << "每个块最大维度: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
//     std::cout << "每个网格最大维度: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
    
//     // 计算 GPU 理论最大并发线程数（满载阈值）
//     int maxConcurrentThreads = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
//     std::cout << "GPU 理论最大并发线程数（满载）: " << maxConcurrentThreads << std::endl;
// }

// int main() {
//     getGPUInfo();
//     // 1. 设置参数 (Dimensions)
//     int batch_size = 8;
//     int tgt_seq_len = 128; // 输出序列长度
//     int src_seq_len = 128; // 输入序列长度
//     int query_heads = 8;
//     int kv_heads = 8;      // 简单的 Multi-Head Attention (非 GQA)
//     int head_dim = 64;
//     bool is_causal = true; // 开启因果掩码测试

//     std::cout << "Initializing data..." << std::endl;
//     std::cout << "Batch: " << batch_size << ", SeqLen: " << tgt_seq_len 
//               << ", Heads: " << query_heads << ", Dim: " << head_dim << std::endl;

//     // 2. 计算 Tensor 大小
//     size_t q_size = batch_size * tgt_seq_len * query_heads * head_dim;
//     size_t k_size = batch_size * src_seq_len * kv_heads * head_dim;
//     size_t v_size = batch_size * src_seq_len * kv_heads * head_dim;
//     size_t o_size = batch_size * tgt_seq_len * query_heads * head_dim;

//     // 3. 初始化 Host 数据 (使用随机数)
//     std::vector<float> h_q(q_size);
//     std::vector<float> h_k(k_size);
//     std::vector<float> h_v(v_size);
//     std::vector<float> h_o(o_size, 0.0f);

//     std::mt19937 gen(42); // 固定种子以便复现
//     std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

//     for (auto& x : h_q) x = dist(gen);
//     for (auto& x : h_k) x = dist(gen);
//     for (auto& x : h_v) x = dist(gen);

//     // 4. 调用 Flash Attention
//     // 建议：在 ncu 分析前，通常可以先运行一次作为 Warmup，
//     // 但 ncu 默认会捕捉所有 kernel launch，所以直接调用即可。
//     std::cout << "Launching Flash Attention Kernel..." << std::endl;
    
//     flashAttention(h_q, h_k, h_v, h_o, 
//                    batch_size, tgt_seq_len, src_seq_len, 
//                    query_heads, kv_heads, head_dim, is_causal);
//     cudaDeviceSynchronize(); // 确保 GPU 完成

//     // 5. 简单验证输出 (打印前几个值，确保不是全0或NaN)
//     std::cout << "Kernel execution finished." << std::endl;
//     std::cout << "First 5 output values: ";
//     for (int i = 0; i < 5; ++i) {
//         std::cout << std::fixed << std::setprecision(4) << h_o[i] << " ";
//     }
//     std::cout << std::endl;

//     // --- 调用 Trace 进行校验 ---
//     std::cout << "Calculating Trace (Checksum)..." << std::endl;
    
//     // 为了演示 trace，我们将输出视为一个大矩阵。
//     // 只要 rows * cols == o_size 即可。
//     // 这里我们简单地把整个输出看作 (Batch*Seq*Head, Dim) 的矩阵
//     size_t rows = batch_size * tgt_seq_len * query_heads;
//     size_t cols = head_dim;
    
//     float checksum = trace(h_o, rows, cols);

//     std::cout << "------------------------------------------------" << std::endl;
//     std::cout << "Output Trace Checksum: " << checksum << std::endl;
//     std::cout << "------------------------------------------------" << std::endl;


//     return 0;
// }