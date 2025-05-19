#include <stdio.h>
#include <cuda_runtime.h> // CUDA运行时API
#include <cublas_v2.h>    // cuBLAS库API (虽然GEMM被替换，但保留以防未来使用或对比)
#include <vector>         // 用于初始化
#include <numeric>        // For std::accumulate
#include <algorithm>      // For std::min_element, std::max_element
#include <limits>         // For std::numeric_limits
#include <cstdlib>        // For rand, srand
#include <ctime>          // For time (seeding rand)

// CUDA API调用错误检查宏
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误位于 %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// cuBLAS API调用错误检查宏
#define CHECK_CUBLAS_ERROR(call) do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS 错误位于 %s:%d: 状态码 %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// --- 定义卷积参数 (与提供的C++代码一致) ---
const int KERNEL_NUM_CONST = 64;  // 输出通道数 (卷积核数量)
const int KERNEL_H_CONST = 7;     // 卷积核高度 k_h
const int KERNEL_W_CONST = 7;     // 卷积核宽度 k_w
const int IN_HEIGHT_CONST = 224;   // 输入特征图高度 i_h
const int IN_WIDTH_CONST = 224;    // 输入特征图宽度 i_w
const int IN_CHANNELS_CONST = 1;  // 输入特征图通道数 i_c (C++代码是单通道)

const int NUM_WARMUP_RUNS = 10; // 预热运行次数
const int NUM_TIMED_RUNS = 100; // 计时运行次数

// --- 定义GEMM分块大小 ---
#define TILE_DIM 16 // 分块维度，TILE_M, TILE_N, TILE_K 都设为这个值

/**
 * @brief 融合了im2col和GEMM的CUDA核函数
 * 计算 C[oh][m][n] = sum_k A[m][k] * B_oh[k][n]
 * A 是 d_kernel_flat (M_orig x K_orig)
 * B_oh[k][n] 是从 d_src 通过im2col逻辑动态计算得到的元素，
 * 对应于原始 d_src_im2col_cpp_style[oh + k_row_in_B][n_col_in_B]
 *
 * @param d_kernel_flat 扁平化的卷积核数据 (设备指针, M_orig x K_orig, 行主序)
 * @param d_src 输入特征图数据 (设备指针, HW格式)
 * @param d_output 输出矩阵 (设备指针)
 * @param M_orig A的行数 (kernel_num)
 * @param N_orig B的列数 (outWidth_cpp)
 * @param K_orig A的列数 / B的行数 (kernel_h * kernel_w)
 * @param outHeight_cpp C的批处理维度 (blockIdx.z的范围)
 * @param inHeight 原始输入特征图的高度
 * @param inWidth 原始输入特征图的宽度
 * @param kernel_w_param 卷积核的宽度 (用于im2col计算)
 */
__global__ void fused_im2col_gemm_kernel(const float* d_kernel_flat,
                                         const float* d_src,
                                         float* d_output,
                                         int M_orig, int N_orig, int K_orig,
                                         int outHeight_cpp,
                                         int inHeight_param, int inWidth_param, int kernel_w_param) {
    // --- 共享内存声明 ---
    __shared__ float A_tile[TILE_DIM][TILE_DIM]; // A的子块 (M方向 x K方向)
    __shared__ float B_tile[TILE_DIM][TILE_DIM]; // B的子块 (K方向 x N方向)

    // --- 线程索引和块索引 ---
    int oh = blockIdx.z; // 当前处理的输出高度切片 (C++循环中的 'oh')

    // 目标C矩阵中的全局行索引 (m维度)
    int m_global = blockIdx.y * TILE_DIM + threadIdx.y;
    // 目标C矩阵中的全局列索引 (n维度)
    int n_global = blockIdx.x * TILE_DIM + threadIdx.x;

    float C_val = 0.0f; // 每个线程计算的C元素的部分和

    // --- 遍历K维度进行分块矩阵乘法 ---
    for (int k_block = 0; k_block < (K_orig + TILE_DIM - 1) / TILE_DIM; ++k_block) {
        // --- 加载 A_tile ---
        // A_tile[threadIdx.y][threadIdx.x] 对应 A[m_global][k_base + threadIdx.x]
        int k_a_local = threadIdx.x; // A_tile的列索引 (K方向)
        int k_a_global = k_block * TILE_DIM + k_a_local; // A在K维度的全局索引

        if (m_global < M_orig && k_a_global < K_orig) {
            A_tile[threadIdx.y][k_a_local] = d_kernel_flat[m_global * K_orig + k_a_global];
        } else {
            A_tile[threadIdx.y][k_a_local] = 0.0f;
        }

        // --- 加载 B_tile ---
        // B_tile[threadIdx.y][threadIdx.x] 对应 B_oh[k_base + threadIdx.y][n_global]
        int k_b_local = threadIdx.y; // B_tile的行索引 (K方向)
        int k_b_global = k_block * TILE_DIM + k_b_local; // B_oh在K维度的全局索引

        if (n_global < N_orig && k_b_global < K_orig) {
            // 从 d_src 计算 B_tile[k_b_local][threadIdx.x] 的值
            // B_oh[k_b_global][n_global] 对应于原始 d_src_im2col_cpp_style[oh + k_b_global][n_global]
            int im2col_src_row = oh + k_b_global;
            int im2col_src_col = n_global;

            // 应用im2col逻辑
            int ih_idx_for_src = im2col_src_row / kernel_w_param;
            int kw_offset_for_src = im2col_src_row % kernel_w_param;
            int src_w_unpadded = im2col_src_col + kw_offset_for_src;

            if (src_w_unpadded >= 0 && src_w_unpadded < inWidth_param &&
                ih_idx_for_src >= 0 && ih_idx_for_src < inHeight_param &&
                im2col_src_row < (inHeight_param * kernel_w_param)) { // 确保im2col_src_row在有效范围内
                B_tile[k_b_local][threadIdx.x] = d_src[ih_idx_for_src * inWidth_param + src_w_unpadded];
            } else {
                B_tile[k_b_local][threadIdx.x] = 0.0f;
            }
        } else {
            B_tile[k_b_local][threadIdx.x] = 0.0f;
        }
        __syncthreads(); // 等待所有线程加载完A_tile和B_tile

        // --- 计算子块的点积 ---
        // C_val += sum_k_dot (A_tile[threadIdx.y][k_dot] * B_tile[k_dot][threadIdx.x])
        if (m_global < M_orig && n_global < N_orig) { // 再次检查，因为C_val累加
            for (int k_dot = 0; k_dot < TILE_DIM; ++k_dot) {
                C_val += A_tile[threadIdx.y][k_dot] * B_tile[k_dot][threadIdx.x];
            }
        }
        __syncthreads(); // 等待所有线程完成点积计算，再加载下一批瓦片
    }

    // --- 将结果写回全局内存 ---
    if (m_global < M_orig && n_global < N_orig && oh < outHeight_cpp) {
        d_output[(size_t)oh * M_orig * N_orig + m_global * N_orig + n_global] = C_val;
    }
}


int main() {
    unsigned int seed = 42;
    srand(seed);

    size_t baseline_free_cuda_mem_bytes, baseline_total_cuda_mem_bytes;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&baseline_free_cuda_mem_bytes, &baseline_total_cuda_mem_bytes));
    size_t baseline_used_cuda_mem_bytes = baseline_total_cuda_mem_bytes - baseline_free_cuda_mem_bytes;

    // 使用常量
    int kernel_num = KERNEL_NUM_CONST;
    int kernel_h = KERNEL_H_CONST;
    int kernel_w = KERNEL_W_CONST;
    int inHeight = IN_HEIGHT_CONST;
    int inWidth = IN_WIDTH_CONST;
    int in_channels = IN_CHANNELS_CONST; // C++代码是单通道

    // 计算C++版本定义的输出尺寸
    int outHeight_cpp = inHeight - kernel_h + 1;
    int outWidth_cpp = inWidth - kernel_w + 1;

    printf("基于C++逻辑的参数 (已融合im2col与GEMM):\n");
    printf("输入尺寸: %dx%dx%d (高x宽x通道)\n", inHeight, inWidth, in_channels);
    printf("卷积核: %dx%d, 数量: %d\n", kernel_h, kernel_w, kernel_num);
    printf("C++风格输出尺寸: %dx%d (高x宽), 每个输出点有 %d 个通道\n", outHeight_cpp, outWidth_cpp, kernel_num);

    // --- CPU端数据初始化 (单通道输入) ---
    std::vector<float> h_src_vec(inHeight * inWidth * in_channels);
    for (size_t i = 0; i < h_src_vec.size(); i++) {
        h_src_vec[i] = 0.1f;
    }
    std::vector<float> h_kernel_flat_vec(kernel_num * kernel_h * kernel_w);
    int cnt = 0;
    for (int i = 0; i < kernel_num; i++) {
        for (int j = 0; j < kernel_h; j++) {
            for (int k = 0; k < kernel_w; k++) {
                h_kernel_flat_vec[cnt++] = 0.2f;
            }
        }
    }

    // --- GPU端数据分配 ---
    float *d_src, *d_kernel_flat, *d_output; // d_src_im2col_cpp_style 不再需要

    size_t src_size_bytes = (size_t)inHeight * inWidth * in_channels * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_src, src_size_bytes));

    size_t kernel_flat_size_bytes = (size_t)kernel_num * kernel_h * kernel_w * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_kernel_flat, kernel_flat_size_bytes));

    // 输出矩阵 d_output: outHeight_cpp x kernel_num x outWidth_cpp
    size_t M_orig_gemm = kernel_num;
    size_t N_orig_gemm = outWidth_cpp;
    size_t output_elements = (size_t)outHeight_cpp * M_orig_gemm * N_orig_gemm;
    size_t output_size_bytes = output_elements * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size_bytes));
    printf("输出矩阵 (d_output) 大小: %.2f MB\n", output_size_bytes / (1024.0 * 1024.0));
    printf("中间im2col矩阵不再分配到全局内存。\n");

    // --- 数据从CPU拷贝到GPU ---
    CHECK_CUDA_ERROR(cudaMemcpy(d_src, h_src_vec.data(), src_size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernel_flat, h_kernel_flat_vec.data(), kernel_flat_size_bytes, cudaMemcpyHostToDevice));

    // --- CUDA Events for Timing ---
    cudaEvent_t start_event, stop_event; // 只需要一对事件来计时融合后的核函数
    CHECK_CUDA_ERROR(cudaEventCreate(&start_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_event));

    // --- GEMM 参数 (用于融合核函数) ---
    int M_param = kernel_num;       // A的行数, C的行数
    int N_param = outWidth_cpp;     // B的列数, C的列数
    int K_param = kernel_h * kernel_w; // A的列数, B的行数

    // --- 融合核函数的启动配置 ---
    dim3 block_dim(TILE_DIM, TILE_DIM, 1); // TILE_N, TILE_M
    dim3 grid_dim( (N_param + TILE_DIM - 1) / TILE_DIM, // X方向对应N_param
                   (M_param + TILE_DIM - 1) / TILE_DIM, // Y方向对应M_param
                   outHeight_cpp);                      // Z方向对应outHeight_cpp (oh)

    printf("融合核函数启动配置: Grid(%u,%u,%u), Block(%u,%u,%u)\n", grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y, block_dim.z);

    // --- Warm-up Run ---
    printf("开始预热 (%d 次运行)...\n", NUM_WARMUP_RUNS);
    for (int i = 0; i < NUM_WARMUP_RUNS; ++i) {
        fused_im2col_gemm_kernel<<<grid_dim, block_dim>>>(
                d_kernel_flat, d_src, d_output,
                M_param, N_param, K_param,
                outHeight_cpp, inHeight, inWidth, kernel_w);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    printf("预热完成。\n");

    // --- Timed Execution (Multiple Runs) ---
    printf("开始计时执行 (%d 次运行)...\n", NUM_TIMED_RUNS);
    std::vector<float> fused_kernel_times_ms(NUM_TIMED_RUNS);

    for (int i = 0; i < NUM_TIMED_RUNS; ++i) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaEventRecord(start_event, 0));

        fused_im2col_gemm_kernel<<<grid_dim, block_dim>>>(
                d_kernel_flat, d_src, d_output,
                M_param, N_param, K_param,
                outHeight_cpp, inHeight, inWidth, kernel_w);
        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaEventRecord(stop_event, 0));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&fused_kernel_times_ms[i], start_event, stop_event));

        if ((i + 1) % (NUM_TIMED_RUNS / 10 == 0 ? 1 : NUM_TIMED_RUNS / 10) == 0) {
            printf("  完成运行 %d/%d\n", i + 1, NUM_TIMED_RUNS);
        }
    }

    // 计算平均、最小、最大时间
    auto calculate_stats = [&](const std::vector<float>& times) {
        if (times.empty()) return std::make_tuple(0.0, 0.0, 0.0);
        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        double avg = sum / times.size();
        double min_val = *std::min_element(times.begin(), times.end());
        double max_val = *std::max_element(times.begin(), times.end());
        return std::make_tuple(avg, min_val, max_val);
    };

    auto [avg_fused_time_ms, min_fused_time_ms, max_fused_time_ms] = calculate_stats(fused_kernel_times_ms);

    printf("\n--- 计时结果 (%d 次运行) ---\n", NUM_TIMED_RUNS);
    printf("融合im2col+GEMM核函数时间:\n  平均: %.3f ms, 最小: %.3f ms, 最大: %.3f ms\n",
           avg_fused_time_ms, min_fused_time_ms, max_fused_time_ms);

    // --- 显存占用统计 ---
    size_t final_free_cuda_mem_bytes, final_total_cuda_mem_bytes;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&final_free_cuda_mem_bytes, &final_total_cuda_mem_bytes));
    size_t final_used_cuda_mem_bytes = final_total_cuda_mem_bytes - final_free_cuda_mem_bytes;
    size_t program_gpu_mem_increase_bytes = final_used_cuda_mem_bytes - baseline_used_cuda_mem_bytes;

    printf("\n--- 显存占用统计 ---\n");
    double bytes_to_mb = 1.0 / (1024.0 * 1024.0);
    double initial_main_buffers_mb = (src_size_bytes + kernel_flat_size_bytes) * bytes_to_mb;
    // d_src_im2col_cpp_style 不再分配
    double peak_main_buffers_mb = (src_size_bytes + kernel_flat_size_bytes + output_size_bytes) * bytes_to_mb;
    double main_buffers_increment_mb = (output_size_bytes) * bytes_to_mb; // 仅输出作为增量

    printf("初始主要缓冲区占用 (输入+卷积核): %.2f MB\n", initial_main_buffers_mb);
    printf("峰值主要缓冲区占用 (输入+卷积核+输出): %.2f MB\n", peak_main_buffers_mb);
    printf("主要缓冲区显存增量 (输出, 相对于初始缓冲区): %.2f MB\n", main_buffers_increment_mb);
    printf("\n--- GPU 总显存使用变化 ---\n");
    printf("程序启动前GPU已用总显存: %.2f MB\n", baseline_used_cuda_mem_bytes * bytes_to_mb);
    printf("程序主要计算完成后GPU已用总显存: %.2f MB\n", final_used_cuda_mem_bytes * bytes_to_mb);
    printf("本程序运行导致的GPU总显存增加 (包括库分配): %.2f MB\n", program_gpu_mem_increase_bytes * bytes_to_mb);

    // --- 清理 ---
    // cublasDestroy 不再需要，因为没有创建cuBLAS句柄用于GEMM
    CHECK_CUDA_ERROR(cudaFree(d_src));
    CHECK_CUDA_ERROR(cudaFree(d_kernel_flat));
    // CHECK_CUDA_ERROR(cudaFree(d_src_im2col_cpp_style)); // 已移除
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_event));
    // 其他事件也不再需要

    printf("\nCUDA卷积基准测试 (融合im2col与GEMM) 完成。\n");
    printf("注意: 此融合核函数实现了与原始C++代码等效的特定im2col和GEMM逻辑。\n");
    printf("性能可能因TILE_DIM选择和GPU架构而异。\n");
    return 0;
}
