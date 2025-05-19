#include <stdio.h>
#include <cuda_runtime.h> // CUDA运行时API
#include <cublas_v2.h>    // cuBLAS库API
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

/**
 * @brief CUDA核函数，用于模拟C++版本的im2col_mec逻辑。
 * 将输入特征图 src (HW格式, C=1) 转换为 srcIm2col 矩阵。
 * srcIm2col 的维度为 (inHeight * kernel_w) 行 x outWidth_cpp 列，按行主序存储。
 *
 * @param src 输入特征图数据 (设备指针, HW格式)
 * @param inHeight 原始输入特征图的高度
 * @param inWidth 原始输入特征图的宽度
 * @param kernel_w 卷积核的宽度 (C++ im2col_mec主要使用这个)
 * @param srcIm2col 输出的列化矩阵 (设备指针)
 * @param outWidth_cpp 输出特征图的宽度 (inWidth - kernel_w + 1)
 */
__global__ void im2col_mec_cpp_equivalent(const float* src,
                                          int inHeight, int inWidth,
                                          int kernel_w,
                                          float* srcIm2col,
                                          int outWidth_cpp) {
    // 每个线程计算 srcIm2col 中的一个元素
    // srcIm2col 总元素数: (inHeight * kernel_w) * outWidth_cpp
    // 线程索引映射到 (dest_row_idx, dest_col_idx)
    int dest_col_idx = blockIdx.x * blockDim.x + threadIdx.x; // ow_idx
    int dest_row_idx = blockIdx.y * blockDim.y + threadIdx.y; // iterates 0 to inHeight * kernel_w - 1

    int num_rows_im2col = inHeight * kernel_w;

    if (dest_col_idx < outWidth_cpp && dest_row_idx < num_rows_im2col) {
        // 从 dest_row_idx 反推 ih_idx 和 kw_offset
        int ih_idx = dest_row_idx / kernel_w;
        int kw_offset = dest_row_idx % kernel_w;

        // C++版 im2col_mec 的逻辑: src_w_unpadded = ow_idx + kw_offset (stride=1, padding=0)
        // 这里我们使用 C++ 定义的 outWidth_cpp，它已经假设了 stride=1, padding=0
        // ow_idx (即 dest_col_idx) 是输出列的索引
        int src_w_unpadded = dest_col_idx + kw_offset; // 这是输入图像中的列索引

        float val = 0.0f;
        // 边界检查 (ih_idx 应该总是在 [0, inHeight-1] 范围内，因为 dest_row_idx < inHeight*kernel_w)
        if (src_w_unpadded >= 0 && src_w_unpadded < inWidth &&
            ih_idx >= 0 && ih_idx < inHeight) {
            // src 是 HW 格式 (in_channels = 1)
            val = src[ih_idx * inWidth + src_w_unpadded];
        }

        // srcIm2col 是行主序: (inHeight * kernel_w) 行, outWidth_cpp 列
        srcIm2col[dest_row_idx * outWidth_cpp + dest_col_idx] = val;
    }
}


// 定义卷积参数 (与提供的C++代码一致)
const int kernel_num = 64;  // 输出通道数 (卷积核数量)
const int kernel_h = 7;     // 卷积核高度 k_h
const int kernel_w = 7;     // 卷积核宽度 k_w
const int inHeight = 224;   // 输入特征图高度 i_h
const int inWidth = 224;    // 输入特征图宽度 i_w
const int in_channels = 1;  // 输入特征图通道数 i_c (C++代码是单通道)

const int num_warmup_runs = 10; // 预热运行次数
const int num_timed_runs = 100; // 计时运行次数


int main() {
    unsigned int seed = 42;
    srand(seed);

    size_t baseline_free_cuda_mem_bytes, baseline_total_cuda_mem_bytes;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&baseline_free_cuda_mem_bytes, &baseline_total_cuda_mem_bytes));
    size_t baseline_used_cuda_mem_bytes = baseline_total_cuda_mem_bytes - baseline_free_cuda_mem_bytes;

    // 计算C++版本定义的输出尺寸
    int outHeight_cpp = inHeight - kernel_h + 1;
    int outWidth_cpp = inWidth - kernel_w + 1;

    printf("基于C++逻辑的参数:\n");
    printf("输入尺寸: %dx%dx%d (高x宽x通道)\n", inHeight, inWidth, in_channels);
    printf("卷积核: %dx%d, 数量: %d\n", kernel_h, kernel_w, kernel_num);
    printf("C++风格输出尺寸: %dx%d (高x宽), 每个输出点有 %d 个通道\n", outHeight_cpp, outWidth_cpp, kernel_num);

    // --- CPU端数据初始化 (单通道输入) ---
    std::vector<float> h_src_vec(inHeight * inWidth * in_channels); // in_channels is 1
    for (size_t i = 0; i < h_src_vec.size(); i++) {
        h_src_vec[i] = 0.1f; // 与C++一致
    }
    // 卷积核数据 (kernel_num 个 kernel_h x kernel_w 卷积核)
    std::vector<float> h_kernel_flat_vec(kernel_num * kernel_h * kernel_w);
    int cnt = 0;
    for (int i = 0; i < kernel_num; i++) {
        for (int j = 0; j < kernel_h; j++) {
            for (int k = 0; k < kernel_w; k++) {
                h_kernel_flat_vec[cnt++] = 0.2f; // 与C++一致
            }
        }
    }

    // --- GPU端数据分配 ---
    float *d_src, *d_kernel_flat, *d_src_im2col_cpp_style, *d_output;

    size_t src_size_bytes = (size_t)inHeight * inWidth * in_channels * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_src, src_size_bytes));

    size_t kernel_flat_size_bytes = (size_t)kernel_num * kernel_h * kernel_w * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_kernel_flat, kernel_flat_size_bytes));

    // d_src_im2col_cpp_style 内存分配: (inHeight * kernel_w) 行 x outWidth_cpp 列, 行主序
    size_t src_im2col_rows = (size_t)inHeight * kernel_w;
    size_t src_im2col_cols = (size_t)outWidth_cpp;
    size_t src_im2col_elements = src_im2col_rows * src_im2col_cols;
    size_t src_im2col_size_bytes = src_im2col_elements * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_src_im2col_cpp_style, src_im2col_size_bytes));
    printf("C++风格im2col矩阵 (d_src_im2col_cpp_style) 大小: %.2f MB (%zu rows, %zu cols)\n",
           src_im2col_size_bytes / (1024.0 * 1024.0), src_im2col_rows, src_im2col_cols);

    // 输出矩阵 d_output: outHeight_cpp x kernel_num x outWidth_cpp
    // 存储为 outHeight_cpp 个 (kernel_num x outWidth_cpp) 的行主序矩阵块
    size_t output_elements = (size_t)outHeight_cpp * kernel_num * outWidth_cpp;
    size_t output_size_bytes = output_elements * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size_bytes));
    printf("输出矩阵 (d_output) 大小: %.2f MB\n", output_size_bytes / (1024.0 * 1024.0));


    // --- 数据从CPU拷贝到GPU ---
    CHECK_CUDA_ERROR(cudaMemcpy(d_src, h_src_vec.data(), src_size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernel_flat, h_kernel_flat_vec.data(), kernel_flat_size_bytes, cudaMemcpyHostToDevice));

    // --- CUDA Events for Timing ---
    cudaEvent_t start_event, stop_event, start_gemm_event, stop_gemm_event, start_total_event, stop_total_event;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&start_gemm_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_gemm_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&start_total_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_total_event));


    // --- im2col_mec_cpp_equivalent 内核的启动配置 ---
    // 每个线程处理 srcIm2col 的一个元素
    dim3 block_im2col_cpp(16, 16); // 可调整
    dim3 grid_im2col_cpp(((unsigned int)outWidth_cpp + block_im2col_cpp.x - 1) / block_im2col_cpp.x,
                         ( (unsigned int)(inHeight * kernel_w) + block_im2col_cpp.y - 1) / block_im2col_cpp.y);


    // --- GEMM 参数 (用于cublasSgemm, 模拟C++的行主序操作) ---
    // C_rm(M,N) = A_rm(M,K) * B_rm(K,N)
    // cuBLAS (col-major) using NVIDIA row-major strategy:
    // C_rm_T_as_cm(N,M) = B_rm_T_as_cm(N,K) * A_rm_T_as_cm(K,M)
    // Call cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N_orig, M_orig, K_orig, alpha, B_rm_ptr, ldB_rm_cols, A_rm_ptr, ldA_rm_cols, beta, C_rm_ptr, ldC_rm_cols)

    // M_orig: rows of A_rm and C_rm
    // N_orig: cols of B_rm and C_rm
    // K_orig: cols of A_rm and rows of B_rm
    int M_orig = kernel_num;
    int N_orig = outWidth_cpp;
    int K_orig = kernel_h * kernel_w;

    // For A_rm (d_kernel_flat): M_orig rows, K_orig cols. ldA_rm_cols = K_orig.
    int ldA_rm_cols = K_orig;

    // For B_rm (slice from d_src_im2col_cpp_style): K_orig rows, N_orig cols. ldB_rm_cols = N_orig.
    int ldB_rm_cols = N_orig;

    // For C_rm (slice from d_output): M_orig rows, N_orig cols. ldC_rm_cols = N_orig.
    int ldC_rm_cols = N_orig;

    float alpha = 1.0f;
    float beta = 0.0f;

    // --- Warm-up Run ---
    printf("开始预热 (%d 次运行)...\n", num_warmup_runs);
    cublasHandle_t temp_handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&temp_handle));

    for (int i = 0; i < num_warmup_runs; ++i) {
        im2col_mec_cpp_equivalent<<<grid_im2col_cpp, block_im2col_cpp>>>(
                d_src, inHeight, inWidth, kernel_w,
                d_src_im2col_cpp_style, outWidth_cpp);
        CHECK_CUDA_ERROR(cudaGetLastError());

        for (int oh = 0; oh < outHeight_cpp; ++oh) {
            // B_rm_ptr: Points to the start of the K_orig x N_orig submatrix within d_src_im2col_cpp_style
            // d_src_im2col_cpp_style is (inHeight * kernel_w) rows by outWidth_cpp (N_orig) columns.
            // Each B_rm matrix uses K_orig rows from d_src_im2col_cpp_style, starting at row 'oh'.
            float* current_B_rm_ptr = d_src_im2col_cpp_style + (size_t)oh * N_orig; // Offset to the oh-th row
            float* current_C_rm_ptr = d_output + (size_t)oh * M_orig * N_orig; // Offset for C_rm slice

            // Corrected cublasSgemm call for row-major C = A * B
            // cublasSgemm(handle, op_B, op_A, N_C, M_C, K_AB, alpha, B_ptr, ldB_cols, A_ptr, ldA_cols, beta, C_ptr, ldC_cols)
            CHECK_CUBLAS_ERROR(cublasSgemm(temp_handle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,       // opB, opA (both NoTrans for row-major strategy)
                                           N_orig, M_orig, K_orig,        // N_C, M_C, K_AB
                                           &alpha,
                                           current_B_rm_ptr,  ldB_rm_cols, // B_rm pointer, its number of columns
                                           d_kernel_flat,     ldA_rm_cols, // A_rm pointer, its number of columns
                                           &beta,
                                           current_C_rm_ptr,  ldC_rm_cols  // C_rm pointer, its number of columns
            ));
        }
    }
    CHECK_CUBLAS_ERROR(cublasDestroy(temp_handle));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    printf("预热完成。\n");


    // --- Timed Execution (Multiple Runs) ---
    printf("开始计时执行 (%d 次运行)...\n", num_timed_runs);
    std::vector<float> im2col_times_ms(num_timed_runs);
    std::vector<float> gemm_loop_times_ms(num_timed_runs);
    std::vector<float> total_times_ms(num_timed_runs);

    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));

    for (int i = 0; i < num_timed_runs; ++i) {
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaEventRecord(start_total_event, 0));

        CHECK_CUDA_ERROR(cudaEventRecord(start_event, 0)); // Time im2col
        im2col_mec_cpp_equivalent<<<grid_im2col_cpp, block_im2col_cpp>>>(
                d_src, inHeight, inWidth, kernel_w,
                d_src_im2col_cpp_style, outWidth_cpp);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaEventRecord(stop_event, 0));

        CHECK_CUDA_ERROR(cudaEventRecord(start_gemm_event, 0)); // Time GEMM loop
        for (int oh = 0; oh < outHeight_cpp; ++oh) {
            float* current_B_rm_ptr = d_src_im2col_cpp_style + (size_t)oh * N_orig;
            float* current_C_rm_ptr = d_output + (size_t)oh * M_orig * N_orig;

            CHECK_CUBLAS_ERROR(cublasSgemm(handle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           N_orig, M_orig, K_orig,
                                           &alpha,
                                           current_B_rm_ptr,  ldB_rm_cols,
                                           d_kernel_flat,     ldA_rm_cols,
                                           &beta,
                                           current_C_rm_ptr,  ldC_rm_cols
            ));
        }
        CHECK_CUDA_ERROR(cudaEventRecord(stop_gemm_event, 0));
        CHECK_CUDA_ERROR(cudaEventRecord(stop_total_event, 0));


        CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&im2col_times_ms[i], start_event, stop_event));

        CHECK_CUDA_ERROR(cudaEventSynchronize(stop_gemm_event));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&gemm_loop_times_ms[i], start_gemm_event, stop_gemm_event));

        CHECK_CUDA_ERROR(cudaEventSynchronize(stop_total_event));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&total_times_ms[i], start_total_event, stop_total_event));

        if ((i + 1) % (num_timed_runs / 10 == 0 ? 1 : num_timed_runs / 10) == 0) {
            printf("  完成运行 %d/%d\n", i + 1, num_timed_runs);
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

    auto [avg_im2col_time_ms, min_im2col_time_ms, max_im2col_time_ms] = calculate_stats(im2col_times_ms);
    auto [avg_gemm_loop_time_ms, min_gemm_loop_time_ms, max_gemm_loop_time_ms] = calculate_stats(gemm_loop_times_ms);
    auto [avg_total_time_ms, min_total_time_ms, max_total_time_ms] = calculate_stats(total_times_ms);


    printf("\n--- 计时结果 (%d 次运行) ---\n", num_timed_runs);
    printf("im2col_mec_cpp_equivalent 核函数时间:\n  平均: %.3f ms, 最小: %.3f ms, 最大: %.3f ms\n",
           avg_im2col_time_ms, min_im2col_time_ms, max_im2col_time_ms);
    printf("cublasSgemm 循环时间:\n  平均: %.3f ms, 最小: %.3f ms, 最大: %.3f ms\n",
           avg_gemm_loop_time_ms, min_gemm_loop_time_ms, max_gemm_loop_time_ms);
    printf("CUDA 总耗时 (im2col + GEMM循环):\n  平均: %.3f ms, 最小: %.3f ms, 最大: %.3f ms\n",
           avg_total_time_ms, min_total_time_ms, max_total_time_ms);

    // --- 显存占用统计 ---
    size_t final_free_cuda_mem_bytes, final_total_cuda_mem_bytes;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&final_free_cuda_mem_bytes, &final_total_cuda_mem_bytes));
    size_t final_used_cuda_mem_bytes = final_total_cuda_mem_bytes - final_free_cuda_mem_bytes;
    size_t program_gpu_mem_increase_bytes = final_used_cuda_mem_bytes - baseline_used_cuda_mem_bytes;

    printf("\n--- 显存占用统计 ---\n");
    double bytes_to_mb = 1.0 / (1024.0 * 1024.0);
    double initial_main_buffers_mb = (src_size_bytes + kernel_flat_size_bytes) * bytes_to_mb;
    double peak_main_buffers_mb = (src_size_bytes + kernel_flat_size_bytes + src_im2col_size_bytes + output_size_bytes) * bytes_to_mb;
    double main_buffers_increment_mb = (src_im2col_size_bytes + output_size_bytes) * bytes_to_mb;

    printf("初始主要缓冲区占用 (输入+卷积核): %.2f MB\n", initial_main_buffers_mb);
    printf("峰值主要缓冲区占用 (所有已分配主要缓冲区总和): %.2f MB\n", peak_main_buffers_mb);
    printf("主要缓冲区显存增量 (C++风格im2col+输出, 相对于初始缓冲区): %.2f MB\n", main_buffers_increment_mb);
    printf("\n--- GPU 总显存使用变化 ---\n");
    printf("程序启动前GPU已用总显存: %.2f MB\n", baseline_used_cuda_mem_bytes * bytes_to_mb);
    printf("程序主要计算完成后GPU已用总显存: %.2f MB\n", final_used_cuda_mem_bytes * bytes_to_mb);
    printf("本程序运行导致的GPU总显存增加 (包括库分配): %.2f MB\n", program_gpu_mem_increase_bytes * bytes_to_mb);

    // --- 清理 ---
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
    CHECK_CUDA_ERROR(cudaFree(d_src));
    CHECK_CUDA_ERROR(cudaFree(d_kernel_flat));
    CHECK_CUDA_ERROR(cudaFree(d_src_im2col_cpp_style));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_gemm_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_gemm_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_total_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_total_event));

    printf("\nCUDA卷积基准测试 (基于C++ MEC逻辑) 完成。\n");
    printf("注意: 此实现严格复制了提供的C++代码的im2col和GEMM循环结构。\n");
    printf("其im2col逻辑和GEMM中的B矩阵切片方式特定于该C++代码，可能与标准2D卷积优化方法不同。\n");
    return 0;
}
