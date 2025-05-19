#include <stdio.h>
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <ctime>

#define BLOCK_SIZE 32

// CUDA API调用错误检查宏
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误位于 %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// CUTLASS GEMM 配置
using Gemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,  // ElementA, LayoutA
        float, cutlass::layout::RowMajor,  // ElementB, LayoutB
        float, cutlass::layout::RowMajor,  // ElementC, LayoutC
        float,                             // ElementAccumulator
        cutlass::arch::OpClassSimt,        // OperatorClass
        cutlass::arch::Sm89                // ArchTag (改为Sm89以匹配RTX 4070)
>;

// MEC im2col 核函数
__global__ void im2col_cuda_mec_optimized(const float* src, int inHeight, int inWidth, int inChannels,
                                          int kernel_w, int stride_w, int padding_w,
                                          float* mecL, int outWidth_new) {
    extern __shared__ float shared_mem[];
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int ih_orig = blockIdx.y * blockDim.y + threadIdx.y;

    if (ow < outWidth_new && ih_orig < inHeight) {
        int shared_idx = threadIdx.y * blockDim.x + threadIdx.x;
        int src_col_start = ow * stride_w - padding_w;
        for (int kw = 0; kw < kernel_w; ++kw) {
            int src_col_unpadded = src_col_start + kw;
            if (src_col_unpadded >= 0 && src_col_unpadded < inWidth) {
                for (int ic = 0; ic < inChannels; ++ic) {
                    int src_idx = ih_orig * (inWidth * inChannels) + src_col_unpadded * inChannels + ic;
                    shared_mem[shared_idx] = src[src_idx];
                }
            } else {
                shared_mem[shared_idx] = 0.0f;
            }
            __syncthreads();

            int mecL_idx = ow * (inHeight * kernel_w * inChannels) +
                           ih_orig * (kernel_w * inChannels) +
                           kw * inChannels + threadIdx.x % inChannels;
            mecL[mecL_idx] = shared_mem[shared_idx];
        }
    }
}

// 卷积参数
const int batch_size = 1;
const int kernel_num = 64;
const int kernel_h = 7;
const int kernel_w = 7;
const int inHeight = 224;
const int inWidth = 224;
const int in_channels = 3;
const int stride_h = 1;
const int stride_w = 1;
const int padding_h = 1;
const int padding_w = 1;
const int num_warmup_runs = 10;
const int num_timed_runs = 100;

int main() {
    srand(42);

    // 计算输出尺寸
    int outHeight_new = (inHeight - kernel_h + 2 * padding_h) / stride_h + 1;
    int outWidth_new = (inWidth - kernel_w + 2 * padding_w) / stride_w + 1;

    // CPU 数据初始化
    std::vector<float> h_src_vec(inHeight * inWidth * in_channels);
    std::vector<float> h_kernel_vec(kernel_num * in_channels * kernel_h * kernel_w);
    for (size_t i = 0; i < h_src_vec.size(); i++) h_src_vec[i] = static_cast<float>(rand()) / (RAND_MAX / 2.0f) - 1.0f;
    for (size_t i = 0; i < h_kernel_vec.size(); i++) h_kernel_vec[i] = static_cast<float>(rand()) / (RAND_MAX / 2.0f) - 1.0f;

    float *h_src = h_src_vec.data();
    float *h_kernel = h_kernel_vec.data();

    // GPU 内存分配
    float *d_src, *d_kernel, *d_mec_L, *d_output;
    size_t src_size_bytes = (size_t)inHeight * inWidth * in_channels * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_src, src_size_bytes));

    size_t kernel_size_bytes = (size_t)kernel_num * in_channels * kernel_h * kernel_w * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, kernel_size_bytes));

    size_t mecL_size_bytes = (size_t)outWidth_new * inHeight * kernel_w * in_channels * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_mec_L, mecL_size_bytes));

    size_t output_size_bytes = (size_t)outHeight_new * outWidth_new * kernel_num * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size_bytes));

    // 拷贝数据到 GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_src, h_src, src_size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel, kernel_size_bytes, cudaMemcpyHostToDevice));

    // CUDA 事件
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Grid 和 Block 尺寸
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_mec((outWidth_new + block.x - 1) / block.x, (inHeight + block.y - 1) / block.y);
    size_t shared_mem_size = BLOCK_SIZE * BLOCK_SIZE * sizeof(float);

    // 预热
    for (int i = 0; i < num_warmup_runs; ++i) {
        im2col_cuda_mec_optimized<<<grid_mec, block, shared_mem_size>>>(d_src, inHeight, inWidth, in_channels,
                                                                        kernel_w, stride_w, padding_w, d_mec_L, outWidth_new);
        typename Gemm::Arguments arguments{
                {outWidth_new, kernel_num, kernel_h * kernel_w * in_channels}, // 问题尺寸 m, n, k
                d_mec_L,    // 矩阵A
                d_kernel,   // 矩阵B
                d_output,   // 矩阵C（输入）
                d_output,   // 矩阵D（输出）
                {1.0f, 0.0f} // alpha, beta
        };
        Gemm gemm_op;
        gemm_op(arguments);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 计时运行
    std::vector<float> kernel_times(num_timed_runs);
    std::vector<float> gemm_times(num_timed_runs);

    for (int i = 0; i < num_timed_runs; ++i) {
        CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
        im2col_cuda_mec_optimized<<<grid_mec, block, shared_mem_size>>>(d_src, inHeight, inWidth, in_channels,
                                                                        kernel_w, stride_w, padding_w, d_mec_L, outWidth_new);
        CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernel_times[i], start, stop));

        CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
        typename Gemm::Arguments arguments{
                {outWidth_new, kernel_num, kernel_h * kernel_w * in_channels}, // 问题尺寸 m, n, k
                d_mec_L,    // 矩阵A
                d_kernel,   // 矩阵B
                d_output,   // 矩阵C（输入）
                d_output,   // 矩阵D（输出）
                {1.0f, 0.0f} // alpha, beta
        };
        Gemm gemm_op;
        gemm_op(arguments);
        CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&gemm_times[i], start, stop));
    }

    // 计算统计数据
    double avg_kernel_time = std::accumulate(kernel_times.begin(), kernel_times.end(), 0.0) / num_timed_runs;
    double avg_gemm_time = std::accumulate(gemm_times.begin(), gemm_times.end(), 0.0) / num_timed_runs;

    // 输出结果
    printf("--- Performance with CUTLASS GEMM ---\n");
    printf("MEC im2col:\n");
    printf("  Avg Time: %.3f ms\n", avg_kernel_time);
    printf("CUTLASS GEMM:\n");
    printf("  Avg Time: %.3f ms\n", avg_gemm_time);

    // 清理
    CHECK_CUDA_ERROR(cudaFree(d_src));
    CHECK_CUDA_ERROR(cudaFree(d_kernel));
    CHECK_CUDA_ERROR(cudaFree(d_mec_L));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return 0;
}