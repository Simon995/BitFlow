#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <ctime>

#define BLOCK_SIZE 32

#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误位于 %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_CUBLAS_ERROR(call) do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS 错误位于 %s:%d: 状态码 %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Original im2col CUDA kernel
__global__ void im2col_cuda_original(const float* src, int inHeight, int inWidth, int inChannels,
                                     int kernel_h, int kernel_w, int stride_h, int stride_w,
                                     int padding_h, int padding_w, float* im2col,
                                     int outHeight_new, int outWidth_new) {
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;

    if (oh < outHeight_new && ow < outWidth_new) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                for (int ic = 0; ic < inChannels; ++ic) {
                    int ih = oh * stride_h + kh - padding_h;
                    int iw = ow * stride_w + kw - padding_w;
                    int im2col_idx = (oh * outWidth_new + ow) * (kernel_h * kernel_w * inChannels) +
                                     (kh * kernel_w + kw) * inChannels + ic;
                    if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                        int src_idx = ih * (inWidth * inChannels) + iw * inChannels + ic;
                        im2col[im2col_idx] = src[src_idx];
                    } else {
                        im2col[im2col_idx] = 0.0f;
                    }
                }
            }
        }
    }
}

// Optimized MEC im2col CUDA kernel
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

// Convolution parameters
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

    // Calculate output dimensions
    int outHeight_new = (inHeight - kernel_h + 2 * padding_h) / stride_h + 1;
    int outWidth_new = (inWidth - kernel_w + 2 * padding_w) / stride_w + 1;

    // CPU data initialization
    std::vector<float> h_src_vec(inHeight * inWidth * in_channels);
    std::vector<float> h_kernel_vec(kernel_num * in_channels * kernel_h * kernel_w);
    for (size_t i = 0; i < h_src_vec.size(); i++) h_src_vec[i] = static_cast<float>(rand()) / RAND_MAX;
    for (size_t i = 0; i < h_kernel_vec.size(); i++) h_kernel_vec[i] = static_cast<float>(rand()) / RAND_MAX;

    float *h_src = h_src_vec.data();
    float *h_kernel = h_kernel_vec.data();

    // GPU memory allocation
    float *d_src, *d_kernel, *d_im2col_original, *d_mec_L, *d_output;
    size_t src_size_bytes = (size_t)inHeight * inWidth * in_channels * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_src, src_size_bytes));

    size_t kernel_size_bytes = (size_t)kernel_num * in_channels * kernel_h * kernel_w * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, kernel_size_bytes));

    size_t im2col_original_size_bytes = (size_t)outHeight_new * outWidth_new * kernel_h * kernel_w * in_channels * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_im2col_original, im2col_original_size_bytes));

    size_t mecL_size_bytes = (size_t)outWidth_new * inHeight * kernel_w * in_channels * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_mec_L, mecL_size_bytes));

    size_t output_size_bytes = (size_t)outHeight_new * outWidth_new * kernel_num * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size_bytes));

    // Copy data to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_src, h_src, src_size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel, kernel_size_bytes, cudaMemcpyHostToDevice));

    // CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Grid and block dimensions
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_original((outWidth_new + block.x - 1) / block.x, (outHeight_new + block.y - 1) / block.y);
    dim3 grid_mec((outWidth_new + block.x - 1) / block.x, (inHeight + block.y - 1) / block.y);
    size_t shared_mem_size = BLOCK_SIZE * BLOCK_SIZE * sizeof(float);

    // Warmup runs
    for (int i = 0; i < num_warmup_runs; ++i) {
        im2col_cuda_original<<<grid_original, block>>>(d_src, inHeight, inWidth, in_channels,
                                                       kernel_h, kernel_w, stride_h, stride_w,
                                                       padding_h, padding_w, d_im2col_original,
                                                       outHeight_new, outWidth_new);
        im2col_cuda_mec_optimized<<<grid_mec, block, shared_mem_size>>>(d_src, inHeight, inWidth, in_channels,
                                                                        kernel_w, stride_w, padding_w, d_mec_L, outWidth_new);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Timed runs
    std::vector<float> original_times(num_timed_runs);
    std::vector<float> mec_times(num_timed_runs);

    for (int i = 0; i < num_timed_runs; ++i) {
        CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
        im2col_cuda_original<<<grid_original, block>>>(d_src, inHeight, inWidth, in_channels,
                                                       kernel_h, kernel_w, stride_h, stride_w,
                                                       padding_h, padding_w, d_im2col_original,
                                                       outHeight_new, outWidth_new);
        CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&original_times[i], start, stop));

        CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
        im2col_cuda_mec_optimized<<<grid_mec, block, shared_mem_size>>>(d_src, inHeight, inWidth, in_channels,
                                                                        kernel_w, stride_w, padding_w, d_mec_L, outWidth_new);
        CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&mec_times[i], start, stop));
    }

    // Compute statistics
    double avg_original_time = std::accumulate(original_times.begin(), original_times.end(), 0.0) / num_timed_runs;
    double avg_mec_time = std::accumulate(mec_times.begin(), mec_times.end(), 0.0) / num_timed_runs;

    // Output results
    printf("--- Performance Comparison ---\n");
    printf("Original im2col:\n");
    printf("  Avg Time: %.3f ms\n", avg_original_time);
    printf("  Memory Usage: %.2f MB\n", im2col_original_size_bytes / (1024.0 * 1024.0));
    printf("Optimized MEC im2col:\n");
    printf("  Avg Time: %.3f ms\n", avg_mec_time);
    printf("  Memory Usage: %.2f MB\n", mecL_size_bytes / (1024.0 * 1024.0));

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_src));
    CHECK_CUDA_ERROR(cudaFree(d_kernel));
    CHECK_CUDA_ERROR(cudaFree(d_im2col_original));
    CHECK_CUDA_ERROR(cudaFree(d_mec_L));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return 0;
}