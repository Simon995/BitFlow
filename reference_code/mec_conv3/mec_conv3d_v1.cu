#include <cuda_runtime.h>
#include <stdio.h>

// CUDA API call error checking macro
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d (%s): %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Define tile size
// 恢复 TILE_DIM 到 16
#define TILE_DIM 16

// Fused im2col, GEMM, and bias addition kernel for 3D convolution
__global__ void fused_im2col_gemm_bias_kernel_3d(
        const float* __restrict__ d_kernel_flat, // Weights (out_channels, in_channels * kernel_d * kernel_h * kernel_w)
        const float* __restrict__ d_src,         // Input (in_channels, in_depth, in_height, in_width) for one batch
        const float* __restrict__ d_bias,        // Bias (out_channels), can be nullptr if no bias
        float* __restrict__ d_output,            // Output (out_channels, out_depth, out_height, out_width) for one batch
        int M_orig, int N_orig, int K_orig,      // GEMM dimensions: M=out_channels, N=out_width, K=in_c*k_d*k_h*k_w
        int outDepth_param, int outHeight_param, // N_orig is outWidth_param
        int inDepth_param, int inHeight_param, int inWidth_param,
        int kernel_d_param, int kernel_h_param, int kernel_w_param,
        int inChannels_param,
        int padding_d_param, int padding_h_param, int padding_w_param,
        int stride_d_param, int stride_h_param, int stride_w_param,
        int od_current) { // Current output depth index

    // Shared memory declaration
    __shared__ float A_tile[TILE_DIM][TILE_DIM];
    __shared__ float B_tile[TILE_DIM][TILE_DIM];

    int oh = blockIdx.z;
    int m_global = blockIdx.y * TILE_DIM + threadIdx.y;
    int n_global = blockIdx.x * TILE_DIM + threadIdx.x;

    // 使用 float 类型进行累加
    float C_val = 0.0f;

    for (int k_block_outer = 0; k_block_outer < (K_orig + TILE_DIM - 1) / TILE_DIM; ++k_block_outer) {
        // Load A_tile (weights)
        int k_a_local = threadIdx.x;
        int m_a_local = threadIdx.y;
        int k_a_global = k_block_outer * TILE_DIM + k_a_local;

        if (m_global < M_orig && k_a_global < K_orig) {
            A_tile[m_a_local][k_a_local] = d_kernel_flat[(size_t)m_global * K_orig + k_a_global];
        } else {
            A_tile[m_a_local][k_a_local] = 0.0f;
        }

        // Load B_tile (im2col input)
        // 性能提示：此处的 im2col 索引计算和全局内存加载是潜在的性能瓶颈。
        // 使用 NVIDIA Nsight Compute 分析此部分的内存访问模式（是否合并）至关重要。
        int k_b_local = threadIdx.y;
        int n_b_local = threadIdx.x;
        int k_b_global = k_block_outer * TILE_DIM + k_b_local;

        if (n_global < N_orig && k_b_global < K_orig) {
            // im2col logic: derive input channel, kernel d,h,w offsets from k_b_global
            // k_b_global iterates through in_channels * kernel_d * kernel_h * kernel_w
            int k_d_x_k_h_x_k_w = kernel_d_param * kernel_h_param * kernel_w_param;
            int channel_idx = k_b_global / k_d_x_k_h_x_k_w;
            int k_idx_in_channel_vol = k_b_global % k_d_x_k_h_x_k_w;

            int kd_offset = k_idx_in_channel_vol / (kernel_h_param * kernel_w_param);
            int k_idx_in_channel_area = k_idx_in_channel_vol % (kernel_h_param * kernel_w_param);
            int kh_offset = k_idx_in_channel_area / kernel_w_param;
            int kw_offset = k_idx_in_channel_area % kernel_w_param;

            // Calculate actual coordinates in the input feature map (considering padding and stride)
            int id_idx = od_current * stride_d_param + kd_offset - padding_d_param;
            int ih_idx = oh * stride_h_param + kh_offset - padding_h_param;
            int iw_idx = n_global * stride_w_param + kw_offset - padding_w_param;

            // Boundary check and load data
            // 数值偏差提示：如果在大卷积核时出现偏差，需要仔细检查此处的索引计算和边界条件
            // 是否与 PyTorch 的 im2col (unfold) 行为完全一致。
            if (id_idx >= 0 && id_idx < inDepth_param &&
                ih_idx >= 0 && ih_idx < inHeight_param &&
                iw_idx >= 0 && iw_idx < inWidth_param &&
                channel_idx < inChannels_param) {
                size_t src_offset = (size_t)channel_idx * inDepth_param * inHeight_param * inWidth_param +
                                    (size_t)id_idx * inHeight_param * inWidth_param +
                                    (size_t)ih_idx * inWidth_param +
                                    iw_idx;
                B_tile[k_b_local][n_b_local] = d_src[src_offset];
            } else {
                B_tile[k_b_local][n_b_local] = 0.0f; // Padding region or out-of-bounds is 0
            }
        } else {
            B_tile[k_b_local][n_b_local] = 0.0f;
        }
        __syncthreads();

        // Compute dot product of sub-blocks
        if (m_global < M_orig && n_global < N_orig) {
#pragma unroll
            for (int k_dot = 0; k_dot < TILE_DIM; ++k_dot) {
                C_val += A_tile[threadIdx.y][k_dot] * B_tile[k_dot][threadIdx.x];
            }
        }
        __syncthreads();
    }

    // Write result back to global memory and fuse bias addition
    if (m_global < M_orig && n_global < N_orig && oh < outHeight_param && od_current < outDepth_param) {
        if (d_bias != nullptr) {
            C_val += d_bias[m_global];
        }
        size_t output_idx = (size_t)m_global * outDepth_param * outHeight_param * N_orig +
                            (size_t)od_current * outHeight_param * N_orig +
                            (size_t)oh * N_orig +
                            n_global;
        d_output[output_idx] = C_val;
    }
}

extern "C" void custom_conv3d_gpu(
        float* d_input, float* d_weight_flat, float* d_bias, float* d_output,
        int batch_size,
        int in_depth, int in_height, int in_width, int in_channels,
        int kernel_depth, int kernel_height, int kernel_width,
        int out_channels, int out_depth, int out_height, int out_width,
        int padding_d, int padding_h, int padding_w,
        int stride_d, int stride_h, int stride_w) {

    int M_param = out_channels;
    int N_param = out_width;
    int K_param = kernel_depth * kernel_height * kernel_width * in_channels;

    dim3 block_dim_conv(TILE_DIM, TILE_DIM, 1); // TILE_DIM is now 16

    for (int b = 0; b < batch_size; ++b) {
        float* current_input_ptr = d_input + (size_t)b * in_channels * in_depth * in_height * in_width;
        float* current_output_ptr = d_output + (size_t)b * out_channels * out_depth * out_height * out_width;

        for (int od = 0; od < out_depth; ++od) {
            dim3 grid_dim_conv(
                    (N_param + TILE_DIM - 1) / TILE_DIM,
                    (M_param + TILE_DIM - 1) / TILE_DIM,
                    out_height
            );

            fused_im2col_gemm_bias_kernel_3d<<<grid_dim_conv, block_dim_conv>>>(
                    d_weight_flat,
                    current_input_ptr,
                    d_bias,
                    current_output_ptr,
                    M_param, N_param, K_param,
                    out_depth, out_height,
                    in_depth, in_height, in_width,
                    kernel_depth, kernel_height, kernel_width,
                    in_channels,
                    padding_d, padding_h, padding_w,
                    stride_d, stride_h, stride_w,
                    od
            );
            CHECK_CUDA_ERROR(cudaGetLastError());
        }
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

/*
// Example main function (for testing purposes)
int main() {
    // --- Define problem dimensions ---
    int batch_size = 1;
    int in_channels = 3;
    int in_depth = 8;
    int in_height = 32;
    int in_width = 32;

    int out_channels = 16;
    int kernel_d = 3;
    int kernel_h = 3;
    int kernel_w = 3;

    int stride_d = 1;
    int stride_h = 1;
    int stride_w = 1;

    int padding_d = (kernel_d - 1) / 2;
    int padding_h = (kernel_h - 1) / 2;
    int padding_w = (kernel_w - 1) / 2;


    int out_depth = (in_depth - kernel_d + 2 * padding_d) / stride_d + 1;
    int out_height = (in_height - kernel_h + 2 * padding_h) / stride_h + 1;
    int out_width = (in_width + 2 * padding_w) / stride_w + 1;

    printf("Input: %d x %d x %d x %d x %d (NCDHW)\n", batch_size, in_channels, in_depth, in_height, in_width);
    printf("Kernel: %d x %d x %d x %d (OCICDKHW)\n", out_channels, in_channels, kernel_d, kernel_h, kernel_w);
    printf("Padding (DHW): %d, %d, %d\n", padding_d, padding_h, padding_w);
    printf("Stride (DHW): %d, %d, %d\n", stride_d, stride_h, stride_w);
    printf("Output: %d x %d x %d x %d x %d (NCDHW)\n", batch_size, out_channels, out_depth, out_height, out_width);


    size_t input_size_bytes = (size_t)batch_size * in_channels * in_depth * in_height * in_width * sizeof(float);
    size_t weight_flat_size_bytes = (size_t)out_channels * in_channels * kernel_d * kernel_h * kernel_w * sizeof(float);
    size_t bias_size_bytes = (size_t)out_channels * sizeof(float);
    size_t output_size_bytes = (size_t)batch_size * out_channels * out_depth * out_height * out_width * sizeof(float);

    float *h_input, *h_weight_flat, *h_bias, *h_output_gpu;
    h_input = (float*)malloc(input_size_bytes);
    h_weight_flat = (float*)malloc(weight_flat_size_bytes);
    h_bias = (float*)malloc(bias_size_bytes);
    h_output_gpu = (float*)malloc(output_size_bytes);

    for(size_t i = 0; i < input_size_bytes / sizeof(float); ++i) h_input[i] = (float)(i % 10 + 1) * 0.1f;
    for(size_t i = 0; i < weight_flat_size_bytes / sizeof(float); ++i) h_weight_flat[i] = (float)((i % 7 + 1) * 0.05f);
    for(size_t i = 0; i < out_channels; ++i) h_bias[i] = (float)(i * 0.02f);


    float *d_input, *d_weight_flat, *d_bias, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_weight_flat, weight_flat_size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bias, bias_size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size_bytes));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_weight_flat, h_weight_flat, weight_flat_size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bias, h_bias, bias_size_bytes, cudaMemcpyHostToDevice));

    printf("Running 3D convolution on GPU (TILE_DIM=%d, Float Accumulation)...\n", TILE_DIM);
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    custom_conv3d_gpu(d_input, d_weight_flat, d_bias, d_output,
                      batch_size,
                      in_depth, in_height, in_width, in_channels,
                      kernel_depth, kernel_height, kernel_width,
                      out_channels, out_depth, out_height, out_width,
                      padding_d, padding_h, padding_w,
                      stride_d, stride_h, stride_w);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("GPU execution time: %f ms\n", milliseconds);

    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu, d_output, output_size_bytes, cudaMemcpyDeviceToHost));

    printf("First few output values from GPU:\n");
    for (int i = 0; i < 10 && i < (output_size_bytes / sizeof(float)); ++i) {
        printf("%f ", h_output_gpu[i]);
    }
    printf("\n");

    free(h_input);
    free(h_weight_flat);
    free(h_bias);
    free(h_output_gpu);

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_weight_flat));
    CHECK_CUDA_ERROR(cudaFree(d_bias));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    printf("Done.\n");
    return 0;
}
*/
