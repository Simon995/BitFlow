#include <iostream>
#include <vector>
#include <stdexcept> // 用于异常处理

#include <cuda_runtime.h>
#include <device_launch_parameters.h> // 用于 CUDA 核函数

// CUTLASS Includes - 确保您的包含路径已正确设置
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/epilogue/thread/linear_combination.h" // Ensure this is included

// CUDA API 调用错误检查宏
#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err = call;                                                           \
        if (err != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA Error at %s:%d - %s: %s\n", __FILE__, __LINE__,          \
                    #call, cudaGetErrorString(err));                                      \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while (0)

// Forward declaration for the bias kernel
__global__ void add_bias_kernel_batched(float* d_output, const float* d_bias,
                                        int batch_size, int out_channels, int out_height, int out_width);

// ------------------------------------------------------------------------------------
// Multi-Channel im2col GPU Kernel
// ------------------------------------------------------------------------------------
__global__ void im2col_gpu_kernel_multichannel(const float* __restrict__ d_input_batch,
                                               float* __restrict__ d_im2col_data,
                                               int in_channels, int in_height, int in_width,
                                               int kernel_h, int kernel_w,
                                               int out_height, int out_width,
                                               int stride_h, int stride_w,
                                               int pad_h, int pad_w) {
    int K_gemm_dim = in_channels * kernel_h * kernel_w;
    int N_gemm_dim = out_height * out_width;
    int total_elements_im2col = K_gemm_dim * N_gemm_dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < total_elements_im2col) {
        int row_idx_im2col = tid / N_gemm_dim;
        int col_idx_output_patch = tid % N_gemm_dim;

        int c_in = row_idx_im2col / (kernel_h * kernel_w);
        int kh_kw_offset = row_idx_im2col % (kernel_h * kernel_w);
        int kh = kh_kw_offset / kernel_w;
        int kw = kh_kw_offset % kernel_w;

        int out_y = col_idx_output_patch / out_width;
        int out_x = col_idx_output_patch % out_width;

        int start_in_y = out_y * stride_h - pad_h;
        int start_in_x = out_x * stride_w - pad_w;

        int current_in_y = start_in_y + kh;
        int current_in_x = start_in_x + kw;

        float val = 0.0f;
        if (current_in_y >= 0 && current_in_y < in_height &&
            current_in_x >= 0 && current_in_x < in_width) {
            val = d_input_batch[c_in * (in_height * in_width) + current_in_y * in_width + current_in_x];
        }
        d_im2col_data[tid] = val;
    }
}

// ------------------------------------------------------------------------------------
// CUTLASS GEMM 调用封装函数
// ------------------------------------------------------------------------------------
void run_cutlass_gemm_for_conv(int M, int N, int K,
                               const float* __restrict__ d_A_weights,
                               const float* __restrict__ d_B_im2col,
                               float* __restrict__ d_C_output,
                               float alpha = 1.0f, float beta = 0.0f) {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = float; // Type for alpha/beta in linear combination
    using ElementInputA = float;
    using ElementInputB = float;
    using ElementOutput = float;
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::RowMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    using ArchTag = cutlass::arch::Sm89; // Target Ada Lovelace (RTX 4070)
    using OperatorClass = cutlass::arch::OpClassSimt;

    // Tile shapes for SIMT operations
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>; // M, N, K per Threadblock
    using WarpShape        = cutlass::gemm::GemmShape<32, 32, 8>;   // M, N, K per Warp
    // This gives 4x4 = 16 warps, 512 threads/TB
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;     // For SIMT FMA

    // Epilogue for SIMT must operate on scalars (ElementsPerAccess = 1)
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
            ElementOutput,
            1,                                                // Elements per access for C/D
            ElementAccumulator,
            ElementComputeEpilogue,
            cutlass::epilogue::thread::ScaleType::Nothing,    // No elementwise scaling
            cutlass::FloatRoundStyle::round_to_nearest        // Default rounding for float
    >;

    // Define the GEMM operator
    using GemmUniversal = cutlass::gemm::device::GemmUniversal<
            ElementInputA, LayoutInputA,
            ElementInputB, LayoutInputB,
            ElementOutput, LayoutOutput,
            ElementAccumulator,
            OperatorClass,
            ArchTag,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            EpilogueOutputOp,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, // Default swizzle
            2 // Number of stages (pipeline depth for shared memory loading)
    >;

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // Arguments for the GEMM operator
    typename GemmUniversal::Arguments arguments{
            problem_size,
            {d_A_weights, K},    // TensorRef for A (Weights), lda = K (stride for RowMajor)
            {d_B_im2col, N},     // TensorRef for B (Im2Col), ldb = N (stride for RowMajor)
            {d_C_output, N},     // TensorRef for C (Source, if beta != 0), ldc = N
            {d_C_output, N},     // TensorRef for D (Destination), ldd = N
            {alpha, beta}        // Epilogue parameters
    };

    // Instantiate and run the GEMM operator
    GemmUniversal gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM cannot implement for arguments: " << cutlassGetStatusString(status) << std::endl;
        throw std::runtime_error("CUTLASS GEMM configuration error.");
    }
    status = gemm_op.initialize(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM initialization failed: " << cutlassGetStatusString(status) << std::endl;
        throw std::runtime_error("CUTLASS GEMM initialization error.");
    }
    status = gemm_op(); // Execute the GEMM
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM execution failed: " << cutlassGetStatusString(status) << std::endl;
        throw std::runtime_error("CUTLASS GEMM execution error.");
    }
}

// ------------------------------------------------------------------------------------
// Bias Addition Kernel
// ------------------------------------------------------------------------------------
__global__ void add_bias_kernel_batched(float* d_output, const float* __restrict__ d_bias_vector,
                                        int batch_size, int out_channels, int out_height, int out_width) {
    int ow_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int oh_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b_oc_combined_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (ow_idx < out_width && oh_idx < out_height && b_oc_combined_idx < batch_size * out_channels) {
        int b = b_oc_combined_idx / out_channels;
        int oc = b_oc_combined_idx % out_channels;

        size_t output_offset = (size_t)b * out_channels * out_height * out_width +
                               (size_t)oc * out_height * out_width +
                               (size_t)oh_idx * out_width +
                               ow_idx;
        d_output[output_offset] += d_bias_vector[oc];
    }
}

// ------------------------------------------------------------------------------------
// 主卷积函数 (Extern "C" for C++ wrapper)
// ------------------------------------------------------------------------------------
extern "C" void custom_conv2d_gpu_cutlass_approach(
        float* d_input,
        float* d_weight_flat,
        float* d_bias,
        float* d_output,
        int batch_size,
        int in_channels, int in_height, int in_width,
        int kernel_h, int kernel_w, int out_channels,
        int pad_h, int pad_w,
        int stride_h, int stride_w) {

    const int out_height_conv = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int out_width_conv = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;

    if (out_height_conv <= 0 || out_width_conv <= 0) {
        printf("Error: Output dimensions are non-positive in custom_conv2d_gpu_cutlass_approach.\n");
        return;
    }

    const int M_gemm = out_channels;
    const int K_gemm = in_channels * kernel_h * kernel_w;
    const int N_gemm = out_height_conv * out_width_conv;

    float* d_im2col_buffer;
    size_t im2col_buffer_size_bytes = (size_t)K_gemm * N_gemm * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_im2col_buffer, im2col_buffer_size_bytes));

    int total_elements_im2col = K_gemm * N_gemm;
    int threads_per_block_im2col = 256; // Can be tuned
    int blocks_im2col = (total_elements_im2col + threads_per_block_im2col - 1) / threads_per_block_im2col;

    for (int b = 0; b < batch_size; ++b) {
        float* current_input_ptr = d_input + (size_t)b * in_channels * in_height * in_width;
        float* current_output_ptr = d_output + (size_t)b * out_channels * out_height_conv * out_width_conv;

        im2col_gpu_kernel_multichannel<<<blocks_im2col, threads_per_block_im2col>>>(
                current_input_ptr, d_im2col_buffer,
                in_channels, in_height, in_width,
                kernel_h, kernel_w,
                out_height_conv, out_width_conv,
                stride_h, stride_w, pad_h, pad_w
        );
        CUDA_CHECK(cudaGetLastError()); // Check after im2col kernel

        run_cutlass_gemm_for_conv(
                M_gemm, N_gemm, K_gemm,
                d_weight_flat,
                d_im2col_buffer,
                current_output_ptr,
                1.0f,
                0.0f
        );
    }

    if (d_bias != nullptr) {
        dim3 bias_grid_dim( (out_width_conv + 15) / 16,
                            (out_height_conv + 15) / 16,
                            (unsigned int)batch_size * out_channels );
        dim3 bias_block_dim(16, 16, 1);

        add_bias_kernel_batched<<<bias_grid_dim, bias_block_dim>>>(
                d_output, d_bias,
                batch_size, out_channels, out_height_conv, out_width_conv
        );
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_im2col_buffer));
}
