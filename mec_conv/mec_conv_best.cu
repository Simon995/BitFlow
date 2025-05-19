#include <cuda_runtime.h>
#include <stdio.h>

// CUDA API调用错误检查宏
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 错误位于 %s:%d (%s): %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// 定义分块大小 (可以根据目标GPU和具体问题进行调优)
#define TILE_DIM 16

// 融合im2col、GEMM和偏置加法的核函数
__global__ void fused_im2col_gemm_bias_kernel(
        const float* __restrict__ d_kernel_flat, // 权重 (out_channels, in_channels * kernel_h * kernel_w)
        const float* __restrict__ d_src,         // 输入 (in_channels, in_height, in_width) for one batch
        const float* __restrict__ d_bias,        // 偏置 (out_channels)，如果不需要偏置，可以传入 nullptr
        float* __restrict__ d_output,            // 输出 (out_channels, out_height, out_width) for one batch
        int M_orig, int N_orig, int K_orig, // GEMM维度: M=out_channels, N=out_width, K=in_c*k_h*k_w
        int outHeight_param, int inHeight_param, int inWidth_param,
        int kernel_h_param, int kernel_w_param,
        int inChannels_param,
        int padding_h_param, int padding_w_param,
        int stride_h_param, int stride_w_param) {

    // 共享内存声明
    __shared__ float A_tile[TILE_DIM][TILE_DIM]; // A的子块 (M方向 x K方向) -> 权重
    __shared__ float B_tile[TILE_DIM][TILE_DIM]; // B的子块 (K方向 x N方向) -> im2col后的输入

    // 线程索引和块索引
    // blockIdx.z 对应输出高度 oh
    // blockIdx.y * TILE_DIM + threadIdx.y 对应输出通道 m_global (M_orig)
    // blockIdx.x * TILE_DIM + threadIdx.x 对应输出宽度 n_global (N_orig)
    int oh = blockIdx.z; // 输出特征图的 y 坐标 (高度方向)
    // m_global 对应输出特征图的通道索引
    int m_global = blockIdx.y * TILE_DIM + threadIdx.y;
    // n_global 对应输出特征图的 x 坐标 (宽度方向)
    int n_global = blockIdx.x * TILE_DIM + threadIdx.x;

    float C_val = 0.0f; // 每个线程计算的C元素的部分和

    // 遍历K维度进行分块矩阵乘法 (K_orig = kernel_h * kernel_w * inChannels)
    // K_orig 是输入特征的深度 (in_channels * kernel_h * kernel_w)
    for (int k_block_outer = 0; k_block_outer < (K_orig + TILE_DIM - 1) / TILE_DIM; ++k_block_outer) {
        // --- 加载 A_tile (权重) ---
        // A_tile[threadIdx.y][threadIdx.x]
        // threadIdx.y 对应 M (输出通道) 方向的块内偏移
        // threadIdx.x 对应 K (输入特征深度) 方向的块内偏移
        int k_a_local = threadIdx.x; // A_tile的列索引 (K方向)
        int m_a_local = threadIdx.y; // A_tile的行索引 (M方向)

        int k_a_global = k_block_outer * TILE_DIM + k_a_local; // A在K维度的全局索引

        if (m_global < M_orig && k_a_global < K_orig) {
            // d_kernel_flat布局: (M_orig, K_orig) -> (out_channels, in_channels * kernel_h * kernel_w)
            A_tile[m_a_local][k_a_local] = d_kernel_flat[(size_t)m_global * K_orig + k_a_global];
        } else {
            A_tile[m_a_local][k_a_local] = 0.0f;
        }

        // --- 加载 B_tile (im2col后的输入) ---
        // B_tile[threadIdx.y][threadIdx.x]
        // threadIdx.y 对应 K (输入特征深度) 方向的块内偏移
        // threadIdx.x 对应 N (输出宽度) 方向的块内偏移
        int k_b_local = threadIdx.y; // B_tile的行索引 (K方向)
        int n_b_local = threadIdx.x; // B_tile的列索引 (N方向)

        int k_b_global = k_block_outer * TILE_DIM + k_b_local; // B在K维度的全局索引

        if (n_global < N_orig && k_b_global < K_orig) {
            // im2col逻辑: 从k_b_global反推输入通道、卷积核kh、kw偏移
            // k_b_global 遍历的是 in_channels * kernel_h * kernel_w 这个维度
            int k_h_x_k_w = kernel_h_param * kernel_w_param;
            int channel_idx = k_b_global / k_h_x_k_w;
            int k_idx_in_channel = k_b_global % k_h_x_k_w;
            int kh_offset = k_idx_in_channel / kernel_w_param;
            int kw_offset = k_idx_in_channel % kernel_w_param;

            // 计算在输入特征图中的实际坐标 (考虑padding和stride)
            // oh 是当前输出像素的行号
            // n_global 是当前输出像素的列号
            int ih_idx = oh * stride_h_param + kh_offset - padding_h_param;
            int iw_idx = n_global * stride_w_param + kw_offset - padding_w_param;

            // 边界检查并加载数据
            if (ih_idx >= 0 && ih_idx < inHeight_param &&
                iw_idx >= 0 && iw_idx < inWidth_param &&
                channel_idx < inChannels_param) { // 确保channel_idx有效
                // d_src布局 (对于单batch): (inChannels_param, inHeight_param, inWidth_param)
                B_tile[k_b_local][n_b_local] = d_src[(size_t)channel_idx * inHeight_param * inWidth_param +
                                                     (size_t)ih_idx * inWidth_param + iw_idx];
            } else {
                B_tile[k_b_local][n_b_local] = 0.0f; // padding区域或越界为0
            }
        } else {
            B_tile[k_b_local][n_b_local] = 0.0f;
        }
        __syncthreads(); // 等待所有线程加载完A_tile和B_tile

        // --- 计算子块的点积 ---
        // 每个线程计算C_val中对应的一个元素
        // m_global < M_orig 检查输出通道是否越界
        // n_global < N_orig 检查输出宽度是否越界
        if (m_global < M_orig && n_global < N_orig) {
            // 尝试展开此循环，因为 TILE_DIM 通常是编译时常量且较小
#pragma unroll
            for (int k_dot = 0; k_dot < TILE_DIM; ++k_dot) {
                // A_tile[threadIdx.y][k_dot] * B_tile[k_dot][threadIdx.x]
                // A_tile 的行索引是 threadIdx.y (对应M)
                // B_tile 的列索引是 threadIdx.x (对应N)
                C_val += A_tile[threadIdx.y][k_dot] * B_tile[k_dot][threadIdx.x];
            }
        }
        __syncthreads(); // 等待所有线程完成点积计算，并确保共享内存写入完成，准备下一轮k_block_outer
    }

    // --- 将结果写回全局内存，并融合偏置加法 ---
    // 目标d_output布局 (对于单batch): (M_orig, outHeight_param, N_orig) -> (out_channels, out_height, out_width)
    if (m_global < M_orig && n_global < N_orig && oh < outHeight_param) {
        if (d_bias != nullptr) { // 只有当偏置指针非空时才添加偏置
            C_val += d_bias[m_global]; // m_global 对应输出通道索引
        }
        d_output[(size_t)m_global * outHeight_param * N_orig + // M_orig是out_channels
                 (size_t)oh * N_orig +
                 n_global] = C_val;
    }
}


// 导出的C函数，用于执行卷积操作 (已融合偏置)
// 注意：d_weight_flat 参数名保持不变，但其内容和核函数的期望一致
extern "C" void custom_conv2d_gpu(float* d_input, float* d_weight_flat, float* d_bias, float* d_output,
                                  int batch_size, int in_height, int in_width, int in_channels,
                                  int kernel_height, int kernel_width, int out_channels,
                                  int out_height, int out_width,
                                  int padding_h, int padding_w,
                                  int stride_h, int stride_w) {
    // GEMM参数
    int M_param = out_channels; // 输出通道数
    int N_param = out_width;    // 输出宽度
    int K_param = kernel_height * kernel_width * in_channels; // 卷积核元素总数 (K维度)

    // 卷积核的启动配置
    // blockDim.x 对应 N (输出宽度方向的分块)
    // blockDim.y 对应 M (输出通道方向的分块)
    // blockDim.z 固定为1，因为Z方向的 gridDim 用于遍历输出高度
    dim3 block_dim_conv(TILE_DIM, TILE_DIM, 1);

    // 遍历批处理维度
    for (int b = 0; b < batch_size; ++b) {
        // gridDim.x for N_param (输出宽度)
        // gridDim.y for M_param (输出通道)
        // gridDim.z for out_height (输出高度的每个像素位置)
        dim3 grid_dim_conv(
                (N_param + TILE_DIM - 1) / TILE_DIM,
                (M_param + TILE_DIM - 1) / TILE_DIM,
                out_height // 每个z索引的block处理输出特征图的一行 (或一个oh值)
        );

        // 计算当前批次输入和输出的指针偏移
        float* current_input_ptr = d_input + (size_t)b * in_channels * in_height * in_width;
        float* current_output_ptr = d_output + (size_t)b * out_channels * out_height * out_width;

        // 调用融合了偏置加法的卷积核函数
        fused_im2col_gemm_bias_kernel<<<grid_dim_conv, block_dim_conv>>>(
                d_weight_flat,
                current_input_ptr,
                d_bias, // 直接传递偏置指针，核函数内部会检查是否为nullptr
                current_output_ptr,
                M_param, N_param, K_param,
                out_height, in_height, in_width,
                kernel_height, kernel_width,
                in_channels, padding_h, padding_w,
                stride_h, stride_w
        );
        // 每次核函数启动后都检查错误是一个好习惯，尤其是在开发阶段
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
    // 所有批次的卷积计算完成后，进行一次全局同步，确保所有操作完成
    // 这对于后续依赖此输出的操作或准确计时是重要的
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 由于偏置已经融合到主核函数中，不再需要单独的 add_bias_kernel
}
