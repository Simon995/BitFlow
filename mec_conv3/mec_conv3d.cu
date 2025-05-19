#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> // For malloc/free

// CUDA API 调用错误检查宏
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d (%s): %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// 定义Tile大小
// TILE_DIM 是一个关键的可调参数。对于Lovelace架构 (如RTX 4070)，
// 可以尝试不同的值 (例如16, 32)。
// TILE_DIM = 16 (256 threads/block) 是一个常见的起点。
// TILE_DIM = 32 (1024 threads/block) 可能会提高占用率，但会增加共享内存和寄存器压力。
// 需要使用Nsight Compute进行分析以确定最佳值。
#define TILE_DIM 16

// 融合im2col, GEMM和偏置加法的3D卷积核函数
// 针对Lovelace架构的优化提示：
// 1. 编译器优化：编译时请使用针对Lovelace架构的标志，例如 -arch=sm_89 (RTX 4070通常是AD104核心，属于sm_89)。
//    这将使nvcc编译器生成针对特定架构优化的代码。
// 2. Tensor Cores：Lovelace的Tensor Cores对FP16, BF16, TF32甚至FP8提供硬件加速。
//    当前kernel使用FP32进行累加。若要充分利用Tensor Cores，需要使用mma指令 (matrix multiply-accumulate)
//    并可能转换数据类型，这将是更大幅度的代码修改。对于FP32 FMA操作，硬件可能会使用TF32路径（如果CUDA版本和硬件支持），
//    这能提供一定的加速而无需代码更改，但Nsight Compute可以确认计算单元的利用情况。
// 3. L2缓存：Lovelace拥有更大的L2缓存。优化数据局部性，减少全局内存访问，可以更好地利用L2缓存。
//    im2col操作的数据重用模式是关键。
// 4. Shader Execution Reordering (SER)：硬件特性，有助于处理线程束分歧。但仍应尽量减少代码中的分歧。
__global__ void fused_im2col_gemm_bias_kernel_3d(
        const float* __restrict__ d_kernel_flat, // 权重 (out_channels, in_channels * kernel_d * kernel_h * kernel_w)
        const float* __restrict__ d_src,         // 输入 (in_channels, in_depth, in_height, in_width) - 单个batch
        const float* __restrict__ d_bias,        // 偏置 (out_channels), 如果没有偏置则为nullptr
        float* __restrict__ d_output,            // 输出 (out_channels, out_depth, out_height, out_width) - 单个batch
        int M_orig, int N_orig, int K_orig,      // GEMM维度: M=out_channels, N=out_width (即输出宽度), K=in_c*k_d*k_h*k_w
        int outDepth_param, int outHeight_param, // N_orig 是 outWidth_param
        int inDepth_param, int inHeight_param, int inWidth_param,
        int kernel_d_param, int kernel_h_param, int kernel_w_param,
        int inChannels_param,
        int padding_d_param, int padding_h_param, int padding_w_param,
        int stride_d_param, int stride_h_param, int stride_w_param,
        int od_current) { // 当前处理的输出深度索引

    // 共享内存声明
    // A_tile 用于存储权重矩阵的子块
    // B_tile 用于存储im2col转换后输入数据的子块
    __shared__ float A_tile[TILE_DIM][TILE_DIM];
    __shared__ float B_tile[TILE_DIM][TILE_DIM];

    // 计算当前线程块在输出高度维度上的索引
    int oh = blockIdx.z; // 输出高度索引

    // 计算当前线程在输出矩阵C中的全局行索引 (m_global) 和列索引 (n_global)
    // threadIdx.y 对应 M (输出通道) 方向的tile内偏移
    // threadIdx.x 对应 N (输出宽度) 方向的tile内偏移
    int m_global = blockIdx.y * TILE_DIM + threadIdx.y;
    int n_global = blockIdx.x * TILE_DIM + threadIdx.x;

    // 使用float类型进行累加，以保证精度
    float C_val = 0.0f;

    // 外层循环遍历K维度上的所有tile块
    // K_orig = in_channels * kernel_d * kernel_h * kernel_w
    for (int k_block_outer = 0; k_block_outer < (K_orig + TILE_DIM - 1) / TILE_DIM; ++k_block_outer) {
        // --- 加载 A_tile (权重数据) ---
        // threadIdx.x 对应 K (kernel/input channel) 方向的tile内偏移
        // threadIdx.y 对应 M (output channel) 方向的tile内偏移
        int k_a_local = threadIdx.x; // A_tile的列索引 (K方向)
        int m_a_local = threadIdx.y; // A_tile的行索引 (M方向)
        int k_a_global = k_block_outer * TILE_DIM + k_a_local; // K维度的全局索引

        // 边界检查并加载权重数据
        if (m_global < M_orig && k_a_global < K_orig) {
            A_tile[m_a_local][k_a_local] = d_kernel_flat[(size_t)m_global * K_orig + k_a_global];
        } else {
            A_tile[m_a_local][k_a_local] = 0.0f; // 超出边界则补零
        }

        // --- 加载 B_tile (im2col转换后的输入数据) ---
        // 性能提示：此处的 im2col 索引计算和全局内存加载是主要的潜在性能瓶颈。
        // 使用 NVIDIA Nsight Compute 分析此部分的内存访问模式（是否合并，L1/L2缓存命中率）至关重要。
        // 目标是实现合并的全局内存读取，以最大限度地提高带宽利用率。
        // threadIdx.y 对应 K (kernel/input channel) 方向的tile内偏移
        // threadIdx.x 对应 N (output width) 方向的tile内偏移
        int k_b_local = threadIdx.y; // B_tile的行索引 (K方向)
        int n_b_local = threadIdx.x; // B_tile的列索引 (N方向)
        int k_b_global = k_block_outer * TILE_DIM + k_b_local; // K维度的全局索引

        if (n_global < N_orig && k_b_global < K_orig) {
            // im2col 逻辑: 从 k_b_global (K维度全局索引) 推导出输入通道、卷积核d,h,w的偏移量
            // k_b_global 遍历的范围是 in_channels * kernel_d * kernel_h * kernel_w
            int k_d_x_k_h_x_k_w = kernel_d_param * kernel_h_param * kernel_w_param; // 单个输入通道的卷积核元素数量
            int channel_idx = k_b_global / k_d_x_k_h_x_k_w; // 当前输入通道索引
            int k_idx_in_channel_vol = k_b_global % k_d_x_k_h_x_k_w; // 在单个通道卷积核内的索引

            int kd_offset = k_idx_in_channel_vol / (kernel_h_param * kernel_w_param); // 卷积核深度方向的偏移
            int k_idx_in_channel_area = k_idx_in_channel_vol % (kernel_h_param * kernel_w_param);
            int kh_offset = k_idx_in_channel_area / kernel_w_param; // 卷积核高度方向的偏移
            int kw_offset = k_idx_in_channel_area % kernel_w_param; // 卷积核宽度方向的偏移

            // 根据输出位置 (od_current, oh, n_global) 和卷积核偏移，计算在输入特征图中的实际坐标 (考虑padding和stride)
            // od_current: 当前处理的输出深度切片，由主机传入
            // oh: blockIdx.z，当前处理的输出高度
            // n_global: 当前线程对应的输出宽度
            int id_idx = od_current * stride_d_param + kd_offset - padding_d_param; // 输入深度索引
            int ih_idx = oh * stride_h_param + kh_offset - padding_h_param;         // 输入高度索引
            int iw_idx = n_global * stride_w_param + kw_offset - padding_w_param;   // 输入宽度索引

            // 边界检查并加载输入数据
            // 数值偏差提示：如果在大卷积核或特定参数下出现偏差，需要仔细检查此处的索引计算和边界条件
            // 是否与 PyTorch 的 im2col (unfold) 等标准库的行为完全一致。
            // 内存访问模式：iw_idx 是最内层的维度。当 threadIdx.x (影响 n_global) 变化时，
            // 如果 stride_w_param == 1，则对 d_src 的访问可能是部分合并的。
            // 如果 stride_w_param > 1，则是跨步访问。
            // channel_idx, id_idx, ih_idx 的变化由 k_b_global (影响 threadIdx.y) 控制，
            // 这使得跨线程束的完美合并访问变得复杂。Nsight Compute 将揭示实际的内存事务。
            if (id_idx >= 0 && id_idx < inDepth_param &&
                ih_idx >= 0 && ih_idx < inHeight_param &&
                iw_idx >= 0 && iw_idx < inWidth_param &&
                channel_idx < inChannels_param) {
                // 计算源数据在d_src中的扁平化索引
                size_t src_offset = (size_t)channel_idx * inDepth_param * inHeight_param * inWidth_param +
                                    (size_t)id_idx * inHeight_param * inWidth_param +
                                    (size_t)ih_idx * inWidth_param +
                                    iw_idx;
                B_tile[k_b_local][n_b_local] = d_src[src_offset];
            } else {
                B_tile[k_b_local][n_b_local] = 0.0f; // Padding区域或越界，填充0
            }
        } else {
            B_tile[k_b_local][n_b_local] = 0.0f; // 超出N或K边界则补零
        }

        // 同步线程块内的所有线程，确保A_tile和B_tile都已加载完毕
        __syncthreads();

        // --- 计算子块的点积 ---
        // 只有在有效的输出M和N范围内的线程才执行计算
        if (m_global < M_orig && n_global < N_orig) {
            // #pragma unroll 指示编译器展开此循环，以减少循环开销，增加指令级并行性。
            // Lovelace架构具有较强的指令流水线能力，展开可能有效。
            // 最终效果需通过Nsight Compute分析。
#pragma unroll
            for (int k_dot = 0; k_dot < TILE_DIM; ++k_dot) {
                // A_tile[threadIdx.y][k_dot] (m_a_local = threadIdx.y)
                // B_tile[k_dot][threadIdx.x] (n_b_local = threadIdx.x)
                // 这是标准的tile GEMM共享内存访问模式，有助于避免共享内存bank冲突。
                C_val += A_tile[threadIdx.y][k_dot] * B_tile[k_dot][threadIdx.x];
            }
        }
        // 同步线程块内的所有线程，确保当前子块的点积计算完成，
        // 并且C_val的更新不会与下一次迭代的A_tile/B_tile加载冲突。
        __syncthreads();
    }

    // --- 写回结果到全局内存并融合偏置加法 ---
    // 确保线程在有效的输出元素范围内
    if (m_global < M_orig && n_global < N_orig && oh < outHeight_param && od_current < outDepth_param) {
        if (d_bias != nullptr) {
            C_val += d_bias[m_global]; // m_global是输出通道索引
        }
        // 计算输出数据在d_output中的扁平化索引
        // 输出格式为 NCDHW, 但这里处理的是单个batch, 所以是 CDHW
        // M_orig = out_channels
        // N_orig = out_width
        size_t output_idx = (size_t)m_global * outDepth_param * outHeight_param * N_orig + // N_orig is out_width
                            (size_t)od_current * outHeight_param * N_orig +
                            (size_t)oh * N_orig +
                            n_global;
        d_output[output_idx] = C_val;
    }
}

// 主机端启动CUDA核函数的封装函数
extern "C" void custom_conv3d_gpu(
        float* d_input, float* d_weight_flat, float* d_bias, float* d_output,
        int batch_size,
        int in_depth, int in_height, int in_width, int in_channels,
        int kernel_depth, int kernel_height, int kernel_width,
        int out_channels, int out_depth, int out_height, int out_width,
        int padding_d, int padding_h, int padding_w,
        int stride_d, int stride_h, int stride_w) {

    // GEMM维度参数
    // M: 输出特征图的通道数 (out_channels)
    // N: 输出特征图的宽度 (out_width)
    // K: kernel_size_flat * in_channels (kernel_depth * kernel_height * kernel_width * in_channels)
    int M_param = out_channels;
    int N_param = out_width;
    int K_param = kernel_depth * kernel_height * kernel_width * in_channels;

    // 定义线程块维度 (TILE_DIM x TILE_DIM x 1)
    // blockDim.x = TILE_DIM, blockDim.y = TILE_DIM, blockDim.z = 1
    // 每个线程块处理输出特征图的一个TILE_DIM x TILE_DIM的子块
    dim3 block_dim_conv(TILE_DIM, TILE_DIM, 1);

    // 遍历batch中的每个样本
    for (int b = 0; b < batch_size; ++b) {
        // 计算当前batch样本的输入和输出数据指针
        float* current_input_ptr = d_input + (size_t)b * in_channels * in_depth * in_height * in_width;
        float* current_output_ptr = d_output + (size_t)b * out_channels * out_depth * out_height * out_width;

        // 遍历输出特征图的深度维度
        // 优化提示: 如果out_depth非常小，多次启动kernel的开销可能会比较显著。
        // 在这种情况下，可以考虑将这个od循环也融合到kernel内部，
        // 让一个kernel处理整个输出深度，或者至少是多个输出深度切片。
        // 这会增加kernel的复杂度，但可能减少启动开销。
        for (int od = 0; od < out_depth; ++od) {
            // 定义Grid维度
            // gridDim.x: 对应输出宽度的tile数量
            // gridDim.y: 对应输出通道数的tile数量
            // gridDim.z: 直接映射到输出高度 (out_height)，每个z平面是一个2D的tile网格
            dim3 grid_dim_conv(
                    (N_param + TILE_DIM - 1) / TILE_DIM,  // X-dimension: tiles for output width
                    (M_param + TILE_DIM - 1) / TILE_DIM,  // Y-dimension: tiles for output channels
                    out_height                            // Z-dimension: one plane for each output height row
            );

            // 启动CUDA核函数
            fused_im2col_gemm_bias_kernel_3d<<<grid_dim_conv, block_dim_conv>>>(
                    d_weight_flat,
                    current_input_ptr,
                    d_bias,
                    current_output_ptr,
                    M_param, N_param, K_param,
                    out_depth, out_height, // outWidth_param is N_param
                    in_depth, in_height, in_width,
                    kernel_depth, kernel_height, kernel_width,
                    in_channels,
                    padding_d, padding_h, padding_w,
                    stride_d, stride_h, stride_w,
                    od // 传入当前处理的输出深度索引
            );
            // 每次kernel启动后检查错误（主要用于调试）
            // 在生产环境中，频繁的GetLastError可能影响性能，可以考虑在循环外或关键点检查。
            CHECK_CUDA_ERROR(cudaGetLastError());
        }
    }
    // 同步设备，确保所有kernel执行完毕，这对于准确计时和后续操作是必要的。
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


// To compile this main function for standalone testing, define COMPILE_MAIN_TEST
// For example: nvcc -DCOMPILE_MAIN_TEST your_file.cu -o standalone_test -arch=sm_89
// When building as a Python extension, this main function should generally not be compiled.
#ifdef COMPILE_MAIN_TEST

// 示例 main 函数 (用于测试目的)
// 编译命令示例 (请根据您的CUDA安装路径和目标架构调整):
// nvcc -DCOMPILE_MAIN_TEST -o conv3d_test your_file.cu -O3 -arch=sm_89  (sm_89 适用于Lovelace架构, 如RTX 4070)
int main() {
    // --- 定义问题维度 ---
    int batch_size = 1;
    int in_channels = 3;    // 输入通道数
    int in_depth = 16;      // 输入深度
    int in_height = 64;     // 输入高度
    int in_width = 64;      // 输入宽度

    int out_channels = 32;  // 输出通道数
    int kernel_d = 3;       // 卷积核深度 (局部变量)
    int kernel_h = 3;       // 卷积核高度 (局部变量)
    int kernel_w = 3;       // 卷积核宽度 (局部变量)

    // 步长 (stride)
    int stride_d = 1;
    int stride_h = 1;
    int stride_w = 1;

    // 填充 (padding) - 'SAME' padding for these kernel sizes and strides
    // 确保卷积核的中心对齐输入元素
    int padding_d = (kernel_d - 1) / 2;
    int padding_h = (kernel_h - 1) / 2;
    int padding_w = (kernel_w - 1) / 2;


    // 计算输出维度
    int out_depth = (in_depth - kernel_d + 2 * padding_d) / stride_d + 1;
    int out_height = (in_height - kernel_h + 2 * padding_h) / stride_h + 1;
    int out_width = (in_width - kernel_w + 2 * padding_w) / stride_w + 1;

    printf("编译提示: 请使用 nvcc -O3 -arch=sm_XX (例如 sm_89 for RTX 4070) 进行编译以获得最佳性能。\n");
    printf("TILE_DIM: %d\n", TILE_DIM);
    printf("输入: %d x %d x %d x %d x %d (NCDHW)\n", batch_size, in_channels, in_depth, in_height, in_width);
    // 在此 printf 中，我们使用局部变量 kernel_d, kernel_h, kernel_w
    printf("权重 (OC x IC x KD x KH x KW): %d x %d x %d x %d x %d\n", out_channels, in_channels, kernel_d, kernel_h, kernel_w);
    printf("填充 (DHW): %d, %d, %d\n", padding_d, padding_h, padding_w);
    printf("步长 (DHW): %d, %d, %d\n", stride_d, stride_h, stride_w);
    printf("输出: %d x %d x %d x %d x %d (NCDHW)\n", batch_size, out_channels, out_depth, out_height, out_width);
    // 在此 printf (即您错误信息指向的行号附近)，我们使用局部变量 kernel_d, kernel_h, kernel_w
    // 如果您的代码在这里使用了 kernel_depth, kernel_height, kernel_width，那就会导致 "identifier undefined" 错误，
    // 因为这些名称是 custom_conv3d_gpu 函数的参数，并非 main 函数的局部变量。
    printf("GEMM M=%d, N=%d, K=%d\n", out_channels, out_width, in_channels * kernel_d * kernel_h * kernel_w);


    // 分配主机内存
    size_t input_num_elements = (size_t)batch_size * in_channels * in_depth * in_height * in_width;
    size_t weight_flat_num_elements = (size_t)out_channels * in_channels * kernel_d * kernel_h * kernel_w;
    size_t bias_num_elements = (size_t)out_channels;
    size_t output_num_elements = (size_t)batch_size * out_channels * out_depth * out_height * out_width;

    size_t input_size_bytes = input_num_elements * sizeof(float);
    size_t weight_flat_size_bytes = weight_flat_num_elements * sizeof(float);
    size_t bias_size_bytes = bias_num_elements * sizeof(float);
    size_t output_size_bytes = output_num_elements * sizeof(float);

    float *h_input, *h_weight_flat, *h_bias, *h_output_gpu;
    h_input = (float*)malloc(input_size_bytes);
    h_weight_flat = (float*)malloc(weight_flat_size_bytes);
    h_bias = (float*)malloc(bias_size_bytes);
    h_output_gpu = (float*)malloc(output_size_bytes);

    if (!h_input || !h_weight_flat || !h_bias || !h_output_gpu) {
        printf("主机内存分配失败!\n");
        return EXIT_FAILURE;
    }

    // 初始化主机数据 (简单示例)
    for(size_t i = 0; i < input_num_elements; ++i) h_input[i] = (float)((i % 100) + 1) * 0.01f;
    for(size_t i = 0; i < weight_flat_num_elements; ++i) h_weight_flat[i] = (float)((i % 70 + 1) * 0.005f);
    for(size_t i = 0; i < bias_num_elements; ++i) h_bias[i] = (float)(i * 0.002f);


    // 分配设备内存
    float *d_input, *d_weight_flat, *d_bias, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_weight_flat, weight_flat_size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bias, bias_size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size_bytes));

    // 将数据从主机复制到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_weight_flat, h_weight_flat, weight_flat_size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bias, h_bias, bias_size_bytes, cudaMemcpyHostToDevice));

    printf("在GPU上运行3D卷积 (TILE_DIM=%d, Float累加)...\n", TILE_DIM);
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // 记录开始时间
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    // 调用自定义3D卷积GPU函数
    // 注意：这里我们将 main 中的局部变量 kernel_d, kernel_h, kernel_w
    // 作为参数传递给 custom_conv3d_gpu，该函数对应的参数名为 kernel_depth, kernel_height, kernel_width
    custom_conv3d_gpu(d_input, d_weight_flat, d_bias, d_output,
                      batch_size,
                      in_depth, in_height, in_width, in_channels,
                      kernel_d, kernel_h, kernel_w, // 传递局部变量
                      out_channels, out_depth, out_height, out_width,
                      padding_d, padding_h, padding_w,
                      stride_d, stride_h, stride_w);

    // 记录结束时间并同步
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop)); // 等待GPU执行完成

    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("GPU 执行时间: %f ms\n", milliseconds);
    double gflops = (2.0 * batch_size * out_channels * out_depth * out_height * out_width * (in_channels * kernel_d * kernel_h * kernel_w)) / (milliseconds / 1000.0) / 1e9;
    printf("GPU 性能: %f GFLOPS\n", gflops);


    // 将结果从设备复制回主机
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu, d_output, output_size_bytes, cudaMemcpyDeviceToHost));

    // 打印部分GPU输出结果以供验证
    printf("GPU输出的前10个值:\n");
    for (int i = 0; i < 10 && i < (int)output_num_elements; ++i) {
        printf("%f ", h_output_gpu[i]);
    }
    printf("\n");

    // 释放主机内存
    free(h_input);
    free(h_weight_flat);
    free(h_bias);
    free(h_output_gpu);

    // 释放设备内存
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_weight_flat));
    CHECK_CUDA_ERROR(cudaFree(d_bias));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    // 销毁CUDA事件
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    printf("完成.\n");
    return 0;
}
#endif // COMPILE_MAIN_TEST
