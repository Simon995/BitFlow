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

// 使用只读缓存加载数据
#if __CUDA_ARCH__ >= 350
#define LOAD(ptr) __ldg(ptr)
#else
#define LOAD(ptr) (*(ptr))
#endif

// 小卷积核(1x1, 3x3)优化 - 模拟PyTorch布局和计算顺序
__global__ void small_kernel_conv(
        const float* __restrict__ d_input,
        const float* __restrict__ d_filter,
        const float* __restrict__ d_bias,
        float* __restrict__ d_output,
        int batch_size, int in_h, int in_w, int in_c,
        int kernel_h, int kernel_w, int out_c,
        int out_h, int out_w,
        int pad_h, int pad_w,
        int stride_h, int stride_w) {

    // 输出位置
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const int oc = blockIdx.z;

    if (ow >= out_w || oh >= out_h || oc >= out_c) return;

    // 获取偏置
    const float bias_val = d_bias != nullptr ? LOAD(&d_bias[oc]) : 0.0f;

    // 输入位置基准点
    const int ih_base = oh * stride_h - pad_h;
    const int iw_base = ow * stride_w - pad_w;

    // 输出偏移量预计算
    size_t output_offset = (size_t)oc * out_h * out_w + (size_t)oh * out_w + ow;

    // 按批次处理
    for (int b = 0; b < batch_size; ++b) {
        float sum = bias_val;

        // 严格按照PyTorch计算顺序
        for (int ic = 0; ic < in_c; ++ic) {
            // 预计算通道偏移
            size_t input_c_offset = (size_t)b * in_c * in_h * in_w + (size_t)ic * in_h * in_w;
            size_t filter_c_offset = (size_t)oc * in_c * kernel_h * kernel_w + (size_t)ic * kernel_h * kernel_w;

            for (int kh = 0; kh < kernel_h; ++kh) {
                const int ih = ih_base + kh;

                // 跳过边界外的行
                if (ih < 0 || ih >= in_h) continue;

                // 预计算行偏移
                size_t input_h_offset = input_c_offset + (size_t)ih * in_w;
                size_t filter_h_offset = filter_c_offset + (size_t)kh * kernel_w;

                for (int kw = 0; kw < kernel_w; ++kw) {
                    const int iw = iw_base + kw;

                    // 仅处理有效位置
                    if (iw >= 0 && iw < in_w) {
                        sum += LOAD(&d_input[input_h_offset + iw]) *
                               LOAD(&d_filter[filter_h_offset + kw]);
                    }
                }
            }
        }

        // 写回结果
        d_output[(size_t)b * out_c * out_h * out_w + output_offset] = sum;
    }
}

// 中等卷积核(5x5)优化 - 使用分块和共享内存
__global__ void medium_kernel_conv(
        const float* __restrict__ d_input,
        const float* __restrict__ d_filter,
        const float* __restrict__ d_bias,
        float* __restrict__ d_output,
        int batch_size, int in_h, int in_w, int in_c,
        int kernel_h, int kernel_w, int out_c,
        int out_h, int out_w,
        int pad_h, int pad_w,
        int stride_h, int stride_w) {

    // 共享内存缓存小块卷积核权重
    __shared__ float s_filter[8][8];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int ow = blockIdx.x * blockDim.x + tx;
    const int oh = blockIdx.y * blockDim.y + ty;
    const int oc = blockIdx.z;

    if (ow >= out_w || oh >= out_h || oc >= out_c) return;

    // 获取偏置
    const float bias_val = d_bias != nullptr ? LOAD(&d_bias[oc]) : 0.0f;

    // 输入基准位置
    const int ih_base = oh * stride_h - pad_h;
    const int iw_base = ow * stride_w - pad_w;

    // 输出偏移量
    size_t output_offset = (size_t)oc * out_h * out_w + (size_t)oh * out_w + ow;

    // 按批次处理
    for (int b = 0; b < batch_size; ++b) {
        float sum = bias_val;

        // 按输入通道分块处理
        for (int ic = 0; ic < in_c; ++ic) {
            size_t input_c_offset = (size_t)b * in_c * in_h * in_w + (size_t)ic * in_h * in_w;
            size_t filter_c_offset = (size_t)oc * in_c * kernel_h * kernel_w + (size_t)ic * kernel_h * kernel_w;

            // 协作加载卷积核权重到共享内存
            for (int i = ty; i < kernel_h; i += blockDim.y) {
                for (int j = tx; j < kernel_w; j += blockDim.x) {
                    if (i < kernel_h && j < kernel_w) {
                        s_filter[i][j] = LOAD(&d_filter[filter_c_offset + (size_t)i * kernel_w + j]);
                    }
                }
            }

            __syncthreads();

            // 计算卷积
            for (int kh = 0; kh < kernel_h; ++kh) {
                const int ih = ih_base + kh;

                if (ih >= 0 && ih < in_h) {
                    size_t input_h_offset = input_c_offset + (size_t)ih * in_w;

                    for (int kw = 0; kw < kernel_w; ++kw) {
                        const int iw = iw_base + kw;

                        if (iw >= 0 && iw < in_w) {
                            sum += LOAD(&d_input[input_h_offset + iw]) * s_filter[kh][kw];
                        }
                    }
                }
            }

            __syncthreads();
        }

        // 写回结果
        d_output[(size_t)b * out_c * out_h * out_w + output_offset] = sum;
    }
}

// 大卷积核(7x7, 9x9)优化 - 线程块协作
__global__ void large_kernel_conv(
        const float* __restrict__ d_input,
        const float* __restrict__ d_filter,
        const float* __restrict__ d_bias,
        float* __restrict__ d_output,
        int batch_size, int in_h, int in_w, int in_c,
        int kernel_h, int kernel_w, int out_c,
        int out_h, int out_w,
        int pad_h, int pad_w,
        int stride_h, int stride_w) {

    // 使用常量内存缓存一些常用大小的卷积核
    const int TILE_SIZE_X = 2;
    const int TILE_SIZE_Y = 2;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // 基础输出位置
    const int ow_base = blockIdx.x * blockDim.x * TILE_SIZE_X + tx * TILE_SIZE_X;
    const int oh_base = blockIdx.y * blockDim.y * TILE_SIZE_Y + ty * TILE_SIZE_Y;
    const int oc = blockIdx.z;

    if (oc >= out_c) return;

    // 获取偏置
    const float bias_val = d_bias != nullptr ? LOAD(&d_bias[oc]) : 0.0f;

    // 卷积核基址
    const size_t filter_oc_offset = (size_t)oc * in_c * kernel_h * kernel_w;

    // 分块加载优化
    __shared__ float s_filter_block[5][5]; // 5x5子块缓存

    // 处理每个批次
    for (int b = 0; b < batch_size; ++b) {
        const size_t input_b_offset = (size_t)b * in_c * in_h * in_w;
        const size_t output_b_offset = (size_t)b * out_c * out_h * out_w + (size_t)oc * out_h * out_w;

        // 每个线程处理TILE_SIZE_Y x TILE_SIZE_X个输出点
        for (int oh_offset = 0; oh_offset < TILE_SIZE_Y; ++oh_offset) {
            const int oh = oh_base + oh_offset;

            if (oh >= out_h) continue;

            const int ih_base = oh * stride_h - pad_h;
            const size_t output_h_offset = output_b_offset + (size_t)oh * out_w;

            for (int ow_offset = 0; ow_offset < TILE_SIZE_X; ++ow_offset) {
                const int ow = ow_base + ow_offset;

                if (ow >= out_w) continue;

                const int iw_base = ow * stride_w - pad_w;
                float sum = bias_val;

                // 分块计算
                for (int ic = 0; ic < in_c; ++ic) {
                    const size_t input_c_offset = input_b_offset + (size_t)ic * in_h * in_w;
                    const size_t filter_ic_offset = filter_oc_offset + (size_t)ic * kernel_h * kernel_w;

                    // 逐块处理卷积核
                    for (int kh_block = 0; kh_block < kernel_h; kh_block += 5) {
                        for (int kw_block = 0; kw_block < kernel_w; kw_block += 5) {
                            // 协作加载子块权重
                            if (tx < 5 && ty < 5) {
                                int kh = kh_block + ty;
                                int kw = kw_block + tx;

                                if (kh < kernel_h && kw < kernel_w) {
                                    s_filter_block[ty][tx] = LOAD(&d_filter[
                                            filter_ic_offset + (size_t)kh * kernel_w + kw]);
                                } else {
                                    s_filter_block[ty][tx] = 0.0f;
                                }
                            }

                            __syncthreads();

                            // 处理当前5x5块
                            for (int sub_kh = 0; sub_kh < 5 && kh_block + sub_kh < kernel_h; ++sub_kh) {
                                const int ih = ih_base + kh_block + sub_kh;

                                if (ih >= 0 && ih < in_h) {
                                    const size_t input_h_offset = input_c_offset + (size_t)ih * in_w;

                                    for (int sub_kw = 0; sub_kw < 5 && kw_block + sub_kw < kernel_w; ++sub_kw) {
                                        const int iw = iw_base + kw_block + sub_kw;

                                        if (iw >= 0 && iw < in_w) {
                                            sum += s_filter_block[sub_kh][sub_kw] *
                                                   LOAD(&d_input[input_h_offset + iw]);
                                        }
                                    }
                                }
                            }

                            __syncthreads();
                        }
                    }
                }

                // 写出结果
                d_output[output_h_offset + ow] = sum;
            }
        }
    }
}

// 卷积算子核心函数
extern "C" void custom_conv2d_gpu(float* d_input, float* d_weight_flat, float* d_bias, float* d_output,
                                  int batch_size, int in_height, int in_width, int in_channels,
                                  int kernel_height, int kernel_width, int out_channels,
                                  int out_height, int out_width,
                                  int padding_h, int padding_w,
                                  int stride_h, int stride_w) {

    // 根据卷积核大小选择最优实现
    if (kernel_height <= 3 && kernel_width <= 3) {
        // 小卷积核 (1x1, 3x3)
        dim3 block(16, 16);
        dim3 grid(
                (out_width + block.x - 1) / block.x,
                (out_height + block.y - 1) / block.y,
                out_channels
        );

        small_kernel_conv<<<grid, block>>>(
                d_input, d_weight_flat, d_bias, d_output,
                batch_size, in_height, in_width, in_channels,
                kernel_height, kernel_width, out_channels,
                out_height, out_width,
                padding_h, padding_w, stride_h, stride_w
        );
    }
    else if (kernel_height <= 5 && kernel_width <= 5) {
        // 中等卷积核 (5x5)
        dim3 block(8, 8);
        dim3 grid(
                (out_width + block.x - 1) / block.x,
                (out_height + block.y - 1) / block.y,
                out_channels
        );

        medium_kernel_conv<<<grid, block>>>(
                d_input, d_weight_flat, d_bias, d_output,
                batch_size, in_height, in_width, in_channels,
                kernel_height, kernel_width, out_channels,
                out_height, out_width,
                padding_h, padding_w, stride_h, stride_w
        );
    }
    else {
        // 大卷积核 (7x7, 9x9)
        dim3 block(8, 8);
        dim3 grid(
                (out_width + block.x * 2 - 1) / (block.x * 2),
                (out_height + block.y * 2 - 1) / (block.y * 2),
                out_channels
        );

        large_kernel_conv<<<grid, block>>>(
                d_input, d_weight_flat, d_bias, d_output,
                batch_size, in_height, in_width, in_channels,
                kernel_height, kernel_width, out_channels,
                out_height, out_width,
                padding_h, padding_w, stride_h, stride_w
        );
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}