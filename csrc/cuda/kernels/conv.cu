#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <vector>

// CUDA 宏定义
#define CUDA_NUM_THREADS 1024
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

inline int GET_BLOCKS(const int N) { return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS; }

#define CUDA_POST_KERNEL_CHECK TORCH_CHECK(cudaGetLastError() == cudaSuccess)

namespace bitflow {
namespace ops {
namespace cuda {

// CUDA kernel for im2col
__global__ void im2col_gpu_kernel(const int n, const float* data_im, const int height,
                                  const int width, const int ksize_h, const int ksize_w,
                                  const int pad_h, const int pad_w, const int stride_h,
                                  const int stride_w, const int dilation_h, const int dilation_w,
                                  const int height_col, const int width_col, float* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * ksize_h * ksize_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;

    float* data_col_ptr = data_col + (c_col * height_col + h_col) * width_col + w_col;
    const float* data_im_ptr = data_im + (c_im * height + h_offset) * width + w_offset;

    for (int i = 0; i < ksize_h; ++i) {
      for (int j = 0; j < ksize_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                            ? data_im_ptr[i * dilation_h * width + j * dilation_w]
                            : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

// CUDA kernel for col2im
__global__ void col2im_gpu_kernel(const int n, const float* data_col, const int height,
                                  const int width, const int channels, const int ksize_h,
                                  const int ksize_w, const int pad_h, const int pad_w,
                                  const int stride_h, const int stride_w, const int dilation_h,
                                  const int dilation_w, const int height_col, const int width_col,
                                  float* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    float val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);

    // compute the start and end of the output
    const int w_col_start =
        (w_im < ksize_w * dilation_w) ? 0 : (w_im - ksize_w * dilation_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < ksize_h * dilation_h) ? 0 : (h_im - ksize_h * dilation_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);

    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index =
              (((c_im * ksize_h + h_k) * ksize_w + w_k) * height_col + h_col) * width_col + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = val;
  }
}

// CUDA kernel for adding bias
__global__ void add_bias_kernel(float* output, float bias_val, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] += bias_val;
  }
}

// CUDA kernel for bias gradient computation
__global__ void bias_grad_kernel(const float* grad_output, float* grad_bias, int batch_size,
                                 int channels, int spatial_size, int channel_idx) {
  __shared__ float shared_sum[CUDA_NUM_THREADS];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  shared_sum[tid] = 0.0f;

  // 累加所有 batch 和 spatial 位置的梯度
  while (idx < batch_size * spatial_size) {
    int batch_idx = idx / spatial_size;
    int spatial_idx = idx % spatial_size;
    int grad_idx = batch_idx * channels * spatial_size + channel_idx * spatial_size + spatial_idx;
    shared_sum[tid] += grad_output[grad_idx];
    idx += blockDim.x * gridDim.x;
  }

  __syncthreads();

  // Reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_sum[tid] += shared_sum[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(grad_bias + channel_idx, shared_sum[0]);
  }
}

void im2col_cuda(const float* data_im, const int channels, const int height, const int width,
                 const int ksize_h, const int ksize_w, const int pad_h, const int pad_w,
                 const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
                 float* data_col) {
  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;

  im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w,
      dilation_h, dilation_w, height_col, width_col, data_col);

  CUDA_POST_KERNEL_CHECK;
}

void col2im_cuda(const float* data_col, const int channels, const int height, const int width,
                 const int ksize_h, const int ksize_w, const int pad_h, const int pad_w,
                 const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
                 float* data_im) {
  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height * width;

  // Clear output
  cudaMemset(data_im, 0, sizeof(float) * num_kernels);

  col2im_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, ksize_h, ksize_w, pad_h, pad_w, stride_h,
      stride_w, dilation_h, dilation_w, height_col, width_col, data_im);

  CUDA_POST_KERNEL_CHECK;
}

// 计算输出尺寸的辅助函数
int compute_output_size_cuda(int input_size, int kernel_size, int stride, int padding,
                             int dilation) {
  return (input_size + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1;
}

torch::Tensor conv2d_forward_cuda(const torch::Tensor& input, const torch::Tensor& weight,
                                  const torch::Tensor& bias, const std::vector<int64_t>& stride,
                                  const std::vector<int64_t>& padding,
                                  const std::vector<int64_t>& dilation, int groups) {
  TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");

  auto input_contig = input.contiguous();
  auto weight_contig = weight.contiguous();

  int batch_size = input_contig.size(0);
  int input_channels = input_contig.size(1);
  int input_height = input_contig.size(2);
  int input_width = input_contig.size(3);

  int output_channels = weight_contig.size(0);
  int kernel_height = weight_contig.size(2);
  int kernel_width = weight_contig.size(3);

  int stride_h = stride[0];
  int stride_w = stride[1];
  int pad_h = padding[0];
  int pad_w = padding[1];
  int dilation_h = dilation[0];
  int dilation_w = dilation[1];

  int output_height =
      compute_output_size_cuda(input_height, kernel_height, stride_h, pad_h, dilation_h);
  int output_width =
      compute_output_size_cuda(input_width, kernel_width, stride_w, pad_w, dilation_w);

  auto output =
      torch::zeros({batch_size, output_channels, output_height, output_width}, input.options());

  int channels_per_group = input_channels / groups;
  int output_channels_per_group = output_channels / groups;

  // Use cuBLAS for GEMM operations
  cublasHandle_t handle;
  cublasCreate(&handle);

  for (int b = 0; b < batch_size; ++b) {
    for (int g = 0; g < groups; ++g) {
      // im2col
      int col_height = channels_per_group * kernel_height * kernel_width;
      int col_width = output_height * output_width;
      auto col_data = torch::zeros({col_height, col_width}, input.options());

      const float* input_data = input_contig.data_ptr<float>() +
                                b * input_channels * input_height * input_width +
                                g * channels_per_group * input_height * input_width;

      im2col_cuda(input_data, channels_per_group, input_height, input_width, kernel_height,
                  kernel_width, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                  col_data.data_ptr<float>());

      // GEMM using cuBLAS
      const float* weight_data = weight_contig.data_ptr<float>() + g * output_channels_per_group *
                                                                       channels_per_group *
                                                                       kernel_height * kernel_width;

      float* output_data = output.data_ptr<float>() +
                           b * output_channels * output_height * output_width +
                           g * output_channels_per_group * output_height * output_width;

      const float alpha = 1.0f, beta = 0.0f;
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, col_width, output_channels_per_group,
                  col_height, &alpha, col_data.data_ptr<float>(), col_width, weight_data,
                  col_height, &beta, output_data, col_width);
    }

    // Add bias
    if (bias.defined()) {
      auto bias_cuda = bias.cuda();
      for (int c = 0; c < output_channels; ++c) {
        float bias_val = bias_cuda.data_ptr<float>()[c];
        float* output_channel = output.data_ptr<float>() +
                                b * output_channels * output_height * output_width +
                                c * output_height * output_width;

        int num_elements = output_height * output_width;
        add_bias_kernel<<<GET_BLOCKS(num_elements), CUDA_NUM_THREADS>>>(output_channel, bias_val,
                                                                        num_elements);
      }
    }
  }

  cublasDestroy(handle);
  return output;
}

std::vector<torch::Tensor> conv2d_backward_cuda(
    const torch::Tensor& grad_output, const torch::Tensor& input, const torch::Tensor& weight,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, int groups, const std::array<bool, 3>& output_mask) {
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be CUDA tensor");
  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");

  std::vector<torch::Tensor> grads;

  auto grad_output_contig = grad_output.contiguous();
  auto input_contig = input.contiguous();
  auto weight_contig = weight.contiguous();

  int batch_size = input_contig.size(0);
  int input_channels = input_contig.size(1);
  int input_height = input_contig.size(2);
  int input_width = input_contig.size(3);

  int output_channels = weight_contig.size(0);
  int kernel_height = weight_contig.size(2);
  int kernel_width = weight_contig.size(3);

  int stride_h = stride[0];
  int stride_w = stride[1];
  int pad_h = padding[0];
  int pad_w = padding[1];
  int dilation_h = dilation[0];
  int dilation_w = dilation[1];

  int output_height = grad_output_contig.size(2);
  int output_width = grad_output_contig.size(3);

  int channels_per_group = input_channels / groups;
  int output_channels_per_group = output_channels / groups;

  // Create cuBLAS handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // 计算输入梯度
  if (output_mask[0]) {
    auto grad_input = torch::zeros_like(input_contig);

    for (int b = 0; b < batch_size; ++b) {
      for (int g = 0; g < groups; ++g) {
        // col_data 用于存储 weight^T × grad_output
        int col_height = channels_per_group * kernel_height * kernel_width;
        int col_width = output_height * output_width;
        auto col_data = torch::zeros({col_height, col_width}, input.options());

        const float* weight_data =
            weight_contig.data_ptr<float>() +
            g * output_channels_per_group * channels_per_group * kernel_height * kernel_width;

        const float* grad_output_data =
            grad_output_contig.data_ptr<float>() +
            b * output_channels * output_height * output_width +
            g * output_channels_per_group * output_height * output_width;

        // GEMM: weight^T × grad_output
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, col_width, col_height,
                    output_channels_per_group, &alpha, grad_output_data, col_width, weight_data,
                    col_height, &beta, col_data.data_ptr<float>(), col_width);

        // col2im
        float* grad_input_data = grad_input.data_ptr<float>() +
                                 b * input_channels * input_height * input_width +
                                 g * channels_per_group * input_height * input_width;

        col2im_cuda(col_data.data_ptr<float>(), channels_per_group, input_height, input_width,
                    kernel_height, kernel_width, pad_h, pad_w, stride_h, stride_w, dilation_h,
                    dilation_w, grad_input_data);
      }
    }
    grads.push_back(grad_input);
  } else {
    grads.push_back(torch::Tensor());
  }

  // 计算权重梯度
  if (output_mask[1]) {
    auto grad_weight = torch::zeros_like(weight_contig);

    for (int b = 0; b < batch_size; ++b) {
      for (int g = 0; g < groups; ++g) {
        // im2col on input
        int col_height = channels_per_group * kernel_height * kernel_width;
        int col_width = output_height * output_width;
        auto col_data = torch::zeros({col_height, col_width}, input.options());

        const float* input_data = input_contig.data_ptr<float>() +
                                  b * input_channels * input_height * input_width +
                                  g * channels_per_group * input_height * input_width;

        im2col_cuda(input_data, channels_per_group, input_height, input_width, kernel_height,
                    kernel_width, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                    col_data.data_ptr<float>());

        // GEMM: grad_output × col_data^T
        const float* grad_output_data =
            grad_output_contig.data_ptr<float>() +
            b * output_channels * output_height * output_width +
            g * output_channels_per_group * output_height * output_width;

        float* grad_weight_data = grad_weight.data_ptr<float>() + g * output_channels_per_group *
                                                                      channels_per_group *
                                                                      kernel_height * kernel_width;

        const float alpha = 1.0f, beta = (b == 0 && g == 0) ? 0.0f : 1.0f;  // 第一次设为0，后续累加
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, col_height, output_channels_per_group,
                    col_width, &alpha, col_data.data_ptr<float>(), col_width, grad_output_data,
                    col_width, &beta, grad_weight_data, col_height);
      }
    }
    grads.push_back(grad_weight);
  } else {
    grads.push_back(torch::Tensor());
  }

  // 计算偏置梯度
  if (output_mask[2]) {
    auto grad_bias = torch::zeros({output_channels}, input.options());

    // 清零偏置梯度
    cudaMemset(grad_bias.data_ptr<float>(), 0, sizeof(float) * output_channels);

    int spatial_size = output_height * output_width;

    for (int c = 0; c < output_channels; ++c) {
      // 使用 CUDA kernel 并行计算每个通道的偏置梯度
      bias_grad_kernel<<<GET_BLOCKS(batch_size * spatial_size), CUDA_NUM_THREADS>>>(
          grad_output_contig.data_ptr<float>(), grad_bias.data_ptr<float>(), batch_size,
          output_channels, spatial_size, c);
    }

    CUDA_POST_KERNEL_CHECK;
    grads.push_back(grad_bias);
  } else {
    grads.push_back(torch::Tensor());
  }

  cublasDestroy(handle);
  return grads;
}

}  // namespace cuda
}  // namespace ops
}  // namespace bitflow