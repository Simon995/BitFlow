#include "bitflow/ops/conv.h"

#include <omp.h>

#include <algorithm>
#include <cstring>

// CUDA前向声明
#ifdef TORCH_CUDA_AVAILABLE
namespace bitflow {
namespace ops {
namespace cuda {
torch::Tensor conv2d_forward_cuda(const torch::Tensor& input, const torch::Tensor& weight,
                                  const torch::Tensor& bias, const std::vector<int64_t>& stride,
                                  const std::vector<int64_t>& padding,
                                  const std::vector<int64_t>& dilation, int groups);

std::vector<torch::Tensor> conv2d_backward_cuda(
    const torch::Tensor& grad_output, const torch::Tensor& input, const torch::Tensor& weight,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, int groups, const std::array<bool, 3>& output_mask);
}  // namespace cuda
}  // namespace ops
}  // namespace bitflow
#endif

namespace bitflow {
namespace ops {

// 添加计算输出尺寸的辅助函数
int compute_output_size(int input_size, int kernel_size, int stride, int padding, int dilation) {
  return (input_size + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1;
}

// im2col 实现
void im2col_cpu(const float* data_im, int channels, int height, int width, int kernel_h,
                int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
                int dilation_w, float* data_col) {
  int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;

#pragma omp parallel for
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    int w_offset = c_col % kernel_w;
    int h_offset = (c_col / kernel_w) % kernel_h;
    int c_im = c_col / kernel_h / kernel_w;

    for (int h_col = 0; h_col < output_h; ++h_col) {
      for (int w_col = 0; w_col < output_w; ++w_col) {
        int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
        int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;

        data_col[(c_col * output_h + h_col) * output_w + w_col] =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                ? data_im[(c_im * height + h_im) * width + w_im]
                : 0;
      }
    }
  }
}

// col2im 实现
void col2im_cpu(const float* data_col, int channels, int height, int width, int kernel_h,
                int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
                int dilation_w, float* data_im) {
  std::memset(data_im, 0, sizeof(float) * height * width * channels);

  int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;

#pragma omp parallel for
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    int w_offset = c_col % kernel_w;
    int h_offset = (c_col / kernel_w) % kernel_h;
    int c_im = c_col / kernel_h / kernel_w;

    for (int h_col = 0; h_col < output_h; ++h_col) {
      for (int w_col = 0; w_col < output_w; ++w_col) {
        int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
        int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;

        if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
#pragma omp atomic
          data_im[(c_im * height + h_im) * width + w_im] +=
              data_col[(c_col * output_h + h_col) * output_w + w_col];
        }
      }
    }
  }
}

// GEMM 实现（简化版）
void gemm_cpu(bool trans_a, bool trans_b, int m, int n, int k, float alpha, const float* a,
              const float* b, float beta, float* c) {
#pragma omp parallel for
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int l = 0; l < k; ++l) {
        float val_a = trans_a ? a[l * m + i] : a[i * k + l];
        float val_b = trans_b ? b[j * k + l] : b[l * n + j];
        sum += val_a * val_b;
      }
      c[i * n + j] = alpha * sum + beta * c[i * n + j];
    }
  }
}

torch::Tensor conv2d_forward_cpu(const torch::Tensor& input, const torch::Tensor& weight,
                                 const torch::Tensor& bias, const std::vector<int64_t>& stride,
                                 const std::vector<int64_t>& padding,
                                 const std::vector<int64_t>& dilation, int groups) {
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

  int output_height = compute_output_size(input_height, kernel_height, stride_h, pad_h, dilation_h);
  int output_width = compute_output_size(input_width, kernel_width, stride_w, pad_w, dilation_w);

  auto output =
      torch::zeros({batch_size, output_channels, output_height, output_width}, input.options());

  int channels_per_group = input_channels / groups;
  int output_channels_per_group = output_channels / groups;

  // 为每个 batch 处理
  for (int b = 0; b < batch_size; ++b) {
    for (int g = 0; g < groups; ++g) {
      // im2col
      int col_height = channels_per_group * kernel_height * kernel_width;
      int col_width = output_height * output_width;
      auto col_data = torch::zeros({col_height, col_width}, input.options());

      const float* input_data = input_contig.data_ptr<float>() +
                                b * input_channels * input_height * input_width +
                                g * channels_per_group * input_height * input_width;

      im2col_cpu(input_data, channels_per_group, input_height, input_width, kernel_height,
                 kernel_width, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                 col_data.data_ptr<float>());

      // GEMM: weight × col_data
      const float* weight_data = weight_contig.data_ptr<float>() + g * output_channels_per_group *
                                                                       channels_per_group *
                                                                       kernel_height * kernel_width;

      float* output_data = output.data_ptr<float>() +
                           b * output_channels * output_height * output_width +
                           g * output_channels_per_group * output_height * output_width;

      gemm_cpu(false, false, output_channels_per_group, col_width, col_height, 1.0f, weight_data,
               col_data.data_ptr<float>(), 0.0f, output_data);
    }

    // 添加偏置
    if (bias.defined()) {
      for (int c = 0; c < output_channels; ++c) {
        float bias_val = bias.data_ptr<float>()[c];
        float* output_channel = output.data_ptr<float>() +
                                b * output_channels * output_height * output_width +
                                c * output_height * output_width;

#pragma omp parallel for
        for (int i = 0; i < output_height * output_width; ++i) {
          output_channel[i] += bias_val;
        }
      }
    }
  }

  return output;
}

std::vector<torch::Tensor> conv2d_backward_cpu(
    const torch::Tensor& grad_output, const torch::Tensor& input, const torch::Tensor& weight,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, int groups, const std::array<bool, 3>& output_mask) {
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
        gemm_cpu(true, false, col_height, col_width, output_channels_per_group, 1.0f, weight_data,
                 grad_output_data, 0.0f, col_data.data_ptr<float>());

        // col2im
        float* grad_input_data = grad_input.data_ptr<float>() +
                                 b * input_channels * input_height * input_width +
                                 g * channels_per_group * input_height * input_width;

        col2im_cpu(col_data.data_ptr<float>(), channels_per_group, input_height, input_width,
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

        im2col_cpu(input_data, channels_per_group, input_height, input_width, kernel_height,
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

        gemm_cpu(false, true, output_channels_per_group, col_height, col_width, 1.0f,
                 grad_output_data, col_data.data_ptr<float>(), 1.0f,
                 grad_weight_data);  // 累加到现有梯度
      }
    }
    grads.push_back(grad_weight);
  } else {
    grads.push_back(torch::Tensor());
  }

  // 计算偏置梯度
  if (output_mask[2]) {
    auto grad_bias = torch::zeros({output_channels}, input.options());

    for (int c = 0; c < output_channels; ++c) {
      float sum = 0.0f;
      for (int b = 0; b < batch_size; ++b) {
        const float* grad_output_channel = grad_output_contig.data_ptr<float>() +
                                           b * output_channels * output_height * output_width +
                                           c * output_height * output_width;
        for (int i = 0; i < output_height * output_width; ++i) {
          sum += grad_output_channel[i];
        }
      }
      grad_bias.data_ptr<float>()[c] = sum;
    }
    grads.push_back(grad_bias);
  } else {
    grads.push_back(torch::Tensor());
  }

  return grads;
}

// 主接口函数
torch::Tensor conv2d_forward(const torch::Tensor& input, const torch::Tensor& weight,
                             const torch::Tensor& bias, const std::vector<int64_t>& stride,
                             const std::vector<int64_t>& padding,
                             const std::vector<int64_t>& dilation, int groups) {
  if (input.is_cuda()) {
#ifdef TORCH_CUDA_AVAILABLE
    return cuda::conv2d_forward_cuda(input, weight, bias, stride, padding, dilation, groups);
#else
    // 如果没有CUDA支持，将tensor移到CPU进行计算
    auto cpu_input = input.cpu();
    auto cpu_weight = weight.cpu();
    auto cpu_bias = bias.defined() ? bias.cpu() : torch::Tensor();
    auto result =
        conv2d_forward_cpu(cpu_input, cpu_weight, cpu_bias, stride, padding, dilation, groups);
    return result.to(input.device());
#endif
  } else {
    return conv2d_forward_cpu(input, weight, bias, stride, padding, dilation, groups);
  }
}

std::vector<torch::Tensor> conv2d_backward(const torch::Tensor& grad_output,
                                           const torch::Tensor& input, const torch::Tensor& weight,
                                           const std::vector<int64_t>& stride,
                                           const std::vector<int64_t>& padding,
                                           const std::vector<int64_t>& dilation, int groups,
                                           const std::array<bool, 3>& output_mask) {
  if (input.is_cuda()) {
#ifdef TORCH_CUDA_AVAILABLE
    return cuda::conv2d_backward_cuda(grad_output, input, weight, stride, padding, dilation, groups,
                                      output_mask);
#else
    // 如果没有CUDA支持，将tensor移到CPU进行计算
    auto cpu_grad_output = grad_output.cpu();
    auto cpu_input = input.cpu();
    auto cpu_weight = weight.cpu();
    auto results = conv2d_backward_cpu(cpu_grad_output, cpu_input, cpu_weight, stride, padding,
                                       dilation, groups, output_mask);

    // 将结果移回原设备
    for (auto& result : results) {
      if (result.defined()) {
        result = result.to(input.device());
      }
    }
    return results;
#endif
  } else {
    return conv2d_backward_cpu(grad_output, input, weight, stride, padding, dilation, groups,
                               output_mask);
  }
}

}  // namespace ops
}  // namespace bitflow