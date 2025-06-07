#pragma once

#include <torch/torch.h>

#include <array>
#include <vector>

namespace bitflow {
namespace ops {

// 辅助函数声明（需要提前声明，因为被其他函数使用）
int compute_output_size(int input_size, int kernel_size, int stride, int padding, int dilation);

// CPU 实现的辅助函数声明
void im2col_cpu(const float* data_im, int channels, int height, int width, int kernel_h,
                int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
                int dilation_w, float* data_col);

void col2im_cpu(const float* data_col, int channels, int height, int width, int kernel_h,
                int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
                int dilation_w, float* data_im);

void gemm_cpu(bool trans_a, bool trans_b, int m, int n, int k, float alpha, const float* a,
              const float* b, float beta, float* c);

// CPU 实现
torch::Tensor conv2d_forward_cpu(const torch::Tensor& input, const torch::Tensor& weight,
                                 const torch::Tensor& bias, const std::vector<int64_t>& stride,
                                 const std::vector<int64_t>& padding,
                                 const std::vector<int64_t>& dilation, int groups);

std::vector<torch::Tensor> conv2d_backward_cpu(
    const torch::Tensor& grad_output, const torch::Tensor& input, const torch::Tensor& weight,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, int groups, const std::array<bool, 3>& output_mask);

// 主接口函数
torch::Tensor conv2d_forward(const torch::Tensor& input, const torch::Tensor& weight,
                             const torch::Tensor& bias, const std::vector<int64_t>& stride,
                             const std::vector<int64_t>& padding,
                             const std::vector<int64_t>& dilation, int groups);

std::vector<torch::Tensor> conv2d_backward(const torch::Tensor& grad_output,
                                           const torch::Tensor& input, const torch::Tensor& weight,
                                           const std::vector<int64_t>& stride,
                                           const std::vector<int64_t>& padding,
                                           const std::vector<int64_t>& dilation, int groups,
                                           const std::array<bool, 3>& output_mask);

// CUDA 实现（如果启用）
#ifdef TORCH_CUDA_AVAILABLE
namespace cuda {

// CUDA 辅助函数声明
void im2col_cuda(const float* data_im, const int channels, const int height, const int width,
                 const int ksize_h, const int ksize_w, const int pad_h, const int pad_w,
                 const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
                 float* data_col);

void col2im_cuda(const float* data_col, const int channels, const int height, const int width,
                 const int ksize_h, const int ksize_w, const int pad_h, const int pad_w,
                 const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
                 float* data_im);

int compute_output_size_cuda(int input_size, int kernel_size, int stride, int padding,
                             int dilation);

// CUDA 主要函数
torch::Tensor conv2d_forward_cuda(const torch::Tensor& input, const torch::Tensor& weight,
                                  const torch::Tensor& bias, const std::vector<int64_t>& stride,
                                  const std::vector<int64_t>& padding,
                                  const std::vector<int64_t>& dilation, int groups);

std::vector<torch::Tensor> conv2d_backward_cuda(
    const torch::Tensor& grad_output, const torch::Tensor& input, const torch::Tensor& weight,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, int groups, const std::array<bool, 3>& output_mask);

}  // namespace cuda
#endif

}  // namespace ops
}  // namespace bitflow