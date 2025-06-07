#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "bitflow/ops/conv.h"

void bind_conv_ops(pybind11::module& m) {
  // 卷积前向传播
  m.def("conv2d_forward", &bitflow::ops::conv2d_forward, "Conv2d forward pass",
        pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
        pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("dilation"),
        pybind11::arg("groups"));

  // 卷积反向传播
  m.def(
      "conv2d_backward",
      [](const torch::Tensor& grad_output, const torch::Tensor& input, const torch::Tensor& weight,
         const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
         const std::vector<int64_t>& dilation, int groups,
         const std::vector<bool>& output_mask) -> std::vector<torch::Tensor> {
        // 转换为数组
        std::array<bool, 3> mask_array = {false, false, false};
        for (size_t i = 0; i < std::min(static_cast<size_t>(3), output_mask.size()); ++i) {
          mask_array[i] = output_mask[i];
        }

        return bitflow::ops::conv2d_backward(grad_output, input, weight, stride, padding, dilation,
                                             groups, mask_array);
      },
      "Conv2d backward pass", pybind11::arg("grad_output"), pybind11::arg("input"),
      pybind11::arg("weight"), pybind11::arg("stride"), pybind11::arg("padding"),
      pybind11::arg("dilation"), pybind11::arg("groups"), pybind11::arg("output_mask"));

  // CPU专用函数
  m.def("conv2d_forward_cpu", &bitflow::ops::conv2d_forward_cpu, "Conv2d forward pass (CPU)",
        pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
        pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("dilation"),
        pybind11::arg("groups"));

  m.def(
      "conv2d_backward_cpu",
      [](const torch::Tensor& grad_output, const torch::Tensor& input, const torch::Tensor& weight,
         const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
         const std::vector<int64_t>& dilation, int groups,
         const std::vector<bool>& output_mask) -> std::vector<torch::Tensor> {
        std::array<bool, 3> mask_array = {false, false, false};
        for (size_t i = 0; i < std::min(static_cast<size_t>(3), output_mask.size()); ++i) {
          mask_array[i] = output_mask[i];
        }
        return bitflow::ops::conv2d_backward_cpu(grad_output, input, weight, stride, padding,
                                                 dilation, groups, mask_array);
      },
      "Conv2d backward pass (CPU)", pybind11::arg("grad_output"), pybind11::arg("input"),
      pybind11::arg("weight"), pybind11::arg("stride"), pybind11::arg("padding"),
      pybind11::arg("dilation"), pybind11::arg("groups"), pybind11::arg("output_mask"));

#ifdef TORCH_CUDA_AVAILABLE
  // CUDA 版本
  m.def("conv2d_forward_cuda", &bitflow::ops::cuda::conv2d_forward_cuda,
        "Conv2d forward pass (CUDA)", pybind11::arg("input"), pybind11::arg("weight"),
        pybind11::arg("bias"), pybind11::arg("stride"), pybind11::arg("padding"),
        pybind11::arg("dilation"), pybind11::arg("groups"));

  m.def(
      "conv2d_backward_cuda",
      [](const torch::Tensor& grad_output, const torch::Tensor& input, const torch::Tensor& weight,
         const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
         const std::vector<int64_t>& dilation, int groups,
         const std::vector<bool>& output_mask) -> std::vector<torch::Tensor> {
        std::array<bool, 3> mask_array = {false, false, false};
        for (size_t i = 0; i < std::min(static_cast<size_t>(3), output_mask.size()); ++i) {
          mask_array[i] = output_mask[i];
        }
        return bitflow::ops::cuda::conv2d_backward_cuda(grad_output, input, weight, stride, padding,
                                                        dilation, groups, mask_array);
      },
      "Conv2d backward pass (CUDA)", pybind11::arg("grad_output"), pybind11::arg("input"),
      pybind11::arg("weight"), pybind11::arg("stride"), pybind11::arg("padding"),
      pybind11::arg("dilation"), pybind11::arg("groups"), pybind11::arg("output_mask"));
#endif

  // 设备检查函数
  m.def(
      "is_cuda_available",
      []() {
#ifdef TORCH_CUDA_AVAILABLE
        return torch::cuda::is_available();
#else
        return false;
#endif
      },
      "Check if CUDA is available");

  // 工具函数
  m.def("compute_output_size", &bitflow::ops::compute_output_size,
        "Compute convolution output size", pybind11::arg("input_size"),
        pybind11::arg("kernel_size"), pybind11::arg("stride"), pybind11::arg("padding"),
        pybind11::arg("dilation"));
}