#include <pybind11/pybind11.h>
#include <torch/extension.h>

// 声明绑定函数
void bind_conv_ops(pybind11::module& m);

PYBIND11_MODULE(_C, m) {
  m.doc() = "BitFlow C++ Extension";

  // 绑定操作
  bind_conv_ops(m);
}