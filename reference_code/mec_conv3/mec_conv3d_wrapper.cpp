#include <torch/extension.h>
#include <vector>

// 声明在 .cu 文件中定义的 CUDA 3D 卷积函数
// extern "C" 确保 C++ 编译器使用 C 链接约定
extern "C" void custom_conv3d_gpu(
        float* d_input, float* d_weight_flat, float* d_bias, float* d_output,
        int batch_size,
        int in_depth, int in_height, int in_width, int in_channels,
        int kernel_depth, int kernel_height, int kernel_width,
        int out_channels, int out_depth, int out_height, int out_width,
        int padding_d, int padding_h, int padding_w,
        int stride_d, int stride_h, int stride_w);

// 宏定义，用于检查张量是否在 CUDA 设备上并且是 float 类型且连续
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_FLOAT(x); CHECK_CONTIGUOUS(x)

// PyTorch 包装函数 for 3D convolution
torch::Tensor mec_conv3d_forward(
        torch::Tensor input,      // (batch_size, in_channels, in_depth, in_height, in_width)
        torch::Tensor weight,     // (out_channels, in_channels, kernel_d, kernel_h, kernel_w)
        torch::Tensor bias,       // (out_channels) or empty tensor
        int padding_d, int padding_h, int padding_w,
        int stride_d, int stride_h, int stride_w) {

    // 检查输入张量
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined() && bias.numel() > 0) { // 只有当 bias 被定义且非空时才检查
        CHECK_INPUT(bias);
    }

    // 从输入张量获取维度信息 (NCDHW)
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);

    // 从权重张量获取维度信息 (KCDHW_kernel)
    const int out_channels = weight.size(0);
    TORCH_CHECK(weight.size(1) == in_channels, "weight.size(1) (in_channels for weight) must match input.size(1)");
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);


    // 计算输出维度
    // out_depth = (in_depth + 2 * padding_d - kernel_d) / stride_d + 1
    // out_height = (in_height + 2 * padding_h - kernel_h) / stride_h + 1
    // out_width = (in_width + 2 * padding_w - kernel_w) / stride_w + 1
    const int out_depth = (in_depth + 2 * padding_d - kernel_d) / stride_d + 1;
    const int out_height = (in_height + 2 * padding_h - kernel_h) / stride_h + 1;
    const int out_width = (in_width + 2 * padding_w - kernel_w) / stride_w + 1;

    TORCH_CHECK(out_depth > 0 && out_height > 0 && out_width > 0,
                "Output depth, height, and width must be positive. Check padding, stride, and kernel size.");

    // 准备权重张量：将其扁平化为 (out_channels, in_channels * kernel_d * kernel_h * kernel_w)
    torch::Tensor weight_flat = weight.contiguous().view({out_channels, in_channels * kernel_d * kernel_h * kernel_w});
    CHECK_CONTIGUOUS(weight_flat); // 确保变形后的张量仍然是连续的

    // 创建输出张量 (NCDHW)
    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    CHECK_INPUT(output); // 确保输出张量也在CUDA上，是float类型且连续

    // 获取原始数据指针
    float* d_input_ptr = input.data_ptr<float>();
    float* d_weight_flat_ptr = weight_flat.data_ptr<float>();
    float* d_bias_ptr = nullptr;
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.size(0) == out_channels, "bias.size(0) must match out_channels");
        d_bias_ptr = bias.data_ptr<float>();
    }
    float* d_output_ptr = output.data_ptr<float>();

    // 调用 CUDA C 函数
    custom_conv3d_gpu(
            d_input_ptr, d_weight_flat_ptr, d_bias_ptr, d_output_ptr,
            batch_size,
            in_depth, in_height, in_width, in_channels,
            kernel_d, kernel_h, kernel_w,
            out_channels, out_depth, out_height, out_width,
            padding_d, padding_h, padding_w,
            stride_d, stride_h, stride_w
    );

    return output;
}

// 将C++函数绑定到Python模块
// TORCH_EXTENSION_NAME 会被 setup.py 中的 name 参数替换
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("forward", // 在Python中调用的函数名
&mec_conv3d_forward, // C++中对应的函数指针
"MEC Custom Conv3D forward (CUDA)" // 函数的文档字符串
// 使用 pybind11::arg 来命名参数，使其在 Python 中更友好
, py::arg("input"), py::arg("weight"), py::arg("bias")
, py::arg("padding_d"), py::arg("padding_h"), py::arg("padding_w")
, py::arg("stride_d"), py::arg("stride_h"), py::arg("stride_w")
);
}
