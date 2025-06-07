import torch
import torch.nn.functional as F
from torch.autograd import Function
import warnings


class Conv2dFunction(Function):
    @staticmethod
    def forward(
        ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
    ):
        # 确保参数是正确的类型
        if isinstance(stride, int):
            stride = [stride, stride]
        if isinstance(padding, int):
            padding = [padding, padding]
        if isinstance(dilation, int):
            dilation = [dilation, dilation]

        # 保存用于反向传播的变量
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        # 尝试使用 C++ 实现
        try:
            from bitflow import _C

            # 调用 C++ 实现
            if bias is None:
                bias = torch.empty(0, dtype=input.dtype, device=input.device)

            output = _C.conv2d_forward(
                input, weight, bias, stride, padding, dilation, groups
            )
            ctx.used_cpp = True
            return output
        except (ImportError, AttributeError) as e:
            warnings.warn(
                f"C++ extension not available: {e}, falling back to PyTorch implementation"
            )

        # 回退到 PyTorch 实现
        ctx.used_cpp = False
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # 确定需要计算哪些梯度
        needs_input_grad = (
            ctx.needs_input_grad[0] if len(ctx.needs_input_grad) > 0 else False
        )
        needs_weight_grad = (
            ctx.needs_input_grad[1] if len(ctx.needs_input_grad) > 1 else False
        )
        needs_bias_grad = (
            ctx.needs_input_grad[2]
            if len(ctx.needs_input_grad) > 2 and bias is not None
            else False
        )

        if hasattr(ctx, "used_cpp") and ctx.used_cpp:
            # 使用 C++ 反向传播
            try:
                from bitflow import _C

                output_mask = [needs_input_grad, needs_weight_grad, needs_bias_grad]

                grads = _C.conv2d_backward(
                    grad_output,
                    input,
                    weight,
                    ctx.stride,
                    ctx.padding,
                    ctx.dilation,
                    ctx.groups,
                    output_mask,
                )

                if needs_input_grad:
                    grad_input = grads[0]
                if needs_weight_grad:
                    grad_weight = grads[1]
                if needs_bias_grad:
                    grad_bias = grads[2]

            except Exception as e:
                warnings.warn(f"C++ backward failed: {e}, falling back to PyTorch")
                # 回退到 PyTorch 实现
                return _pytorch_conv2d_backward(ctx, grad_output, input, weight, bias)
        else:
            # 使用 PyTorch 反向传播
            return _pytorch_conv2d_backward(ctx, grad_output, input, weight, bias)

        return grad_input, grad_weight, grad_bias, None, None, None, None


def _pytorch_conv2d_backward(ctx, grad_output, input, weight, bias):
    """PyTorch 反向传播实现"""
    grad_input = grad_weight = grad_bias = None

    needs_input_grad = (
        ctx.needs_input_grad[0] if len(ctx.needs_input_grad) > 0 else False
    )
    needs_weight_grad = (
        ctx.needs_input_grad[1] if len(ctx.needs_input_grad) > 1 else False
    )
    needs_bias_grad = (
        ctx.needs_input_grad[2]
        if len(ctx.needs_input_grad) > 2 and bias is not None
        else False
    )

    if needs_input_grad:
        grad_input = torch.nn.grad.conv2d_input(
            input.shape,
            weight,
            grad_output,
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            ctx.groups,
        )

    if needs_weight_grad:
        grad_weight = torch.nn.grad.conv2d_weight(
            input,
            weight.shape,
            grad_output,
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            ctx.groups,
        )

    if needs_bias_grad:
        grad_bias = grad_output.sum(dim=(0, 2, 3))

    return grad_input, grad_weight, grad_bias, None, None, None, None


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    自定义2D卷积操作

    Args:
        input: 输入张量，形状为 (N, C_in, H_in, W_in)
        weight: 权重张量，形状为 (C_out, C_in/groups, kH, kW)
        bias: 偏置张量，形状为 (C_out,)，可选
        stride: 步长
        padding: 填充
        dilation: 膨胀
        groups: 分组卷积的组数

    Returns:
        输出张量，形状为 (N, C_out, H_out, W_out)
    """
    return Conv2dFunction.apply(input, weight, bias, stride, padding, dilation, groups)


# 提供一个简单的测试函数
def test_conv2d():
    """测试卷积功能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建测试数据
    input_tensor = torch.randn(1, 3, 8, 8, device=device, requires_grad=True)
    weight = torch.randn(16, 3, 3, 3, device=device, requires_grad=True)
    bias = torch.randn(16, device=device, requires_grad=True)

    # 使用自定义卷积
    output = conv2d(input_tensor, weight, bias, stride=1, padding=1)

    # 使用 PyTorch 卷积对比
    output_ref = F.conv2d(input_tensor, weight, bias, stride=1, padding=1)

    # 计算差异
    diff = torch.abs(output - output_ref).max().item()
    print(f"Max difference: {diff:.2e}")

    # 测试反向传播
    loss = output.sum()
    loss.backward()

    print("Forward and backward pass completed successfully!")

    return output, output_ref, diff


if __name__ == "__main__":
    test_conv2d()
