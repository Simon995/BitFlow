import torch
import torch.nn.functional as F
import time


def test_conv_correctness():
    """测试卷积操作的正确性"""
    print("=" * 60)
    print("卷积正确性测试")
    print("=" * 60)

    # 测试用例
    test_cases = [
        # (batch, in_channels, height, width, out_channels, kernel_size, stride, padding)
        (1, 3, 8, 8, 16, 3, 1, 0),
        (2, 16, 16, 16, 32, 3, 1, 1),
        (4, 32, 32, 32, 64, 5, 2, 2),
        (1, 64, 56, 56, 128, 3, 1, 1),
    ]

    for i, (batch, in_ch, h, w, out_ch, k, s, p) in enumerate(test_cases):
        print(
            f"\n测试用例 {i + 1}: [{batch}, {in_ch}, {h}, {w}] -> [{out_ch}, {k}x{k}], stride={s}, padding={p}"
        )

        # 创建测试数据
        input_tensor = torch.randn(batch, in_ch, h, w, requires_grad=True)
        weight = torch.randn(out_ch, in_ch, k, k, requires_grad=True)
        bias = torch.randn(out_ch, requires_grad=True)

        # PyTorch 参考实现
        input_pt = input_tensor.clone().detach().requires_grad_(True)
        weight_pt = weight.clone().detach().requires_grad_(True)
        bias_pt = bias.clone().detach().requires_grad_(True)

        output_pt = F.conv2d(input_pt, weight_pt, bias_pt, stride=s, padding=p)
        loss_pt = output_pt.sum()
        loss_pt.backward()

        # BitFlow 实现
        try:
            from bitflow.ops import conv2d

            input_bf = input_tensor.clone().detach().requires_grad_(True)
            weight_bf = weight.clone().detach().requires_grad_(True)
            bias_bf = bias.clone().detach().requires_grad_(True)

            output_bf = conv2d(input_bf, weight_bf, bias_bf, stride=s, padding=p)
            loss_bf = output_bf.sum()
            loss_bf.backward()

            # 检查输出一致性
            output_diff = torch.max(torch.abs(output_bf - output_pt)).item()

            # 检查梯度一致性
            input_grad_diff = torch.max(torch.abs(input_bf.grad - input_pt.grad)).item()
            weight_grad_diff = torch.max(
                torch.abs(weight_bf.grad - weight_pt.grad)
            ).item()
            bias_grad_diff = torch.max(torch.abs(bias_bf.grad - bias_pt.grad)).item()

            print(f"  输出差异: {output_diff:.2e}")
            print(f"  输入梯度差异: {input_grad_diff:.2e}")
            print(f"  权重梯度差异: {weight_grad_diff:.2e}")
            print(f"  偏置梯度差异: {bias_grad_diff:.2e}")

            # 判断是否通过
            tolerance = 1e-4
            if (
                output_diff < tolerance
                and input_grad_diff < tolerance
                and weight_grad_diff < tolerance
                and bias_grad_diff < tolerance
            ):
                print("  ✅ 测试通过")
            else:
                print("  ❌ 测试失败")

        except Exception as e:
            print(f"  ❌ BitFlow 测试失败: {e}")


def test_conv_performance():
    """性能测试"""
    print("\n" + "=" * 60)
    print("卷积性能测试")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 性能测试配置
    configs = [
        # (batch, in_channels, height, width, out_channels, kernel_size)
        (1, 64, 56, 56, 64, 3),
        (8, 128, 28, 28, 128, 3),
        (16, 256, 14, 14, 256, 3),
        (32, 512, 7, 7, 512, 3),
    ]

    for batch, in_ch, h, w, out_ch, k in configs:
        print(f"\n配置: [{batch}, {in_ch}, {h}, {w}] -> [{out_ch}, {k}x{k}]")

        # 创建测试数据
        input_tensor = torch.randn(batch, in_ch, h, w, device=device)
        weight = torch.randn(out_ch, in_ch, k, k, device=device)
        bias = torch.randn(out_ch, device=device)

        # PyTorch 性能测试
        torch.cuda.empty_cache() if device == "cuda" else None

        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = F.conv2d(input_tensor, weight, bias, padding=1)

        if device == "cuda":
            torch.cuda.synchronize()

        # 测量时间
        num_runs = 100
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = F.conv2d(input_tensor, weight, bias, padding=1)

        if device == "cuda":
            torch.cuda.synchronize()

        pytorch_time = (time.time() - start_time) / num_runs * 1000
        print(f"  PyTorch: {pytorch_time:.2f} ms")

        # BitFlow 性能测试
        try:
            from bitflow.ops import conv2d

            # 预热
            for _ in range(10):
                with torch.no_grad():
                    _ = conv2d(input_tensor, weight, bias, padding=1)

            if device == "cuda":
                torch.cuda.synchronize()

            # 测量时间
            start_time = time.time()
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = conv2d(input_tensor, weight, bias, padding=1)

            if device == "cuda":
                torch.cuda.synchronize()

            bitflow_time = (time.time() - start_time) / num_runs * 1000
            speedup = pytorch_time / bitflow_time

            print(f"  BitFlow: {bitflow_time:.2f} ms")
            print(f"  加速比: {speedup:.2f}x")

        except Exception as e:
            print(f"  BitFlow 测试失败: {e}")


def test_conv_edge_cases():
    """边界情况测试"""
    print("\n" + "=" * 60)
    print("边界情况测试")
    print("=" * 60)

    # 测试不同的参数组合
    edge_cases = [
        # 描述, (input_shape, weight_shape, stride, padding, dilation, groups)
        ("基本卷积", ((1, 1, 5, 5), (1, 1, 3, 3), 1, 0, 1, 1)),
        ("大步长", ((1, 1, 8, 8), (1, 1, 3, 3), 2, 0, 1, 1)),
        ("大填充", ((1, 1, 5, 5), (1, 1, 3, 3), 1, 2, 1, 1)),
        ("膨胀卷积", ((1, 1, 7, 7), (1, 1, 3, 3), 1, 0, 2, 1)),
        ("分组卷积", ((1, 4, 5, 5), (4, 2, 3, 3), 1, 1, 1, 2)),
        ("1x1卷积", ((1, 64, 32, 32), (128, 64, 1, 1), 1, 0, 1, 1)),
    ]

    for desc, (
        input_shape,
        weight_shape,
        stride,
        padding,
        dilation,
        groups,
    ) in edge_cases:
        print(f"\n{desc}:")

        # 创建测试数据
        input_tensor = torch.randn(*input_shape, requires_grad=True)
        weight = torch.randn(*weight_shape, requires_grad=True)
        bias = torch.randn(weight_shape[0], requires_grad=True)

        try:
            # PyTorch 实现
            output_pt = F.conv2d(
                input_tensor, weight, bias, stride, padding, dilation, groups
            )

            # BitFlow 实现
            from bitflow.ops import conv2d

            input_bf = input_tensor.clone().detach().requires_grad_(True)
            weight_bf = weight.clone().detach().requires_grad_(True)
            bias_bf = bias.clone().detach().requires_grad_(True)

            output_bf = conv2d(
                input_bf, weight_bf, bias_bf, stride, padding, dilation, groups
            )

            # 检查一致性
            diff = torch.max(torch.abs(output_bf - output_pt)).item()
            print(f"  输出形状: {output_bf.shape}")
            print(f"  最大差异: {diff:.2e}")

            if diff < 1e-4:
                print("  ✅ 测试通过")
            else:
                print("  ❌ 测试失败")

        except Exception as e:
            print(f"  ❌ 测试失败: {e}")


if __name__ == "__main__":
    test_conv_correctness()
    test_conv_performance()
    test_conv_edge_cases()
