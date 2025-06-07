"""BitFlow基本使用示例"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from bitflow.ops import conv2d


def example_basic_conv():
    """基本卷积示例"""
    print("=== 基本卷积示例 ===")

    # 创建输入数据
    batch_size = 1
    in_channels = 3
    height, width = 32, 32

    input_tensor = torch.randn(batch_size, in_channels, height, width)

    # 创建卷积参数
    out_channels = 16
    kernel_size = 3
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
    bias = torch.randn(out_channels)

    # 使用BitFlow卷积
    output = conv2d(input_tensor, weight, bias, stride=1, padding=1)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"Output shape: {output.shape}")

    # 验证与PyTorch的一致性
    pytorch_output = F.conv2d(input_tensor, weight, bias, stride=1, padding=1)
    max_diff = torch.max(torch.abs(output - pytorch_output))

    print(f"Max difference with PyTorch: {max_diff.item():.2e}")
    assert max_diff < 1e-4, "Results differ too much from PyTorch"
    print("✓ Results match PyTorch implementation")


def example_autograd():
    """自动求导示例"""
    print("\n=== 自动求导示例 ===")

    # 创建需要梯度的张量
    input_tensor = torch.randn(1, 3, 32, 32, requires_grad=True)
    weight = torch.randn(16, 3, 3, 3, requires_grad=True)
    bias = torch.randn(16, requires_grad=True)

    # 前向传播
    output = conv2d(input_tensor, weight, bias, stride=1, padding=1)
    loss = output.sum()

    # 反向传播
    loss.backward()

    print(f"Input gradient shape: {input_tensor.grad.shape}")
    print(f"Weight gradient shape: {weight.grad.shape}")
    print(f"Bias gradient shape: {bias.grad.shape}")
    print("✓ Autograd working correctly")


def example_cuda():
    """CUDA示例"""
    if not torch.cuda.is_available():
        print("\n=== CUDA not available, skipping ===")
        return

    print("\n=== CUDA示例 ===")

    device = torch.device('cuda')

    # 创建CUDA张量
    input_tensor = torch.randn(2, 64, 64, 64, device=device)
    weight = torch.randn(128, 64, 3, 3, device=device)
    bias = torch.randn(128, device=device)

    # 使用BitFlow CUDA卷积
    output = conv2d(input_tensor, weight, bias)

    print(f"Input device: {input_tensor.device}")
    print(f"Output device: {output.device}")
    print(f"Output shape: {output.shape}")
    print("✓ CUDA computation successful")


def example_performance():
    """性能测试示例"""
    print("\n=== 性能测试示例 ===")

    from bitflow.ops.conv import benchmark_conv2d
    benchmark_conv2d()


class BitFlowConvNet(nn.Module):
    """使用BitFlow卷积的简单网络"""

    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        # 定义卷积层参数
        self.conv1_weight = nn.Parameter(torch.randn(32, in_channels, 3, 3))
        self.conv1_bias = nn.Parameter(torch.randn(32))

        self.conv2_weight = nn.Parameter(torch.randn(64, 32, 3, 3))
        self.conv2_bias = nn.Parameter(torch.randn(64))

        # 分类层
        self.classifier = nn.Linear(64 * 6 * 6, num_classes)

    def forward(self, x):
        # 第一个卷积层 + ReLU + MaxPool
        x = conv2d(x, self.conv1_weight, self.conv1_bias, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # 第二个卷积层 + ReLU + MaxPool
        x = conv2d(x, self.conv2_weight, self.conv2_bias, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # 展平并分类
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def example_training():
    """训练示例"""
    print("\n=== 训练示例 ===")

    # 创建网络和优化器
    model = BitFlowConvNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 创建虚拟数据
    batch_size = 4
    input_data = torch.randn(batch_size, 3, 28, 28)
    target = torch.randint(0, 10, (batch_size,))

    # 训练一个步骤
    optimizer.zero_grad()

    output = model(input_data)
    loss = criterion(output, target)

    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.4f}")

    loss.backward()
    optimizer.step()

    print("✓ Training step completed successfully")


if __name__ == "__main__":
    print("BitFlow 使用示例\n")

    # 运行所有示例
    example_basic_conv()
    example_autograd()
    example_cuda()
    example_performance()
    example_training()

    print("\n🎉 所有示例运行成功！")
