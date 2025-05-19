import torch
import torch.nn as nn
import time
import random
import numpy as np

# 1. 固定随机种子以保证可复现性
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    # 为了在不同运行中获得更一致的结果（但可能会牺牲一些性能），可以设置以下选项：
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False # 通常在输入尺寸固定时设为True以优化
else:
    print("警告: 未检测到CUDA设备，将在CPU上运行。")

# 确保使用GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 卷积参数 (与您的C++代码保持一致)
batch_size = 1
in_channels = 3
out_channels = 64
height = 224
width = 224
kernel_size = 7
stride = 1
padding = 0 # 确保与您的C++代码中的padding设置一致
dilation = 1 # 假设dilation为1
groups = 1   # 假设groups为1

# 5. (可选) 优化cuDNN算法选择
# 对于固定的输入尺寸，设置benchmark = True通常可以提升性能
# 但这可能会增加初次调用的开销，并且如果输入尺寸变化，可能需要重新选择算法
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = False
    print(f"torch.backends.cudnn.benchmark 设置为: {torch.backends.cudnn.benchmark}")

# 创建随机输入张量和卷积层
input_tensor = torch.randn(batch_size, in_channels, height, width, device=device)
conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                       dilation=dilation, groups=groups, bias=False).to(device)

# --- 2. GPU 预热 (Warm-up) ---
num_warmup_runs = 10
print(f"开始预热 ({num_warmup_runs} 次)...")
for _ in range(num_warmup_runs):
    _ = conv_layer(input_tensor)
# 确保预热操作完成
if device.type == 'cuda':
    torch.cuda.synchronize()
print("预热完成。")

# --- 4. 显存占用测量 ---
bytes_to_mb = 1 / (1024 * 1024) # 转换因子：bytes to MB

if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats(device=device) # 重置峰值显存统计
    start_memory_allocated_bytes = torch.cuda.memory_allocated(device=device) # 记录初始PyTorch张量分配的显存

# --- 3. 多次运行并取平均值 (结合2. GPU同步) ---
num_runs = 100 # 可以根据需要调整运行次数
timings = []
print(f"开始正式测试 ({num_runs} 次)...")

for i in range(num_runs):
    if device.type == 'cuda':
        torch.cuda.synchronize() # 确保所有先前GPU操作完成
    iter_start_time = time.time()

    # 执行卷积操作
    with torch.no_grad(): # 在推理时关闭梯度计算以节省显存和计算
        output_tensor = conv_layer(input_tensor)

    if device.type == 'cuda':
        torch.cuda.synchronize() # 确保卷积操作完成
    iter_end_time = time.time()
    timings.append(iter_end_time - iter_start_time)
    if (i + 1) % (num_runs // 10) == 0 : # 每10%打印一次进度
        print(f"  完成 {(i + 1) / num_runs * 100:.0f}%")


avg_time_ms = (sum(timings) / num_runs) * 1000 # 转换为毫秒
min_time_ms = min(timings) * 1000
max_time_ms = max(timings) * 1000

# 显存占用 - 结束后的显存使用
if device.type == 'cuda':
    end_memory_allocated_bytes = torch.cuda.memory_allocated(device=device)
    peak_memory_allocated_bytes = torch.cuda.max_memory_allocated(device=device)


# 输出结果
print("\n--- 结果 ---")
print(f"输入尺寸: {input_tensor.size()}")
print(f"输出尺寸: {output_tensor.size()}") # 确保输出尺寸与预期一致

if device.type == 'cuda':
    start_memory_allocated_mb = start_memory_allocated_bytes * bytes_to_mb
    end_memory_allocated_mb = end_memory_allocated_bytes * bytes_to_mb
    peak_memory_allocated_mb = peak_memory_allocated_bytes * bytes_to_mb

    print(f"初始PyTorch张量显存: {start_memory_allocated_mb:.2f} MB")
    print(f"结束PyTorch张量显存: {end_memory_allocated_mb:.2f} MB")
    print(f"峰值PyTorch张量显存: {peak_memory_allocated_mb:.2f} MB")
    # 这个差值可以大致反映输出张量和可能的中间缓存所占用的显存
    print(f"近似PyTorch张量显存增量: {(peak_memory_allocated_bytes - start_memory_allocated_bytes) * bytes_to_mb:.2f} MB")
    # 更详细的显存报告
    # print("\n详细显存报告:")
    # print(torch.cuda.memory_summary(device=device, abbreviated=True))


print(f"\n运行时间 ({num_runs} 次):")
print(f"  平均: {avg_time_ms:.3f} ms")
print(f"  最短: {min_time_ms:.3f} ms")
print(f"  最长: {max_time_ms:.3f} ms")

print("\n测试完成。")
