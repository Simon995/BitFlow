import torch
import torch.nn as nn
import time
import numpy as np
import pandas as pd # 用于Excel导出
import os

# 尝试导入您编译的自定义 CUDA 模块
try:
    import mec_conv_cuda
    print("Successfully imported mec_conv_cuda module.")
except ImportError as e:
    print(f"Error importing mec_conv_cuda: {e}")
    print("Please make sure you have compiled the CUDA extension using setup.py and it's in the PYTHONPATH.")
    exit()

def get_gpu_memory_usage_mb(device=None):
    """获取当前设备上已分配的峰值GPU显存（MB）"""
    if device is None:
        device = torch.cuda.current_device()
    return torch.cuda.max_memory_allocated(device) / (1024**2) # 转换为 MB

def reset_gpu_memory_stats(device=None):
    """重置当前设备上的GPU显存统计"""
    if device is None:
        device = torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(device)

def profile_conv_op(op_name, conv_op, input_tensor, num_runs=10, device="cuda"):
    """
    对卷积操作进行性能分析。

    参数:
        op_name (str): 操作的名称 (例如 "Custom Conv" 或 "PyTorch Conv").
        conv_op (callable): 要执行的卷积操作 (自定义算子包装器或 PyTorch nn.Module).
        input_tensor (torch.Tensor): 输入张量.
        num_runs (int): 运行次数.
        device (str): "cuda" 或 "cpu".

    返回:
        tuple: (avg_time_ms, min_time_ms, max_time_ms, avg_mem_mb, min_mem_mb, max_mem_mb, output_tensor)
    """
    times = []
    memories_mb = []
    output_tensor = None

    if device == "cuda":
        torch.cuda.synchronize()

    # 预热运行 (warm-up)
    _ = conv_op(input_tensor)

    if device == "cuda":
        torch.cuda.synchronize()

    for _ in range(num_runs):
        if device == "cuda":
            reset_gpu_memory_stats(device)
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        current_output = conv_op(input_tensor)

        if device == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            times.append(elapsed_time_ms)
            memories_mb.append(get_gpu_memory_usage_mb(device))
        else:
            pass # CPU timing not implemented

        if output_tensor is None:
            output_tensor = current_output.detach().clone() # 保存副本以防被修改

    if not times: # 如果是CPU或其他情况没有记录时间
        return 0, 0, 0, 0, 0, 0, output_tensor

    avg_time_ms = np.mean(times)
    min_time_ms = np.min(times)
    max_time_ms = np.max(times)

    avg_mem_mb = np.mean(memories_mb) if memories_mb else 0
    min_mem_mb = np.min(memories_mb) if memories_mb else 0
    max_mem_mb = np.max(memories_mb) if memories_mb else 0

    return avg_time_ms, min_time_ms, max_time_ms, avg_mem_mb, min_mem_mb, max_mem_mb, output_tensor

def compare_outputs(output1, output2, op1_name="Output 1", op2_name="Output 2"):
    """比较两个输出张量，返回差异统计字典"""
    comparison_results = {
        "output_allclose": False,
        "max_abs_diff": float('nan'),
        "mean_abs_diff": float('nan'),
        "mse": float('nan')
    }
    print(f"\n--- 输出比较 ({op1_name} vs {op2_name}) ---")
    if output1 is None or output2 is None:
        print("一个或两个输出为空，无法比较。")
        return comparison_results

    print(f"  {op1_name} shape: {output1.shape}, dtype: {output1.dtype}, device: {output1.device}")
    print(f"  {op2_name} shape: {output2.shape}, dtype: {output2.dtype}, device: {output2.device}")

    if output1.shape != output2.shape:
        print("  输出形状不匹配，无法进行元素级比较。")
        return comparison_results

    all_close = torch.allclose(output1, output2, atol=1e-5, rtol=1e-4) # 可以调整容忍度
    comparison_results["output_allclose"] = all_close
    print(f"  torch.allclose (atol=1e-5, rtol=1e-4): {all_close}")

    if not all_close:
        abs_diff = torch.abs(output1 - output2)
        comparison_results["max_abs_diff"] = torch.max(abs_diff).item()
        comparison_results["mean_abs_diff"] = torch.mean(abs_diff).item()
        comparison_results["mse"] = torch.mean(abs_diff**2).item()
        print(f"  最大绝对差异: {comparison_results['max_abs_diff']:.6e}")
        print(f"  平均绝对差异 (L1): {comparison_results['mean_abs_diff']:.6e}")
        print(f"  均方误差 (MSE): {comparison_results['mse']:.6e}")
    return comparison_results

class CustomConvOpWrapper:
    def __init__(self, weight_4d_tensor, bias_tensor, padding_h, padding_w, stride_h, stride_w):
        self.weight_4d = weight_4d_tensor
        self.bias = bias_tensor
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.stride_h = stride_h
        self.stride_w = stride_w

    def __call__(self, inp_tensor):
        return mec_conv_cuda.forward(inp_tensor,
                                     self.weight_4d,
                                     self.bias,
                                     self.padding_h, self.padding_w,
                                     self.stride_h, self.stride_w)

def run_comparison(test_config, device="cuda"):
    """
    运行单个测试配置的比较，并返回结果字典。
    """
    batch_size = test_config["batch_size"]
    in_channels = test_config["in_channels"]
    in_height = test_config["in_height"]
    in_width = test_config["in_width"]
    out_channels = test_config["out_channels"]
    kernel_h, kernel_w = test_config["kernel_size"]
    padding_h, padding_w = test_config["padding"]
    stride_h, stride_w = test_config["stride"]
    use_bias = test_config.get("use_bias", False)

    config_str = (f"B{batch_size}_InC{in_channels}_InH{in_height}W{in_width}_"
                  f"OutC{out_channels}_K{kernel_h}x{kernel_w}_P{padding_h}x{padding_w}_S{stride_h}x{stride_w}_Bias{use_bias}")
    print("="*80)
    print(f"测试配置: {config_str}")
    print("="*80)

    results_row = {"config": config_str} # 初始化结果行

    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA 不可用，跳过此测试。")
        results_row.update({"error": "CUDA not available"})
        return results_row

    input_tensor = torch.randn(batch_size, in_channels, in_height, in_width,
                               dtype=torch.float32, device=device)
    weight_pytorch_format = torch.randn(out_channels, in_channels, kernel_h, kernel_w,
                                        dtype=torch.float32, device=device)
    bias_tensor = None
    if use_bias:
        bias_tensor = torch.randn(out_channels, dtype=torch.float32, device=device)
    else:
        bias_tensor = torch.empty(0, dtype=torch.float32, device=device)


    # --- 自定义算子 ---
    print("\n--- 分析自定义算子 (mec_conv_cuda.forward) ---")
    custom_op_runner = CustomConvOpWrapper(weight_pytorch_format,
                                           bias_tensor,
                                           padding_h, padding_w,
                                           stride_h, stride_w)

    avg_time_custom, min_time_custom, max_time_custom, \
        avg_mem_custom_mb, min_mem_custom_mb, max_mem_custom_mb, \
        output_custom = profile_conv_op("Custom Conv", custom_op_runner, input_tensor, device=device)

    results_row.update({
        "custom_avg_time_ms": avg_time_custom,
        "custom_min_time_ms": min_time_custom,
        "custom_max_time_ms": max_time_custom,
        "custom_avg_mem_mb": avg_mem_custom_mb,
        "custom_min_mem_mb": min_mem_custom_mb,
        "custom_max_mem_mb": max_mem_custom_mb,
    })
    print(f"  平均执行时间: {avg_time_custom:.3f} ms")
    print(f"  最小执行时间: {min_time_custom:.3f} ms")
    print(f"  最大执行时间: {max_time_custom:.3f} ms")
    if device == "cuda":
        print(f"  平均峰值显存: {avg_mem_custom_mb:.2f} MB")
        print(f"  最小峰值显存: {min_mem_custom_mb:.2f} MB")
        print(f"  最大峰值显存: {max_mem_custom_mb:.2f} MB")

    # time.sleep(2)
    # --- PyTorch 内置算子 ---
    print("\n--- 分析 PyTorch 内置算子 (torch.nn.Conv2d) ---")
    pytorch_conv = nn.Conv2d(in_channels, out_channels, (kernel_h, kernel_w),
                             stride=(stride_h, stride_w),
                             padding=(padding_h, padding_w),
                             bias=use_bias,
                             dtype=torch.float32,
                             device=device)

    with torch.no_grad():
        pytorch_conv.weight.copy_(weight_pytorch_format)
        if use_bias and bias_tensor.numel() > 0 :
            pytorch_conv.bias.copy_(bias_tensor)
        elif use_bias and pytorch_conv.bias is not None:
            pytorch_conv.bias.zero_()


    avg_time_torch, min_time_torch, max_time_torch, \
        avg_mem_torch_mb, min_mem_torch_mb, max_mem_torch_mb, \
        output_torch = profile_conv_op("PyTorch Conv", pytorch_conv, input_tensor, device=device)

    results_row.update({
        "torch_avg_time_ms": avg_time_torch,
        "torch_min_time_ms": min_time_torch,
        "torch_max_time_ms": max_time_torch,
        "torch_avg_mem_mb": avg_mem_torch_mb,
        "torch_min_mem_mb": min_mem_torch_mb,
        "torch_max_mem_mb": max_mem_torch_mb,
    })
    print(f"  平均执行时间: {avg_time_torch:.3f} ms")
    print(f"  最小执行时间: {min_time_torch:.3f} ms")
    print(f"  最大执行时间: {max_time_torch:.3f} ms")
    if device == "cuda":
        print(f"  平均峰值显存: {avg_mem_torch_mb:.2f} MB")
        print(f"  最小峰值显存: {min_mem_torch_mb:.2f} MB")
        print(f"  最大峰值显存: {max_mem_torch_mb:.2f} MB")

    # --- 比较输出 ---
    output_diff_stats = compare_outputs(output_custom, output_torch, "Custom Output", "PyTorch Output")
    results_row.update(output_diff_stats)
    print("\n")
    return results_row


if __name__ == "__main__":
    all_results_data = [] # 存储所有测试配置的结果

    # 定义基础参数
    base_batch_size = 1
    base_in_channels = 3 # 可以根据需要调整，例如 3, 16, 32, 64
    base_out_channels = 64 # 可以根据需要调整
    base_padding_mode = "same" # "same" 或 "valid" (0)
    base_stride = 1
    use_bias_options = [False, True]

    # 定义要测试的卷积核大小和输入图像大小
    kernel_sizes_to_test = [1, 3, 5, 7, 9]
    input_hw_sizes_to_test = [128, 256, 512, 1024]

    test_configurations = []

    for use_bias in use_bias_options:
        for k_size in kernel_sizes_to_test:
            for hw_size in input_hw_sizes_to_test:
                # 为了保持输出尺寸大致相同（对于 "same" padding），或者进行有效卷积
                if base_padding_mode == "same":
                    # "same" padding for stride 1: (kernel_size - 1) / 2
                    # PyTorch Conv2d padding takes a single int for symmetric padding
                    # or a tuple for (padH, padW)
                    padding_val = (k_size - 1) // 2
                    padding = (padding_val, padding_val)
                else: # "valid" padding
                    padding = (0, 0)

                # 确保输入尺寸至少和卷积核一样大
                if hw_size < k_size:
                    print(f"Skipping config: input size {hw_size} < kernel size {k_size}")
                    continue

                # 限制一下输出通道数，避免显存爆炸，特别是对于大输入和卷积核
                current_out_channels = base_out_channels
                if hw_size > 512 or k_size > 7:
                    current_out_channels = max(16, base_out_channels // 2) # 减少大输入的输出通道
                if hw_size > 256 and k_size > 5:
                    current_out_channels = max(32, base_out_channels //2)


                test_configurations.append({
                    "batch_size": base_batch_size,
                    "in_channels": base_in_channels,
                    "in_height": hw_size,
                    "in_width": hw_size,
                    "out_channels": current_out_channels,
                    "kernel_size": (k_size, k_size),
                    "padding": padding,
                    "stride": (base_stride, base_stride),
                    "use_bias": use_bias
                })

    # 添加一些之前定义的多样化配置（可选）
    # test_configurations.extend([
    #     {
    #         "batch_size": 8, "in_channels": 3, "in_height": 224, "in_width": 224,
    #         "out_channels": 64, "kernel_size": (3, 3), "padding": (1, 1), "stride": (1, 1), "use_bias": True
    #     },
    #     {
    #         "batch_size": 1, "in_channels": 128, "in_height": 28, "in_width": 28,
    #         "out_channels": 256, "kernel_size": (3, 3), "padding": (1, 1), "stride": (2, 2), "use_bias": True
    #     },
    # ])


    target_device = "cuda"
    if not torch.cuda.is_available() and target_device == "cuda":
        print("CUDA is not available on this system. Please check your CUDA installation and PyTorch setup.")
        exit()
    if target_device == "cuda":
        print(f"Using device: {torch.cuda.get_device_name(0)}")
        print(f"Total CUDA devices: {torch.cuda.device_count()}")


    for config in test_configurations:
        try:
            result_row = run_comparison(config, device=target_device)
            all_results_data.append(result_row)
        except Exception as e:
            config_str = (f"B{config['batch_size']}_InC{config['in_channels']}_InH{config['in_height']}W{config['in_width']}_"
                          f"OutC{config['out_channels']}_K{config['kernel_size'][0]}x{config['kernel_size'][1]}_P{config['padding'][0]}x{config['padding'][1]}_S{config['stride'][0]}x{config['stride'][1]}_Bias{config['use_bias']}")
            print(f"测试配置 {config_str} 发生严重错误: {e}")
            import traceback
            traceback.print_exc()
            all_results_data.append({"config": config_str, "error": str(e)})
            print("-" * 80)

    # 将结果保存到 Excel
    if all_results_data:
        results_df = pd.DataFrame(all_results_data)
        excel_path = "conv_comparison_results.xlsx"
        try:
            results_df.to_excel(excel_path, index=False, engine='openpyxl')
            print(f"\n比较结果已保存到: {os.path.abspath(excel_path)}")
        except Exception as e:
            print(f"\n保存 Excel 文件失败: {e}")
            print("请确保已安装 'openpyxl' 库: pip install openpyxl")
            print("尝试保存为 CSV 文件...")
            csv_path = "conv_comparison_results.csv"
            try:
                results_df.to_csv(csv_path, index=False)
                print(f"比较结果已保存为 CSV 文件: {os.path.abspath(csv_path)}")
            except Exception as e_csv:
                print(f"保存 CSV 文件也失败: {e_csv}")
    else:
        print("\n没有收集到任何结果数据。")
