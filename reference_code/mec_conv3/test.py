import torch
import torch.nn as nn
import time
import numpy as np
import pandas as pd # 用于Excel导出
import os

# 尝试导入您编译的自定义 CUDA 模块
try:
    import mec_conv3d_cuda # <--- 确保这是您编译的3D卷积模块名
    print("Successfully imported mec_conv3d_cuda module.")
except ImportError as e:
    print(f"Error importing mec_conv3d_cuda: {e}")
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

def profile_conv_op(op_name, conv_op, input_tensor, num_runs=20, device="cuda"):
    """
    对卷积操作进行性能分析。

    参数:
        op_name (str): 操作的名称 (例如 "Custom Conv3D" 或 "PyTorch Conv3D").
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

    # 预热运行 (warm-up) - 运行几次以稳定 GPU 时钟等
    for _ in range(max(5, num_runs // 4)): # 至少预热5次
        _ = conv_op(input_tensor)

    if device == "cuda":
        torch.cuda.synchronize()

    for i in range(num_runs):
        if device == "cuda":
            reset_gpu_memory_stats(device)
            torch.cuda.synchronize() # 确保之前的操作完成
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        current_output = conv_op(input_tensor)

        if device == "cuda":
            end_event.record()
            torch.cuda.synchronize() # 确保卷积操作完成
            elapsed_time_ms = start_event.elapsed_time(end_event)
            times.append(elapsed_time_ms)
            memories_mb.append(get_gpu_memory_usage_mb(device))
        else:
            start_time = time.time()
            if output_tensor is None:
                current_output = conv_op(input_tensor)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)

        if output_tensor is None:
            output_tensor = current_output.detach().clone()


    if not times:
        return 0, 0, 0, 0, 0, 0, output_tensor

    times_to_consider = times
    avg_time_ms = np.mean(times_to_consider) if times_to_consider else 0
    min_time_ms = np.min(times_to_consider) if times_to_consider else 0
    max_time_ms = np.max(times_to_consider) if times_to_consider else 0

    avg_mem_mb = np.mean(memories_mb) if memories_mb else 0
    min_mem_mb = np.min(memories_mb) if memories_mb else 0
    max_mem_mb = np.max(memories_mb) if memories_mb else 0

    return avg_time_ms, min_time_ms, max_time_ms, avg_mem_mb, min_mem_mb, max_mem_mb, output_tensor

def compare_outputs(output1, output2, op1_name="Output 1", op2_name="Output 2", atol=1e-5, rtol=1e-4):
    """比较两个输出张量，返回差异统计字典"""
    comparison_results = {
        "output_allclose": False,
        "max_abs_diff": float('nan'),
        "mean_abs_diff": float('nan'),
        "mse": float('nan')
    }
    print(f"\n--- 输出比较 ({op1_name} vs {op2_name}) ---")
    if output1 is None or output2 is None:
        print("  一个或两个输出为空，无法比较。")
        return comparison_results

    print(f"  {op1_name} shape: {output1.shape}, dtype: {output1.dtype}, device: {output1.device}")
    print(f"  {op2_name} shape: {output2.shape}, dtype: {output2.dtype}, device: {output2.device}")

    if output1.shape != output2.shape:
        print(f"  输出形状不匹配: {output1.shape} vs {output2.shape}，无法进行元素级比较。")
        return comparison_results

    all_close = torch.allclose(output1, output2, atol=atol, rtol=rtol)
    comparison_results["output_allclose"] = all_close
    print(f"  torch.allclose (atol={atol}, rtol={rtol}): {all_close}")

    if not all_close:
        abs_diff = torch.abs(output1 - output2)
        comparison_results["max_abs_diff"] = torch.max(abs_diff).item()
        comparison_results["mean_abs_diff"] = torch.mean(abs_diff).item()
        comparison_results["mse"] = torch.mean(torch.square(abs_diff)).item()
        print(f"  最大绝对差异: {comparison_results['max_abs_diff']:.6e}")
        print(f"  平均绝对差异 (L1): {comparison_results['mean_abs_diff']:.6e}")
        print(f"  均方误差 (MSE): {comparison_results['mse']:.6e}")
    return comparison_results

class CustomConv3DOpWrapper:
    def __init__(self, weight_5d_tensor, bias_tensor,
                 padding_d, padding_h, padding_w,
                 stride_d, stride_h, stride_w):
        self.weight_5d = weight_5d_tensor
        self.bias = bias_tensor
        self.padding_d = padding_d
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.stride_d = stride_d
        self.stride_h = stride_h
        self.stride_w = stride_w

    def __call__(self, inp_tensor):
        return mec_conv3d_cuda.forward(inp_tensor,
                                       self.weight_5d,
                                       self.bias,
                                       self.padding_d, self.padding_h, self.padding_w,
                                       self.stride_d, self.stride_h, self.stride_w)

def run_comparison(test_config, device="cuda"):
    """
    运行单个测试配置的比较，并返回结果字典 (适配3D)。
    """
    batch_size = test_config["batch_size"]
    in_channels = test_config["in_channels"]
    # D, H, W 都是一样的
    in_spatial_size = test_config["in_spatial_size"]
    in_depth, in_height, in_width = in_spatial_size, in_spatial_size, in_spatial_size

    out_channels = test_config["out_channels"]
    # Kernel D, H, W 都是一样的
    kernel_spatial_size = test_config["kernel_spatial_size"]
    kernel_d, kernel_h, kernel_w = kernel_spatial_size, kernel_spatial_size, kernel_spatial_size

    # Padding D, H, W 都是一样的
    padding_spatial_val = test_config["padding_spatial_val"]
    padding_d, padding_h, padding_w = padding_spatial_val, padding_spatial_val, padding_spatial_val

    # Stride D, H, W 都是一样的
    stride_spatial_val = test_config["stride_spatial_val"]
    stride_d, stride_h, stride_w = stride_spatial_val, stride_spatial_val, stride_spatial_val

    use_bias = test_config.get("use_bias", False)

    config_str = (f"B{batch_size}_InC{in_channels}_InS{in_spatial_size}_"
                  f"OutC{out_channels}_KS{kernel_spatial_size}_PS{padding_spatial_val}_SS{stride_spatial_val}_Bias{use_bias}")
    print("="*80)
    print(f"测试配置: {config_str}")
    print("="*80)

    results_row = {"config": config_str}

    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA 不可用，跳过此测试。")
        results_row.update({"error": "CUDA not available"})
        return results_row

    input_tensor = torch.randn(batch_size, in_channels, in_depth, in_height, in_width,
                               dtype=torch.float32, device=device)
    weight_pytorch_format = torch.randn(out_channels, in_channels, kernel_d, kernel_h, kernel_w,
                                        dtype=torch.float32, device=device)
    bias_tensor = None
    if use_bias:
        bias_tensor = torch.randn(out_channels, dtype=torch.float32, device=device)
    else:
        bias_tensor = torch.empty(0, dtype=torch.float32, device=device)

    # --- 自定义算子 ---
    print("\n--- 分析自定义算子 (mec_conv3d_cuda.forward) ---")
    custom_op_runner = CustomConv3DOpWrapper(weight_pytorch_format,
                                             bias_tensor,
                                             padding_d, padding_h, padding_w,
                                             stride_d, stride_h, stride_w)

    avg_time_custom, min_time_custom, max_time_custom, \
        avg_mem_custom_mb, min_mem_custom_mb, max_mem_custom_mb, \
        output_custom = profile_conv_op("Custom Conv3D", custom_op_runner, input_tensor, device=device)

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

    # --- PyTorch 内置算子 ---
    print("\n--- 分析 PyTorch 内置算子 (torch.nn.Conv3d) ---")
    pytorch_conv = nn.Conv3d(in_channels, out_channels, (kernel_d, kernel_h, kernel_w),
                             stride=(stride_d, stride_h, stride_w),
                             padding=(padding_d, padding_h, padding_w),
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
        output_torch = profile_conv_op("PyTorch Conv3D", pytorch_conv, input_tensor, device=device)

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

    output_diff_stats = compare_outputs(output_custom, output_torch, "Custom Output", "PyTorch Output", atol=1e-4, rtol=1e-3)
    results_row.update(output_diff_stats)
    print("\n")
    return results_row


if __name__ == "__main__":
    all_results_data = []

    base_batch_size = 1
    base_in_channels = 3
    base_out_channels = 32
    base_padding_mode = "same"
    base_stride_val = 1 # Stride D, H, W 将相同
    use_bias_options = [False, True]

    # 定义卷积核和输入的统一空间尺寸
    kernel_uniform_sizes_to_test = [1, 3, 5]
    # 之前 input_depth_sizes_to_test = [8, 16, 32]
    # 之前 input_hw_sizes_to_test = [32, 64, 128]
    # 为了统一，我们可以选择一个范围，或者合并这些值。
    # 这里我们选择之前 hw 使用的较大尺寸，因为通常 D,H,W 一致时会取较大的值。
    # 如果需要测试更小的统一尺寸，可以加入例如 8, 16
    input_uniform_sizes_to_test = [32, 64, 128] # D=H=W
    # 若要包含更小的尺寸: input_uniform_sizes_to_test = [8, 16, 32, 64, 128]


    test_configurations = []

    for use_bias in use_bias_options:
        for k_spatial_size in kernel_uniform_sizes_to_test:
            for in_spatial_size in input_uniform_sizes_to_test:
                padding_spatial_val = 0
                if base_padding_mode == "same":
                    # "same" padding for stride 1: (kernel_size - 1) // 2
                    padding_spatial_val = (k_spatial_size - 1) // 2
                # else "valid" padding, padding_spatial_val remains 0

                if in_spatial_size < k_spatial_size:
                    print(f"Skipping config: input spatial size {in_spatial_size} < kernel spatial size {k_spatial_size}")
                    continue

                current_out_channels = base_out_channels
                # 简化后的输出通道调整逻辑
                if in_spatial_size > 64 or k_spatial_size >= 5:
                    current_out_channels = max(16, base_out_channels // 2)
                elif in_spatial_size > 32 or k_spatial_size >= 3:
                    current_out_channels = max(16, int(base_out_channels * 0.75))


                test_configurations.append({
                    "batch_size": base_batch_size,
                    "in_channels": base_in_channels,
                    "in_spatial_size": in_spatial_size, # D_in = H_in = W_in
                    "out_channels": current_out_channels,
                    "kernel_spatial_size": k_spatial_size, # D_k = H_k = W_k
                    "padding_spatial_val": padding_spatial_val, # P_d = P_h = P_w
                    "stride_spatial_val": base_stride_val, # S_d = S_h = S_w
                    "use_bias": use_bias
                })

    target_device = "cuda"
    if not torch.cuda.is_available() and target_device == "cuda":
        print("CUDA is not available on this system. Please check your CUDA installation and PyTorch setup.")
        exit()

    if target_device == "cuda":
        print(f"Using device: {torch.cuda.get_device_name(0)}")
        print(f"Total CUDA devices: {torch.cuda.device_count()}")
        current_cuda_version = torch.version.cuda
        print(f"PyTorch compiled with CUDA version: {current_cuda_version}")

    for i, config in enumerate(test_configurations):
        print(f"\n--- Running Test {i+1} of {len(test_configurations)} ---")
        try:
            result_row = run_comparison(config, device=target_device)
            all_results_data.append(result_row)
        except Exception as e:
            cfg = config
            config_str_err = (f"B{cfg['batch_size']}_InC{cfg['in_channels']}_InS{cfg['in_spatial_size']}_"
                              f"OutC{cfg['out_channels']}_KS{cfg['kernel_spatial_size']}_PS{cfg['padding_spatial_val']}_SS{cfg['stride_spatial_val']}_Bias{cfg['use_bias']}")
            print(f"测试配置 {config_str_err} 发生严重错误: {e}")
            import traceback
            traceback.print_exc()
            all_results_data.append({"config": config_str_err, "error": str(e)})
            if device == "cuda":
                torch.cuda.empty_cache()
            print("-" * 80)

    if all_results_data:
        results_df = pd.DataFrame(all_results_data)
        excel_path = "conv3d_comparison_results_uniform_dims.xlsx"
        try:
            results_df.to_excel(excel_path, index=False, engine='openpyxl')
            print(f"\n比较结果已保存到: {os.path.abspath(excel_path)}")
        except Exception as e:
            print(f"\n保存 Excel 文件失败: {e}")
            print("请确保已安装 'openpyxl' 库: pip install openpyxl")
            print("尝试保存为 CSV 文件...")
            csv_path = "conv3d_comparison_results_uniform_dims.csv"
            try:
                results_df.to_csv(csv_path, index=False)
                print(f"比较结果已保存为 CSV 文件: {os.path.abspath(csv_path)}")
            except Exception as e_csv:
                print(f"保存 CSV 文件也失败: {e_csv}")
    else:
        print("\n没有收集到任何结果数据。")

