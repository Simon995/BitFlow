# MEC:（ICML 2017)

MEC: Memory-efficient Convolution for Deep Neural Network C++个人实现，[论文地址](https://arxiv.org/abs/1706.06873v1)

# 测试环境和依赖

- Ubuntu18.04
- BLAS的免费版本ATLAS（安装命令：`sudo apt-get install libatlas-base-dev`）
- OpenMP

# 编译&运行
- g++ -o demo1 Im2ColOrigin.cpp /usr/lib/x86_64-linux-gnu/libcblas.so.3 -fopenmp 
- ./demo1
- g++ -o demo2 Im2ColMEC.cpp /usr/lib/x86_64-linux-gnu/libcblas.so.3 -fopenmp 
- ./demo2

# 速度测试、内存占用测试

|方法|速度|内存占用|
|--|--|--|
|Im2Col+Origin|35ms|26.5M|
|MEC(4线程)|7ms|15.7M|
|MEC(1线程)|28ms|11.7M|

* conv_cuda_test.cu是测试使用gemm矩阵乘法修改为nvidia的cutlass
nvcc -o conv_cuda conv_cuda.cu -lcublas
nvcc -o conv_cuda_test conv_cuda_test.cu -lcublas
nvcc -std=c++17 -o conv_cuda_test conv_cuda_test.cu -I cutlass/include -gencode arch=compute_89,code=sm_89

一、性能差异分析 (自定义 MEC: 0.690 ms vs. PyTorch: 0.145 ms)
您自定义的 MEC 卷积实现的总耗时（平均 0.690 ms）显著高于 PyTorch 的 nn.Conv2d（平均 0.145 ms）。主要原因如下：

cuDNN 的强大优化能力:

PyTorch 的卷积操作底层依赖于 NVIDIA 的 cuDNN 库。cuDNN 是一个为深度学习原语（如卷积、池化等）高度优化的库。

特别是当 torch.backends.cudnn.benchmark = True 时，cuDNN 会在首次遇到特定尺寸的卷积时，测试多种内部卷积算法（例如 Winograd、FFT、基于不同 GEMM 的实现、直接卷积等），并选择针对当前输入尺寸、卷积核参数和硬件最快的那一种。

您的自定义实现采用了“MEC变换 + 单一cublasSgemmStridedBatched策略”。虽然 cuBLAS 本身是优化的 GEMM 库，但对于特定的卷积配置，它可能不是绝对最优的卷积实现方法，而 cuDNN 有更多选择。

算法选择:

MEC 算法的核心在于通过更紧凑的中间表示来减少内存占用和潜在的内存带宽瓶颈，尤其是在传统 im2col 产生的矩阵非常大时。

对于您测试的这个特定参数组合（输入 224x224x3，卷积核 7x7，输出 64 通道），在现代 GPU 上，计算资源可能比内存带宽更为充裕，cuDNN 选择的某种计算密集型但高度优化的算法（如特定形式的 Winograd 或高度优化的直接卷积）可能会胜过基于 GEMM 的间接方法。

Kernel Launch 和同步开销:

您的代码包含两个主要的GPU操作：自定义的 im2col_cuda_mec 核函数（平均 0.065 ms）和 cublasSgemmStridedBatched 调用（平均 0.618 ms）。虽然您已经做了预热，但多次独立的核函数/库函数调用及其间的同步（即使是隐式的）也可能累积开销。

cuDNN 可能会将一些操作进行更深层次的融合或采用更优化的执行调度。

数据布局和编译优化:

PyTorch 内部通常使用 NCHW 数据格式。您的自定义核函数目前处理的是 HWC 格式的输入，并将 d_mec_L 构造成适配这种格式的结构。虽然 cuBLAS 可以处理不同的数据布局，但与硬件内存访问模式和库内部优化最契合的布局通常能带来最佳性能。cuDNN 针对深度学习常用的数据布局进行了深度调优。

PyTorch 后端和 cuDNN 的编译可能采用了比标准 nvcc 编译自定义代码时更激进的优化选项和针对特定GPU架构的微调。

二、显存占用分析
MEC 算法的内存效率已体现：

您的日志清晰显示：MEC Lowered 矩阵 (d_mec_L) 大小: 3.95 MB，而 传统 im2col 矩阵大小 (对比用): 27.14 MB。这充分证明了 MEC 算法在减少 im2col 步骤产生的中间数据量方面的有效性，这是其设计初衷。

自定义 CUDA 代码显存构成:

主要显式分配的缓冲区总和：

d_src (输入): 224*224*3*4 bytes ≈ 0.57 MB

d_kernel (卷积核): 64*3*7*7*4 bytes ≈ 0.04 MB

d_mec_L (MEC 中间矩阵): 3.95 MB

d_output (输出): 220*220*64*4 bytes ≈ 11.8 MB

总计: 0.57 + 0.04 + 3.95 + 11.8 ≈ 16.36 MB (与您输出的“峰值主要缓冲区占用”基本一致)

程序运行导致的 GPU 总显存增加: 60.00 MB (这是 cudaMemGetInfo 在程序分配主要缓冲区后相比程序启动前的增量)。

cuBLAS 等库的隐式工作空间: 60.00 MB (总增加) - 16.36 MB (显式分配) ≈ 43.64 MB。这部分主要是 cublasSgemmStridedBatched 在执行时，cuBLAS 库为存储中间计算结果或优化计算而自行分配的临时 GPU 工作区。

PyTorch 显存构成:

初始PyTorch张量显存: 12.61 MB (主要包含输入张量和模型权重)。

峰值PyTorch张量显存: 48.67 MB (PyTorch 管理的所有张量，包括输入、输出、权重以及 cuDNN 可能使用并由 PyTorch 管理的工作空间或中间张量，在运行期间达到的最高点)。

近似PyTorch张量显存增量: 36.06 MB (峰值 - 初始)。这部分增量主要包括：

输出张量: 1*64*220*220*4 bytes ≈ 11.8 MB

cuDNN 工作空间及其他中间张量 (由 PyTorch 管理的部分): 36.06 MB - 11.8 MB ≈ 24.26 MB。

显存占用对比总结:

显式主要缓冲区: 您的自定义代码显式分配了约 16.36 MB。PyTorch 在加载数据和模型后，初始张量占用了 12.61 MB (不完全等同，但可作参考)。

“工作空间”对比 (估算):

您的 cuBLAS 工作空间 (隐式): 约 43.64 MB。

PyTorch/cuDNN 工作空间 (从其报告估算): 约 24.26 MB。

程序总影响: 您的程序使 GPU 总已用显存增加了 60 MB。PyTorch 的峰值张量占用为 48.67 MB (注意：PyTorch 的这个值不直接等同于 cudaMemGetInfo 的总增量，因为它只追踪 PyTorch 的分配)。

从“程序运行导致的GPU总显存增加 (60.00 MB)”来看，这个数值确实比 PyTorch 报告的“峰值PyTorch张量显存 (48.67 MB)”要高一些。这主要是因为 cudaMemGetInfo 统计的是全局的GPU显存，而 PyTorch 的统计是其自身分配的部分。您的 cuBLAS 调用可能申请了比 cuDNN 所选算法更多的工作空间。

三、可能的优化方向
性能优化：
im2col_cuda_mec 核函数：

目前耗时约 0.065 ms，相对总时间占比较小，优化优先级可能不高。

但仍可考虑：使用共享内存（如果 kernel_w 较大或存在更多数据复用）、向量化加载/存储（如 float2/float4，需注意对齐）。

GEMM 策略 (核心)：

cuBLASLt：cuBLASLt (Light) 库提供了对 GEMM 操作更细粒度的控制，包括一些算法选择和工作空间管理的可能性。但这会增加使用复杂度。

CUTLASS：CUTLASS 是 NVIDIA 开源的 C++ 模板库，用于构建高性能的 GEMM 和卷积核。通过 CUTLASS 可以生成高度定制和优化的 GEMM 实现，有可能获得比标准 cuBLAS 更好的性能或更可控的显存使用。集成 CUTLASS 的工作量较大。

直接卷积核/手写融合核：编写完全自定义的直接卷积核，或者将 im2col 与 GEMM 步骤融合到一个或少数几个 CUDA 核中，是最高难度的优化路径，但潜力也最大。这通常是专业库（如 cuDNN）开发者所做的工作。

编译器优化选项：确保使用 nvcc 编译时开启了合适的优化标志（如 -O3）、针对您的 GPU 架构的特定代码生成（如 -arch=sm_XX）以及在精度允许的情况下的快速数学库（-use_fast_math）。

显存优化：
cuBLAS 工作空间:

标准 cublasSgemmStridedBatched 函数通常不提供对工作空间大小的直接用户控制。

如上所述，cuBLASLt 可能提供一些管理工作空间的选项，允许您传入预分配的工作区。如果那约 43.64 MB 的 cuBLAS 工作空间是主要的额外显存负担，并且对您的应用场景（如部署到显存受限设备或运行更大模型/批量）造成了显著影响，那么研究 cuBLASLt 或其他 GEMM 实现是主要方向。

数据类型:

如果应用场景允许精度降低，可以考虑使用半精度浮点数（FP16/half），并利用 Tensor Cores（如果您的 GPU 支持）。这将使数据相关的显存占用减半，并可能大幅提升计算速度。这需要对代码进行较大修改，包括数据转换和可能的数值稳定性处理。

四、总结与建议
正视 cuDNN 的优势: 对于标准的卷积层和常见的参数配置，在性能上超越 cuDNN 是非常具有挑战性的。cuDNN 是 NVIDIA 投入大量资源进行优化的成果。

MEC 算法的价值: 您的实验结果已经证明了 MEC 在减少 im2col 产生的中间数据方面的核心价值。这在传统方法会产生极大中间矩阵，导致显存不足或带宽瓶颈的场景中尤其重要。

性能与显存的权衡:

您的自定义 MEC 实现目前在显存效率（特指中间 im2col 矩阵）上优于传统 im2col。

在运行速度上，当前不如 PyTorch/cuDNN。

在总显存增加（包括工作空间）方面，目前略高于 PyTorch 报告的峰值张量，主要是因为 cuBLAS 的工作空间占用。

后续步骤:

明确优化目标: 您是更看重极致的运行速度，还是在保持 MEC 内存效率的前提下进一步优化速度和总显存？

深度剖析: 使用 NVIDIA Nsight Systems / Nsight Compute 工具对您的 CUDA 代码进行深度剖析，精确了解 cublasSgemmStridedBatched 内部的耗时分布和资源使用情况，以及您的 im2col_cuda_mec 核函数是否存在如内存访问瓶颈、低 SM 占用率等问题。

小步尝试: 如果决定进一步优化，可以从更容易入手的编译器选项开始，然后评估 cuBLASLt 的适用性，最后再考虑 CUTLASS 这种更重量级的方案。

总而言之，您的 MEC 实现思路是正确的，并且在核心的中间数据压缩方面取得了预期效果。与 PyTorch/cuDNN 的性能差距主要源于后者拥有更成熟和广泛的底层算法选择与优化策略。










