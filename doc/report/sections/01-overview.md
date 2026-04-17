# 项目概述

## 开发环境

| 项目 | 说明 |
| --- | --- |
| 操作系统 | Linux 6.6.87.2-microsoft-standard-WSL2 x86_64 GNU/Linux。 |
| 编译器 | `g++ 13.3.0`，使用 C++20 标准编译。 |
| 第三方库 | 使用 BLAS 进行矩阵乘法加速；使用 OpenMP 并行化部分循环；使用 `inih` 读取配置；使用 `stb_image` 读取 BMP 图像；使用 `doctest` 编写单元测试。<br/><br/>核心训练、反向传播和参数更新由项目代码自主实现。 |

## 仓库结构

项目中三个主要训练入口分别对应实验要求中的三个任务：`src/regression.cpp` 用于正弦函数回归，`src/classification.cpp` 用于基于全连接网络的手写汉字分类，`src/cnn.cpp` 用于卷积神经网络分类。

本项目没有直接依赖现成深度学习框架，而是围绕自定义的 `NDArray` 和 `Module` 两个抽象搭建训练流程。


`NDArray` 是项目中的基础张量类型，内部使用 `shared_ptr<vector<float>>` 管理实际存储，并用 C++20 的 `std::span<float>` 表示当前视图。这样做的好处是可以在不复制底层数据的情况下实现轻量级切片。例如 mini-batch 训练中调用 `slice(row_start, row_end)` 时，只会让新 `NDArray` 共享原始数据并指向对应的连续区间，从而减少频繁拷贝带来的开销。`NDArray` 同时维护 `shape`、`stride`、`ndim` 和 `numel`，支持索引访问、reshape、逐元素运算、广播、按轴求和/均值和二维矩阵乘法。

矩阵乘法是神经网络训练中的主要计算瓶颈，因此 `NDArray::matmul` 调用 BLAS 的 `cblas_sgemm` 完成二维矩阵乘法。全连接层、卷积层中的 `im2col` 变换后矩阵乘法都可以复用这一接口。逐元素激活函数、Dropout mask 和部分张量循环使用 OpenMP 并行化，以降低大 batch 或多通道特征图上的循环开销。

在代码组织方面，我们采用了接近 PyTorch 的模块化设计。所有网络层继承统一的 `Module` 接口，训练入口只需要按顺序调用各层的 `forward`、`backward`、`parameters` 和优化器更新逻辑即可。可训练参数通过 `Parameter` 结构暴露给优化器，在使用优化器时，由它接管负责 `step` 操作。

`Module` 是所有网络层的统一接口，主要包含以下方法：

| 方法 | 作用 |
| --- | --- |
| `forward` | 执行前向传播，并在需要时缓存输入或输出。 |
| `backward` | 接收上游梯度，计算本层参数梯度和传给前一层的梯度。 |
| `parameters` | 返回本层可训练参数及其梯度指针，供优化器统一管理。 |
| `train` / `eval` | 切换训练和推理行为，主要用于 Dropout、BatchNorm 等层。 |

在此基础上，MLP 由 `Linear + Activation + Dropout` 组合而成，CNN 由 `Conv2D + BatchNorm2D + Activation + MaxPool2D + Dropout` 组合而成。训练入口只负责数据读取、batch 划分、损失计算、反向传播和优化器更新，具体层的前向/反向细节则封装在各模块内部。

## 训练与复现流程

实验超参数通过 `ini/` 目录中的配置文件控制。`regression.ini` 配置正弦函数采样数量、隐藏层规模、学习率和训练轮数；`classification.ini` 配置 MLP 分类器的隐藏层、Dropout、Adam 参数、数据增强和 checkpoint；`cnn-bs64.ini` 配置 CNN 的卷积块数量、通道数、学习率调度器和数据增强强度。

训练流程整体如下：

1. 从配置文件读取数据路径、模型结构和优化器参数。
2. 构造数据集：回归任务随机采样 $x \in [-\pi, \pi]$；分类任务读取 28×28 BMP 图像并展平或 reshape 为 NCHW。
3. 构造模型和优化器，并在每个 epoch 内按 mini-batch 前向传播。
4. 计算损失函数，调用 `backward` 逐层反传梯度。
5. 调用优化器 `step` 更新所有参数，并通过 `zero_grad` 清空梯度。
6. 在验证集上计算指标，必要时保存验证指标最优的 checkpoint。

为了保证实现可靠性，项目使用 `doctest` 对关键模块进行单元测试，覆盖 `NDArray`、全连接层、Dropout、Conv2D、图像读取和 checkpoint 等功能。报告中的实验结果均基于这些自定义模块运行得到。
