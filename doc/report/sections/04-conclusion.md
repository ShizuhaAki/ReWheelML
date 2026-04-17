# 总结

## 主要结果
任务 1：

回归任务：MAE 满足 <0.01> 要求

分类任务：
```shell
$ build/classification --inference checkpoints/best_classification.ckpt 
Loaded checkpoint checkpoints/best_classification.ckpt (epoch 50, eval_loss = 0.127526)
Loading inference dataset from data/test...
Inference loss: 0.127526, accuracy: 0.9637
```

任务 2：
```shell
$ build/cnn --inference checkpoints/best_cnn_bs64.ckpt
Loaded checkpoint checkpoints/best_cnn_bs64.ckpt (epoch 11, eval_loss = 0.016389)
Loading inference dataset from data/test...
Inference loss: 0.016389, accuracy: 0.9966
```

## 经验与不足

本项目实现了从基础 MLP 到 CNN 的完整训练流程。BP 部分展示了全连接网络如何通过前向传播、损失函数和链式法则完成参数学习；CNN 部分进一步利用局部连接、权值共享和池化结构，更好地匹配图像数据的空间特征。

从表达能力看，MLP 可以拟合一般函数，也可以处理展平后的图像输入，但它没有显式利用二维邻域结构；CNN 则通过卷积核提取局部笔画和部件特征，同一组参数可以在不同位置复用，因此在图像分类任务上通常具有更好的参数效率和泛化能力。

从收敛角度看，Adam、BatchNorm 和学习率调度器可以提升训练稳定性；从泛化角度看，数据增强、Dropout 和最佳 checkpoint 保存可以缓解过拟合。

同时，我也深刻认识到了如何正确地利用好计算资源。通过实现看似神秘的 im2col 机制，我了解了如何使用通用矩阵乘法（GEMM）来处理看似和矩阵乘法无关的运算需求。这帮助了我理解未来进行真实世界的底层算子开发所需要的知识。