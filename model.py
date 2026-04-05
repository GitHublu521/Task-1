"""MLP model built from scratch in PyTorch."""
"""从零开始用PyTorch构建的多层感知机模型"""

import torch.nn as nn

# 支持的激活函数字典，映射字符串到对应的PyTorch模块类
ACTIVATIONS = {
    "relu": nn.ReLU,      # 线性整流单元，最常用，计算简单，缓解梯度消失
    "sigmoid": nn.Sigmoid, # Sigmoid函数，输出范围(0,1)，但容易梯度饱和
    "gelu": nn.GELU,       # 高斯误差线性单元，Transformer中常用，性能优于ReLU
}


class MLP(nn.Module):
    """Multi-layer perceptron with configurable depth, width, and activation."""
    """可配置深度、宽度和激活函数的多层感知机"""

    def __init__(
        self,
        input_dim: int = 784,        # 输入维度，MNIST图像28x28=784像素
        hidden_dims: list[int] = [256, 128],  # 隐藏层维度列表，例如[256,128]表示两个隐藏层
        output_dim: int = 10,        # 输出维度，MNIST有10个数字类别(0-9)
        activation: str = "relu",    # 激活函数名称，必须是ACTIVATIONS字典中的键
    ):
        super().__init__()  # 调用父类nn.Module的初始化方法
        
        # 验证激活函数名称是否合法
        assert activation in ACTIVATIONS, f"Unknown activation: {activation}. Choose from {list(ACTIVATIONS)}"

        # 获取对应的激活函数类（注意：这里获取的是类，不是实例）
        act_fn = ACTIVATIONS[activation]
        
        # 构建网络层列表
        layers = []
        prev_dim = input_dim  # 当前层的输入维度，初始为输入维度
        
        # 遍历每个隐藏层的维度，构建全连接层 + 激活函数
        for h in hidden_dims:
            # 添加线性层：输入维度prev_dim，输出维度h
            layers.append(nn.Linear(prev_dim, h))
            # 添加激活函数层（实例化）
            layers.append(act_fn())
            # 更新prev_dim为当前层的输出维度，作为下一层的输入
            prev_dim = h
        
        # 添加输出层：输入维度为最后一个隐藏层的维度，输出维度为类别数10
        # 注意：输出层没有激活函数，因为后面会使用CrossEntropyLoss（它内部包含softmax）
        layers.append(nn.Linear(prev_dim, output_dim))

        # 使用Sequential容器将所有层串联起来
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten: (B, 1, 28, 28) -> (B, 784)
        # 将输入的4维张量展平为2维
        # x的形状：[batch_size, 通道数, 高度, 宽度] -> [batch_size, 784]
        x = x.view(x.size(0), -1)  # -1表示自动计算该维度大小
        # 通过网络前向传播
        return self.net(x)
