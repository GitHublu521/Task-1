"""End-to-end experiment pipeline for MLP on MNIST.

Sweeps over architectures, activations, optimizers, and learning rates.
Produces a comparison table (CSV) and training curves (PNG) in results/.

Usage:
    python run.py                    # uses config.json
    python run.py --config my.json   # uses custom config
"""

"""
端到端的MNIST多层感知机实验流水线

遍历架构、激活函数、优化器和学习率的不同组合
在results/目录下生成对比表格(CSV)和训练曲线(PNG)

使用方法：
    python run.py                    # 使用config.json
    python run.py --config my.json   # 使用自定义配置
"""

import argparse
import itertools
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from dataset import get_mnist_loaders
from model import MLP
from train import evaluate, train_one_epoch

DEFAULT_CONFIG = "config.json"


def load_config(path):
    """从JSON文件加载配置"""
    with open(path) as f:
        return json.load(f)


def get_device(requested):
    """
    获取计算设备（CPU/CUDA/MPS）
    
    参数:
        requested: 请求的设备类型，可以是 "auto", "cpu", "cuda", "mps"
    
    返回:
        实际使用的设备名称
    """
    # 如果指定了具体设备且不是"auto"，直接使用
    if requested and requested != "auto":
        return requested
    # 自动检测：优先CUDA，然后MPS（Apple Silicon），最后CPU
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_optimizer(name, params, lr):
    """
    根据名称创建优化器
    
    参数:
        name: 优化器名称 ("sgd" 或 "adam")
        params: 模型参数
        lr: 学习率
    
    返回:
        优化器实例
    """
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def run_single(cfg, train_loader, test_loader, device, epochs):
    """
    训练单个配置并返回每个epoch的评估指标
    
    参数:
        cfg: 配置字典，包含 hidden_dims, activation, optimizer, lr
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        device: 计算设备
        epochs: 训练轮数
    
    返回:
        history: 包含训练/测试损失和准确率的字典
    """
    # 创建模型并移动到指定设备
    model = MLP(
        hidden_dims=cfg["hidden_dims"],
        activation=cfg["activation"],
    ).to(device)

    # 创建优化器
    optimizer = make_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"])

    # 记录每个epoch的指标
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    
    # 训练epochs轮
    for epoch in range(1, epochs + 1):
        # 训练一个epoch
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        # 在测试集上评估
        te_loss, te_acc = evaluate(model, test_loader, device)
        
        # 保存结果
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        
        # 打印进度
        print(
            f"  Epoch {epoch}/{epochs}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"test_loss={te_loss:.4f}  test_acc={te_acc:.4f}"
        )
    return history


def cfg_label(cfg):
    """
    生成配置的标签字符串，用于标识不同的实验组合
    
    示例: "256x128_relu_adam_lr0.001"
    """
    # 将隐藏层维度用"x"连接，如 [256, 128] -> "256x128"
    dims = "x".join(map(str, cfg["hidden_dims"]))
    return f"{dims}_{cfg['activation']}_{cfg['optimizer']}_lr{cfg['lr']}"


# ── 绘图辅助函数 ─────────────────────────────────────────────────────────
def plot_training_curves(all_results, epochs, results_dir):
    """
    绘制训练曲线图（一个包含两个子图的图形）
    
    参数:
        all_results: 所有配置的训练结果字典
        epochs: 训练轮数
        results_dir: 结果保存目录
    """
    # 创建1行2列的子图，大小为14x5英寸
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 为每个配置绘制曲线
    for label, hist in all_results.items():
        # 左图：训练损失曲线
        ax1.plot(range(1, epochs + 1), hist["train_loss"], label=label)
        # 右图：测试准确率曲线
        ax2.plot(range(1, epochs + 1), hist["test_acc"], label=label)
    
    # 设置坐标轴标签和标题
    ax1.set(xlabel="Epoch", ylabel="Train Loss", title="Training Loss")
    ax2.set(xlabel="Epoch", ylabel="Test Accuracy", title="Test Accuracy")
    
    # 添加图例，字体调小以便容纳多个配置
    ax1.legend(fontsize=6)
    ax2.legend(fontsize=6)
    
    # 自动调整布局并保存
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "training_curves.png"), dpi=150)
    plt.close(fig)  # 关闭图形释放内存


def plot_by_factor(all_results, factor, values, results_dir):
    """
    绘制按某个因素分组的条形图
    
    参数:
        all_results: 所有配置的训练结果字典
        factor: 要分组的因素名称（如 "activation" 或 "optimizer"）
        values: 该因素的所有可能取值列表
        results_dir: 结果保存目录
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 对每个取值，计算平均最终测试准确率
    for v in values:
        # 找出所有标签中包含当前值的配置
        accs = [hist["test_acc"][-1] for label, hist in all_results.items() if str(v) in label]
        # 计算平均值（如果没有配置则返回0）
        avg_acc = sum(accs) / len(accs) if accs else 0
        ax.bar(str(v), avg_acc)
    
    # 设置图表标题和标签
    ax.set(xlabel=factor, ylabel="Avg Final Test Acc", title=f"Test Accuracy by {factor}")
    fig.tight_layout()
    # 保存图片，文件名如 "compare_activation.png"
    fig.savefig(os.path.join(results_dir, f"compare_{factor}.png"), dpi=150)
    plt.close(fig)


# ── 主函数 ─────────────────────────────────────────────────────────────────────
def main():
    """主函数：执行完整的实验流程"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MLP MNIST experiment sweep")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, 
                       help=f"Path to config JSON (default: {DEFAULT_CONFIG})")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    
    # 确定计算设备
    device = get_device(config.get("device", "auto"))
    
    # 提取超参数
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    results_dir = config.get("results_dir", "results")
    grid = config["grid"]  # 实验网格配置

    # 计算总共需要运行的实验次数
    n_runs = 1
    for v in grid.values():
        n_runs *= len(v)
    
    # 打印实验配置信息
    print(f"Config: {args.config}")
    print(f"Device: {device} | Epochs: {epochs} | Batch size: {batch_size} | Total runs: {n_runs}")

    # 创建结果目录（如果不存在）
    os.makedirs(results_dir, exist_ok=True)
    
    # 加载MNIST数据集
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    # 存储所有实验结果
    all_results = {}
    summary_rows = []  # 用于生成CSV表格的行

    # 生成所有参数组合（笛卡尔积）
    keys = list(grid.keys())  # 例如: ["hidden_dims", "activation", "optimizer", "lr"]
    
    # 使用itertools.product遍历所有组合
    for i, combo in enumerate(itertools.product(*grid.values()), 1):
        # 将组合转换为配置字典
        cfg = dict(zip(keys, combo))
        # 生成标签
        label = cfg_label(cfg)
        
        print(f"\n{'='*60}\n[RUN {i}/{n_runs}] {label}\n{'='*60}")

        # 运行单个配置的训练
        history = run_single(cfg, train_loader, test_loader, device, epochs)
        
        # 保存结果
        all_results[label] = history
        summary_rows.append({
            "config": label,
            "hidden_dims": str(cfg["hidden_dims"]),
            "activation": cfg["activation"],
            "optimizer": cfg["optimizer"],
            "lr": cfg["lr"],
            "best_test_acc": max(history["test_acc"]),      # 最佳测试准确率
            "final_test_acc": history["test_acc"][-1],      # 最终测试准确率
            "final_train_loss": history["train_loss"][-1],  # 最终训练损失
        })

    # 保存对比表格（按最佳测试准确率降序排序）
    df = pd.DataFrame(summary_rows).sort_values("best_test_acc", ascending=False)
    df.to_csv(os.path.join(results_dir, "comparison.csv"), index=False)
    
    print(f"\n{'='*60}\nComparison Table\n{'='*60}")
    print(df.to_string(index=False))

    # 保存原始结果（JSON格式）
    with open(os.path.join(results_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # 生成图表
    plot_training_curves(all_results, epochs, results_dir)
    plot_by_factor(all_results, "activation", grid["activation"], results_dir)
    plot_by_factor(all_results, "optimizer", grid["optimizer"], results_dir)

    print(f"\nResults saved to {results_dir}/")


# 程序入口
if __name__ == "__main__":
    main()
