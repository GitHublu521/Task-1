"""End-to-end experiment pipeline for MLP on MNIST.

Sweeps over architectures, activations, optimizers, and learning rates.
Produces a comparison table (CSV) and training curves (PNG) in results/.

Usage:
    python run.py                    # uses config.json
    python run.py --config my.json   # uses custom config
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
    with open(path) as f:
        return json.load(f)


def get_device(requested):
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_optimizer(name, params, lr):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def run_single(cfg, train_loader, test_loader, device, epochs):
    """Train one configuration and return per-epoch metrics."""
    model = MLP(
        hidden_dims=cfg["hidden_dims"],
        activation=cfg["activation"],
    ).to(device)

    optimizer = make_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"])

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        print(
            f"  Epoch {epoch}/{epochs}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"test_loss={te_loss:.4f}  test_acc={te_acc:.4f}"
        )
    return history


def cfg_label(cfg):
    dims = "x".join(map(str, cfg["hidden_dims"]))
    return f"{dims}_{cfg['activation']}_{cfg['optimizer']}_lr{cfg['lr']}"


# ── Plotting helpers ─────────────────────────────────────────────────────────
def plot_training_curves(all_results, epochs, results_dir):
    """One figure with train loss and test accuracy subplots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for label, hist in all_results.items():
        ax1.plot(range(1, epochs + 1), hist["train_loss"], label=label)
        ax2.plot(range(1, epochs + 1), hist["test_acc"], label=label)
    ax1.set(xlabel="Epoch", ylabel="Train Loss", title="Training Loss")
    ax2.set(xlabel="Epoch", ylabel="Test Accuracy", title="Test Accuracy")
    ax1.legend(fontsize=6)
    ax2.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def plot_by_factor(all_results, factor, values, results_dir):
    """Bar chart of final test accuracy grouped by one factor."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in values:
        accs = [hist["test_acc"][-1] for label, hist in all_results.items() if str(v) in label]
        ax.bar(str(v), sum(accs) / len(accs) if accs else 0)
    ax.set(xlabel=factor, ylabel="Avg Final Test Acc", title=f"Test Accuracy by {factor}")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"compare_{factor}.png"), dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MLP MNIST experiment sweep")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help=f"Path to config JSON (default: {DEFAULT_CONFIG})")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config.get("device", "auto"))
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    results_dir = config.get("results_dir", "results")
    grid = config["grid"]

    n_runs = 1
    for v in grid.values():
        n_runs *= len(v)
    print(f"Config: {args.config}")
    print(f"Device: {device} | Epochs: {epochs} | Batch size: {batch_size} | Total runs: {n_runs}")

    os.makedirs(results_dir, exist_ok=True)
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    all_results = {}
    summary_rows = []

    keys = list(grid.keys())
    for i, combo in enumerate(itertools.product(*grid.values()), 1):
        cfg = dict(zip(keys, combo))
        label = cfg_label(cfg)
        print(f"\n{'='*60}\n[RUN {i}/{n_runs}] {label}\n{'='*60}")

        history = run_single(cfg, train_loader, test_loader, device, epochs)
        all_results[label] = history
        summary_rows.append({
            "config": label,
            "hidden_dims": str(cfg["hidden_dims"]),
            "activation": cfg["activation"],
            "optimizer": cfg["optimizer"],
            "lr": cfg["lr"],
            "best_test_acc": max(history["test_acc"]),
            "final_test_acc": history["test_acc"][-1],
            "final_train_loss": history["train_loss"][-1],
        })

    # Save comparison table
    df = pd.DataFrame(summary_rows).sort_values("best_test_acc", ascending=False)
    df.to_csv(os.path.join(results_dir, "comparison.csv"), index=False)
    print(f"\n{'='*60}\nComparison Table\n{'='*60}")
    print(df.to_string(index=False))

    # Save raw results
    with open(os.path.join(results_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate figures
    plot_training_curves(all_results, epochs, results_dir)
    plot_by_factor(all_results, "activation", grid["activation"], results_dir)
    plot_by_factor(all_results, "optimizer", grid["optimizer"], results_dir)

    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
