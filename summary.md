# Lab 1: MLP on MNIST — Results Summary

## Setup

- **Dataset**: MNIST (60k train / 10k test), normalized with mean=0.1307, std=0.3081
- **Epochs**: 10
- **Batch size**: 64
- **Device**: mps

## Experiment Grid

| Factor | Values |
|---|---|
| Hidden dims | `[256]`, `[256, 128]`, `[512, 256, 128]` |
| Activation | ReLU, Sigmoid, GELU |
| Optimizer | SGD, Adam |
| Learning rate | 0.01, 0.001 |

**Total configurations**: 3 x 3 x 2 x 2 = **36 runs**

## Key Findings

### 1. Activation Functions

ReLU and GELU performed best, achieving similar high test accuracy (above 95%). Sigmoid performed significantly worse (around 78%) because it suffers from the vanishing gradient problem. In deeper networks, Sigmoid saturates easily, causing gradients to become extremely small during backpropagation, which hinders effective weight updates in earlier layers.
### 2. Optimizers

Adam outperformed SGD by a large margin in terms of both convergence speed and final accuracy. Adam utilizes adaptive learning rates and momentum, allowing it to navigate the loss landscape more efficiently.

### 3. Architecture Depth & Width

Increasing the network depth (from 1 to 3 hidden layers) generally improved the model's capacity to represent complex patterns in the MNIST dataset, leading to lower training loss. However, deeper models like [512, 256, 128] also required more computational time per epoch.

### 4. Learning Rate

A higher learning rate (0.01) with Adam sometimes led to slight oscillations in loss, while a smaller rate (0.001) provided smoother convergence. For SGD, the 0.01 learning rate was necessary as 0.001 resulted in excessively slow learning.

## Best Configuration

| Metric | Value |
|---|---|
| Config | [512, 256, 128] / gelu / adam / 0.001 |
| Best test accuracy | 0.9826 |
| Final train loss | 0.021667001707347422 |

## Training Curves

![Training curves](results/training_curves.png)

## Comparison Figures

![Activation comparison](results/compare_activation.png)
![Optimizer comparison](results/compare_optimizer.png)

## Full Comparison Table

See [results/comparison.csv](results/comparison.csv) for all 36 configurations.

---

## Bonus: Cosine Annealing LR Scheduler with Warm-up and Warm Restarts

<!-- Complete this section if you implemented the bonus scheduler -->

### LR Curve

<!-- Plot the learning rate over training steps. Describe the warm-up, cosine cycles, and decay across restarts. -->

### Comparison with Constant LR

<!-- Does the scheduler improve final test accuracy compared to a constant learning rate? Show numbers. -->

### Effect of Each Component

<!-- How do warm-up, warm restarts, and cross-restart decay each contribute to the result? -->
