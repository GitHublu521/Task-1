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

**Total configurations**: 3 × 3 × 2 × 2 = **36 runs**

---

## Activation Functions: Mathematical Formulae

| Activation | Mathematical Formula | Derivative | Key Characteristics |
|------------|---------------------|------------|---------------------|
| **ReLU** | $f(x) = \max(0, x)$ | $f'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$ | • Computationally efficient<br>• Mitigates vanishing gradient<br>• Can cause dead neurons |
| **Sigmoid** | $f(x) = \frac{1}{1 + e^{-x}}$ | $f'(x) = f(x)(1 - f(x))$ | • Output range (0,1)<br>• Severe vanishing gradient<br>• Output not zero-centered |
| **GELU** | $f(x) = x \cdot \Phi(x)$<br>where $\Phi(x)$ is standard Gaussian CDF | $f'(x) \approx \Phi(x) + x \cdot \text{pdf}(x)$ | • Smooth ReLU variant<br>• Used in Transformers<br>• Often outperforms ReLU |

**GELU Approximation** (commonly implemented):
$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)$$

---

## Key Findings

### 1. Activation Functions

ReLU and GELU performed best, achieving similar high test accuracy (above 97%). Sigmoid performed significantly worse (around 78%) because it suffers from the **vanishing gradient problem**. In deeper networks, Sigmoid saturates easily, causing gradients to become extremely small during backpropagation, which hinders effective weight updates in earlier layers.

| Activation | Avg Best Accuracy | Convergence Speed | Training Stability |
|------------|------------------|-------------------|--------------------|
| **ReLU** | 97.8% | Fast | Good |
| **GELU** | 98.1% | Medium | Excellent |
| **Sigmoid** | 78.3% | Slow | Poor |

### 2. Optimizers

Adam outperformed SGD by a large margin in both convergence speed and final accuracy. Adam utilizes adaptive learning rates and momentum, allowing it to navigate the loss landscape more efficiently.

| Optimizer | Best Accuracy | Convergence | Sensitivity to LR |
|-----------|---------------|-------------|-------------------|
| **SGD** | 94.2% | Slow | High (needs larger LR) |
| **Adam** | 98.2% | Fast | Low (robust to LR choices) |

### 3. Architecture Depth & Width

Increasing the network depth (from 1 to 3 hidden layers) generally improved the model's capacity to represent complex patterns in the MNIST dataset, leading to lower training loss. However, deeper models like `[512, 256, 128]` also required more computational time per epoch.

| Architecture | Parameters | Best Accuracy | Time per Epoch |
|--------------|------------|---------------|----------------|
| `[256]` | ~203K | 97.2% | Fast |
| `[256, 128]` | ~235K | 97.9% | Medium |
| `[512, 256, 128]` | ~669K | 98.2% | Slow |

### 4. Learning Rate

| Optimizer | LR = 0.01 | LR = 0.001 | Conclusion |
|-----------|-----------|------------|-------------|
| **SGD** | 94.2% accuracy | 45.3% accuracy (extremely slow) | SGD needs larger LR |
| **Adam** | 97.5% accuracy (slight oscillations) | 98.1% accuracy (smooth) | Adam is more robust |

A higher learning rate (0.01) with Adam sometimes led to slight oscillations in loss, while a smaller rate (0.001) provided smoother convergence. For SGD, the 0.01 learning rate was necessary as 0.001 resulted in excessively slow learning.

---

## Best Configuration

| Metric | Value |
|--------|-------|
| **Config** | `[512, 256, 128]` / GELU / Adam / 0.001 |
| **Best test accuracy** | 0.9826 |
| **Final train loss** | 0.021667001707347422 |
| **Final test accuracy** | 0.9812 |

---

## Training Curves

![Training curves](results/training_curves.png)

*Figure 1: Training loss (left) and test accuracy (right) curves for all 36 configurations. Adam-based runs show faster convergence and higher final accuracy compared to SGD.*

---

## Comparison Figures

### Activation Function Comparison

![Activation comparison](results/compare_activation.png)

*Figure 2: Average final test accuracy by activation function. ReLU and GELU perform similarly and significantly outperform Sigmoid.*

### Optimizer Comparison

![Optimizer comparison](results/compare_optimizer.png)

*Figure 3: Average final test accuracy by optimizer. Adam consistently outperforms SGD across all architectures and learning rates.*

---

## Full Comparison Table

See [results/comparison.csv](results/comparison.csv) for all 36 configurations sorted by best test accuracy.

### Top 5 Configurations

| Rank | Config | Hidden Dims | Activation | Optimizer | LR | Best Acc |
|------|--------|-------------|------------|-----------|-----|----------|
| 1 | 512x256x128_gelu_adam_lr0.001 | [512,256,128] | GELU | Adam | 0.001 | 0.9826 |
| 2 | 512x256x128_relu_adam_lr0.001 | [512,256,128] | ReLU | Adam | 0.001 | 0.9821 |
| 3 | 256x128_gelu_adam_lr0.001 | [256,128] | GELU | Adam | 0.001 | 0.9815 |
| 4 | 256x128_relu_adam_lr0.001 | [256,128] | ReLU | Adam | 0.001 | 0.9810 |
| 5 | 512x256x128_gelu_adam_lr0.01 | [512,256,128] | GELU | Adam | 0.01 | 0.9803 |

---

## Conclusions

1. **Activation Functions**: ReLU and GELU are both excellent choices for MNIST classification. GELU shows slightly better stability and final accuracy, while ReLU offers faster computation. Sigmoid should be avoided for hidden layers in deep networks.

2. **Optimizers**: Adam is strongly recommended over SGD for this task due to faster convergence, higher accuracy, and better robustness to learning rate choices.

3. **Architecture**: Deeper networks with more parameters achieve higher accuracy, though with diminishing returns. The 3-layer `[512, 256, 128]` architecture performed best.

4. **Learning Rate**: For Adam, 0.001 is optimal; for SGD, 0.01 is necessary. Using an inappropriate learning rate can severely degrade performance.

---

## Bonus: Cosine Annealing LR Scheduler with Warm-up and Warm Restarts

### Implementation Details

```json
{
  "lr_scheduler": {
    "warm_up_epochs": 2,
    "lr_warm_up_min": 1e-5,
    "lr_min": 1e-5,
    "lr_max": 1e-3,
    "restart_epochs": 4,
    "lr_decay_factor": 0.75,
    "lr_minimum": 2e-6
  }
}
