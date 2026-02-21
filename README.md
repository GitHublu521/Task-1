# Lab 1: MLP from Scratch on MNIST

Build a multi-layer perceptron (MLP) from scratch in PyTorch, train it on MNIST, and compare different architectures, activations, optimizers, and learning rates.

## Repository Structure

```
├── config.json        # Experiment configuration (edit this!)
├── dataset.py         # MNIST dataloader (auto-downloads to data/)
├── model.py           # MLP model (configurable layers, activations)
├── train.py           # Training and evaluation loops
├── run.py             # Main entry point — runs full experiment sweep
├── summary.md         # Template to summarize your findings
├── requirements.txt   # Python dependencies
├── data/              # MNIST data (auto-downloaded on first run)
└── results/           # Generated outputs
    ├── comparison.csv        # Full comparison table
    ├── all_results.json      # Raw per-epoch metrics
    ├── training_curves.png   # Loss & accuracy curves
    ├── compare_activation.png
    └── compare_optimizer.png
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Run with default config

```bash
python run.py
```

This reads `config.json` and runs **36 configurations** (3 architectures x 3 activations x 2 optimizers x 2 learning rates), each for **10 epochs** with **batch size 64**. Device is auto-detected (CUDA > MPS > CPU).

### Run with a custom config

```bash
python run.py --config my_config.json
```

## Configuration

All experiment settings live in a single JSON file. The default `config.json`:

```json
{
  "epochs": 10,
  "batch_size": 64,
  "device": "auto",
  "results_dir": "results",
  "grid": {
    "hidden_dims": [[256], [256, 128], [512, 256, 128]],
    "activation": ["relu", "sigmoid", "gelu"],
    "optimizer": ["sgd", "adam"],
    "lr": [0.01, 0.001]
  }
}
```

| Field | Description |
|---|---|
| `epochs` | Training epochs per configuration |
| `batch_size` | Mini-batch size for DataLoader |
| `device` | `"auto"`, `"cpu"`, `"cuda"`, or `"mps"` |
| `results_dir` | Output directory for figures and tables |
| `grid.hidden_dims` | List of architectures, each a list of layer widths |
| `grid.activation` | Activation functions: `"relu"`, `"sigmoid"`, `"gelu"` |
| `grid.optimizer` | Optimizers: `"sgd"`, `"adam"` |
| `grid.lr` | Learning rates to sweep |

### Example: quick smoke test

Create `config_quick.json`:

```json
{
  "epochs": 2,
  "batch_size": 128,
  "device": "auto",
  "results_dir": "results_quick",
  "grid": {
    "hidden_dims": [[256, 128]],
    "activation": ["relu"],
    "optimizer": ["adam"],
    "lr": [0.001]
  }
}
```

```bash
python run.py --config config_quick.json
```

### Example: compare activations only

```json
{
  "epochs": 10,
  "batch_size": 64,
  "device": "auto",
  "results_dir": "results_activations",
  "grid": {
    "hidden_dims": [[256, 128]],
    "activation": ["relu", "sigmoid", "gelu"],
    "optimizer": ["adam"],
    "lr": [0.001]
  }
}
```

## Outputs

After running, check `results/`:

| File | Description |
|---|---|
| `comparison.csv` | All configurations sorted by best test accuracy |
| `all_results.json` | Per-epoch train/test loss and accuracy for every run |
| `training_curves.png` | Training loss and test accuracy curves (all configs) |
| `compare_activation.png` | Average test accuracy by activation function |
| `compare_optimizer.png` | Average test accuracy by optimizer |

## What to Do

1. Run the full sweep: `python run.py`
2. Examine the results in `results/`
3. Fill in `summary.md` with your observations on:
   - Which activation function works best and why
   - SGD vs Adam performance differences
   - Effect of network depth and width
   - Learning rate sensitivity

## Bonus: Cosine Annealing LR Scheduler with Warm-up and Warm Restarts

For bonus credit, implement a custom learning rate scheduler with the following behavior and integrate it into the training pipeline.

### Scheduler Specification

The scheduler operates **per training step** (not per epoch) and consists of three phases:

#### 1. Warm-up Phase
- Duration: `warm_up_epochs` epochs (converted to steps)
- The learning rate ramps up from `lr_warm_up_min` to `lr_max` following a **cosine** curve (not linear)

#### 2. Cosine Annealing with Warm Restarts
- After warm-up, the LR follows repeating cosine cycles
- Each cycle lasts `restart_epochs` epochs (converted to steps)
- Within each cycle, the LR oscillates between `lr_max` and `lr_min` following a cosine curve

#### 3. Decay Across Restarts
- At each restart boundary, the peak LR is multiplied by `lr_decay_factor`
- This means later cycles have progressively lower peak LRs
- The LR is clamped to never go below `lr_minimum`

### Implementation Requirements

- Use `torch.optim.lr_scheduler.LambdaLR` with a custom lambda function that maps `step -> lr_scale`
- Use `AdamW` optimizer (with configurable `weight_decay`) instead of the base Adam
- The scheduler should step **once per training step** (once per batch), not once per epoch
- Add the scheduler configuration to `config.json` under a new `"lr_scheduler"` section
- Plot the LR curve over training steps as an additional figure

### Suggested Config Structure

```json
{
  "lr_scheduler": {
    "warm_up_epochs": 2,
    "lr_warm_up_min": 1e-5,
    "lr_min": 1e-5,
    "lr_max": 1e-4,
    "restart_epochs": 4,
    "lr_decay_factor": 0.75,
    "lr_minimum": 2e-6
  }
}
```

### In Your Summary, Discuss

- How does the LR curve look over the course of training? Plot it.
- Compare the bonus scheduler against a constant LR baseline — does it improve accuracy?
- How do the warm-up, restart, and decay components each contribute to the result?
