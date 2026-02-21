"""MLP model built from scratch in PyTorch."""

import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
}


class MLP(nn.Module):
    """Multi-layer perceptron with configurable depth, width, and activation."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list[int] = [256, 128],
        output_dim: int = 10,
        activation: str = "relu",
    ):
        super().__init__()
        assert activation in ACTIVATIONS, f"Unknown activation: {activation}. Choose from {list(ACTIVATIONS)}"

        act_fn = ACTIVATIONS[activation]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_fn())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten: (B, 1, 28, 28) -> (B, 784)
        x = x.view(x.size(0), -1)
        return self.net(x)
