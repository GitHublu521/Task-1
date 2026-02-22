"""MNIST dataloader utilities."""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(batch_size: int = 64, data_dir: str = "./data"):
    """Return train and test DataLoaders for MNIST.

    Images are normalized to mean=0.1307, std=0.3081 (MNIST statistics).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
