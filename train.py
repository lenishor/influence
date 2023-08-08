import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from commons import IndexedDataset
from models import make_mlp


# MNIST dataset
IN_FEATURES: int = 28 * 28
OUT_FEATURES: int = 10

# model architecture
HIDDEN_FEATURES: int = 128
NUM_HIDDEN_LAYERS: int = 4

# training hyperparameters
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 1
LEARNING_RATE: float = 1e-3


if __name__ == "__main__":
    # make datasets and dataloaders
    train_dataset = IndexedDataset(
        MNIST(root="data/", train=True, transform=ToTensor(), download=True)
    )
    test_dataset = IndexedDataset(
        MNIST(root="data/", train=False, transform=ToTensor(), download=True)
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # make model
    model = make_mlp(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
    )

    # make optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # training loop
    for epoch in range(NUM_EPOCHS):
        for indices, inputs, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
