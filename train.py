import wandb

import torch
import torch.nn.functional as F

from einops import rearrange
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import trange

from commons import DEVICE, DATA_PATH, RUNS_PATH
from data import FancyDataset
from models import make_mlp


RUN_NAME: str = "MNIST-10-epochs"

# MNIST dataset
IN_FEATURES: int = 28 * 28
OUT_FEATURES: int = 10

# model architecture
HIDDEN_FEATURES: int = 128
NUM_HIDDEN_LAYERS: int = 4

# training hyperparameters
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 10
LEARNING_RATE: float = 1e-3


if __name__ == "__main__":
    # initialize wandb
    wandb.init(
        name=RUN_NAME,
        project="influence",
        entity="poggio-lab",
    )

    # make run directory
    run_path = RUNS_PATH / RUN_NAME
    run_path.mkdir(exist_ok=True)

    # make datasets and dataloaders
    train_set = FancyDataset(
        MNIST(
            root=DATA_PATH,
            train=True,
            transform=ToTensor(),
            download=True,
        )
    )
    test_set = FancyDataset(
        MNIST(
            root=DATA_PATH,
            train=False,
            transform=ToTensor(),
            download=True,
        )
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # make model
    model = make_mlp(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        device=DEVICE,
    )

    # make optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # training loop
    for epoch in trange(1, NUM_EPOCHS + 1):
        step = 1

        for indices, inputs, targets in train_loader:
            # move data to device
            indices = indices.to(device=DEVICE)
            inputs = inputs.to(device=DEVICE)
            targets = targets.to(device=DEVICE)

            # flatten input images
            inputs = rearrange(inputs, "b c h w -> b (c h w)")

            # take one optimization step
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

            # calculate accuracy
            predictions = torch.argmax(outputs, dim=-1)
            accuracy = (predictions == targets).float().mean()

            # log metrics
            wandb.log(
                {
                    "loss": loss.item(),
                    "accuracy": accuracy.item(),
                },
                step=step,
            )

            step += 1

        # save model and train indices
        state_dict = model.state_dict()
        torch.save(indices, run_path / f"{epoch:02}_indices.pt")
        torch.save(state_dict, run_path / f"{epoch:02}_state_dict.pt")
