import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from commons import DEVICE


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str = DEVICE,
):
    losses = torch.zeros(size=(len(loader),), device="cpu")
    correctnesses = torch.zeros(size=(len(loader),), dtype=bool, device="cpu")

    model.eval()
    for indices, inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        batch_losses = F.cross_entropy(logits, targets, reduction="none")
        predictions = logits.argmax(dim=-1)
        batch_correctnesses = predictions == targets
        losses[indices] = batch_losses.detach().cpu()
        correctnesses[indices] = batch_correctnesses.detach().cpu()

    return losses, correctnesses
