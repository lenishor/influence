import torch.nn as nn

from commons import DEVICE


def make_fc_layer(hidden_features: int, device: str = DEVICE) -> nn.Module:
    return nn.Sequential(
        nn.Linear(hidden_features, hidden_features),
        nn.ReLU(),
    ).to(device=device)


def make_mlp(
    in_features: int,
    out_features: int,
    hidden_features: int,
    num_hidden_layers: int = 1,
    device: str = DEVICE,
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.ReLU(),
        *[make_fc_layer(hidden_features) for _ in range(num_hidden_layers)],
        nn.Linear(hidden_features, out_features),
    ).to(device=device)
