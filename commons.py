from pathlib import Path

import torch
import torch.nn as nn

from jaxtyping import Int, Float


Array = torch.Tensor
Params = dict[str, Array]

TrainingState = tuple[
    nn.Module, Int[Array, "b"]
]  # (model_checkpoint, training_indices)


DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH: Path = Path("/om2/user/leni/influence/data")
RUNS_PATH: Path = Path("/om2/user/leni/influence/runs")
