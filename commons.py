import torch
import torch.nn as nn

from jaxtyping import Int


Array = torch.Tensor
Params = dict[str, Array]

TrainingState = tuple[
    nn.Module, Int[Array, "b"]
]  # (model_checkpoint, training_indices)


DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR: str = "/om2/user/leni/influence/data"
MODEL_DIR: str = "/om2/user/leni/influence/models"
