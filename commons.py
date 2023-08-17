import torch


Array = torch.Tensor
Params = dict[str, Array]


DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR: str = "/om2/user/leni/influence/data"
MODEL_DIR: str = "/om2/user/leni/influence/models"
