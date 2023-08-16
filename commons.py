import torch

from typing import Any, Optional

from jaxtyping import Float, Int
from torch.utils.data import Dataset


Array = torch.Tensor
Params = dict[str, Array]


DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class Batch:
    """
    inputs: Float[Array, "b ..."]
    targets: Float[Array, "b ..."]
    indices: Int[Array, "b"]]
    batch_size: int
    """

    def __init__(
        self,
        inputs: Float[Array, "b ..."],
        targets: Float[Array, "b ..."],
        indices: Optional[Int[Array, "b"]] = None,
    ) -> None:
        self.batch_size = len(inputs)
        if indices is None:
            indices = torch.arange(self.batch_size)
        # batch size must be the same for all tensors
        assert len(inputs) == len(targets) == len(indices)
        self.inputs = inputs
        self.targets = targets
        self.indices = indices

    def __len__(self) -> int:
        return self.batch_size


class IndexedDataset(Dataset):
    def __init__(self, base_dataset: Dataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Any:
        input, target = self.base_dataset[index]
        return index, input, target
