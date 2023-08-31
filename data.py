from typing import Callable

import torch

from jaxtyping import Float, Int
from torch.utils.data import Dataset

from commons import Array


Batch = tuple[
    Int[Array, "b"],  # indices
    Float[Array, "b ..."],  # inputs
    Float[Array, "b ..."],  # targets
]


class FunctionDataset(Dataset):
    def __init__(
        self,
        fn: Callable[[Float[Array, "b *input"]], Float[Array, "b *output"]],
        domain: Float[Array, "n *input"],
    ) -> None:
        super().__init__()
        self.domain = domain
        self.range = fn(domain)

    def __len__(self) -> int:
        return len(self.domain)

    def __getitem__(self, indices: Int[Array, "b"]) -> Batch:
        return self.domain[indices], self.range[indices]


class FancyDataset(Dataset):
    def __init__(self, plain_dataset: Dataset) -> None:
        self.plain_dataset = plain_dataset

    def __len__(self) -> int:
        return len(self.plain_dataset)

    def __getitem__(self, indices: Int[Array, "b"]) -> Batch:
        inputs, targets = self.plain_dataset[indices]
        return indices, inputs, targets
