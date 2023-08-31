from typing import Any, Callable

import torch

from jaxtyping import Float, Int
from torch.utils.data import Dataset

from commons import Array


Batch = tuple[
    Int[Array, "b"],  # indices
    Float[Array, "b ..."],  # inputs
    Float[Array, "b ..."],  # targets
]
Indices = int | Int[Array, "n"]


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


class IndexedDataset(Dataset):
    def __init__(self, base_dataset: Dataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, indices: Int[Array, "b"]) -> Batch:
        inputs, targets = self.base_dataset[indices]
        return indices, inputs, targets


class IndexedSubset(IndexedDataset):
    """
    A subset of a dataset that is indexed by a tensor of indices.

    Returns the index of samples into the subset, not into the base dataset.
    """

    def __init__(self, dataset: IndexedDataset, indices: Int[Array, "n"]) -> None:
        self.dataset = dataset
        self.dataset_indices = indices  # indices into the base dataset

    def __len__(self) -> int:
        return len(self.dataset_indices)

    def __getitem__(self, indices: Indices) -> Batch:
        _, inputs, targets = self.dataset[self.dataset_indices[indices]]
        return indices, inputs, targets
