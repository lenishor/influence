import torch

from jaxtyping import Float, Int
from torch.utils.data import Dataset

from commons import DEVICE, Array


Batch = tuple[
    Int[Array, "b"],  # indices
    Float[Array, "b ..."],  # inputs
    Float[Array, "b ..."],  # targets
]


class FancyDataset(Dataset):
    def __init__(self, plain_dataset: Dataset, device: str = DEVICE) -> None:
        self.plain_dataset = plain_dataset
        self.device = device

    def __len__(self) -> int:
        return len(self.plain_dataset)

    def __getitem__(self, indices: Int[Array, "b"]) -> Batch:
        inputs, targets = self.plain_dataset[indices]

        indices = indices.to(self.device)
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        return indices, inputs, targets


class UnionDataset(FancyDataset):
    def __init__(self, left: FancyDataset, right: FancyDataset) -> None:
        if not left.device == right.device:
            raise ValueError("datasets must use same device")
        self.left = left
        self.right = right

    def __len__(self) -> int:
        return len(self.left) + len(self.right)

    def __getitem__(self, indices: Int[Array, "b"]) -> Batch:
        # compute masks over indices
        left_mask = indices < len(self.left)
        right_mask = ~left_mask

        # compute indices into left and right datasets
        left_indices = indices[left_mask]
        right_indices = indices[right_mask] - len(self.left)

        # get batches from left and right datasets
        _, left_inputs, left_targets = self.left[left_indices]
        _, right_inputs, right_targets = self.right[right_indices]

        # get shape of each input and target
        # left dataset is arbitrarily chosen to be the reference
        _, *input_shape = left_inputs.shape
        _, *target_shape = left_targets.shape

        # initialize empty tensors for inputs and targets
        inputs = torch.empty(size=(len(indices), *input_shape))
        targets = torch.empty(size=(len(indices), *target_shape))

        # fill in inputs and targets
        inputs[left_mask] = left_inputs
        inputs[right_mask] = right_inputs
        targets[left_mask] = left_targets
        targets[right_mask] = right_targets

        return indices, inputs, targets
