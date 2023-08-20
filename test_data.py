import pytest

import torch

from commons import DEVICE
from data import FancyDataset, FunctionDataset, UnionDataset


@pytest.mark.repeat(10)
def test_union_dataset(
    left_size: int = 32,
    right_size: int = 32,
    batch_size: int = 16,
    device: str = DEVICE,
) -> None:
    """
    Test if UnionDataset works as expected on a special case.

    The left dataset is of the form `[0, ..., left_size - 1]` and the right dataset is of the form `[left_size, ..., right_size - 1]`.
    The union dataset should be of the form `[0, ..., right_size - 1]`.
    """
    left = FancyDataset(
        FunctionDataset(
            lambda x: x, torch.arange(start=0, end=left_size, device=device)
        )
    )
    right = FancyDataset(
        FunctionDataset(
            lambda x: x, torch.arange(start=left_size, end=right_size, device=device)
        )
    )
    union = UnionDataset(left, right)

    indices = torch.randint(low=0, high=right_size, size=(batch_size,), device=device)
    indices, inputs, _ = union[indices]
    assert torch.all(indices == inputs)
