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
