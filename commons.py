import torch

from typing import Optional

from jaxtyping import Float, Int


Array = torch.Tensor
Params = dict[str, Array]


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
