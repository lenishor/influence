import torch

from jaxtyping import Float, Int

from influence import Array


Array = torch.Tensor
Params = dict[str, Array]


class Batch:
    """
    inputs: Float[Array, "b ..."]
    targets: Float[Array, "b ..."]
    indices: Int[Array, "b"]
    batch_size: int
    """

    def __init__(
        self,
        inputs: Float[Array, "b ..."],
        targets: Float[Array, "b ..."],
        indices: Int[Array, "b"],
    ) -> None:
        # batch size must be the same for all tensors
        assert len(inputs) == len(targets) == len(indices)
        self.inputs = inputs
        self.targets = targets
        self.indices = indices
        self.batch_size = len(indices)

    def __len__(self) -> int:
        return self.batch_size
