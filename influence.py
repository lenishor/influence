import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Any, Iterable

from einops import pack
from jaxtyping import Float
from torch.func import functional_call, grad, vmap


Array = torch.Tensor
PyTree = dict[str, Any]


@dataclass
class Batch:
    inputs: Float[Array, "b ..."]
    targets: Float[Array, "b 1"]
    indices: Float[Array, "b"]

    def __len__(self):
        return len(self.indices)


def make_loss_fn(model: nn.Module, loss_fn_name: str = "cross-entropy") -> callable:
    match loss_fn_name:
        case "cross-entropy":
            raise NotImplementedError
        case "l2":
            _loss_fn = lambda outputs, targets: 0.5 * ((outputs - targets) ** 2).mean()
        case _:
            raise ValueError(
                f"loss_fn_name must be one of 'cross-entropy' or 'l2', got {loss_fn_name}"
            )

    def loss_fn(
        params: PyTree, inputs: Float[Array, "1 ..."], targets: Float[Array, "1 1"]
    ) -> Float[Array, ""]:
        # ensure that batch size is 1, as we will vmap over the batch dimension
        batch_size, *_ = inputs.shape
        assert batch_size == 1

        outputs = functional_call(model, params, inputs, strict=True)
        loss = _loss_fn(outputs, targets).squeeze()
        return loss

    return loss_fn


def reshape_grads(grads: dict[str, Array]) -> Array:
    reshaped_grads, _ = pack(list(grads.values()), "b *")
    return reshaped_grads


def tracincp(
    models: Iterable[nn.Module],
    source_data: Batch,
    target_data: Batch,
    loss_fn_name: str = "cross-entropy",
) -> Array:
    """
    Return the influence of the source data on the target data using the TracInCP method.

    Assumes that the model checkpoints are taken once every epoch and that the task is supervised learning using cross-entropy loss.
    """
    influences = torch.zeros(size=(len(source_data), len(target_data)))

    for model in models:
        params = {name: param.detach() for name, param in model.named_parameters()}
        loss_fn = make_loss_fn(model, loss_fn_name)
        grad_fn = lambda params, inputs, targets: reshape_grads(
            vmap(grad(loss_fn), in_dims=(None, 0, 0))(params, inputs, targets)
        )

        source_grads = grad_fn(params, source_data.inputs, source_data.targets)
        target_grads = grad_fn(params, target_data.inputs, target_data.targets)
        inner_prods = torch.einsum("s p, t p -> s t", source_grads, target_grads)
        influences += inner_prods

    return influences
