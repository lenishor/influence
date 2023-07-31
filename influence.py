import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Iterable

from einops import pack, rearrange
from jaxtyping import Float
from torch.func import functional_call, grad, vmap


Array = torch.Tensor


@dataclass
class Batch:
    inputs: Float[Array, "b ..."]
    targets: Float[Array, "b ..."]
    indices: Float[Array, "b"]

    def __len__(self):
        return len(self.indices)


def make_pure_loss_fn(
    model: nn.Module, loss_fn_name: str = "cross-entropy"
) -> callable:
    match loss_fn_name:
        case "cross-entropy":
            loss_fn = F.cross_entropy
        case "l2":
            loss_fn = lambda *args, **kwargs: 0.5 * F.mse_loss(*args, **kwargs)
        case _:
            assert ValueError(
                f"loss_fn_name must be one of 'cross-entropy' or 'l2', got {loss_fn_name}"
            )

    def pure_loss_fn(params, inputs, targets):
        outputs = functional_call(model, params, inputs)
        loss = loss_fn(outputs, targets)
        return loss

    return pure_loss_fn


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
        loss_fn = make_pure_loss_fn(model, loss_fn_name)
        grad_fn = lambda params, inputs, targets: reshape_grads(
            vmap(grad(loss_fn), in_dims=(None, 0, 0))(params, inputs, targets)
        )

        source_grads = grad_fn(params, source_data.inputs, source_data.targets)
        target_grads = grad_fn(params, target_data.inputs, target_data.targets)
        inner_prods = torch.einsum("s p, t p -> s t", source_grads, target_grads)
        influences += inner_prods

    return influences
