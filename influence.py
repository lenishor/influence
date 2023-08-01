import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any

from jaxtyping import Float
from torch.func import functional_call, grad


Array = torch.Tensor
PyTree = dict[str, Any]


def make_loss_fn(model: nn.Module) -> callable:
    """
    Return a pure function that computes the loss of the given model on a given sample.
    """

    def loss_fn(
        params: PyTree, input: Float[Array, "..."], target: Float[Array, ""]
    ) -> Float[Array, ""]:
        """
        Return the loss of the model with the given parameters on the given sample.

        Assumes that the input and target tensors are not batched.
        """
        output = functional_call(model, params, input)
        loss = 0.5 * F.mse_loss(output, target)
        return loss

    return loss_fn


def make_grad_fn(model: nn.Module) -> callable:
    """
    Return a pure function that computes the gradient of the loss w.r.t. the model parameters on a given sample.
    """
    loss_fn = make_loss_fn(model)
    return grad(loss_fn, argnums=0)
