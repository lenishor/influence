import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Iterable

from einops import pack, repeat
from jaxtyping import Float
from torch.func import functional_call, grad, vmap

from commons import Array, Params, Batch


def make_loss_fn(model: nn.Module) -> callable:
    """
    Return a pure function that computes the loss of the given model on a given sample.
    """

    def loss_fn(
        params: Params, input: Float[Array, "..."], target: Float[Array, "..."]
    ) -> Float[Array, ""]:
        """
        Return the loss of the model with the given parameters on the given sample.

        Assumes that the input and target tensors are not batched.
        Uses a singleton batch internally.
        """
        inputs, targets = repeat(input, "... -> 1 ..."), repeat(target, "... -> 1 ...")
        outputs = functional_call(model, params, inputs, strict=True)
        loss = 0.5 * F.mse_loss(outputs, targets)
        return loss

    return loss_fn


def make_grad_fn(model: nn.Module) -> callable:
    """
    Return a pure function that computes the gradient of the loss w.r.t. the model parameters on a given sample.
    """
    loss_fn = make_loss_fn(model)

    def grad_fn(
        params: Params, input: Float[Array, "..."], target: Float[Array, ""]
    ) -> Float[Array, "..."]:
        """
        Return the gradient of the loss w.r.t. the model parameters on the given sample.

        Assumes that the input and target tensors are not batched. The gradient is returned as a flattened tensor.
        """
        grad_dict = grad(loss_fn, argnums=0)(params, input, target)
        _grad, _ = pack(list(grad_dict.values()), "*")
        return _grad

    return vmap(grad_fn, in_dims=(None, 0, 0))


def get_influences(
    models: Iterable[nn.Module],
    train_samples: Batch,
    test_samples: Batch,
    learning_rate: float = 1.0,
) -> Float[Array, "n_train n_test"]:
    """
    Return the influence matrix of the given train samples on the given test samples w.r.t. the given model checkpoints using the TracInCP method.

    Assumes that the checkpoints are taken once every epoch.
    """
    influences = torch.zeros(size=(len(train_samples), len(test_samples)))

    for model in models:
        params = {name: param.detach() for name, param in model.named_parameters()}
        grad_fn = make_grad_fn(model)
        train_grads = grad_fn(params, train_samples.inputs, train_samples.targets)
        test_grads = grad_fn(params, test_samples.inputs, test_samples.targets)
        influences += learning_rate * torch.einsum(
            "i p, j p -> i j", train_grads, test_grads
        )

    return influences
