import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable

from einops import pack, repeat
from jaxtyping import Float
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader

from commons import DEVICE, Array, Params, TrainingState


def make_loss_fn(model: nn.Module, loss_fn_name: str = "cross-entropy") -> Callable:
    """
    Return a pure function that computes the loss of the given model on a given sample.
    """
    if loss_fn_name == "cross-entropy":
        _loss_fn = F.cross_entropy
    elif loss_fn_name == "mse":
        _loss_fn = lambda outputs, targets: 0.5 * F.mse_loss(outputs, targets)
    else:
        raise ValueError(f"unknown loss function name '{loss_fn_name}'")

    def loss_fn(
        params: Params,
        input: Float[Array, "..."],
        target: Float[Array, "..."],
    ) -> Float[Array, ""]:
        """
        Return the loss of the model with the given parameters on the given sample.

        Assumes that the input and target tensors are not batched.
        Uses a singleton batch internally.
        """
        inputs, targets = repeat(input, "... -> 1 ..."), repeat(target, "... -> 1 ...")
        outputs = functional_call(model, params, inputs, strict=True)
        loss = _loss_fn(outputs, targets)
        return loss

    return loss_fn


def make_grad_fn(model: nn.Module, loss_fn_name: str = "cross-entropy") -> Callable:
    """
    Return a pure function that computes the gradient of the loss w.r.t. the model parameters on a given sample.
    """
    loss_fn = make_loss_fn(model, loss_fn_name)

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


def tracincp(
    training_states: list[TrainingState],
    catalyst_loader: DataLoader,
    reactant_loader: DataLoader,
    learning_rate: float = 1.0,
    loss_fn_name: str = "cross-entropy",
    device: str = DEVICE,
) -> Float[Array, "n_train n_test"]:
    """
    Return the influence matrix of the given catalyst samples on the given reactant samples w.r.t. the given model checkpoints using the TracInCP method.
    """
    catalyst_set, reactant_set = catalyst_loader.dataset, reactant_loader.dataset

    influences = torch.zeros(
        size=(len(catalyst_set), len(reactant_set)),
        device="cpu",  # too large to fit on GPU
    )

    for training_state in training_states:
        model, training_batch_indices = training_state
        # ignore training batch indices for now
        training_batch_indices = torch.arange(len(catalyst_set))
        model = model.to(device)

        params = {n: p for n, p in model.named_parameters()}
        grad_fn = make_grad_fn(model, loss_fn_name)

        # calculate the gradients of the catalyst samples
        # _, catalyst_inputs, catalyst_targets = catalyst_set[training_batch_indices]
        catalyst_data = [catalyst_set[index] for index in training_batch_indices]
        _, catalyst_inputs, catalyst_targets = zip(*catalyst_data)
        catalyst_inputs = torch.stack(catalyst_inputs).to(device)
        catalyst_targets = torch.tensor(catalyst_targets).to(device)
        catalyst_grads = grad_fn(params, catalyst_inputs, catalyst_targets)

        # calculate the inner product of the catalyst gradients with the reactant gradients
        inner_products = torch.zeros(
            size=(len(catalyst_inputs), len(reactant_set)),
            device="cpu",  # too large to fit on GPU
        )
        reactant_index_normalizer = {
            index.item(): normalized_index
            for normalized_index, index in enumerate(reactant_set.indices)
        }
        for reactant_indices, reactant_inputs, reactant_targets in reactant_loader:
            # calculate the gradients of a batch of reactant samples
            reactant_inputs = reactant_inputs.to(device)
            reactant_targets = reactant_targets.to(device)
            reactant_grads = grad_fn(params, reactant_inputs, reactant_targets)

            batch_inner_products = torch.einsum(
                "c p, r p -> c r", catalyst_grads, reactant_grads
            )
            normalized_reactant_indices = torch.tensor(
                [reactant_index_normalizer[index.item()] for index in reactant_indices]
            )
            inner_products[:, normalized_reactant_indices] = batch_inner_products.to(
                "cpu"
            )

        # update influences
        influences[training_batch_indices] += learning_rate * inner_products

    return influences
