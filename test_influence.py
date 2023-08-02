import pytest
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from influence import Array, make_loss_fn, make_grad_fn


def make_linear_model_from_weights(weights: Array) -> nn.Module:
    """
    Return a linear model with scalar outputs using the given weights.
    """
    out_features, in_features = weights.shape
    assert out_features == 1

    linear_layer = nn.Linear(
        in_features=in_features, out_features=out_features, bias=False
    )
    linear_layer.weight.data = weights
    model = nn.Sequential(linear_layer, Rearrange("1 ->"))
    return model


def test_make_loss_fn():
    weights = torch.tensor([[1.0, 2.0, 3.0]])  # (1, 3)
    input = torch.tensor([1.0, 5.0, 3.0])  # (3,)
    target = torch.tensor(18.0)  # ()

    model = make_linear_model_from_weights(weights)
    params = {name: param.detach() for name, param in model.named_parameters()}

    # expected:
    # output = weights @ input = 1 * 1 + 2 * 5 + 3 * 3 = 20
    # loss = 0.5 * (output - target) ** 2 = 0.5 * (20 - 18) ** 2 = 2
    expected_loss = torch.tensor(2.0)

    loss_fn = make_loss_fn(model)
    loss = loss_fn(params, input, target)
    assert loss.shape == ()
    assert torch.allclose(loss, expected_loss)


def test_make_grad_fn():
    """Test 'make_grad_fn' on a linear model."""
    weights = torch.tensor([[1.0, 2.0, 3.0]])  # (1, 3)
    input = torch.tensor([1.0, 5.0, 3.0])  # (3,)
    target = torch.tensor(18.0)  # ()

    model = make_linear_model_from_weights(weights)
    params = {name: param.detach() for name, param in model.named_parameters()}

    # expected:
    # output = weights @ input = 1 * 1 + 2 * 5 + 3 * 3 = 20
    # loss = 0.5 * (output - target) ** 2 = 0.5 * (20 - 18) ** 2 = 2
    # grad = dloss/dweights = dloss/doutput * doutput/dweights
    # dloss/doutput = output - target = 20 - 18 = 2
    # doutput/dweights = input = [1, 5, 3]
    # grad = dloss/doutput * doutput/dweights = (output - target) * input = 2 * [1, 5, 3] = [2, 10, 6]
    expected_grad = torch.tensor([2.0, 10.0, 6.0])

    grad_fn = make_grad_fn(model)
    grad = grad_fn(params, input, target)
    assert grad.shape == (3,)
    assert torch.allclose(grad, expected_grad)
