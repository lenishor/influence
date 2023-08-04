import pytest
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from influence import Array, make_loss_fn, make_grad_fn, get_influences


def make_linear_model_from_weights(weights: Array, batching: bool = False) -> nn.Module:
    """
    Return a linear model with scalar outputs using the given weights.
    """
    out_features, in_features = weights.shape
    assert out_features == 1

    linear_layer = nn.Linear(
        in_features=in_features, out_features=out_features, bias=False
    )
    linear_layer.weight.data = weights

    if not batching:
        model = nn.Sequential(linear_layer, Rearrange("... 1 -> ..."))
    else:
        model = linear_layer
    return model


def test_make_loss_fn():
    weights = torch.tensor([[1.0, 2.0, 3.0]])  # (1, 3)
    input = torch.tensor([1.0, 5.0, 3.0])  # (3,)
    target = torch.tensor(18.0)  # ()

    model = make_linear_model_from_weights(weights, batching=False)
    params = {name: param.detach() for name, param in model.named_parameters()}

    # expected:
    # output = weights @ input = 1 * 1 + 2 * 5 + 3 * 3 = 20
    # loss = 0.5 * (output - target) ** 2 = 0.5 * (20 - 18) ** 2 = 2
    expected_loss = torch.tensor(2.0)

    loss_fn = make_loss_fn(model)
    loss = loss_fn(params, input, target)
    assert loss.shape == ()
    assert torch.allclose(loss, expected_loss)


def test_make_grad_fn_manual():
    """
    Test 'make_grad_fn' on a handwritten example.
    """
    weights = torch.tensor([[1.0, 2.0, 3.0]])  # (out=1, dim=3)
    input = torch.tensor([[1.0, 5.0, 3.0]])  # (batch=1, dim=3)
    target = torch.tensor([18.0])  # (batch=1,)

    model = make_linear_model_from_weights(weights, batching=False)
    params = {name: param.detach() for name, param in model.named_parameters()}

    # expected:
    # output = weights @ input = 1 * 1 + 2 * 5 + 3 * 3 = 20
    # loss = 0.5 * (output - target) ** 2 = 0.5 * (20 - 18) ** 2 = 2
    # grad = dloss/dweights = dloss/doutput * doutput/dweights
    # dloss/doutput = output - target = 20 - 18 = 2
    # doutput/dweights = input = [1, 5, 3]
    # grad = dloss/doutput * doutput/dweights = (output - target) * input = 2 * [1, 5, 3] = [2, 10, 6]
    expected_grad = torch.tensor([[2.0, 10.0, 6.0]])

    grad_fn = make_grad_fn(model)
    grad = grad_fn(params, input, target)
    assert grad.shape == (1, 3)
    assert torch.allclose(grad, expected_grad)


@pytest.mark.repeat(1)
@pytest.mark.parametrize("batch_size, in_features", [(1, 3), (5, 10), (10, 10)])
def test_make_grad_fn_auto(batch_size: int, in_features: int):
    """
    Test 'make_grad_fn' on random linear models.
    """
    weights = torch.randn(size=(1, in_features))
    true_weights = torch.randn(size=(1, in_features))
    model = make_linear_model_from_weights(weights, batching=True)
    true_model = make_linear_model_from_weights(true_weights, batching=True)

    inputs = torch.randn(size=(batch_size, in_features))
    outputs = model(inputs)
    targets = true_model(inputs)
    expected_grads = (outputs - targets) * inputs

    params = {name: param.detach() for name, param in model.named_parameters()}
    grad_fn = make_grad_fn(model)
    grads = grad_fn(params, inputs, targets)
    assert grads.shape == (batch_size, in_features)
    assert torch.allclose(grads, expected_grads)


@pytest.mark.repeat(1)
@pytest.mark.parametrize("batch_size, in_features", [(1, 3), (5, 10), (10, 10)])
def test_get_influences(batch_size: int, in_features: int):
    weights = torch.randn(size=(1, in_features))
    true_weights = torch.randn(size=(1, in_features))
    model = make_linear_model_from_weights(weights)
    true_model = make_linear_model_from_weights(true_weights)

    inputs = torch.randn(size=(batch_size, in_features))
    outputs = model(inputs)
    targets = true_model(inputs)
    errors = outputs - targets
    similarities = torch.einsum("i d, j d -> i j", inputs, inputs)
    expected_influences = torch.einsum("i, i j, j -> i j", errors, similarities, errors)

    samples = [(input, target) for input, target in zip(inputs, targets)]
    influences = get_influences([model], samples, samples)
    assert influences.shape == (batch_size, batch_size)
    assert torch.allclose(influences, expected_influences)
