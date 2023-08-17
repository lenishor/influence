import pytest
import torch
import torch.nn as nn

from einops import reduce
from einops.layers.torch import Rearrange

from commons import DEVICE, Array
from data import Batch
from influence import make_loss_fn, make_grad_fn, get_influences


def make_linear_model_from_weights(
    weights: Array,
    batching: bool = False,
    device: str = DEVICE,
) -> nn.Module:
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

    return model.to(device=device)


def test_make_loss_fn(device: str = DEVICE):
    weights = torch.tensor([[1.0, 2.0, 3.0]], device=device)  # (out=1, in=3)
    input = torch.tensor([1.0, 5.0, 3.0], device=device)  # (in=3,)
    target = torch.tensor(18.0, device=device)  # ()

    model = make_linear_model_from_weights(weights, batching=False, device=device)
    params = {name: param.detach() for name, param in model.named_parameters()}

    # expected:
    # output = weights @ input = 1 * 1 + 2 * 5 + 3 * 3 = 20
    # loss = 0.5 * (output - target) ** 2 = 0.5 * (20 - 18) ** 2 = 2
    expected_loss = torch.tensor(2.0, device=device)

    loss_fn = make_loss_fn(model, loss_fn_name="mse")
    loss = loss_fn(params, input, target)
    assert loss.shape == ()
    assert torch.allclose(loss, expected_loss)


def test_make_grad_fn_manual(device: str = DEVICE):
    """
    Test 'make_grad_fn' on a handwritten example.
    """
    weights = torch.tensor([[1.0, 2.0, 3.0]], device=device)  # (out=1, in=3)
    input = torch.tensor([[1.0, 5.0, 3.0]], device=device)  # (batch=1, in=3)
    target = torch.tensor([18.0], device=device)  # (batch=1,)

    model = make_linear_model_from_weights(weights, batching=False, device=device)
    params = {name: param.detach() for name, param in model.named_parameters()}

    # expected:
    # output = weights @ input = 1 * 1 + 2 * 5 + 3 * 3 = 20
    # loss = 0.5 * (output - target) ** 2 = 0.5 * (20 - 18) ** 2 = 2
    # grad = dloss/dweights = dloss/doutput * doutput/dweights
    # dloss/doutput = output - target = 20 - 18 = 2
    # doutput/dweights = input = [1, 5, 3]
    # grad = dloss/doutput * doutput/dweights = (output - target) * input = 2 * [1, 5, 3] = [2, 10, 6]
    expected_grad = torch.tensor([[2.0, 10.0, 6.0]], device=device)

    grad_fn = make_grad_fn(model, loss_fn_name="mse")
    grad = grad_fn(params, input, target)
    assert grad.shape == (1, 3)
    assert torch.allclose(grad, expected_grad)


@pytest.mark.repeat(1)
@pytest.mark.parametrize("batch_size, in_features", [(1, 3), (5, 10), (10, 10)])
def test_make_grad_fn_auto(batch_size: int, in_features: int, device: str = DEVICE):
    """
    Test 'make_grad_fn' on random linear models.
    """
    student_weights = torch.randn(size=(1, in_features), device=device)
    teacher_weights = torch.randn(size=(1, in_features), device=device)
    student_model = make_linear_model_from_weights(
        student_weights, batching=True, device=device
    )
    teacher_model = make_linear_model_from_weights(
        teacher_weights, batching=True, device=device
    )

    inputs = torch.randn(size=(batch_size, in_features), device=device)
    outputs = student_model(inputs)
    targets = teacher_model(inputs)
    expected_grads = (outputs - targets) * inputs

    params = {name: param.detach() for name, param in student_model.named_parameters()}
    grad_fn = make_grad_fn(student_model, loss_fn_name="mse")
    grads = grad_fn(params, inputs, targets)
    assert grads.shape == (batch_size, in_features)
    assert torch.allclose(grads, expected_grads)


@pytest.mark.repeat(1)
@pytest.mark.parametrize("batch_size, in_features", [(5, 10), (10, 10)])
def test_get_influences(batch_size: int, in_features: int, device: str = DEVICE):
    student_weights = torch.randn(size=(1, in_features), device=device)
    teacher_weights = torch.randn(size=(1, in_features), device=device)
    student_model = make_linear_model_from_weights(
        student_weights, batching=True, device=device
    )
    teacher_model = make_linear_model_from_weights(
        teacher_weights, batching=True, device=device
    )

    inputs = torch.randn(size=(batch_size, in_features), device=device)
    outputs = student_model(inputs)
    targets = teacher_model(inputs)
    errors = reduce(outputs - targets, "b 1 -> b", reduction="sum")
    similarities = torch.einsum("i d, j d -> i j", inputs, inputs)
    expected_influences = torch.einsum("i, i j, j -> i j", errors, similarities, errors)

    samples = Batch(inputs, targets)
    influences = get_influences(
        [student_model],
        samples,
        samples,
        loss_fn_name="mse",
        device=device,
    )
    assert influences.shape == (batch_size, batch_size)
    assert torch.allclose(influences, expected_influences)
