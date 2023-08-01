import pytest
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from influence import make_loss_fn, make_grad_fn


@pytest.mark.parametrize("in_features", [1, 5, 10])
def test_make_loss_fn(in_features):
    model = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=1, bias=False),
        Rearrange("1 ->"),
    )
    model.zero_grad()
    params = {name: param.detach() for name, param in model.named_parameters()}
    loss_fn = make_loss_fn(model)

    input = torch.randn(size=(in_features,))
    target = torch.randn(size=())

    loss = loss_fn(params, input, target)
    assert loss.shape == ()

    for param in model.parameters():
        assert param.grad is None


@pytest.mark.parametrize("in_features", [1, 5, 10])
def test_make_grad_fn(in_features):
    """
    Test 'make_grad_fn' on a linear model.
    """
    model = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=1, bias=False),
        Rearrange("1 ->"),
    )
    model.zero_grad()
    params = {name: param.detach() for name, param in model.named_parameters()}
    grad_fn = make_grad_fn(model)

    input = torch.randn(size=(in_features,))
    output = model(input)
    target = torch.tensor(0.0)

    # loss = 0.5 * output ** 2
    # dloss/doutput = output
    # dloss/dparams = dloss/doutput * doutput/dparams = output * input
    grads = grad_fn(params, input, target)
    assert grads.shape == (in_features,)
    assert torch.allclose(grads, output * input)
