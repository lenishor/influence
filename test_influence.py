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
    model = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=1, bias=False),
        Rearrange("1 ->"),
    )
    model.zero_grad()
    params = {name: param.detach() for name, param in model.named_parameters()}
    grad_fn = make_grad_fn(model)

    input = torch.randn(size=(in_features,))
    target = torch.randn(size=())

    grad = grad_fn(params, input, target)
    assert grad["0.weight"].shape == (1, in_features)
    assert torch.allclose(grad["0.weight"].flatten(), input.flatten())
