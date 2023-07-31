import pytest

import torch
import torch.nn as nn

from einops import repeat

from influence import Batch, make_pure_loss_fn, tracincp


def test_make_pure_loss_fn(in_features=10):
    # test if loss is zero if model is perfect
    model = nn.Linear(in_features=in_features, out_features=1)
    params = {name: param.detach() for name, param in model.named_parameters()}
    loss_fn = make_pure_loss_fn(model, loss_fn_name="l2")

    inputs = torch.randn(size=(1, in_features))
    targets = model(inputs)
    loss = loss_fn(params, inputs, targets)

    assert torch.allclose(loss, torch.zeros_like(loss))


@pytest.mark.repeat(10)
def test_tracincp(batch_size=10, in_features=20):
    model = nn.Linear(in_features=in_features, out_features=1, bias=True)
    true_model = nn.Linear(in_features=in_features, out_features=1, bias=False)

    inputs = torch.randn(size=(batch_size, in_features))
    targets = true_model(inputs)
    indices = torch.arange(batch_size)
    data = Batch(inputs=inputs, targets=targets, indices=indices)

    with torch.no_grad():
        delta_outputs = (model(inputs) - true_model(inputs)).squeeze()
    similarities = torch.einsum("m d, n d -> m n", inputs, inputs)

    influences = tracincp([model], data, data, loss_fn_name="l2")

    expected_influences = (
        repeat(delta_outputs, "m -> m n", n=batch_size)
        * similarities
        * repeat(delta_outputs, "n -> m n", m=batch_size)
    )

    assert torch.allclose(influences, expected_influences, rtol=1e-1)
