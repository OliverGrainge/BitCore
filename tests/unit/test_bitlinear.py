import torch
import pytest
from itertools import product

from bitcore.bnn.bitlayers.bitlinear import BitLinear
from bitcore.bnn.bitquantizer import ACT_QUANT_REGISTRY, WEIGHT_QUANT_REGISTRY

# Activation quantizers that support dense (non-convolutional) tensors.
SUPPORTED_ACT_TYPES = [
    act for act in ACT_QUANT_REGISTRY.keys() if act != "ai8pc"
]
WEIGHT_TYPES = list(WEIGHT_QUANT_REGISTRY.keys())

QUANT_TYPES = ["none"]
QUANT_TYPES.extend(
    f"{act}_{weight}"
    for act, weight in product(SUPPORTED_ACT_TYPES, WEIGHT_TYPES)
)
# Deduplicate while preserving order.
QUANT_TYPES = list(dict.fromkeys(QUANT_TYPES))


@pytest.fixture(params=QUANT_TYPES, ids=lambda qt: qt)
def bitlinear_layer(request):
    # A small BitLinear layer for the requested quantization scheme.
    return BitLinear(12, 24, quant_type=request.param)


def test_forward_train_2d(bitlinear_layer): 
    bitlinear_layer.train()
    x = torch.randn(2, 12)
    y = bitlinear_layer(x)
    assert y.shape == (2, 24)
    assert not torch.isnan(y).any()

def test_forward_eval_2d(bitlinear_layer): 
    bitlinear_layer.eval()
    x = torch.randn(2, 12)
    y = bitlinear_layer(x)
    assert y.shape == (2, 24)
    assert not torch.isnan(y).any()

def test_forward_train_3d(bitlinear_layer): 
    bitlinear_layer.train()
    x = torch.randn(2, 8, 12)
    y = bitlinear_layer(x)
    assert y.shape == (2, 8, 24)
    assert not torch.isnan(y).any()

def test_forward_eval_3d(bitlinear_layer): 
    bitlinear_layer.eval()
    x = torch.randn(2, 8, 12)
    y = bitlinear_layer(x)
    assert y.shape == (2, 8, 24)
    assert not torch.isnan(y).any()


def test_backward_train_2d(bitlinear_layer):
    bitlinear_layer.train()
    x = torch.randn(2, 12, requires_grad=True)
    y = bitlinear_layer(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()
    assert bitlinear_layer.weight.grad is not None
    assert bitlinear_layer.weight.grad.shape == bitlinear_layer.weight.shape
    assert not torch.isnan(bitlinear_layer.weight.grad).any()
    if bitlinear_layer.bias is not None:
        assert bitlinear_layer.bias.grad is not None
        assert bitlinear_layer.bias.grad.shape == bitlinear_layer.bias.shape
        assert not torch.isnan(bitlinear_layer.bias.grad).any()


def test_backward_eval_2d(bitlinear_layer):
    bitlinear_layer.eval()
    x = torch.randn(2, 12, requires_grad=True)
    y = bitlinear_layer(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()
    assert bitlinear_layer.weight.grad is not None
    assert bitlinear_layer.weight.grad.shape == bitlinear_layer.weight.shape
    assert not torch.isnan(bitlinear_layer.weight.grad).any()
    if bitlinear_layer.bias is not None:
        assert bitlinear_layer.bias.grad is not None
        assert bitlinear_layer.bias.grad.shape == bitlinear_layer.bias.shape
        assert not torch.isnan(bitlinear_layer.bias.grad).any()


def test_backward_train_3d(bitlinear_layer):
    bitlinear_layer.train()
    x = torch.randn(2, 8, 12, requires_grad=True)
    y = bitlinear_layer(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()
    assert bitlinear_layer.weight.grad is not None
    assert bitlinear_layer.weight.grad.shape == bitlinear_layer.weight.shape
    assert not torch.isnan(bitlinear_layer.weight.grad).any()
    if bitlinear_layer.bias is not None:
        assert bitlinear_layer.bias.grad is not None
        assert bitlinear_layer.bias.grad.shape == bitlinear_layer.bias.shape
        assert not torch.isnan(bitlinear_layer.bias.grad).any()


def test_backward_eval_3d(bitlinear_layer):
    bitlinear_layer.eval()
    x = torch.randn(2, 8, 12, requires_grad=True)
    y = bitlinear_layer(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()
    assert bitlinear_layer.weight.grad is not None
    assert bitlinear_layer.weight.grad.shape == bitlinear_layer.weight.shape
    assert not torch.isnan(bitlinear_layer.weight.grad).any()
    if bitlinear_layer.bias is not None:
        assert bitlinear_layer.bias.grad is not None
        assert bitlinear_layer.bias.grad.shape == bitlinear_layer.bias.shape
        assert not torch.isnan(bitlinear_layer.bias.grad).any()


