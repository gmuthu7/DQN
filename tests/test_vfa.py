from typing import Dict

import pytest
import torch

from vfa.neural_network import NeuralNetworkVfa


def test_vfa_val(vfa: NeuralNetworkVfa):
    x = torch.randn(size=(32, 1))
    y = vfa.val(x)
    assert y.shape == (32, 4)


def test_scheduler(vfa):
    for i in range(3):
        lr1 = vfa.scheduler.get_last_lr()
        vfa.scheduler.step()
        lr2 = vfa.scheduler.get_last_lr()
        assert lr2 < lr1


@pytest.mark.usefixtures("seed")
def test_vfa_step(vfa: NeuralNetworkVfa):
    x = torch.zeros(size=(32, 1), dtype=torch.float32)
    target = torch.randn(size=(32, 4), dtype=torch.float32)
    pred = vfa.val(x)

    def fn(d: Dict):
        pred2 = vfa.val(x)
        assert d["train_mean_loss"].dtype == torch.float32
        assert d["train_mean_qfn"].dtype == torch.float32
        assert torch.allclose(d["train_mean_loss"], torch.tensor(1.0637), atol=1e-04)
        assert not torch.equal(pred, pred2)
        assert d["train_after_clip_grad"] <= vfa.clip_val

    vfa.step(pred, target, fn)


@pytest.mark.usefixtures("seed")
def test_class_fn_callback(vfa: NeuralNetworkVfa):
    x = torch.zeros(size=(32, 1), dtype=torch.float32)
    target = torch.randn(size=(32, 4), dtype=torch.float32)
    pred = vfa.val(x)
    a = vfa.val
    b = a(torch.tensor([[1.]]))
    c = a(torch.tensor([[1.]]))
    vfa.step(pred, target, lambda x: None)
    d = a(torch.tensor([[1.]]))
    assert torch.equal(b, c)
    assert not torch.equal(d, c)
