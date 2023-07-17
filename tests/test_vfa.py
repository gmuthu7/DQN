from typing import Dict

import pytest
import torch

from vfa.neural_network import NeuralNetworkVfa


def test_vfa_val(vfa: NeuralNetworkVfa):
    x = torch.randn(size=(32, 10))
    y = vfa.val(x)
    assert y.shape == (32, 3)


@pytest.mark.usefixtures("myseed")
def test_vfa_step(vfa: NeuralNetworkVfa):
    x = torch.zeros(size=(32, 10), dtype=torch.float32)
    target = torch.randn(size=(32, 3), dtype=torch.float32)
    pred = vfa.val(x)

    def fn(d: Dict):
        assert d["train_mean_loss"].dtype == torch.float32
        assert d["train_mean_qfn"].dtype == torch.float32
        assert torch.allclose(d["train_mean_loss"], torch.tensor(0.9896), atol=1e-04)

    vfa.step(pred, target, fn)
