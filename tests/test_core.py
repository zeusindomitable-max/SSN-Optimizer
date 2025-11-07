# tests/test_core.py
import pytest
import torch
import torch.nn as nn

from ssn.core import SSN


@pytest.fixture
def model():
    return nn.Linear(10, 1)


def test_ssn_step(model):
    optimizer = SSN(model.parameters(), lr=0.1)
    x = torch.randn(8, 10)
    y = torch.randn(8, 1)
    criterion = nn.MSELoss()

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    assert optimizer.state[model.weight]["step"] == 1
    assert not torch.isnan(loss)
