import torch
import torch.nn as nn
import pytest
from ssn.core import SSN

@pytest.fixture
def linear_model():
    return nn.Linear(10, 1)

@pytest.fixture
def data_batch():
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    return x, y

def test_ssn_stability_multiple_steps(linear_model, data_batch):
    x, y = data_batch
    criterion = nn.MSELoss()
    optimizer = SSN(linear_model.parameters(), lr=0.3)
    losses = []

    for _ in range(15):
        optimizer.zero_grad()
        out = linear_model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5, f"Loss tidak turun cukup: {losses}"
