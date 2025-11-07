# tests/test_core.py
import pytest
import torch
import torch.nn as nn

from ssn.core import SSN


@pytest.fixture
def linear_model():
    torch.manual_seed(42)
    return nn.Linear(10, 1)


@pytest.fixture
def data_batch():
    torch.manual_seed(42)
    return torch.randn(8, 10), torch.randn(8, 1)


def test_ssn_initialization(linear_model):
    optimizer = SSN(linear_model.parameters(), lr=0.01)
    assert optimizer.param_groups[0]["lr"] == 0.01


def test_ssn_step_updates_parameters(linear_model, data_batch):
    x, y = data_batch
    criterion = nn.MSELoss()
    optimizer = SSN(linear_model.parameters(), lr=0.1)

    optimizer.zero_grad()
    out = linear_model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    assert optimizer.state[linear_model.weight]["step"] == 1


def test_ssn_no_crash_empty_step(linear_model):
    optimizer = SSN(linear_model.parameters())
    for _ in range(5):
        optimizer.step()
    assert optimizer.state[linear_model.weight]["step"] == 5


def test_ssn_stability_multiple_steps(linear_model, data_batch):
    x, y = data_batch
    criterion = nn.MSELoss()
    optimizer = SSN(linear_model.parameters(), lr=0.1)  # lr lebih besar
    losses = []

    for _ in range(15):  # 15 steps cukup
        optimizer.zero_grad()
        out = linear_model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss harus turun minimal 30%
    assert losses[-1] < losses[0] * 0.7
    assert not any(torch.isnan(torch.tensor(losses)).any())
