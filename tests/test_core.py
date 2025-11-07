# tests/test_core.py
import pytest
import torch
import torch.nn as nn

from ssn.core import SSN


@pytest.fixture
def linear_model():
    model = nn.Linear(10, 1)
    torch.manual_seed(42)
    return model


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
    assert not torch.isnan(linear_model.weight).any()


def test_ssn_no_crash_empty_step(linear_model):
    optimizer = SSN(linear_model.parameters())
    for _ in range(5):
        optimizer.step()  # no grad
    assert optimizer.state[linear_model.weight]["step"] == 5


def test_ssn_stability_multiple_steps(linear_model, data_batch):
    x, y = data_batch
    criterion = nn.MSELoss()
    optimizer = SSN(linear_model.parameters(), lr=0.05)
    losses = []

    for _ in range(20):  # 20 steps cukup untuk CI
        optimizer.zero_grad()
        out = linear_model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss harus turun minimal 10%
    assert losses[-1] < losses[0] * 0.9
    assert all(not torch.isnan(torch.tensor(l)) for l in losses)
