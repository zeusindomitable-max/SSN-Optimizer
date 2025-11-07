import pytest
import torch
import torch.nn as nn
from ssn.core import SSN

@pytest.fixture
def linear_model():
    """Simple linear model for testing."""
    return nn.Linear(10, 1)

@pytest.fixture
def data_batch():
    """Random batch."""
    return torch.randn(8, 10), torch.randn(8, 1)

def test_ssn_initialization(linear_model):
    """Test optimizer initializes correctly."""
    optimizer = SSN(linear_model.parameters(), lr=0.01)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert "step" not in optimizer.state[linear_model.weight]

def test_ssn_step_updates_parameters(linear_model, data_batch):
    """Test one optimization step updates weights."""
    x, y = data_batch
    criterion = nn.MSELoss()
    optimizer = SSN(linear_model.parameters(), lr=0.1)

    # Forward + backward
    optimizer.zero_grad()
    out = linear_model(x)
    loss = criterion(out, y)
    loss.backward()

    # Check gradients exist
    assert linear_model.weight.grad is not None

    # Step
    optimizer.step()

    # Check state updated
    state = optimizer.state[linear_model.weight]
    assert state["step"] == 1
    assert "g" in state
    assert "buffer" in state

    # Check no NaN
    assert not torch.isnan(linear_model.weight).any()

def test_ssn_no_crash_empty_step(linear_model):
    """Test optimizer doesn't crash on empty step."""
    optimizer = SSN(linear_model.parameters())
    for _ in range(5):
        optimizer.step()  # No grad, no crash
    assert optimizer.state[linear_model.weight]["step"] == 5

def test_ssn_stability_multiple_steps(linear_model, data_batch):
    """Test 100 steps: no NaN, loss decreases."""
    x, y = data_batch
    criterion = nn.MSELoss()
    optimizer = SSN(linear_model.parameters(), lr=0.05)
    losses = []

    for _ in range(100):
        optimizer.zero_grad()
        out = linear_model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss should decrease overall
    assert losses[-1] < losses[0]
    assert all(not torch.isnan(torch.tensor(l)) for l in losses)
    print(f"Stability test passed: Final loss = {losses[-1]:.6f}")
