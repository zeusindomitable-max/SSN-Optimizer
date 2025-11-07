import pytest
import torch
import torch.nn as nn
from ssn.core import SSN

def test_stress_large_model():
    """Test on large model (1M params) — no OOM."""
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1)
    )
    optimizer = SSN(model.parameters(), lr=1e-3)
    x = torch.randn(16, 1024)
    y = torch.randn(16, 1)
    criterion = nn.MSELoss()

    for _ in range(10):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    assert not torch.isnan(loss)
    print("Stress test passed: 1M+ params, 10 steps.")

def test_numerical_stability_high_lr():
    """Test with high LR — should clip, not explode."""
    model = nn.Linear(5, 1)
    optimizer = SSN(model.parameters(), lr=10.0, delta=0.1)  # High LR
    x = torch.randn(4, 5)
    y = torch.randn(4, 1)
    criterion = nn.MSELoss()

    for _ in range(20):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    assert loss.item() < 1e3  # Should not explode
    print("High LR stability test passed.")
