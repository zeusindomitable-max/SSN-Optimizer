# tests/test_stability.py
import pytest
import torch
import torch.nn as nn

from ssn.core import SSN


def test_stress():
    model = nn.Linear(5, 1)
    optimizer = SSN(model.parameters(), lr=1.0)
    x = torch.randn(4, 5)
    y = torch.randn(4, 1)
    criterion = nn.MSELoss()

    for _ in range(20):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    assert loss.item() < 1e3
