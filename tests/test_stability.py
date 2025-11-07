import torch
import torch.nn as nn
from ssn.core import SSN

def test_stress():
    model = nn.Linear(5, 1)
    optimizer = SSN(model.parameters(), lr=0.5)
    x = torch.randn(4, 5)
    y = torch.randn(4, 1)
    criterion = nn.MSELoss()

    for _ in range(50):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    assert loss.item() < 100.0, f"Loss masih terlalu besar: {loss.item()}"
