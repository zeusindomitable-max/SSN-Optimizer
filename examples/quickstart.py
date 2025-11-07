import torch
import torch.nn as nn
from ssn import SSN

model = nn.Linear(1000, 1)
optimizer = SSN(model.parameters(), lr=1e-3)

x = torch.randn(32, 1000)
y = torch.randn(32, 1)
criterion = nn.MSELoss()

for _ in range(100):
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item():.6f}")
