import torch
from ssn import SSN

def test_ssn_step():
    model = torch.nn.Linear(10, 1)
    optimizer = SSN(model.parameters(), lr=0.01)
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    criterion = torch.nn.MSELoss()

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    assert optimizer.state[model.weight]["step"] == 1
    print("SSN step test passed.")
