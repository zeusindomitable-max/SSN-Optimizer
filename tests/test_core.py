def test_ssn_stability_multiple_steps(linear_model, data_batch):
    x, y = data_batch
    criterion = nn.MSELoss()
    optimizer = SSN(linear_model.parameters(), lr=0.3)  # LR besar biar cepat turun
    losses = []

    for _ in range(10):
        optimizer.zero_grad()
        out = linear_model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss harus turun >50%
    assert losses[-1] < losses[0] * 0.5
