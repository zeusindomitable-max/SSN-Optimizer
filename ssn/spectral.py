# ssn/spectral.py
def spectral_correction(G, grad, k, gamma, lr):
    try:
        U, S, _ = torch.svd_lowrank(G, q=k)
        diag = S / (S**2 + gamma)
        m_hat = grad.flatten()
        correction = lr * (U @ (diag * (U.T @ m_hat))).view_as(grad)
        return correction
    except:
        return torch.zeros_like(grad)


