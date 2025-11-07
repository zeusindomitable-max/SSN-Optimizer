# ssn/spectral.py
import torch


def spectral_correction(G, grad, k, gamma, lr, eps=1e-8):
    try:
        U, S, _ = torch.svd_lowrank(G, q=k)
        damping = S / (S**2 + gamma + eps)
        correction = U @ (damping * (U.T @ grad.flatten()))
        return lr * correction.view_as(grad)
    except Exception:
        return torch.zeros_like(grad)
