# ssn/preconditioner.py
import torch


def fisher_preconditioner(g, lambda_fisher=0.7, eps=1e-8):
    return lambda_fisher / (torch.sqrt(g) + eps)


def trust_region_clip(s, delta, lr, eta=1e-8):
    norm = torch.norm(s) + eta
    rho = torch.clamp_max(delta / norm, 1.0)
    return -lr * rho * s
