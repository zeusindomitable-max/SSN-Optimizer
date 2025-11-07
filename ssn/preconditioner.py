# ssn/preconditioner.py
def fisher_preconditioner(g, lambda_fisher, eps=1e-8):
    return lambda_fisher / (g.sqrt() + eps)

def trust_region_clip(s, delta, lr):
    rho = min(1.0, delta / (s.norm() + 1e-8))
    return -lr * rho * s
