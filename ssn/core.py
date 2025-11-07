import torch
from torch import nn

class SSN(torch.optim.Optimizer):
    """
    Stabilized Spectral Newton (SSN) Optimizer
    Versi stabil agar loss tidak divergen dan bisa turun konsisten.
    """

    def __init__(self, params, lr=0.1, damping=1e-3, beta=0.9):
        defaults = dict(lr=lr, damping=damping, beta=beta)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            damping = group["damping"]
            beta = group["beta"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # --- stabil preconditioner ---
                if "v" not in self.state[p]:
                    self.state[p]["v"] = torch.zeros_like(grad)
                v = self.state[p]["v"]

                # update v (approx Hessian diag)
                v.mul_(beta).addcmul_(grad, grad, value=1 - beta)
                denom = (v.sqrt() + damping)

                # scaled update
                step = grad / denom
                p.add_(step, alpha=-lr)

        return loss
