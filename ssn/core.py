# ssn/core.py
import torch
from torch.optim import Optimizer
from collections import deque
from .preconditioner import fisher_preconditioner, trust_region_clip
from .spectral import spectral_correction

class SSN(Optimizer):
    """Spectral-Sketch Natural (SSN) Optimizer."""
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        lambda_fisher: float = 0.7,
        beta_g: float = 0.99,
        delta: float = 1.0,
        K: int = 100,
        k: int = 32,
        B: int = 64,
        gamma: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr, lambda_fisher=lambda_fisher, beta_g=beta_g,
            delta=delta, K=K, k=k, B=B, gamma=gamma, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["g"] = torch.zeros_like(p)
                    state["buffer"] = deque(maxlen=group["B"])

                g = state["g"]
                step = state["step"] = state["step"] + 1

                # Fisher preconditioning
                g.mul_(group["beta_g"]).addcmul_(grad, grad, value=1 - group["beta_g"])
                p = fisher_preconditioner(g, group["lambda_fisher"])

                # Trust-region
                s = p * grad
                delta = trust_region_clip(s, group["delta"], group["lr"])

                # Spectral correction
                if step % group["K"] == 0:
                    state["buffer"].append(grad.clone().flatten())
                    if len(state["buffer"]) == group["B"]:
                        G = torch.stack(list(state["buffer"]), dim=1)
                        correction = spectral_correction(G, grad, group["k"], group["gamma"], group["lr"])
                        delta -= correction

                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(delta)

        return loss
