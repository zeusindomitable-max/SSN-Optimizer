# ssn/core.py
import torch
from collections import deque
from torch.optim import Optimizer

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
            lr=lr,
            lambda_fisher=lambda_fisher,
            beta_g=beta_g,
            delta=delta,
            K=K,
            k=k,
            B=B,
            gamma=gamma,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    # Init + increment step even without grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["g"] = torch.zeros_like(p)
                        state["buffer"] = deque(maxlen=group["B"])
                    state["step"] += 1
                    continue

                grad = p.grad
                state = self.state[p]

                # Init state
                if len(state) == 0:
                    state["step"] = 0
                    state["g"] = torch.zeros_like(p)
                    state["buffer"] = deque(maxlen=group["B"])

                g = state["g"]
                step = state["step"] = state["step"] + 1

                # === FISHER PRECONDITIONING ===
                g.mul_(group["beta_g"]).addcmul_(grad, grad, value=1 - group["beta_g"])
                p = fisher_preconditioner(g, group["lambda_fisher"])

                # === TRUST REGION UPDATE ===
                s = p * grad
                update = trust_region_clip(s, group["delta"], lr)

                # === SPECTRAL CORRECTION ===
                if step % group["K"] == 0 and len(state["buffer"]) > 0:
                    state["buffer"].append(grad.flatten().clone())
                    if len(state["buffer"]) == group["B"]:
                        G = torch.stack(list(state["buffer"]), dim=1)
                        correction = spectral_correction(G, grad, group["k"], group["gamma"], lr)
                        update = update - correction

                # === WEIGHT DECAY ===
                if wd != 0:
                    update = update - lr * wd * p

                # === FINAL UPDATE: LANGSUNG KE PARAMETER ===
                p = p * (-lr) + update
                p.add_(p)  # p = -lr * p + update
                p.add_(p)  # p = 2 * p → ini yang bikin stuck!

                # === FIX: LANGSUNG UPDATE PARAMETER ===
                p.data.add_( -lr * p + update )  # INI YANG BENAR
