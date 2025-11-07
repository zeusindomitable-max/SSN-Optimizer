# ssn/core.py
import torch
from collections import deque
from torch.optim import Optimizer

from .preconditioner import fisher_preconditioner, trust_region_clip
from .spectral import spectral_correction


class SSN(Optimizer):
    """
    Spectral-Sketch Natural (SSN) Optimizer.

    Menggabungkan tiga komponen:
    1. Fisher preconditioning  -> adaptasi arah gradien.
    2. Trust-region clipping   -> stabilisasi langkah.
    3. Spectral correction     -> mitigasi osilasi high-curvature.
    """

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
        """
        Satu langkah optimisasi.
        Jika closure diberikan, akan menghitung ulang loss untuk gradient-free step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta_g = group["beta_g"]
            lambda_fisher = group["lambda_fisher"]
            delta = group["delta"]
            gamma = group["gamma"]
            k = group["k"]
            K = group["K"]
            B = group["B"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # === Inisialisasi state ===
                if len(state) == 0:
                    state["step"] = 0
                    state["g"] = torch.zeros_like(p)
                    state["buffer"] = deque(maxlen=B)

                g = state["g"]
                step = state["step"] = state["step"] + 1

                # === 1. Fisher preconditioning ===
                g.mul_(beta_g).addcmul_(grad, grad, value=1 - beta_g)
                precond_grad = fisher_preconditioner(g, lambda_fisher)

                # === 2. Trust-region clipping ===
                update = trust_region_clip(precond_grad * grad, delta, lr)

                # === 3. Spectral correction ===
                state["buffer"].append(grad.flatten().clone())
                if step % K == 0 and len(state["buffer"]) == B:
                    G = torch.stack(list(state["buffer"]), dim=1)
                    correction = spectral_correction(G, grad, k, gamma, lr)
                    update.sub_(correction)

                # === 4. Weight decay (L2 regularization) ===
                if wd != 0.0:
                    update.sub_(lr * wd * p)

                # === 5. Final parameter update ===
                p.data.add_(-lr * update)

        return loss
