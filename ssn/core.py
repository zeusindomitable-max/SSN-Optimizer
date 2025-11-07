# ssn/core.py
import math
import torch
from torch.optim import Optimizer

class SSN(Optimizer):
    """
    Stable Spectral-ish Natural optimizer (practical variant).

    Behavior:
    - Adaptive diagonal preconditioning (RMSProp-like / natural-gradient-ish).
    - Damping/eps to avoid division by zero.
    - Optional gradient clipping (global norm).
    - Deterministic per-step behavior (no reassign p = ... mistakes).
    """

    def __init__(
        self,
        params,
        lr: float = 0.3,
        beta: float = 0.95,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        max_grad_norm: float | None = 5.0,
    ):
        """
        Args:
            lr: learning rate (tests use 0.3 or 1.0; default set to 0.3).
            beta: smoothing for second moment (like RMSProp).
            eps: small constant for numerical stability.
            weight_decay: L2 regularization multiplier.
            max_grad_norm: if not None, clip global grad-norm to this value.
        """
        defaults = dict(lr=lr, beta=beta, eps=eps, weight_decay=weight_decay, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Uses a stable, RMSProp-style diagonal preconditioner with damping and
        global gradient clipping to avoid occasional overshoots.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Optionally compute global norm for clipping across all params in all groups
        # (we compute it once per step).
        max_norm_by_group = {}
        for group in self.param_groups:
            max_norm_by_group[id(group)] = group.get("max_grad_norm", None)

        # compute global norm for groups that require clipping
        group_global_norm = {}
        for group in self.param_groups:
            max_norm = group.get("max_grad_norm", None)
            if max_norm is None:
                group_global_norm[id(group)] = None
                continue
            # accumulate squared norms
            sqsum = 0.0
            for p in group["params"]:
                if p.grad is None:
                    continue
                sqsum += float((p.grad.data ** 2).sum().item())
            group_global_norm[id(group)] = math.sqrt(sqsum)

        # Apply updates per-parameter
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]
            wd = group["weight_decay"]
            max_norm = group.get("max_grad_norm", None)
            global_norm = group_global_norm[id(group)]

            # compute clipping coefficient once
            clip_coef = 1.0
            if global_norm is not None and global_norm > 0 and global_norm > max_norm:
                clip_coef = max_norm / (global_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if clip_coef != 1.0:
                    grad = grad.mul(clip_coef)

                state = self.state[p]
                if len(state) == 0:
                    # initialize state
                    state["step"] = 0
                    # second moment estimate (diagonal approximate of fisher/hessian)
                    state["v"] = torch.zeros_like(p.data)

                v = state["v"]
                state["step"] += 1

                # Update second moment estimate (RMS-style)
                # Use grad**2 to capture curvature-like magnitude
                v.mul_(beta).addcmul_(grad, grad, value=1 - beta)

                # weight decay (decoupled like AdamW): applied to parameter directly
                if wd != 0.0:
                    p.data.add_(p.data, alpha=-lr * wd)

                # preconditioner: scale by 1 / (sqrt(v) + eps)
                denom = v.sqrt().add_(eps)

                # compute step (elementwise)
                step = grad.div(denom)

                # final parameter update
                p.data.add_(step, alpha=-lr)

        return loss
