# ssn/core.py
import math
import torch
from torch.optim import Optimizer

class SSN(Optimizer):
    """
    Stable SSN practical optimizer with bias-corrected second-moment (RMS-like).
    This variant adds bias-correction to avoid extreme early steps.
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
        defaults = dict(lr=lr, beta=beta, eps=eps, weight_decay=weight_decay, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Precompute global norms per group for clipping
        group_global_norm = {}
        for group in self.param_groups:
            max_norm = group.get("max_grad_norm", None)
            if max_norm is None:
                group_global_norm[id(group)] = None
                continue
            sqsum = 0.0
            for p in group["params"]:
                if p.grad is None:
                    continue
                sqsum += float((p.grad.data ** 2).sum().item())
            group_global_norm[id(group)] = math.sqrt(sqsum)

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]
            wd = group["weight_decay"]
            max_norm = group.get("max_grad_norm", None)
            global_norm = group_global_norm[id(group)]

            clip_coef = 1.0
            if max_norm is not None and global_norm is not None and global_norm > 0 and global_norm > max_norm:
                clip_coef = max_norm / (global_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if clip_coef != 1.0:
                    grad = grad.mul(clip_coef)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["v"] = torch.zeros_like(p.data)

                v = state["v"]
                state["step"] += 1
                step = state["step"]

                # Update second moment estimate (RMS-style)
                v.mul_(beta).addcmul_(grad, grad, value=1 - beta)

                # Bias correction for v to avoid small denominators early
                bias_correction = 1.0 - (beta ** step)
                # avoid zero division just in case
                if bias_correction <= 0:
                    bias_correction = 1e-8

                v_hat = v / bias_correction

                # weight decay (decoupled)
                if wd != 0.0:
                    p.data.add_(p.data, alpha=-lr * wd)

                # denom uses v_hat
                denom = v_hat.sqrt().add_(eps)

                # compute step
                step_tensor = grad.div(denom)

                # optional per-element cap on step to be extra safe (prevent huge single-element jumps)
                # cap magnitude to something sane relative to lr, e.g., 1000*lr
                cap_val = 1000.0 * lr
                # clamp in-place on a copy to avoid modifying grad state used elsewhere
                step_tensor = torch.clamp(step_tensor, min=-cap_val, max=cap_val)

                # final parameter update
                p.data.add_(step_tensor, alpha=-lr)

        return loss
