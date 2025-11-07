# ssn/utils.py
import os

import matplotlib.pyplot as plt
import pandas as pd


def save_convergence_plot(losses, path="benchmark/plots/convergence.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(losses, "o-", color="#1f77b4", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("SSN Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
