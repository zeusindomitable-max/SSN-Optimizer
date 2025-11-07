import torch
import matplotlib.pyplot as plt
import pandas as pd

def save_convergence_plot(losses, path="benchmark/plots/ssn_convergence.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="SSN Loss", color="#1f77b4", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SSN Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
