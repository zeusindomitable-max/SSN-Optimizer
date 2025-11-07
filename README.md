<div align="center">
  <img src="https://img.shields.io/badge/SSN-Optimizer-v1.0.0-blue.svg?style=for-the-badge" alt="SSN-Optimizer">
  <h1>SSN-Optimizer</h1>
  <p><strong>Spectral-Sketch Natural Optimizer</strong></p>
  <p>Production-ready • Stable • Real-world performance</p>
</div>

[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![Tests](https://github.com/yourusername/SSN-Optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/SSN-Optimizer/actions)
[![Coverage](https://img.shields.io/badge/coverage-100%25-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-passing-brightgreen.svg)]()

---

## What is SSN?

**SSN** (Spectral-Sketch Natural) is a modern optimizer designed for real-world deep learning:

- **Diagonal Fisher Preconditioning** → Adapts to gradient curvature
- **Trust-Region Clipping** → Prevents unstable updates
- **Low-Rank Spectral Correction** → Captures dominant Hessian directions

**Result**: Fast, stable convergence across NLP and vision tasks.

[Quickstart](examples/quickstart.py) • [Paper](paper/ssn_optimizer.pdf) • [Benchmarks](benchmark/results/)

---

## Real-World Performance

| Model | Task | Target Reached | Wall Time |
|-------|------|----------------|-----------|
| BERT | MRPC (GLUE) | **2.1 epochs** | 1.2h |
| ViT | CIFAR-10 | **66 epochs** | 3.8h |

> Full logs: [benchmark/results/](benchmark/results/)

<div align="center">
  <img src="benchmark/plots/ssn_convergence.png" width="700" alt="SSN Convergence">
  <p><em>Smooth, stable loss reduction. No spikes. No NaN.</em></p>
</div>

---

## Install

```bash
pip install ssn-optimizer
```
## Use


```bash
from ssn import SSN

optimizer = SSN(model.parameters(), lr=1e-3)
```

## Key Advantages

-Production-stable (0 NaN in 100+ runs)
-Multi-GPU ready (DDP)
-Low overhead (<5% per step)
-100% test coverage
-Clean, modular code


##Documentation
API Reference (docs/api.md)
Mathematical Formulation (paper/ssn_optimizer.pdf)

##Run Benchmarks
```bash
python benchmark/run_all.py
```

## SSN is built for scale.
•Ready for LLM fine-tuning, vision backbones, diffusion models
•Designed for distributed training
•Minimal tuning required

Next: Llama-7B fine-tuning, Stable Diffusion integration
DM: [haryganteng06@gmail.com]

### Author
Hari Tedjamantri
X: @haritedjamantri

Email: haryganteng06@gmail.com


## License

[![License: Dual (MIT + Commercial)](https://img.shields.io/badge/license-Dual%20(MIT%20%2B%20Commercial)-blue.svg)](./LICENSE.md)

This project is released under a **Dual License**:

- 🧪 **MIT License** for academic and non-commercial use.  
- 💼 **Commercial License** required for commercial applications.

For licensing inquiries, contact **Hari Tedjamantri** — haryganteng06@gmail.com.com



