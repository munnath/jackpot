# Jackpot

Jackpot is a Python package implementing the methods described in [this paper](https://jmlr.org/papers/v26/24-1769.html).

## Installation

```bash
pip install -e .
```

## Some tutorials

Notebooks tutorials are provided in [notebooks](https://github.com/munnath/jackpot/tree/main/notebooks).

A template with all the explanations can be found in [jackpot_template.ipynb](https://github.com/munnath/jackpot/blob/main/notebooks/jackpot_template.ipynb), while a minimal template can be found in [minimal_template.ipynb](https://github.com/munnath/jackpot/blob/main/notebooks/minimal_template.ipynb).

The 3 main experiments provided in the [article](https://jmlr.org/papers/v26/24-1769.html) are accessible here: [expe_1](https://github.com/munnath/jackpot/blob/main/notebooks/solar_system.ipynb), [expe_2](https://github.com/munnath/jackpot/blob/main/notebooks/blind_inv_pb.ipynb) and [expe_3](https://github.com/munnath/jackpot/blob/main/notebooks/inv_pb.ipynb).


## Basic usage of Jackpot

Jackpot approximates the uncertainty region

```math
\mathcal{U}_\varepsilon(x^\star) = \{ x \in \mathbb{R}^N : \|\Phi(x) - \Phi(x^\star)\| \leq \varepsilon \}
```

by constructing a low-dimensional adversarial manifold $\mathcal{M}_\varepsilon(x^\star)$ using Jacobian-based optimization.

Follow these steps:

1. Configure Jackpot parameters: Set experiment metadata (e.g., name), manifold dimension $D$, tolerance  $\varepsilon$, grid resolution, and whether to load/save intermediate results.
2. Define your direct model $\Phi$ and an initial parameter vector $x_{\text{est}}$ near your target  $x^\star$.
3. Run Jackpot in two stages: jacobian spectrum and manifold.

This is summarized here:

```python
jack = Jackpot(Phi, x_est)
jack.set_params(**params)
jack.jacobian_spectrum()
jack.manifold()

```

## Independent interest:

A particular attention can be carried out for the usage of the least singular vectors computation. This is handled either by the standard [torch.svd](https://docs.pytorch.org/docs/stable/generated/torch.svd.html) algorithm or - when the dimensions of the problem is too large - by the [LOBPCG](https://arxiv.org/pdf/1704.07458) algorithm. A matrix-free version compatible to torch auto-differentiation is used [here](https://github.com/munnath/jackpot/blob/main/src/jackpot/torch_lobpcg.py) and is an adaptation from the standard [torch.lobpcg](https://docs.pytorch.org/docs/stable/generated/torch.lobpcg.html).
