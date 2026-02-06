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

A particular attention can be carried out for the usage of the least singular vectors computation. This is handled either by the standard [torch.svd](https://docs.pytorch.org/docs/stable/generated/torch.svd.html) algorithm or - when the dimensions of the problem is too large - by the [LOBPCG](https://arxiv.org/pdf/1704.07458) algorithm. A matrix-free version compatible to torch auto-differentiation is used [here](https://github.com/munnath/jackpot/blob/main/src/jackpot/torch_lobpcg.py) and is an adaptation from the standard [torch.lobpcg](https://docs.pytorch.org/docs/stable/generated/torch.lobpcg.html).
