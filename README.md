# PyTorch-Tempest

[![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/Erlemar/pytorch_tempest/?ref=repository-badge)

This repository has my pipeline for training neural nets.

Main frameworks used:

* [hydra](https://github.com/facebookresearch/hydra)
* [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)

The main ideas of the pipeline:

* all parameters and modules are defined in configs;
* prepare configs beforehand for different optimizers/schedulers and so on, so it is easy to switch between them;
* have templates for different deep learning tasks. Currently, image classification and named entity recognition are supported;

## Setup

Recommended (with [uv](https://docs.astral.sh/uv/)):
```shell
brew install uv          # or see https://docs.astral.sh/uv/getting-started/installation/
uv sync                  # creates .venv from pyproject.toml + uv.lock
```
Then prefix commands with `uv run`, e.g. `uv run python train.py ...`, or `source .venv/bin/activate` to use the env directly.

Alternative (pip):
```shell
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running

This will run training on MNIST (data will be downloaded):
```shell
uv run python train.py --config-name mnist_config model.encoder.params.to_one_channel=True
```

Running on MPS (M1 macbook)
```shell
uv run python train.py --config-name mnist_config model.encoder.params.to_one_channel=True trainer.accelerator=mps +trainer.devices=1 optimizer=adan training.lr=0.001
```
Running on MPS (M1 macbook) with schedule free optimizer https://github.com/facebookresearch/schedule_free/tree/main
```shell
uv run python train.py --config-name mnist_config model.encoder.params.to_one_channel=True trainer.accelerator=mps trainer.devices=1 optimizer=adamwschedulefree training.lr=0.001 scheduler.params.patience=100
```
The default run:

```shell
uv run python train.py
```

The default version of the pipeline is run on imagenette dataset. To do it, download the data from this repository:
https://github.com/fastai/imagenette
unzip it and define the path to it in conf/datamodule/image_classification.yaml path
