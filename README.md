# PRAISE-GAD

This is the PyTorch implementation for **PRAISE-GAD: Posterior-Risk-Aware Contrastive Learning for Unsupervised Graph Anomaly Detection**.

> The paper is currently under review.

<!-- ## Architecture -->

<!-- ![Overview of PRAISE-GAD](framework.png) -->

## Repository structure

```text
.
├── README.md
├── framework.png
├── environment.yml
├── requirements.txt
├── Data/
│   ├── acm.mat
│   ├── blogcatalog.mat
│   ├── citeseer.mat
│   ├── cora.mat
│   ├── pubmed.mat
│   └── reddit.mat
├── configs/
│   ├── praise_main.json
│   └── praise_datasets.json
└── praise_gad/
    ├── run.py
    ├── model.py
    ├── aug.py
    ├── diffusion.py
    ├── posterior.py
    ├── losses.py
    ├── scoring.py
    └── utils.py
```

| Module | Responsibility |
|---|---|
| [`run.py`](praise_gad/run.py) | Configuration loading, training, selection of the best training state, inference, and evaluation. |
| [`model.py`](praise_gad/model.py) | Shared graph encoder, context readout, and node-context bilinear discriminator. |
| [`aug.py`](praise_gad/aug.py) | Redundancy pruning, neighborhood completion, and posterior-calibrated restorative augmentation (PCRA). |
| [`diffusion.py`](praise_gad/diffusion.py) | Fixed anomaly prior, conditional latent denoiser, diffusion training, and stochastic reconstruction statistics. |
| [`posterior.py`](praise_gad/posterior.py) | EMA posterior bank, normality estimates, and predictive confidence used by subsequent epochs. |
| [`losses.py`](praise_gad/losses.py) | Contrastive objectives, posterior pair risk, negative-context allocation, and bounded false-negative weighting. |
| [`scoring.py`](praise_gad/scoring.py) | Attribute-neighborhood residuals, percentile normalization, and fixed complementary score fusion. |
| [`utils.py`](praise_gad/utils.py) | Dataset loading, feature and graph preprocessing, graph operations, and context sampling. |

## Environment

- Python 3.8.20
- PyTorch 1.12.1
- CUDA 11.6
- DGL 0.4.3
- torch-scatter 2.1.2
- NumPy 1.24.4
- SciPy 1.10.1
- scikit-learn 1.3.2
- NetworkX 3.1

[`environment.yml`](environment.yml) records the Conda environment specification, and [`requirements.txt`](requirements.txt) lists the Python package versions. From the directory containing this README:

```bash
conda env create -f environment.yml
conda activate praise-gad
```

## Datasets

The six benchmark files are provided in [`Data/`](Data). Select a dataset using its lowercase identifier:

```text
acm, blogcatalog, citeseer, cora, pubmed, reddit
```

The loader reads `Data/<dataset>.mat` relative to the repository root. Each file contains:

| Content | Accepted MATLAB field names | Shape / meaning |
|---|---|---|
| Node attributes | `Attributes` or `X` | `N × F` feature matrix. |
| Graph adjacency | `Network` or `A` | `N × N` adjacency matrix. |
| Evaluation labels | `Label` or `gnd` | Binary node labels: `0` normal, `1` anomalous. |


## Configuration

Load configuration file to run the formal PRAISE-GAD model:

- [`praise_main.json`](configs/praise_main.json) selects the shared model mechanisms and parameters for all datasets.
- [`praise_datasets.json`](configs/praise_datasets.json) supplies dataset-specific learning rates, training epochs, contrastive regularization weights (`alpha`), and temperatures (`tau`).

## Run PRAISE-GAD

Run the following commands from the repository root, using the configured environment.

### One dataset and one seed

```bash
python praise_gad/run.py --config configs/praise_main.json --dataset_config configs/praise_datasets.json --dataset cora --seed 1 --gpu_id 0 --run_id main_s123
```

### Six datasets with seeds 1, 2, and 3

The following Bash loop runs the 18 dataset-seed combinations sequentially:

```bash
for dataset in blogcatalog acm cora citeseer pubmed reddit; do
  for seed in 1 2 3; do
    python praise_gad/run.py \
      --config configs/praise_main.json \
      --dataset_config configs/praise_datasets.json \
      --dataset "$dataset" \
      --seed "$seed" \
      --gpu_id 0 \
      --run_id main_s123
  done
done
```
