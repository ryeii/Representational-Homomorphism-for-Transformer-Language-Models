# Reproducing the Experiments

This repository is for reproducing the synthetic compositional generalization experiments only. All data is generated on the fly from a small command language, so there is nothing to download.

The codebase has two experiment tracks:

- `he_probe/`: the main probing and ablation experiments that produce the paper figures for model size, training sparsity, noise injection, and the accuracy-vs-HE correlation.
- `he_reg_causal/`: the causal regularization experiment comparing a baseline model against a model trained with modifier homomorphism regularization.

## Environment

Use Python 3.10+ and install the required packages in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy matplotlib scikit-learn scipy
```

The scripts use CUDA automatically if available, otherwise they run on CPU.

## 1. Reproduce the Main Probe Figures

Run this from the `he_probe/` directory:

```bash
cd he_probe
python reproduce_figures.py
```

This script runs three experiment families over 5 seeds and writes the main paper figures to `he_probe/figures/`:

- `figure3_model_size_ablation.pdf`
- `figure4_training_data_sparsity.pdf`
- `figure5_noise_injection.pdf`
- `figure6_noise_acc_vs_he_poly.pdf`

What it sweeps:

- model depth: 1 to 10 transformer layers
- training sparsity: training examples with up to 1 to 4 primitive clauses
- noise injection: 0 to 15 random noise tokens inserted into the training inputs

Evaluation is OOD generalization on held-out inputs with 5 to 12 primitive clauses.

## 2. Reproduce the HE-Regularization Experiment

Run this from the `he_reg_causal/` directory:

```bash
cd he_reg_causal
python experiment.py --outdir results_he_reg_strong
python plot.py --json results_he_reg_strong/rolling_results.json --outdir figures
```

This reproduces the baseline vs `he_reg` comparison and writes:

- raw results to `he_reg_causal/results_he_reg_strong/`
- figures and a summary table to `he_reg_causal/figures/`

Main outputs:

- `scatter_ood_vs_modHE.pdf`
- `bar_he_summary.pdf`
- `ood_accuracy_by_complexity.pdf`
- `layerwise_he_modifier.pdf`
- `layerwise_he_sequence.pdf`
- `training_losses.pdf`
- `summary.tsv`

Default settings in the code:

- seeds: `0,1,2,3,4`
- noise levels: `0,3,6,9,12,15`
- model: 4-layer decoder-only transformer, `d_model=128`
- training: 50 epochs, learning rate `1e-4`
- HE regularization: `lambda_he=0.2` on layers `2,4`

## Notes

- Run the commands from inside each experiment directory. The scripts rely on local module imports.
- The runs are compute-heavy on CPU; a GPU is helpful but not required.
- Checked-in PDFs and JSON files under `he_probe/figures/`, `he_reg_causal/figures/`, and `he_reg_causal/results_he_reg_strong/` can be used as reference outputs.
