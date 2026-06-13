# Instance-to-sample prediction

A workflow for predicting sample-level labels from collections of instance-level measurements to determine most important instances to prediction.

The motivating use case is biomedical data where an outcome is known for a whole sample or patient, but the relevant signal may come from a small subset of cells, clones, or molecular observations.

This repository demonstrates a multiple-instance learning (MIL) workflow on simulated single-cell immune profiling data (RNA, BCR/TCR sequences). Each sample is represented as a bag of instances with RNA-like numeric features plus TCR/BCR sequence features. A gated-attention MIL classifier predicts the sample label and produces attention scores that can be inspected as candidate driver instances.

## Workflow

1. Generate simulated bagged data with known driver instances.
2. Encode RNA-like features and TCR/BCR sequences.
3. Train a gated-attention MIL classifier from sample-level labels.
4. Export sample predictions, instance-level attention scores, performance metrics, and feature diagnostics.

## Setup

Create and activate a Python environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the simulated workflow:

```bash
python scripts/run_pipeline_simulated.py
```

If Matplotlib cannot write its cache in a restricted environment, run with a writable cache directory:

```bash
MPLCONFIGDIR=.cache/matplotlib python scripts/run_pipeline_simulated.py
```

## Outputs

The demo writes outputs to `results/run_pipeline_simulated/`:

- `sample_predictions.csv`: sample labels and predicted probabilities.
- `instance_attention.csv`: attention scores for each instance, including the known simulated driver label.
- `performance_metrics.csv`: sample-level and attention-based performance metrics.
- `performance_metrics.png`: bar plot of available performance metrics.
- `attention_diagnostics.png`: attention by true driver status.
- `volcano_rna.png` and `volcano_repertoire.png`: driver vs non-driver feature diagnostics.
- `feature_stats_all.csv`: combined feature statistics for RNA and repertoire features.

## Repository layout

- `scripts/`: runnable workflow scripts.
- `utils/`: reusable data generation, encoding, MIL, plotting, and table helpers.
- `results/`: generated workflow outputs.
- `tests/manual/`: lightweight manual checks for demo runs.

## Notes

This project is in active development and currently uses simulated data for that. The current workflow is a proof of concept for the modelling structure and diagnostics, not a clinically evaluated model.
