# Instance-to-sample prediction

A workflow for predicting sample-level labels from collections of instance-level measurements and identifying which instances contribute most to the prediction.

The motivating use case is biomedical data where an outcome is known for a whole sample or patient, but the relevant signal may come from a small subset of cells, clones, or molecular observations. The workflow expects a MIL-ready instance table rather than raw sequencing data: upstream preprocessing should already have produced numeric feature or embedding columns.

This repository is a lightweight, more generic adaptation of MultiMIL for prepared instance-level feature or embedding tables.

## Expected input

The core workflow expects one row per instance with sample identifiers, sample-level labels, optional group annotations, and numeric feature or embedding columns. These numeric columns can be original features or upstream embeddings such as PCA, scVI, repertoire embeddings, morphology embeddings, or other model-derived representations.

```text
bag_id  instance_id  bag_label  cell_type  transcriptome_0  transcriptome_1  repertoire_0  repertoire_1
S001    cell_001     1          T          0.12             -0.44            0.88          -0.11
S001    cell_002     1          myeloid    -0.31             0.72            0.00           0.00
S002    cell_003     0          B          1.12              0.03           -0.23           0.61
```

`driver_true` is used only in the simulated demo to evaluate attention recovery; it is not required for real prediction use.

Downstream feature diagnostics use the model input table by default, but can also be run on a separate instance-by-feature table joined by `instance_id`. This is useful when the MIL model was trained on latent embeddings but feature discovery should be done on original gene or marker values.

## Workflow

1. Create or provide a MIL-ready instance table.
2. Select numeric feature or embedding columns by modality.
3. Train a gated-attention MIL classifier from sample-level labels.
4. Export sample predictions, instance-level attention scores, performance metrics, simulation diagnostics, and top-attention downstream feature diagnostics.

The included simulation creates a small MIL-ready example with transcriptome and repertoire feature blocks. Simulated T and B cells have repertoire features; simulated myeloid cells have transcriptome features only and zero-valued repertoire features.

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
MPLBACKEND=Agg MPLCONFIGDIR=.cache/matplotlib python scripts/run_pipeline_simulated.py
```

## Outputs

The demo writes outputs to `results/run_pipeline_simulated/`:

- `sample_predictions.csv`: sample labels and predicted probabilities.
- `instance_attention.csv`: attention scores for each instance, including `bag_label`, `cell_type`, and the known simulated driver label.
- `performance_metrics.csv`: sample-level and attention-based performance metrics.
- `performance_metrics.png`: bar plot of available performance metrics.
- `attention_diagnostics.png`: attention by true driver status.
- `comparison_groups.csv`: driver and matched non-driver counts used for grouped comparisons.
- `feature_stats_grouped.csv`: per-group, per-modality driver vs non-driver feature statistics.
- `top_attention_instances.csv`: top-attention instances selected within each sample.
- `top_attention_pseudobulk.csv`: sample-level pseudobulks for top-attention and rest instances.
- `top_attention_feature_comparisons.csv`: lightweight MultiMIL-style feature comparisons by group and feature type.
- `top_attention_feature_heatmap.png`: feature-by-comparison effect-size heatmap with significant cells outlined.
- `attention_summary_heatmap.csv`: sample-balanced top-attention summaries by feature type, label, and group.
- `attention_summary_heatmap.png`: red heatmap of mean and median attention with SD annotations.

## Repository layout

- `scripts/`: runnable workflow scripts.
- `utils/`: reusable data generation, MIL, plotting, and table helpers.
- `results/`: generated workflow outputs.
- `tests/manual/`: lightweight manual checks for demo runs.

## Notes

This project is in active development and currently uses simulated data. Raw omics or sequence preprocessing belongs upstream of this workflow; this repository demonstrates MIL over prepared instance-level feature or embedding tables.

Inspired by MultiMIL: Litinetskaya et al., *Weakly supervised learning uncovers phenotypic signatures in single-cell data*, bioRxiv 2024.07.29.605625, https://doi.org/10.1101/2024.07.29.605625.
