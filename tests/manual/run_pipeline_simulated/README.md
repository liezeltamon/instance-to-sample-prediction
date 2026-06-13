# Manual check for `run_pipeline_simulated`

This folder contains the minimal instructions and expected outputs to verify the demo pipeline.

## How to run
From the repository root:

```bash
python scripts/run_pipeline_simulated.py
```

## What to inspect
- `results/run_pipeline_simulated/instance_attention.csv`
- `results/run_pipeline_simulated/sample_predictions.csv`
- `results/run_pipeline_simulated/performance_metrics.csv`
- `results/run_pipeline_simulated/performance_metrics.png`
- `results/run_pipeline_simulated/comparison_groups.csv`
- `results/run_pipeline_simulated/feature_stats_grouped.csv`
- `results/run_pipeline_simulated/attention_diagnostics.png`
- `results/run_pipeline_simulated/top_attention_instances.csv`
- `results/run_pipeline_simulated/top_attention_pseudobulk.csv`
- `results/run_pipeline_simulated/top_attention_feature_comparisons.csv`
- `results/run_pipeline_simulated/top_attention_feature_heatmap.png`
- `results/run_pipeline_simulated/attention_summary_heatmap.csv`
- `results/run_pipeline_simulated/attention_summary_heatmap.png`

Check that the attention output includes the columns `bag_id`, `instance_id`, `bag_label`, `cell_type`, `driver_true`, and `attention`.
Check that grouped feature statistics include `group`, `modality`, `feature`, `cohens_d`, `pvalue`, `n_driver`, and `n_nondriver`.
Check that top-attention feature comparisons are sample-aware and include `comparison`, `group`, `feature_type`, `feature`, `effect_size`, `pvalue`, and sample counts.

## Expected summary
- The script should run without errors.
- The summary printed to the console should include `mean_attention_driver`, `attention_auc`, and `sample_accuracy_0_5`.
- Grouped feature statistics should be written to `results/run_pipeline_simulated/feature_stats_grouped.csv`.
- Top-attention downstream diagnostics should be written to `results/run_pipeline_simulated/top_attention_feature_heatmap.png`.
