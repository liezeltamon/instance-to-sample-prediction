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
- `results/run_pipeline_simulated/attention_diagnostics.png`

Check that the attention output includes the columns `bag_id`, `instance_id`, `driver_true`, and `attention`.
Check that the metrics output includes sample-level accuracy, sample ROC AUC, and attention ROC AUC.

## Expected summary
- The script should run without errors.
- The summary printed to the console should include `mean_attention_driver`, `attention_auc`, and `sample_accuracy_0_5`.
- The diagnostics plot should be written to `results/run_pipeline_simulated/attention_diagnostics.png`.
