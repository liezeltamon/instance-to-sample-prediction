from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import polars as pl

# Ensure repo root is on sys.path when running this script directly from scripts/.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.analysis import (
    compute_attention_summary_table,
    compute_grouped_feature_statistics,
    compute_performance_metrics,
    compute_top_attention_feature_comparisons,
    compute_top_attention_pseudobulk,
    plot_attention_diagnostics,
    plot_attention_summary_heatmap,
    plot_performance_metrics,
    plot_top_attention_feature_heatmap,
    sample_driver_and_nondriver_ids,
    select_top_attention_instances,
    summarize_attention_by_truth,
)
from utils.data import generate_synthetic_bag_data
from utils.encoders import concatenate_modalities
from utils.mil import MILClassifier, predict_mil, train_mil_model
from utils.wrangling import ensure_dir, save_table


def columns_with_prefix(table: pl.DataFrame, prefix: str) -> list[str]:
    return [column for column in table.columns if column.startswith(prefix)]


def main() -> None:
    output_root = Path("results/run_pipeline_simulated")
    ensure_dir(output_root)

    group_col = "cell_type"
    data = generate_synthetic_bag_data(
        n_bags=20,
        bag_size=40,
        n_transcriptome_features=12,
        n_repertoire_features=12,
        drivers_per_positive_bag=5,
        group_col=group_col,
        seed=0,
    )
    sample_table = data.select(["bag_id", "bag_label"]).unique().sort("bag_id")

    transcriptome_cols = columns_with_prefix(data, "transcriptome_")
    repertoire_cols = columns_with_prefix(data, "repertoire_")
    x_transcriptome = data.select(transcriptome_cols).to_numpy().astype(np.float32)
    x_repertoire = data.select(repertoire_cols).to_numpy().astype(np.float32)
    x = concatenate_modalities(x_transcriptome, x_repertoire)

    bag_index = data["bag_id"].to_pandas().factorize()[0].astype(np.int64)
    bag_labels = (
        data.group_by("bag_id", maintain_order=True)
        .agg(pl.first("bag_label"))
        .sort("bag_id")
        .select("bag_label")
        .to_numpy()
        .astype(np.float32)
        .ravel()
    )

    model = MILClassifier(input_dim=x.shape[1], hidden_dim=64, attn_dim=32)
    model, history = train_mil_model(model, x, bag_index, bag_labels, n_epochs=20, lr=1e-3)

    probabilities, attention, bag_ids = predict_mil(model, x, bag_index)

    attention_table = data.select(["bag_id", "instance_id", "bag_label", group_col, "driver_true"]).with_columns(
        pl.Series("attention", attention)
    )

    bag_predictions = pl.DataFrame(
        {
            "bag_id": sample_table["bag_id"],
            "bag_label": sample_table["bag_label"],
            "bag_prediction": probabilities,
        }
    )

    save_table(attention_table, output_root / "instance_attention.csv")
    save_table(bag_predictions, output_root / "sample_predictions.csv")

    performance_metrics = compute_performance_metrics(bag_predictions, attention_table)
    save_table(performance_metrics, output_root / "performance_metrics.csv")
    plot_performance_metrics(performance_metrics, output_root / "performance_metrics.png")

    summary = summarize_attention_by_truth(attention_table)
    print("=== Attention summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("=== Performance metrics ===")
    for row in performance_metrics.iter_rows(named=True):
        print(f"{row['metric']}: {row['value']}")

    plot_attention_diagnostics(attention_table, output_root / "attention_diagnostics.png")

    driver_ids, nondriver_ids, comparison_summary = sample_driver_and_nondriver_ids(
        attention_table,
        group_col=group_col,
        seed=0,
    )
    save_table(comparison_summary, output_root / "comparison_groups.csv")

    feature_sets = {
        "transcriptome": transcriptome_cols,
        "repertoire": repertoire_cols,
    }

    grouped_stats = compute_grouped_feature_statistics(
        data,
        driver_ids,
        nondriver_ids,
        feature_sets,
        group_col=group_col,
    )
    save_table(grouped_stats, output_root / "feature_stats_grouped.csv")

    top_attention_instances = select_top_attention_instances(attention_table, top_fraction=0.10)
    save_table(top_attention_instances, output_root / "top_attention_instances.csv")

    top_attention_pseudobulk = compute_top_attention_pseudobulk(
        attention_table,
        feature_sets,
        feature_table=data,
        group_col=group_col,
        top_fraction=0.10,
    )
    save_table(top_attention_pseudobulk, output_root / "top_attention_pseudobulk.csv")

    top_attention_comparisons = compute_top_attention_feature_comparisons(top_attention_pseudobulk)
    save_table(top_attention_comparisons, output_root / "top_attention_feature_comparisons.csv")
    plot_top_attention_feature_heatmap(
        top_attention_comparisons,
        output_root / "top_attention_feature_heatmap.png",
    )

    attention_summary = compute_attention_summary_table(
        attention_table,
        feature_sets,
        feature_table=data,
        group_col=group_col,
        top_fraction=0.10,
    )
    save_table(attention_summary, output_root / "attention_summary_heatmap.csv")
    plot_attention_summary_heatmap(attention_summary, output_root / "attention_summary_heatmap.png")

    print(f"Wrote outputs to {output_root}")
    print("Saved grouped feature tables to:")
    print(f"  - {output_root / 'feature_stats_grouped.csv'}")
    print("Saved top-attention downstream diagnostics to:")
    print(f"  - {output_root / 'top_attention_instances.csv'}")
    print(f"  - {output_root / 'top_attention_pseudobulk.csv'}")
    print(f"  - {output_root / 'top_attention_feature_comparisons.csv'}")
    print(f"  - {output_root / 'top_attention_feature_heatmap.png'}")
    print(f"  - {output_root / 'attention_summary_heatmap.csv'}")
    print(f"  - {output_root / 'attention_summary_heatmap.png'}")


if __name__ == "__main__":
    main()
