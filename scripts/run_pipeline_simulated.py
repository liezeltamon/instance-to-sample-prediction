from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import polars as pl

# Ensure repo root is on sys.path when running this script directly from scripts/.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.analysis import (
    add_repertoire_features_to_dataframe,
    compute_performance_metrics,
    plot_attention_diagnostics,
    plot_performance_metrics,
    plot_volcano,
    summarize_attention_by_truth,
)
from utils.data import generate_synthetic_bag_data
from utils.encoders import NumericEncoder, SequenceEncoder, concatenate_modalities
from utils.mil import MILClassifier, predict_mil, train_mil_model
from utils.wrangling import ensure_dir, save_table


def main() -> None:
    output_root = Path("results/run_pipeline_simulated")
    ensure_dir(output_root)

    data = generate_synthetic_bag_data(n_bags=20, bag_size=40, n_rna_features=12, drivers_per_positive_bag=5)
    sample_table = data.select(["bag_id", "bag_label"]).unique().sort("bag_id")

    numeric_encoder = NumericEncoder([f"rna_{i}" for i in range(12)])
    x_numeric = numeric_encoder.fit_transform(data)

    tcr_encoder = SequenceEncoder(k=2)
    x_tcr = tcr_encoder.encode_dataframe(data, "tcr_sequence")

    bcr_encoder = SequenceEncoder(k=2)
    x_bcr = bcr_encoder.encode_dataframe(data, "bcr_sequence")

    x = concatenate_modalities(x_numeric, x_tcr, x_bcr)
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

    attention_table = data.select(["bag_id", "instance_id", "driver_true"]).with_columns(
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
    print(f"Wrote outputs to {output_root}")
    
    # Feature comparison: volcano plots for each modality (matching concatenation: rna + repertoire)
    driver_ids = set(attention_table.filter(pl.col("driver_true") == 1)["instance_id"].to_list())
    nondriver_ids = set(attention_table.filter(pl.col("driver_true") == 0)["instance_id"].to_list())
    nondriver_sample = set(np.random.choice(list(nondriver_ids), size=len(driver_ids), replace=False))
    
    # RNA modality volcano
    rna_cols = [f"rna_{i}" for i in range(12)]
    rna_stats = plot_volcano(data, driver_ids, nondriver_sample, rna_cols, output_root / "volcano_rna.png", title="RNA Features: Drivers vs Non-drivers", top_n_labels=6)
    rna_stats = rna_stats.with_columns(pl.lit("rna").alias("modality"))
    
    # Repertoire modality volcano (TCR + BCR combined)
    data_with_tcr, tcr_feature_cols = add_repertoire_features_to_dataframe(data, "tcr_sequence", "tcr")
    data_with_repertoire, bcr_feature_cols = add_repertoire_features_to_dataframe(data_with_tcr, "bcr_sequence", "bcr")
    repertoire_feature_cols = tcr_feature_cols + bcr_feature_cols
    
    repertoire_stats = plot_volcano(data_with_repertoire, driver_ids, nondriver_sample, repertoire_feature_cols, output_root / "volcano_repertoire.png", title="Repertoire Features (TCR+BCR): Drivers vs Non-drivers", top_n_labels=12)
    repertoire_stats = repertoire_stats.with_columns(pl.lit("repertoire").alias("modality"))
    
    # Combine all feature statistics into a single file
    all_feature_stats = pl.concat([rna_stats, repertoire_stats])
    save_table(all_feature_stats, output_root / "feature_stats_all.csv")
    
    print("Saved combined feature statistics to:")
    print(f"  - {output_root / 'feature_stats_all.csv'}")


if __name__ == "__main__":
    main()
