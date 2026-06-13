from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from adjustText import adjust_text
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score


def summarize_attention_by_truth(instance_table: pl.DataFrame) -> dict[str, float]:
    true_mask = instance_table["driver_true"] == 1
    false_mask = instance_table["driver_true"] == 0
    true_attention = instance_table.filter(true_mask)["attention"].to_numpy().astype(np.float32)
    false_attention = instance_table.filter(false_mask)["attention"].to_numpy().astype(np.float32)

    summary = {
        "mean_attention_driver": float(np.mean(true_attention)) if true_attention.size else 0.0,
        "mean_attention_non_driver": float(np.mean(false_attention)) if false_attention.size else 0.0,
        "driver_count": int(true_attention.size),
        "non_driver_count": int(false_attention.size),
    }
    if instance_table.height > 0:
        try:
            summary["attention_auc"] = float(
                roc_auc_score(instance_table["driver_true"].to_numpy().astype(np.int64), instance_table["attention"].to_numpy().astype(np.float32))
            )
        except ValueError:
            summary["attention_auc"] = 0.0
    else:
        summary["attention_auc"] = 0.0
    return summary


def plot_attention_diagnostics(
    instance_table: pl.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = instance_table.to_pandas()
    df["driver_true"] = df["driver_true"].astype(int)

    fig, ax = plt.subplots(figsize=(6, 4))
    df.boxplot(column="attention", by="driver_true", ax=ax)
    ax.set_title("Attention by True Driver Status")
    ax.set_xlabel("driver_true")
    ax.set_ylabel("attention")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def compute_performance_metrics(
    bag_predictions: pl.DataFrame,
    attention_table: pl.DataFrame,
) -> pl.DataFrame:
    """Compute sample-level and instance-level metrics for the simulated run."""
    bag_labels = bag_predictions["bag_label"].to_numpy().astype(np.int64)
    bag_scores = bag_predictions["bag_prediction"].to_numpy().astype(np.float32)
    bag_calls = (bag_scores >= 0.5).astype(np.int64)

    if bag_labels.size:
        sample_accuracy = float(accuracy_score(bag_labels, bag_calls))
    else:
        sample_accuracy = float("nan")

    driver_labels = attention_table["driver_true"].to_numpy().astype(np.int64)
    attention_scores = attention_table["attention"].to_numpy().astype(np.float32)

    return pl.DataFrame(
        {
            "metric": [
                "sample_accuracy_0_5",
                "sample_roc_auc",
                "attention_roc_auc",
            ],
            "label": [
                "Sample accuracy",
                "Sample ROC AUC",
                "Attention ROC AUC",
            ],
            "value": [
                sample_accuracy,
                _safe_roc_auc(bag_labels, bag_scores),
                _safe_roc_auc(driver_labels, attention_scores),
            ],
        }
    )


def plot_performance_metrics(
    metrics_table: pl.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = metrics_table["label"].to_list()
    values = metrics_table["value"].to_numpy().astype(np.float32)
    plot_values = np.nan_to_num(values, nan=0.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, plot_values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("Simulated-data performance metrics")
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        label = "NA" if np.isnan(value) else f"{value:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0.0
    return float(d)


def compute_sequence_features(sequences: list[str]) -> dict[str, list[float]]:
    """Compute amino acid frequencies and top di-mer frequencies from sequences.
    
    Returns a dict with instance-level features: amino acid proportions and di-mer counts.
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    
    aa_freqs = {f"aa_{aa}": [] for aa in amino_acids}
    dimer_counts = {}
    
    # First pass: collect all dimers to find top ones
    all_dimers = []
    for seq in sequences:
        for i in range(len(seq) - 1):
            all_dimers.append(seq[i:i+2])
    
    # Get top 5 most common dimers
    if all_dimers:
        from collections import Counter
        top_dimers = [dimer for dimer, _ in Counter(all_dimers).most_common(5)]
    else:
        top_dimers = []
    
    dimer_freqs = {f"dimer_{dimer}": [] for dimer in top_dimers}
    
    # Second pass: compute features per sequence
    for seq in sequences:
        # Amino acid frequencies
        seq_len = len(seq) if seq else 1
        for aa in amino_acids:
            aa_freqs[f"aa_{aa}"].append(seq.count(aa) / seq_len if seq else 0.0)
        
        # Di-mer frequencies
        seq_dimers = [seq[i:i+2] for i in range(len(seq) - 1)] if len(seq) > 1 else []
        for dimer in top_dimers:
            dimer_freqs[f"dimer_{dimer}"].append(seq_dimers.count(dimer) / len(seq_dimers) if seq_dimers else 0.0)
    
    return {**aa_freqs, **dimer_freqs}


def add_repertoire_features_to_dataframe(
    data: pl.DataFrame,
    seq_col: str,
    prefix: str,
) -> tuple[pl.DataFrame, list[str]]:
    """Add sequence composition features to a DataFrame.
    
    Args:
        data: DataFrame with instance_id and sequence column.
        seq_col: Name of sequence column.
        prefix: Prefix for feature names (e.g., "tcr", "bcr").
    
    Returns:
        Tuple of (DataFrame with new repertoire feature columns, list of feature column names).
    """
    sequences = data.select(seq_col).to_pandas()[seq_col].tolist()
    features = compute_sequence_features(sequences)
    
    # Rename features with prefix and ensure numeric type
    prefixed_features = {}
    for key, values in features.items():
        col_name = f"{prefix}_{key}"
        prefixed_features[col_name] = [float(v) for v in values]
    
    features_df = pl.from_dict(prefixed_features)
    
    # Add instance_id to match with original data
    instance_ids = data.select("instance_id").to_pandas()["instance_id"].tolist()
    features_df = features_df.with_columns(
        pl.Series("instance_id", instance_ids)
    )
    
    # Join and return feature column names
    result = data.join(features_df, on="instance_id")
    feature_cols = list(prefixed_features.keys())
    
    return result, feature_cols


def compute_feature_statistics(
    data: pl.DataFrame,
    driver_ids: set[str],
    nondriver_ids: set[str],
    feature_columns: list[str],
) -> pl.DataFrame:
    """Compute statistics (p-value, Cohen's d) for each feature.
    
    Returns a DataFrame with columns: feature, cohens_d, pvalue, neg_log_pval.
    """
    driver_data = data.filter(pl.col("instance_id").is_in(list(driver_ids))).select(feature_columns).to_numpy()
    nondriver_data = data.filter(pl.col("instance_id").is_in(list(nondriver_ids))).select(feature_columns).to_numpy()
    
    stats_list = []
    for i, feat_col in enumerate(feature_columns):
        driver_col = driver_data[:, i]
        nondriver_col = nondriver_data[:, i]
        
        _, pval = stats.ttest_ind(driver_col, nondriver_col)
        d = compute_cohens_d(driver_col, nondriver_col)
        
        stats_list.append({
            "feature": feat_col,
            "cohens_d": d,
            "pvalue": float(pval),
            "neg_log_pval": -np.log10(max(pval, 1e-10)),
        })
    
    return pl.from_dicts(stats_list).sort("neg_log_pval", descending=True)


def plot_volcano(
    data: pl.DataFrame,
    driver_ids: set[str],
    nondriver_ids: set[str],
    feature_columns: list[str],
    output_path: str | Path,
    fc_threshold: float = 0.5,
    pval_threshold: float = 0.05,
    title: str = "Volcano Plot",
    top_n_labels: int = 5,
) -> pl.DataFrame:
    """Plot volcano plot with top feature labels and return statistics.
    
    Args:
        data: DataFrame with feature columns and instance identifiers.
        driver_ids: Set of instance IDs for driver group.
        nondriver_ids: Set of instance IDs for non-driver group.
        feature_columns: List of column names to analyze.
        output_path: Where to save the plot.
        fc_threshold: Effect size threshold for significance marking.
        pval_threshold: P-value threshold for significance marking.
        title: Plot title.
        top_n_labels: Number of top features to label on plot.
    
    Returns:
        DataFrame with feature statistics.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute statistics
    stats_df = compute_feature_statistics(data, driver_ids, nondriver_ids, feature_columns)
    
    cohens_d_vals = stats_df["cohens_d"].to_numpy()
    neg_log_pvals = stats_df["neg_log_pval"].to_numpy()
    features = stats_df["feature"].to_list()
    
    # Create volcano plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Neutral points (gray)
    mask_neutral = (np.abs(cohens_d_vals) < fc_threshold) | (neg_log_pvals < -np.log10(pval_threshold))
    ax.scatter(cohens_d_vals[mask_neutral], neg_log_pvals[mask_neutral], alpha=0.5, s=50, c="gray", label="Not significant")
    
    # Significant points (red) - positive Cohen's d
    mask_sig_pos = (~mask_neutral) & (cohens_d_vals > 0)
    ax.scatter(cohens_d_vals[mask_sig_pos], neg_log_pvals[mask_sig_pos], alpha=0.7, s=80, c="red", label="Enriched in drivers")
    
    # Significant points (blue) - negative Cohen's d
    mask_sig_neg = (~mask_neutral) & (cohens_d_vals < 0)
    ax.scatter(cohens_d_vals[mask_sig_neg], neg_log_pvals[mask_sig_neg], alpha=0.7, s=80, c="blue", label="Enriched in non-drivers")
    
    # Add threshold lines
    ax.axvline(-fc_threshold, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(fc_threshold, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(-np.log10(pval_threshold), color="black", linestyle="--", linewidth=1, alpha=0.5)
    
    # Label top features with repulsion
    top_idx = np.argsort(neg_log_pvals)[-top_n_labels:]
    texts = []
    for idx in top_idx:
        texts.append(ax.text(
            cohens_d_vals[idx], 
            neg_log_pvals[idx], 
            features[idx],
            fontsize=9,
            alpha=0.8,
        ))
    
    # Adjust labels to avoid overlaps
    adjust_text(texts, arrowprops=dict(arrowstyle='-', lw=0.5, alpha=0.5), ax=ax)
    
    ax.set_xlabel("Cohen's d (effect size)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(alpha=0.3)
    
    # Make x-axis symmetric
    max_d = np.max(np.abs(cohens_d_vals))
    ax.set_xlim(-max_d * 1.1, max_d * 1.1)
    
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    
    return stats_df
