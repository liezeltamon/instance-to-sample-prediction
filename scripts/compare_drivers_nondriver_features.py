from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy import stats

# Ensure repo root is on sys.path when running this script directly from scripts/.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data import generate_synthetic_bag_data
from utils.wrangling import ensure_dir, load_table


def analyze_rna_features(
    data: pl.DataFrame,
    drivers: set[str],
    nondriver_sample: set[str],
    n_top: int = 6,
) -> tuple[list[str], np.ndarray, np.ndarray, list[float], list[float]]:
    """Identify RNA features by significance and effect size (Cohen's d).
    
    Returns top features ranked by combined p-value and effect size.
    """
    rna_cols = [c for c in data.columns if c.startswith("rna_")]
    
    driver_data = data.filter(pl.col("instance_id").is_in(list(drivers))).select(rna_cols).to_numpy()
    nondriver_data = data.filter(pl.col("instance_id").is_in(list(nondriver_sample))).select(rna_cols).to_numpy()
    
    # Compute t-test and Cohen's d for each feature
    pvals = []
    cohens_d = []
    for i in range(len(rna_cols)):
        driver_col = driver_data[:, i]
        nondriver_col = nondriver_data[:, i]
        
        _, pval = stats.ttest_ind(driver_col, nondriver_col)
        
        # Cohen's d
        n1, n2 = len(driver_col), len(nondriver_col)
        var1, var2 = np.var(driver_col, ddof=1), np.var(nondriver_col, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (np.mean(driver_col) - np.mean(nondriver_col)) / pooled_std if pooled_std > 0 else 0.0
        
        pvals.append(pval)
        cohens_d.append(np.abs(d))
    
    # Rank by combined score: -log10(p) * |Cohen's d|
    # Replace p=0 with small value to avoid log(0)
    pvals_safe = np.array([max(p, 1e-10) for p in pvals])
    scores = -np.log10(pvals_safe) * np.array(cohens_d)
    
    # Select top N by combined score
    top_idx = np.argsort(scores)[-n_top:][::-1]
    top_features = [rna_cols[i] for i in top_idx]
    top_pvals = [pvals[i] for i in top_idx]
    top_cohens_d = [cohens_d[i] for i in top_idx]
    
    top_driver = driver_data[:, top_idx]
    top_nondriver = nondriver_data[:, top_idx]
    
    return top_features, top_driver, top_nondriver, top_pvals, top_cohens_d


def analyze_sequence_features(
    data: pl.DataFrame,
    drivers: set[str],
    nondriver_sample: set[str],
    seq_col: str = "tcr_sequence",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract sequence properties: motif presence, GC content, length."""
    driver_seqs = data.filter(pl.col("instance_id").is_in(list(drivers))).select(seq_col).to_pandas()[seq_col].tolist()
    nondriver_seqs = data.filter(pl.col("instance_id").is_in(list(nondriver_sample))).select(seq_col).to_pandas()[seq_col].tolist()
    
    def seq_features(seqs: list[str]) -> np.ndarray:
        features = []
        for seq in seqs:
            gc = (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0
            has_motif = 1.0 if seq_col == "tcr_sequence" and "CASS" in seq else (1.0 if seq_col == "bcr_sequence" and "ARDY" in seq else 0.0)
            features.append([len(seq), gc, has_motif])
        return np.array(features)
    
    driver_feats = seq_features(driver_seqs)
    nondriver_feats = seq_features(nondriver_seqs)
    return driver_feats, nondriver_feats


def plot_rna_comparison(
    top_features: list[str],
    driver_data: np.ndarray,
    nondriver_data: np.ndarray,
    output_path: Path,
) -> None:
    """Plot boxplots of top different RNA features."""
    n_features = len(top_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, (feature, ax) in enumerate(zip(top_features, axes)):
        data_to_plot = [driver_data[:, idx], nondriver_data[:, idx]]
        ax.boxplot(data_to_plot, labels=["Driver", "Non-driver"])
        ax.set_ylabel("Feature value")
        ax.set_title(feature)
        ax.grid(axis="y", alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")
    
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_rna_volcano(
    data: pl.DataFrame,
    drivers: set[str],
    nondriver_sample: set[str],
    output_path: Path,
    fc_threshold: float = 0.5,
    pval_threshold: float = 0.05,
) -> None:
    """Plot volcano plot: Cohen's d (x-axis) vs -log10(p-value) (y-axis) for RNA features."""
    rna_cols = [c for c in data.columns if c.startswith("rna_")]
    
    driver_data = data.filter(pl.col("instance_id").is_in(list(drivers))).select(rna_cols).to_numpy()
    nondriver_data = data.filter(pl.col("instance_id").is_in(list(nondriver_sample))).select(rna_cols).to_numpy()
    
    cohens_d = []
    neg_log_pvals = []
    
    for i in range(len(rna_cols)):
        driver_col = driver_data[:, i]
        nondriver_col = nondriver_data[:, i]
        
        _, pval = stats.ttest_ind(driver_col, nondriver_col)
        
        # Cohen's d (signed)
        n1, n2 = len(driver_col), len(nondriver_col)
        var1, var2 = np.var(driver_col, ddof=1), np.var(nondriver_col, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (np.mean(driver_col) - np.mean(nondriver_col)) / pooled_std if pooled_std > 0 else 0.0
        
        cohens_d.append(d)
        neg_log_pvals.append(-np.log10(max(pval, 1e-10)))
    
    cohens_d = np.array(cohens_d)
    neg_log_pvals = np.array(neg_log_pvals)
    
    # Create volcano plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Neutral points (gray)
    mask_neutral = (np.abs(cohens_d) < fc_threshold) | (neg_log_pvals < -np.log10(pval_threshold))
    ax.scatter(cohens_d[mask_neutral], neg_log_pvals[mask_neutral], alpha=0.5, s=50, c="gray", label="Not significant")
    
    # Significant points (red)
    mask_sig = (~mask_neutral) & (cohens_d > 0)
    ax.scatter(cohens_d[mask_sig], neg_log_pvals[mask_sig], alpha=0.7, s=80, c="red", label="Enriched in drivers")
    
    # Significant points (blue) - negative Cohen's d
    mask_sig_neg = (~mask_neutral) & (cohens_d < 0)
    ax.scatter(cohens_d[mask_sig_neg], neg_log_pvals[mask_sig_neg], alpha=0.7, s=80, c="blue", label="Enriched in non-drivers")
    
    # Add threshold lines
    ax.axvline(-fc_threshold, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(fc_threshold, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(-np.log10(pval_threshold), color="black", linestyle="--", linewidth=1, alpha=0.5)
    
    ax.set_xlabel("Cohen's d (driver effect size)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("RNA Features: Volcano Plot (Drivers vs Non-drivers)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_sequence_comparison(
    driver_feats: np.ndarray,
    nondriver_feats: np.ndarray,
    seq_type: str,
    output_path: Path,
) -> None:
    """Plot sequence properties: length, GC content, motif presence."""
    feature_names = ["Length", "GC content", "Motif present"]
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for idx, (ax, feature_name) in enumerate(zip(axes, feature_names)):
        driver_col = driver_feats[:, idx]
        nondriver_col = nondriver_feats[:, idx]
        
        data_to_plot = [driver_col, nondriver_col]
        ax.boxplot(data_to_plot, labels=["Driver", "Non-driver"])
        ax.set_ylabel(feature_name)
        ax.set_title(f"{seq_type} — {feature_name}")
        ax.grid(axis="y", alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def main() -> None:
    # Load original data (regenerate with same seed for reproducibility)
    data = generate_synthetic_bag_data(n_bags=20, bag_size=40, n_rna_features=12, drivers_per_positive_bag=5, seed=0)
    
    # Load attention table
    attention_table = load_table(Path("results/run_pipeline_simulated/instance_attention.parquet"))
    
    # Separate drivers and non-drivers
    drivers = set(attention_table.filter(pl.col("driver_true") == 1)["instance_id"].to_list())
    nondriver_ids = attention_table.filter(pl.col("driver_true") == 0)["instance_id"].to_list()
    
    # Subsample non-drivers to match driver count
    nondriver_sample = set(np.random.choice(nondriver_ids, size=len(drivers), replace=False))
    
    output_root = Path("results/driver_feature_comparison")
    ensure_dir(output_root)
    
    # RNA analysis
    print("Analyzing RNA features...")
    top_rna, driver_rna, nondriver_rna, top_pvals, top_cohens_d = analyze_rna_features(data, drivers, nondriver_sample, n_top=6)
    print(f"  Top differentiating RNA features: {top_rna}")
    for feat, pval, d in zip(top_rna, top_pvals, top_cohens_d):
        print(f"    {feat}: p={pval:.2e}, Cohen's d={d:.3f}")
    plot_rna_comparison(top_rna, driver_rna, nondriver_rna, output_root / "rna_comparison.png")
    plot_rna_volcano(data, drivers, nondriver_sample, output_root / "rna_volcano.png")
    
    # TCR analysis
    print("Analyzing TCR sequence features...")
    driver_tcr, nondriver_tcr = analyze_sequence_features(data, drivers, nondriver_sample, seq_col="tcr_sequence")
    plot_sequence_comparison(driver_tcr, nondriver_tcr, "TCR", output_root / "tcr_comparison.png")
    
    # BCR analysis
    print("Analyzing BCR sequence features...")
    driver_bcr, nondriver_bcr = analyze_sequence_features(data, drivers, nondriver_sample, seq_col="bcr_sequence")
    plot_sequence_comparison(driver_bcr, nondriver_bcr, "BCR", output_root / "bcr_comparison.png")
    
    print(f"Wrote comparison plots to {output_root}")


if __name__ == "__main__":
    main()
