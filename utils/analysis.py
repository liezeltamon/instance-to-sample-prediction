from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from adjustText import adjust_text
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score


def summarize_attention_by_truth(instance_table: pl.DataFrame) -> dict[str, float]:
    true_attention = instance_table.filter(pl.col("driver_true") == 1)["attention"].to_numpy().astype(np.float32)
    false_attention = instance_table.filter(pl.col("driver_true") == 0)["attention"].to_numpy().astype(np.float32)

    summary = {
        "mean_attention_driver": float(np.mean(true_attention)) if true_attention.size else 0.0,
        "mean_attention_non_driver": float(np.mean(false_attention)) if false_attention.size else 0.0,
        "driver_count": int(true_attention.size),
        "non_driver_count": int(false_attention.size),
    }
    if instance_table.height > 0:
        summary["attention_auc"] = _safe_roc_auc(
            instance_table["driver_true"].to_numpy().astype(np.int64),
            instance_table["attention"].to_numpy().astype(np.float32),
        )
    else:
        summary["attention_auc"] = float("nan")
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
    """Compute sample-level and instance-level metrics for simulated runs."""
    bag_labels = bag_predictions["bag_label"].to_numpy().astype(np.int64)
    bag_scores = bag_predictions["bag_prediction"].to_numpy().astype(np.float32)
    bag_calls = (bag_scores >= 0.5).astype(np.int64)

    sample_accuracy = float(accuracy_score(bag_labels, bag_calls)) if bag_labels.size else float("nan")

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


def sample_driver_and_nondriver_ids(
    instance_table: pl.DataFrame,
    *,
    id_col: str = "instance_id",
    label_col: str = "driver_true",
    group_col: str | None = None,
    seed: int = 0,
) -> tuple[set[str], set[str], pl.DataFrame]:
    """Sample matched driver and non-driver IDs, optionally within groups."""
    rng = np.random.default_rng(seed)
    selected_drivers: set[str] = set()
    selected_nondrivers: set[str] = set()
    summary_rows = []

    if group_col is None:
        groups = [("all", instance_table)]
    else:
        group_values = instance_table.select(group_col).unique().sort(group_col)[group_col].to_list()
        groups = [(str(value), instance_table.filter(pl.col(group_col) == value)) for value in group_values]

    for group_name, group_table in groups:
        driver_pool = group_table.filter(pl.col(label_col) == 1)[id_col].to_list()
        nondriver_pool = group_table.filter(pl.col(label_col) == 0)[id_col].to_list()

        if not driver_pool:
            summary_rows.append(
                {
                    "group": group_name,
                    "n_driver_available": 0,
                    "n_nondriver_available": len(nondriver_pool),
                    "n_driver": 0,
                    "n_nondriver": 0,
                    "status": "skipped_no_drivers",
                }
            )
            continue
        if not nondriver_pool:
            raise ValueError(f"Group {group_name!r} has drivers but no non-drivers to compare against.")

        sample_size = min(len(driver_pool), len(nondriver_pool))
        group_drivers = rng.choice(driver_pool, size=sample_size, replace=False).tolist()
        group_nondrivers = rng.choice(nondriver_pool, size=sample_size, replace=False).tolist()

        selected_drivers.update(group_drivers)
        selected_nondrivers.update(group_nondrivers)
        summary_rows.append(
            {
                "group": group_name,
                "n_driver_available": len(driver_pool),
                "n_nondriver_available": len(nondriver_pool),
                "n_driver": sample_size,
                "n_nondriver": sample_size,
                "status": "matched" if sample_size == len(driver_pool) else "downsampled_drivers",
            }
        )

    return selected_drivers, selected_nondrivers, pl.from_dicts(summary_rows)


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    group1 = np.asarray(group1, dtype=np.float64)
    group2 = np.asarray(group2, dtype=np.float64)
    if group1.size == 0 or group2.size == 0:
        return float("nan")

    var1 = np.var(group1, ddof=1) if group1.size > 1 else 0.0
    var2 = np.var(group2, ddof=1) if group2.size > 1 else 0.0
    pooled_n = group1.size + group2.size - 2
    if pooled_n <= 0:
        return 0.0

    pooled_std = np.sqrt(((group1.size - 1) * var1 + (group2.size - 1) * var2) / pooled_n)
    if not np.isfinite(pooled_std) or pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def _safe_ttest_pvalue(group1: np.ndarray, group2: np.ndarray) -> float:
    if group1.size < 2 or group2.size < 2:
        return float("nan")
    _, pvalue = stats.ttest_ind(group1, group2, equal_var=False, nan_policy="omit")
    if not np.isfinite(pvalue):
        return 1.0
    return float(pvalue)


def compute_grouped_feature_statistics(
    data: pl.DataFrame,
    driver_ids: set[str],
    nondriver_ids: set[str],
    feature_sets: dict[str, list[str]],
    *,
    id_col: str = "instance_id",
    group_col: str | None = None,
    active_threshold: float = 1e-12,
) -> pl.DataFrame:
    """Compute feature statistics per modality and optional group."""
    if group_col is None:
        groups = [("all", data)]
    else:
        group_values = data.select(group_col).unique().sort(group_col)[group_col].to_list()
        groups = [(str(value), data.filter(pl.col(group_col) == value)) for value in group_values]

    stats_rows = []
    for group_name, group_data in groups:
        driver_data = group_data.filter(pl.col(id_col).is_in(list(driver_ids)))
        nondriver_data = group_data.filter(pl.col(id_col).is_in(list(nondriver_ids)))
        if driver_data.height == 0 or nondriver_data.height == 0:
            continue

        for modality, feature_columns in feature_sets.items():
            existing_columns = [col for col in feature_columns if col in group_data.columns]
            if not existing_columns:
                continue

            modality_values = group_data.select(existing_columns).to_numpy()
            if not np.any(np.abs(modality_values) > active_threshold):
                continue

            driver_matrix = driver_data.select(existing_columns).to_numpy()
            nondriver_matrix = nondriver_data.select(existing_columns).to_numpy()

            for feature_idx, feature in enumerate(existing_columns):
                driver_col = driver_matrix[:, feature_idx].astype(np.float64)
                nondriver_col = nondriver_matrix[:, feature_idx].astype(np.float64)
                pvalue = _safe_ttest_pvalue(driver_col, nondriver_col)
                if np.isnan(pvalue):
                    neg_log_pval = float("nan")
                else:
                    neg_log_pval = float(-np.log10(max(pvalue, 1e-10)))

                stats_rows.append(
                    {
                        "group": group_name,
                        "modality": modality,
                        "feature": feature,
                        "cohens_d": compute_cohens_d(driver_col, nondriver_col),
                        "pvalue": pvalue,
                        "neg_log_pval": neg_log_pval,
                        "n_driver": driver_data.height,
                        "n_nondriver": nondriver_data.height,
                    }
                )

    if not stats_rows:
        return pl.DataFrame(
            schema={
                "group": pl.String,
                "modality": pl.String,
                "feature": pl.String,
                "cohens_d": pl.Float64,
                "pvalue": pl.Float64,
                "neg_log_pval": pl.Float64,
                "n_driver": pl.Int64,
                "n_nondriver": pl.Int64,
            }
        )
    return pl.from_dicts(stats_rows).sort(["group", "modality", "neg_log_pval"], descending=[False, False, True])


def plot_grouped_volcano(
    stats_table: pl.DataFrame,
    output_path: str | Path,
    *,
    fc_threshold: float = 0.5,
    pval_threshold: float = 0.05,
    top_n_labels: int = 3,
) -> None:
    """Plot one volcano panel per group/modality comparison."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if stats_table.height == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No comparisons available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        return

    panel_keys = []
    for row in stats_table.select(["group", "modality"]).unique(maintain_order=True).iter_rows(named=True):
        panel_keys.append((row["group"], row["modality"]))

    n_panels = len(panel_keys)
    n_cols = min(2, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 4.3 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, (group_name, modality) in zip(axes_flat, panel_keys):
        panel = stats_table.filter((pl.col("group") == group_name) & (pl.col("modality") == modality))
        cohens_d_vals = panel["cohens_d"].to_numpy().astype(np.float64)
        neg_log_pvals = panel["neg_log_pval"].to_numpy().astype(np.float64)
        features = panel["feature"].to_list()
        n_driver = int(panel["n_driver"][0])
        n_nondriver = int(panel["n_nondriver"][0])

        finite_mask = np.isfinite(cohens_d_vals) & np.isfinite(neg_log_pvals)
        neutral_mask = finite_mask & (
            (np.abs(cohens_d_vals) < fc_threshold)
            | (neg_log_pvals < -np.log10(pval_threshold))
        )
        sig_pos_mask = finite_mask & (~neutral_mask) & (cohens_d_vals > 0)
        sig_neg_mask = finite_mask & (~neutral_mask) & (cohens_d_vals < 0)

        ax.scatter(cohens_d_vals[neutral_mask], neg_log_pvals[neutral_mask], alpha=0.5, s=34, c="gray", label="Not significant")
        ax.scatter(cohens_d_vals[sig_pos_mask], neg_log_pvals[sig_pos_mask], alpha=0.75, s=44, c="#D62728", label="Driver-enriched")
        ax.scatter(cohens_d_vals[sig_neg_mask], neg_log_pvals[sig_neg_mask], alpha=0.75, s=44, c="#1F77B4", label="Non-driver-enriched")
        ax.axvline(-fc_threshold, color="black", linestyle="--", linewidth=0.8, alpha=0.45)
        ax.axvline(fc_threshold, color="black", linestyle="--", linewidth=0.8, alpha=0.45)
        ax.axhline(-np.log10(pval_threshold), color="black", linestyle="--", linewidth=0.8, alpha=0.45)

        labelable = np.where(finite_mask)[0]
        if labelable.size:
            top_idx = labelable[np.argsort(neg_log_pvals[labelable])[-min(top_n_labels, labelable.size):]]
            texts = [
                ax.text(
                    cohens_d_vals[idx],
                    neg_log_pvals[idx],
                    features[idx],
                    fontsize=7,
                    alpha=0.85,
                )
                for idx in top_idx
            ]
            adjust_text(texts, arrowprops=dict(arrowstyle="-", lw=0.4, alpha=0.4), ax=ax)

        max_d = np.nanmax(np.abs(cohens_d_vals)) if cohens_d_vals.size else 1.0
        if not np.isfinite(max_d) or max_d == 0:
            max_d = 1.0
        ax.set_xlim(-max_d * 1.15, max_d * 1.15)
        ax.set_xlabel("Cohen's d")
        ax.set_ylabel("-log10(p-value)")
        ax.set_title(f"{group_name} / {modality}", fontsize=10, pad=6)
        ax.text(
            0.02,
            0.98,
            f"drivers={n_driver}, non-drivers={n_nondriver}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
        )
        ax.grid(alpha=0.25)

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(handles)), fontsize=8)
    fig.suptitle("Within-group driver vs matched non-driver feature diagnostics", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
