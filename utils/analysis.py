from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from adjustText import adjust_text
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import leaves_list, linkage
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


def _safe_paired_ttest_pvalue(group1: np.ndarray, group2: np.ndarray) -> float:
    if group1.size < 2 or group2.size < 2:
        return float("nan")
    _, pvalue = stats.ttest_rel(group1, group2, nan_policy="omit")
    if not np.isfinite(pvalue):
        return 1.0
    return float(pvalue)


def _paired_effect_size(group1: np.ndarray, group2: np.ndarray) -> float:
    diff = np.asarray(group1, dtype=np.float64) - np.asarray(group2, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan")
    if diff.size == 1:
        return 0.0
    diff_sd = np.std(diff, ddof=1)
    if not np.isfinite(diff_sd) or diff_sd == 0:
        return 0.0
    return float(np.mean(diff) / diff_sd)


def _neg_log_pvalue(pvalue: float) -> float:
    if not np.isfinite(pvalue):
        return float("nan")
    return float(-np.log10(max(pvalue, 1e-300)))


def _iter_group_tables(data: pl.DataFrame, group_col: str | None) -> list[tuple[str, pl.DataFrame]]:
    if group_col is None:
        return [("all", data)]
    group_values = data.select(group_col).unique().sort(group_col)[group_col].to_list()
    return [(str(value), data.filter(pl.col(group_col) == value)) for value in group_values]


def _active_feature_columns(
    data: pl.DataFrame,
    feature_columns: list[str],
    *,
    active_threshold: float,
) -> list[str]:
    existing_columns = [column for column in feature_columns if column in data.columns]
    if not existing_columns:
        return []

    values = data.select(existing_columns).to_numpy().astype(np.float64)
    values = np.nan_to_num(values, nan=0.0)
    active = np.any(np.abs(values) > active_threshold, axis=0)
    return [column for column, is_active in zip(existing_columns, active) if bool(is_active)]


def _prepare_attention_feature_table(
    attention_table: pl.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    feature_table: pl.DataFrame | None,
    id_col: str,
) -> pl.DataFrame:
    if feature_table is None:
        return attention_table
    if id_col not in feature_table.columns:
        raise ValueError(f"feature_table must include {id_col!r}.")
    if feature_table[id_col].n_unique() != feature_table.height:
        raise ValueError(f"feature_table must contain one row per {id_col!r}.")

    feature_columns = []
    seen = set()
    for columns in feature_sets.values():
        for column in columns:
            if column in feature_table.columns and column not in seen:
                feature_columns.append(column)
                seen.add(column)

    feature_data = feature_table.select([id_col, *feature_columns])
    overlapping_feature_columns = [column for column in feature_columns if column in attention_table.columns]
    base_table = attention_table.drop(overlapping_feature_columns) if overlapping_feature_columns else attention_table
    return base_table.join(feature_data, on=id_col, how="left")


def _scored_top_attention_table(
    attention_table: pl.DataFrame,
    *,
    bag_col: str,
    id_col: str,
    attention_col: str,
    top_fraction: float,
) -> pl.DataFrame:
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be greater than 0 and less than or equal to 1.")
    for column in [bag_col, id_col, attention_col]:
        if column not in attention_table.columns:
            raise ValueError(f"attention_table must include {column!r}.")

    rows = []
    for bag_id in attention_table[bag_col].unique().to_list():
        bag_table = attention_table.filter(pl.col(bag_col) == bag_id).sort(attention_col, descending=True)
        n_in_bag = bag_table.height
        n_top = max(1, int(np.ceil(n_in_bag * top_fraction)))
        for rank, row in enumerate(bag_table.select([bag_col, id_col]).iter_rows(named=True), start=1):
            rows.append(
                {
                    bag_col: row[bag_col],
                    id_col: row[id_col],
                    "attention_rank_in_bag": rank,
                    "n_instances_in_bag": n_in_bag,
                    "n_top_instances_in_bag": n_top,
                    "top_attention": rank <= n_top,
                }
            )

    if not rows:
        return attention_table.with_columns(
            [
                pl.Series("attention_rank_in_bag", [], dtype=pl.Int64),
                pl.Series("n_instances_in_bag", [], dtype=pl.Int64),
                pl.Series("n_top_instances_in_bag", [], dtype=pl.Int64),
                pl.Series("top_attention", [], dtype=pl.Boolean),
            ]
        )

    return attention_table.join(pl.from_dicts(rows), on=[bag_col, id_col], how="left").sort(
        [bag_col, "attention_rank_in_bag"]
    )


def select_top_attention_instances(
    attention_table: pl.DataFrame,
    *,
    bag_col: str = "bag_id",
    id_col: str = "instance_id",
    attention_col: str = "attention",
    top_fraction: float = 0.10,
) -> pl.DataFrame:
    """Select top-attention instances within each bag/sample."""
    return _scored_top_attention_table(
        attention_table,
        bag_col=bag_col,
        id_col=id_col,
        attention_col=attention_col,
        top_fraction=top_fraction,
    ).filter(pl.col("top_attention"))


def _empty_pseudobulk_table() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "bag_id": pl.String,
            "bag_label": pl.String,
            "group": pl.String,
            "feature_type": pl.String,
            "attention_subset": pl.String,
            "feature": pl.String,
            "value": pl.Float64,
            "n_instances": pl.Int64,
            "mean_attention": pl.Float64,
            "median_attention": pl.Float64,
            "sd_attention": pl.Float64,
        }
    )


def compute_top_attention_pseudobulk(
    attention_table: pl.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    feature_table: pl.DataFrame | None = None,
    group_col: str | None = None,
    bag_col: str = "bag_id",
    id_col: str = "instance_id",
    label_col: str = "bag_label",
    attention_col: str = "attention",
    top_fraction: float = 0.10,
    active_threshold: float = 1e-12,
) -> pl.DataFrame:
    """Create sample-level pseudobulks for top-attention and rest instances."""
    for column in [bag_col, id_col, label_col, attention_col]:
        if column not in attention_table.columns:
            raise ValueError(f"attention_table must include {column!r}.")

    scored_attention = _scored_top_attention_table(
        attention_table,
        bag_col=bag_col,
        id_col=id_col,
        attention_col=attention_col,
        top_fraction=top_fraction,
    )
    data = _prepare_attention_feature_table(
        scored_attention,
        feature_sets,
        feature_table=feature_table,
        id_col=id_col,
    )

    rows = []
    for group_name, group_data in _iter_group_tables(data, group_col):
        for feature_type, feature_columns in feature_sets.items():
            active_columns = _active_feature_columns(
                group_data,
                feature_columns,
                active_threshold=active_threshold,
            )
            if not active_columns:
                continue

            for bag_id in group_data[bag_col].unique().to_list():
                bag_data = group_data.filter(pl.col(bag_col) == bag_id)
                if bag_data.height == 0:
                    continue
                bag_label = str(bag_data[label_col][0])

                for subset_name, subset_data in [
                    ("top", bag_data.filter(pl.col("top_attention"))),
                    ("rest", bag_data.filter(~pl.col("top_attention"))),
                ]:
                    if subset_data.height == 0:
                        continue

                    attention_values = subset_data[attention_col].to_numpy().astype(np.float64)
                    feature_values = subset_data.select(active_columns).to_numpy().astype(np.float64)
                    feature_means = np.nanmean(feature_values, axis=0)
                    sd_attention = float(np.std(attention_values, ddof=1)) if attention_values.size > 1 else 0.0

                    for feature, value in zip(active_columns, feature_means):
                        rows.append(
                            {
                                "bag_id": str(bag_id),
                                "bag_label": bag_label,
                                "group": group_name,
                                "feature_type": feature_type,
                                "attention_subset": subset_name,
                                "feature": feature,
                                "value": float(value),
                                "n_instances": subset_data.height,
                                "mean_attention": float(np.mean(attention_values)),
                                "median_attention": float(np.median(attention_values)),
                                "sd_attention": sd_attention,
                            }
                        )

    if not rows:
        return _empty_pseudobulk_table()
    return pl.from_dicts(rows).sort(["group", "feature_type", "bag_id", "attention_subset", "feature"])


def _empty_feature_comparisons_table() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "comparison": pl.String,
            "comparison_label": pl.String,
            "group": pl.String,
            "feature_type": pl.String,
            "feature": pl.String,
            "label_a": pl.String,
            "label_b": pl.String,
            "effect_size": pl.Float64,
            "pvalue": pl.Float64,
            "neg_log_pvalue": pl.Float64,
            "significant": pl.Boolean,
            "n_samples_a": pl.Int64,
            "n_samples_b": pl.Int64,
        }
    )


def _append_comparison_row(
    rows: list[dict[str, object]],
    *,
    comparison: str,
    comparison_label: str,
    group_name: str,
    feature_type: str,
    feature: str,
    label_a: str,
    label_b: str,
    values_a: np.ndarray,
    values_b: np.ndarray,
    pval_threshold: float,
    paired: bool = False,
) -> None:
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    if values_a.size == 0 or values_b.size == 0:
        return

    if paired:
        n = min(values_a.size, values_b.size)
        pair_values = np.column_stack([values_a[:n], values_b[:n]])
        pair_values = pair_values[np.isfinite(pair_values).all(axis=1)]
        if pair_values.shape[0] == 0:
            return
        values_a = pair_values[:, 0]
        values_b = pair_values[:, 1]
        effect_size = _paired_effect_size(values_a, values_b)
        pvalue = _safe_paired_ttest_pvalue(values_a, values_b)
    else:
        values_a = values_a[np.isfinite(values_a)]
        values_b = values_b[np.isfinite(values_b)]
        if values_a.size == 0 or values_b.size == 0:
            return
        effect_size = compute_cohens_d(values_a, values_b)
        pvalue = _safe_ttest_pvalue(values_a, values_b)

    rows.append(
        {
            "comparison": comparison,
            "comparison_label": comparison_label,
            "group": group_name,
            "feature_type": feature_type,
            "feature": feature,
            "label_a": label_a,
            "label_b": label_b,
            "effect_size": effect_size,
            "pvalue": pvalue,
            "neg_log_pvalue": _neg_log_pvalue(pvalue),
            "significant": bool(np.isfinite(pvalue) and pvalue < pval_threshold),
            "n_samples_a": int(values_a.size),
            "n_samples_b": int(values_b.size),
        }
    )


def compute_top_attention_feature_comparisons(
    pseudobulk_table: pl.DataFrame,
    *,
    pval_threshold: float = 0.05,
) -> pl.DataFrame:
    """Compare top-attention pseudobulks with MultiMIL-style lightweight contrasts."""
    required_columns = {"bag_id", "bag_label", "group", "feature_type", "attention_subset", "feature", "value"}
    missing_columns = sorted(required_columns.difference(pseudobulk_table.columns))
    if missing_columns:
        raise ValueError(f"pseudobulk_table is missing required columns: {missing_columns}")
    if pseudobulk_table.height == 0:
        return _empty_feature_comparisons_table()

    rows: list[dict[str, object]] = []
    keys = pseudobulk_table.select(["group", "feature_type", "feature"]).unique().sort(
        ["group", "feature_type", "feature"]
    )

    for key in keys.iter_rows(named=True):
        group_name = key["group"]
        feature_type = key["feature_type"]
        feature = key["feature"]
        feature_data = pseudobulk_table.filter(
            (pl.col("group") == group_name)
            & (pl.col("feature_type") == feature_type)
            & (pl.col("feature") == feature)
        )
        labels = sorted(feature_data["bag_label"].unique().to_list())

        for label in labels:
            label_data = feature_data.filter(pl.col("bag_label") == label)
            top_data = label_data.filter(pl.col("attention_subset") == "top").select(["bag_id", "value"])
            rest_data = label_data.filter(pl.col("attention_subset") == "rest").select(["bag_id", "value"])
            paired_data = top_data.join(rest_data, on="bag_id", how="inner", suffix="_rest").sort("bag_id")
            if paired_data.height:
                _append_comparison_row(
                    rows,
                    comparison="top_vs_rest_within_label",
                    comparison_label=f"{label}: top vs rest",
                    group_name=group_name,
                    feature_type=feature_type,
                    feature=feature,
                    label_a=f"{label}:top",
                    label_b=f"{label}:rest",
                    values_a=paired_data["value"].to_numpy(),
                    values_b=paired_data["value_rest"].to_numpy(),
                    pval_threshold=pval_threshold,
                    paired=True,
                )

        top_feature_data = feature_data.filter(pl.col("attention_subset") == "top")
        for label_b, label_a in combinations(labels, 2):
            values_a = top_feature_data.filter(pl.col("bag_label") == label_a)["value"].to_numpy()
            values_b = top_feature_data.filter(pl.col("bag_label") == label_b)["value"].to_numpy()
            _append_comparison_row(
                rows,
                comparison="top_label_vs_top_label",
                comparison_label=f"top {label_a} vs top {label_b}",
                group_name=group_name,
                feature_type=feature_type,
                feature=feature,
                label_a=f"{label_a}:top",
                label_b=f"{label_b}:top",
                values_a=values_a,
                values_b=values_b,
                pval_threshold=pval_threshold,
            )

        for target_label in labels:
            values_a = top_feature_data.filter(pl.col("bag_label") == target_label)["value"].to_numpy()
            values_b = top_feature_data.filter(pl.col("bag_label") != target_label)["value"].to_numpy()
            _append_comparison_row(
                rows,
                comparison="target_top_vs_other_top",
                comparison_label=f"top {target_label} vs other top",
                group_name=group_name,
                feature_type=feature_type,
                feature=feature,
                label_a=f"{target_label}:top",
                label_b="other:top",
                values_a=values_a,
                values_b=values_b,
                pval_threshold=pval_threshold,
            )

    if not rows:
        return _empty_feature_comparisons_table()
    return pl.from_dicts(rows).sort(["group", "feature_type", "comparison", "neg_log_pvalue"], descending=[False, False, False, True])


def plot_top_attention_feature_heatmap(
    comparisons_table: pl.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot feature-by-comparison effect sizes, outlining significant cells."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if comparisons_table.height == 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No top-attention feature comparisons available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        return

    panel_keys = comparisons_table.select(["group", "feature_type"]).unique(maintain_order=True).to_dicts()
    panel_feature_counts = []
    for key in panel_keys:
        panel = comparisons_table.filter(
            (pl.col("group") == key["group"]) & (pl.col("feature_type") == key["feature_type"])
        )
        sig_features = panel.filter(pl.col("significant"))["feature"].unique().to_list()
        panel_feature_counts.append(max(1, len(sig_features)))

    n_panels = len(panel_keys)
    total_feature_rows = sum(panel_feature_counts)
    fig_height = max(3.2 * n_panels, 0.32 * total_feature_rows + 1.8 * n_panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, fig_height), squeeze=False)
    axes_flat = axes.ravel()
    image = None

    cmap = plt.cm.get_cmap("RdBu_r").copy()
    cmap.set_bad("#F2F2F2")

    for ax, key in zip(axes_flat, panel_keys):
        group_name = key["group"]
        feature_type = key["feature_type"]
        panel = comparisons_table.filter(
            (pl.col("group") == group_name) & (pl.col("feature_type") == feature_type)
        )
        significant_features = panel.filter(pl.col("significant"))["feature"].unique().to_list()
        if not significant_features:
            ax.text(0.5, 0.5, "No significant features", ha="center", va="center")
            ax.set_title(f"{group_name} / {feature_type}", fontsize=10)
            ax.axis("off")
            continue

        panel = panel.filter(pl.col("feature").is_in(significant_features))
        features = panel["feature"].unique(maintain_order=True).to_list()
        comparisons_in_panel = panel["comparison_label"].unique(maintain_order=True).to_list()
        matrix = np.full((len(features), len(comparisons_in_panel)), np.nan, dtype=np.float64)
        significant = np.zeros_like(matrix, dtype=bool)

        feature_index = {feature: idx for idx, feature in enumerate(features)}
        comparison_index = {comparison: idx for idx, comparison in enumerate(comparisons_in_panel)}
        for row in panel.iter_rows(named=True):
            row_idx = feature_index[row["feature"]]
            col_idx = comparison_index[row["comparison_label"]]
            matrix[row_idx, col_idx] = row["effect_size"]
            significant[row_idx, col_idx] = bool(row["significant"])

        clustering_matrix = np.nan_to_num(matrix, nan=0.0)
        if matrix.shape[0] > 1 and np.any(np.std(clustering_matrix, axis=0) > 0):
            order = leaves_list(linkage(clustering_matrix, method="average", metric="euclidean"))
        elif matrix.shape[0] > 1:
            order = np.argsort(-np.nanmax(np.abs(matrix), axis=1))
        else:
            order = np.arange(matrix.shape[0])

        matrix = matrix[order]
        significant = significant[order]
        features = [features[idx] for idx in order]
        max_abs = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
        if not np.isfinite(max_abs) or max_abs == 0:
            max_abs = 1.0

        image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=-max_abs, vmax=max_abs)
        for row_idx in range(significant.shape[0]):
            for col_idx in range(significant.shape[1]):
                if significant[row_idx, col_idx]:
                    ax.add_patch(Rectangle((col_idx - 0.5, row_idx - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.0))

        ax.set_title(f"{group_name} / {feature_type}", fontsize=10)
        ax.set_xticks(np.arange(len(comparisons_in_panel)))
        ax.set_xticklabels(comparisons_in_panel, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels(features, fontsize=8)
        ax.set_xlabel("Comparison")
        ax.set_ylabel("Feature")

    fig.suptitle("Top-attention pseudobulk feature comparisons", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.84, 0.96))
    if image is not None:
        cbar_ax = fig.add_axes([0.88, 0.16, 0.025, 0.68])
        fig.colorbar(image, cax=cbar_ax, label="Effect size")
    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def _empty_attention_summary_table() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "split": pl.String,
            "group": pl.String,
            "feature_type": pl.String,
            "bag_label": pl.String,
            "mean_attention": pl.Float64,
            "median_attention": pl.Float64,
            "sd_attention": pl.Float64,
            "n_samples": pl.Int64,
            "n_instances": pl.Int64,
        }
    )


def compute_attention_summary_table(
    attention_table: pl.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    feature_table: pl.DataFrame | None = None,
    group_col: str | None = None,
    bag_col: str = "bag_id",
    id_col: str = "instance_id",
    label_col: str = "bag_label",
    attention_col: str = "attention",
    top_fraction: float | None = 0.10,
    active_threshold: float = 1e-12,
) -> pl.DataFrame:
    """Summarize attention per feature type, label, and optional group using sample-level summaries."""
    for column in [bag_col, id_col, label_col, attention_col]:
        if column not in attention_table.columns:
            raise ValueError(f"attention_table must include {column!r}.")

    base_attention = (
        _scored_top_attention_table(
            attention_table,
            bag_col=bag_col,
            id_col=id_col,
            attention_col=attention_col,
            top_fraction=top_fraction,
        )
        if top_fraction is not None
        else attention_table
    )
    data = _prepare_attention_feature_table(
        base_attention,
        feature_sets,
        feature_table=feature_table,
        id_col=id_col,
    )
    if top_fraction is not None:
        data = data.filter(pl.col("top_attention"))

    rows = []
    for group_name, group_data in _iter_group_tables(data, group_col):
        for feature_type, feature_columns in feature_sets.items():
            active_columns = _active_feature_columns(
                group_data,
                feature_columns,
                active_threshold=active_threshold,
            )
            if not active_columns:
                continue

            for label in sorted(group_data[label_col].unique().to_list()):
                label_data = group_data.filter(pl.col(label_col) == label)
                sample_means = []
                sample_medians = []
                sample_sds = []
                n_instances = 0

                for bag_id in label_data[bag_col].unique().to_list():
                    bag_data = label_data.filter(pl.col(bag_col) == bag_id)
                    attention_values = bag_data[attention_col].to_numpy().astype(np.float64)
                    if attention_values.size == 0:
                        continue
                    sample_means.append(float(np.mean(attention_values)))
                    sample_medians.append(float(np.median(attention_values)))
                    sample_sds.append(float(np.std(attention_values, ddof=1)) if attention_values.size > 1 else 0.0)
                    n_instances += int(attention_values.size)

                if not sample_means:
                    continue

                label_str = str(label)
                split = f"{feature_type} | label={label_str}"
                if group_col is not None:
                    split = f"{feature_type} | label={label_str} | {group_col}={group_name}"

                rows.append(
                    {
                        "split": split,
                        "group": group_name,
                        "feature_type": feature_type,
                        "bag_label": label_str,
                        "mean_attention": float(np.mean(sample_means)),
                        "median_attention": float(np.median(sample_medians)),
                        "sd_attention": float(np.mean(sample_sds)),
                        "n_samples": len(sample_means),
                        "n_instances": n_instances,
                    }
                )

    if not rows:
        return _empty_attention_summary_table()
    return pl.from_dicts(rows).sort("mean_attention", descending=True)


def plot_attention_summary_heatmap(
    summary_table: pl.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot sample-balanced attention summaries by feature type, label, and optional group."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if summary_table.height == 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No attention summaries available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        return

    heatmap_metrics = ["mean_attention", "median_attention"]
    table = summary_table.sort("mean_attention", descending=True)
    splits = table["split"].to_list()
    matrix = table.select(heatmap_metrics).to_numpy().astype(np.float64)
    sd_values = table["sd_attention"].to_numpy().astype(np.float64)

    fig_height = max(4.0, 0.38 * len(splits) + 1.6)
    fig, ax = plt.subplots(figsize=(8.8, fig_height))
    vmin = float(np.nanmin(matrix)) if np.isfinite(matrix).any() else 0.0
    vmax = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0
    if vmin == vmax:
        vmax = vmin + 1e-6
    image = ax.imshow(matrix, aspect="auto", cmap="Reds", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(heatmap_metrics)))
    ax.set_xticklabels(["mean", "median"], fontsize=9)
    ax.set_yticks(np.arange(len(splits)))
    ax.set_yticklabels(splits, fontsize=8)
    ax.set_title("Sample-balanced top-attention summaries", fontsize=11)
    ax.set_xlim(-0.5, 2.65)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{matrix[row_idx, col_idx]:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )
        sd_label = "sd=NA" if not np.isfinite(sd_values[row_idx]) else f"sd={sd_values[row_idx]:.3f}"
        ax.text(
            2.15,
            row_idx,
            sd_label,
            ha="left",
            va="center",
            fontsize=7,
            color="black",
        )

    fig.colorbar(image, ax=ax, shrink=0.75, label="Attention")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


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
