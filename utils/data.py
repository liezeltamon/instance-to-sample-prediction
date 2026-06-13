from __future__ import annotations

import numpy as np
import polars as pl

DEFAULT_GROUP_PROBABILITIES = {"T": 0.35, "B": 0.30, "myeloid": 0.35}
REPERTOIRE_GROUPS = {"T", "B"}


def generate_synthetic_bag_data(
    n_bags: int = 20,
    bag_size: int = 50,
    n_transcriptome_features: int = 16,
    n_repertoire_features: int = 12,
    drivers_per_positive_bag: int = 5,
    positive_bag_fraction: float = 0.5,
    group_col: str = "cell_type",
    group_probabilities: dict[str, float] | None = None,
    seed: int = 0,
) -> pl.DataFrame:
    """Generate a MIL-ready synthetic instance table.

    The generated table has one row per instance/cell, sample-level labels,
    optional simulated driver labels, group annotations, and numeric feature
    blocks ready for MIL. T and B cells carry repertoire embeddings; myeloid
    cells have zero-valued repertoire embeddings.
    """
    if drivers_per_positive_bag > bag_size:
        raise ValueError("drivers_per_positive_bag cannot exceed bag_size.")

    rng = np.random.default_rng(seed)
    probabilities = group_probabilities or DEFAULT_GROUP_PROBABILITIES
    group_names = list(probabilities)
    group_weights = np.array(list(probabilities.values()), dtype=np.float64)
    if np.any(group_weights < 0) or group_weights.sum() <= 0:
        raise ValueError("group_probabilities must contain non-negative weights with positive total.")
    group_weights = group_weights / group_weights.sum()

    positive_bags = int(round(n_bags * positive_bag_fraction))
    bag_labels = np.array([1] * positive_bags + [0] * (n_bags - positive_bags), dtype=np.int64)
    rng.shuffle(bag_labels)

    records = []
    repertoire_split = max(1, n_repertoire_features // 2)

    for bag_idx, bag_label in enumerate(bag_labels):
        driver_count = drivers_per_positive_bag if bag_label == 1 else 0
        driver_indices = set(rng.choice(bag_size, size=driver_count, replace=False).tolist())

        for instance_idx in range(bag_size):
            group_name = str(rng.choice(group_names, p=group_weights))
            driver_true = int(instance_idx in driver_indices)

            transcriptome = rng.normal(0.0, 1.0, size=n_transcriptome_features)
            if driver_true:
                transcriptome += rng.normal(1.5, 0.5, size=n_transcriptome_features)

            repertoire = np.zeros(n_repertoire_features, dtype=np.float64)
            if group_name in REPERTOIRE_GROUPS:
                repertoire = rng.normal(0.0, 1.0, size=n_repertoire_features)
                if driver_true and n_repertoire_features > 0:
                    if group_name == "T":
                        repertoire[:repertoire_split] += rng.normal(1.5, 0.5, size=repertoire_split)
                    elif group_name == "B":
                        repertoire[repertoire_split:] += rng.normal(
                            1.5,
                            0.5,
                            size=n_repertoire_features - repertoire_split,
                        )

            record = {
                "bag_id": f"bag_{bag_idx:03d}",
                "instance_id": f"bag_{bag_idx:03d}_instance_{instance_idx:03d}",
                "bag_label": int(bag_label),
                "driver_true": driver_true,
                group_col: group_name,
            }
            for feature_idx, value in enumerate(transcriptome):
                record[f"transcriptome_{feature_idx}"] = float(value)
            for feature_idx, value in enumerate(repertoire):
                record[f"repertoire_{feature_idx}"] = float(value)
            records.append(record)

    return pl.from_records(records)
