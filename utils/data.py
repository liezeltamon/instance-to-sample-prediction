from __future__ import annotations

import random
import string
from typing import Optional

import numpy as np
import polars as pl

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def random_sequence(length: int = 12, motif: Optional[str] = None) -> str:
    seq = [random.choice(AMINO_ACIDS) for _ in range(length)]
    if motif:
        motif = motif[:length]
        position = random.randrange(0, length - len(motif) + 1)
        seq[position : position + len(motif)] = list(motif)
    return "".join(seq)


def generate_synthetic_bag_data(
    n_bags: int = 20,
    bag_size: int = 50,
    n_rna_features: int = 16,
    drivers_per_positive_bag: int = 5,
    positive_bag_fraction: float = 0.5,
    seed: int = 0,
) -> pl.DataFrame:
    """Generate synthetic bagged instance data for MIL benchmarking.

    Each bag is labeled positive if it contains driver instances. Driver instances
    receive a numeric signal on RNA features and motif-enriched receptor sequences.
    """
    random.seed(seed)
    np.random.seed(seed)

    positive_bags = int(round(n_bags * positive_bag_fraction))
    bag_labels = [1] * positive_bags + [0] * (n_bags - positive_bags)
    random.shuffle(bag_labels)

    records = []
    for bag_idx, bag_label in enumerate(bag_labels):
        driver_count = drivers_per_positive_bag if bag_label == 1 else 0
        driver_indices = set(random.sample(range(bag_size), driver_count))

        for instance_idx in range(bag_size):
            driver_true = int(instance_idx in driver_indices)
            base_rna = np.random.normal(0.0, 1.0, size=n_rna_features)
            if driver_true:
                base_rna += np.random.normal(1.5, 0.5, size=n_rna_features)

            record = {
                "bag_id": f"bag_{bag_idx:03d}",
                "instance_id": f"bag_{bag_idx:03d}_instance_{instance_idx:03d}",
                "bag_label": bag_label,
                "driver_true": driver_true,
                "tcr_sequence": random_sequence(12, motif="CASS" if driver_true else None),
                "bcr_sequence": random_sequence(12, motif="ARDY" if driver_true else None),
            }
            for feature_idx in range(n_rna_features):
                record[f"rna_{feature_idx}"] = float(base_rna[feature_idx])
            records.append(record)

    return pl.from_records(records)
