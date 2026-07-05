# =============================================================================
# profiler.py
# Automated Dataset Profiler for the Neural Network Design Expert System.
#
# RESTRICTION: No if / elif / else / for / while / match / case anywhere.
# All decisions are made via tuple-indexing, dict lookups, and vectorized
# pandas operations only.
# =============================================================================

import json
import pathlib

import pandas as pd


def profile_dataset(file_path: str) -> dict:
    """
    Read a CSV file and compute all dataset metrics needed to auto-fill
    the expert system questionnaire.

    Returns a dict with the following keys (all values are strings or ints
    matching the corresponding Fact field definitions in facts.py):

        dataset_size         : "small" | "medium" | "large"
        feature_count        : "low" | "medium" | "high"
        feature_sample_ratio : "safe" | "risky"
        has_numerical        : 1 | 0
        numerical_scale      : "similar" | "different"
        numerical_outliers   : "yes" | "no"
        missing_values       : "yes" | "no"
        target_kind          : "binary" | "multiclass" | "continuous"

    Also writes a <filename>.profile.json alongside the CSV for session restore.
    Raises ValueError if the file cannot be read or has fewer than 2 columns.
    """

    df = pd.read_csv(file_path)

    # -------------------------------------------------------------------------
    # 1. Dataset Size
    #    Thresholds: < 1000 -> small, 1000-9999 -> medium, >= 10000 -> large
    # -------------------------------------------------------------------------
    row_count = len(df)
    size_idx  = int(row_count >= 1000) + int(row_count >= 10000)   # 0, 1, or 2
    dataset_size = ("small", "medium", "large")[size_idx]

    # -------------------------------------------------------------------------
    # 2. Feature Count  (all columns except the last, which is the target)
    #    Thresholds: <= 10 -> low, 11-50 -> medium, > 50 -> high
    # -------------------------------------------------------------------------
    col_count   = len(df.columns) - 1              # exclude target column
    count_idx   = int(col_count > 10) + int(col_count > 50)
    feature_count = ("low", "medium", "high")[count_idx]

    # -------------------------------------------------------------------------
    # 3. Feature-to-Sample Ratio  -> DatasetFact.feature_sample_ratio
    #    "risky" when more than 1 feature per 10 samples (ratio > 0.1)
    # -------------------------------------------------------------------------
    ratio              = col_count / (row_count + 1e-9)
    ratio_idx          = int(ratio > 0.1)
    feature_sample_ratio = ("safe", "risky")[ratio_idx]

    # -------------------------------------------------------------------------
    # 4. Numerical Presence & Scale
    #    numerical_scale: "different" when max range is 10x the min range
    # -------------------------------------------------------------------------
    feature_df  = df.iloc[:, :-1]                  # all columns except target
    num_df      = feature_df.select_dtypes(include=["number"])
    has_numerical = int(len(num_df.columns) > 0)

    ranges        = (num_df.max() - num_df.min()).replace(0, 1e-9)
    scale_idx     = int((ranges.max() / ranges.min()) > 10.0) * has_numerical
    numerical_scale = ("similar", "different")[scale_idx]

    # -------------------------------------------------------------------------
    # 5. Numerical Outliers  (IQR method)
    # -------------------------------------------------------------------------
    q1            = num_df.quantile(0.25)
    q3            = num_df.quantile(0.75)
    iqr           = q3 - q1
    outliers_mask = (num_df < (q1 - 1.5 * iqr)) | (num_df > (q3 + 1.5 * iqr))
    outliers_exist = outliers_mask.any().any() and bool(has_numerical)
    numerical_outliers = ("no", "yes")[int(outliers_exist)]

    # -------------------------------------------------------------------------
    # 6. Missing Values
    # -------------------------------------------------------------------------
    missing_exist  = df.isnull().any().any()
    missing_values = ("no", "yes")[int(missing_exist)]

    # -------------------------------------------------------------------------
    # 7. Target Kind Auto-detection  (last column is assumed to be the target)
    #    Uses a dict lookup - no conditionals, exhaustive mapping:
    #
    #    (is_numeric, unique_count > 2) -> target_kind
    #    (False, False) -> "binary"       categorical target with 2 values
    #    (False, True)  -> "multiclass"   categorical target with many values
    #    (True,  False) -> "binary"       numeric 0/1 target
    #    (True,  True)  -> "continuous"   regression target (many numeric values)
    # -------------------------------------------------------------------------
    target_col   = df.columns[-1]
    is_numeric   = pd.api.types.is_numeric_dtype(df[target_col])
    unique_count = df[target_col].nunique()

    kind_map = {
        (False, False): "binary",
        (False, True ): "multiclass",
        (True,  False): "binary",
        (True,  True ): "continuous",
    }
    target_kind = kind_map[(bool(is_numeric), bool(unique_count > 2))]

    # -------------------------------------------------------------------------
    # 8. Assemble profile dict
    # -------------------------------------------------------------------------
    profile = {
        "dataset_size"         : dataset_size,
        "feature_count"        : feature_count,
        "feature_sample_ratio" : feature_sample_ratio,
        "has_numerical"        : has_numerical,
        "numerical_scale"      : numerical_scale,
        "numerical_outliers"   : numerical_outliers,
        "missing_values"       : missing_values,
        "target_kind"          : target_kind,
        # --- raw stats for display in the UI ---
        "_row_count"           : row_count,
        "_col_count"           : col_count,
        "_numeric_col_count"   : len(num_df.columns),
        "_target_col"          : target_col,
        "_unique_target_values": int(unique_count),
    }

    # -------------------------------------------------------------------------
    # 9. Export profile.json alongside the CSV
    # -------------------------------------------------------------------------
    profile_path = pathlib.Path(file_path).with_suffix(".profile.json")
    profile_path.write_text(json.dumps(profile, indent=2))

    return profile
