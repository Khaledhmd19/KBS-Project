# Automated Dataset Profiler

The **Automated Dataset Profiler** (`profiler.py`) is responsible for reading user-supplied CSV files and extracting metadata to automatically populate the questionnaire in the expert system UI. 

Like the rest of the codebase, it is designed with a strict **No-Control-Flow** constraint. It uses vectorized `pandas` operations, boolean logic, and mathematical index mappings rather than loops or conditional branches.

---

## 📊 Functions and Profiling Logic

### 1. Dataset Size Determination
Dataset size is classified into `"small"` ($N < 1000$), `"medium"` ($1000 \leq N < 10000$), or `"large"` ($N \geq 10000$).
* **Implementation**:
  ```python
  row_count = len(df)
  size_idx  = int(row_count >= 1000) + int(row_count >= 10000)   # Resolves to 0, 1, or 2
  dataset_size = ("small", "medium", "large")[size_idx]
  ```

### 2. Feature Count Check
Counts columns (excluding the target column, assumed to be the last column) and classifies features into `"low"` ($\leq 10$), `"medium"` ($11 \text{ to } 50$), or `"high"` ($> 50$).
* **Implementation**:
  ```python
  col_count   = len(df.columns) - 1
  count_idx   = int(col_count > 10) + int(col_count > 50)
  feature_count = ("low", "medium", "high")[count_idx]
  ```

### 3. Feature-to-Sample Ratio (FSR)
Determines whether features are sparse relative to records. It is flagged as `"risky"` if there is more than 1 feature per 10 samples (ratio $> 0.1$), and `"safe"` otherwise.
* **Implementation**:
  ```python
  ratio              = col_count / (row_count + 1e-9)
  ratio_idx          = int(ratio > 0.1)
  feature_sample_ratio = ("safe", "risky")[ratio_idx]
  ```

### 4. Numerical Presence and Scale Differences
Determines if numeric features are present, and checks if their ranges differ by more than 10x (requiring data normalization/scaling).
* **Implementation**:
  ```python
  feature_df  = df.iloc[:, :-1]                  # Exclude target
  num_df      = feature_df.select_dtypes(include=["number"])
  has_numerical = int(len(num_df.columns) > 0)

  ranges        = (num_df.max() - num_df.min()).replace(0, 1e-9)
  scale_idx     = int((ranges.max() / ranges.min()) > 10.0) * has_numerical
  numerical_scale = ("similar", "different")[scale_idx]
  ```

### 5. Outlier Detection (Interquartile Range - IQR)
Identifies if outliers exist within numerical columns. If they do, a robust scaler (e.g. `RobustScaler`) is suggested rather than standard scaling.
* **Implementation**:
  ```python
  q1            = num_df.quantile(0.25)
  q3            = num_df.quantile(0.75)
  iqr           = q3 - q1
  outliers_mask = (num_df < (q1 - 1.5 * iqr)) | (num_df > (q3 + 1.5 * iqr))
  outliers_exist = outliers_mask.any().any() and bool(has_numerical)
  numerical_outliers = ("no", "yes")[int(outliers_exist)]
  ```

### 6. Target Kind Auto-Detection
Categorizes the target variable (the last column of the CSV) into `"binary"`, `"multiclass"`, or `"continuous"` (regression) based on whether it is numeric and its count of unique values.
* **Implementation**:
  ```python
  target_col   = df.columns[-1]
  is_numeric   = pd.api.types.is_numeric_dtype(df[target_col])
  unique_count = df[target_col].nunique()

  kind_map = {
      (False, False): "binary",       # Categorical with 2 values
      (False, True ): "multiclass",   # Categorical with >2 values
      (True,  False): "binary",       # Numeric (e.g. 0/1) with 2 values
      (True,  True ): "continuous",   # Numeric with >2 values (regression)
  }
  target_kind = kind_map[(bool(is_numeric), bool(unique_count > 2))]
  ```

---

## 💾 Profile Cache Export (`.profile.json`)

To prevent re-running calculation-heavy profiling, the tool automatically caches metrics into a `<dataset_name>.profile.json` file in the same directory as the CSV dataset.

Example Output structure:
```json
{
  "dataset_size": "medium",
  "feature_count": "low",
  "feature_sample_ratio": "safe",
  "has_numerical": 1,
  "numerical_scale": "different",
  "numerical_outliers": "yes",
  "missing_values": "no",
  "target_kind": "binary",
  "_row_count": 303,
  "_col_count": 13,
  "_numeric_col_count": 13,
  "_target_col": "target",
  "_unique_target_values": 2
}
```
If this profile file exists, the expert system interface can parse it instantly instead of reading and recomputing metrics from the CSV.
