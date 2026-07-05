# Knowledge-Based Rules Engine

The decision-making heart of the Neural Network Design Expert System is implemented in `engine.py` using `experta` (a Python implementation of CLIPS). It processes user inputs as **Facts** and applies a collection of declarative rules grouped by domain categories.

---

## 🧠 Fact Models (`facts.py`)

Facts represent the structural input models loaded into the reasoning engine:

### 1. `ProblemFact`
Defines the machine learning problem structure:
- `target_kind`: `"continuous"` | `"binary"` | `"multiclass"` | `"multilabel"`
- `label_format`: `"numeric_value"` | `"integer_class_ids"` | `"one_hot"` | `"multiple_binary_labels"`

### 2. `DatasetFact`
Defines the general dataset properties:
- `input_data_type`: `"tabular"` | `"image"` | `"text"` | `"time_series"` | `"mixed"`
- `dataset_size`: `"small"` | `"medium"` | `"large"`
- `feature_count`: `"low"` | `"medium"` | `"high"`
- `feature_sample_ratio`: `"safe"` | `"risky"`

### 3. `FeatureFact`
Detailed feature characteristics (by input datatype):
* **Numerical**:
  - `has_numerical`: `1` | `0`
  - `numerical_scale`: `"similar"` | `"different"`
  - `numerical_distribution`: `"normal"` | `"bounded"` | `"skewed"` | `"unknown"`
  - `numerical_outliers`: `"yes"` | `"no"`
  - `nonlinear_relationships`: `"yes"` | `"no"`
* **Categorical**:
  - `has_categorical`: `1` | `0`
  - `categorical_cardinality`: `"low"` | `"high"`
  - `categorical_order`: `"nominal"` | `"ordinal"`
  - `rare_categories`: `"yes"` | `"no"`
  - `noisy_categories`: `"yes"` | `"no"`
* **Date-Time**:
  - `has_datetime`: `1` | `0`
  - `seasonality`: `"yes"` | `"no"`
  - `time_order`: `"yes"` | `"no"`
  - `cyclic_time`: `"yes"` | `"no"`
* **Text**:
  - `has_text`: `1` | `0`
  - `text_length`: `"short"` | `"long"`
  - `text_dataset_size`: `"small"` | `"large"`
  - `word_order`: `"yes"` | `"no"`
* **Image**:
  - `has_image`: `1` | `0`
  - `image_dataset_size`: `"small"` | `"large"`
  - `image_augmentation`: `"yes"` | `"no"`

### 4. `QualityFact`
Identifies dataset anomalies and quality indicators:
- `missing_values`: `"yes"` | `"no"`
- `class_imbalance`: `"yes"` | `"no"`
- `noisy_labels`: `"yes"` | `"no"`
- `data_leakage`: `"yes"` | `"no"`

### 5. `TrainingFact`
Observed training behaviors (for iteration/debugging):
- `overfitting`: `"yes"` | `"no"`
- `underfitting`: `"yes"` | `"no"`
- `unstable_training`: `"yes"` | `"no"`
- `slow_training`: `"yes"` | `"no"`

---

## 📋 The Rule Groups (A – N)

The engine features 14 logically separated groups of production rules:

### Group A: Problem Type Rules
*Determines the output neurons, activation function, and default loss class.*
- **Regression**: Continuous target $\rightarrow$ 1 output neuron, linear activation, MSE/MAE loss.
- **Binary**: Binary target $\rightarrow$ 1 output neuron, sigmoid activation, binary cross-entropy (BCE) loss.
- **Multiclass**: Multiclass target $\rightarrow$ $N$ output neurons, softmax activation, categorical cross-entropy loss.
- **Multilabel**: Multilabel target $\rightarrow$ $K$ independent output neurons, sigmoid activation, BCE loss.

### Group B: Label Format Rules
*Validates the mapping of label schemas to loss implementations.*
- Ensures that integer class IDs use **sparse** cross-entropy, one-hot uses standard **categorical** cross-entropy, and numeric vectors use regression losses.

### Group C: Architecture by Input Type
*Selects the network backbone architecture.*
- **Tabular**: Dense Multilayer Perceptron (MLP) (2-4 layers).
- **Image**: Convolutional Neural Network (CNN) using convolution and pooling.
- **Text**: Embedding layer followed by LSTMs/GRUs or Transformer blocks.
- **Time-Series**: 1D CNNs or recurrent architectures (LSTMs).
- **Mixed**: Multi-input hybrid architectures.

### Group D: Dataset Size & Ratio
*Addresses regularization and network capacity based on sample density.*
- **Small Dataset**: Suggests shallow networks, strong dropout/weight decay, and small batch sizes to avoid memorization.
- **Large Dataset**: Suggests deeper architectures and larger batch sizes.
- **Risky Ratio**: High feature count relative to samples ($>0.1$) $\rightarrow$ recommends dimensionality reduction (PCA, feature selection) or L1/L2 regularization.

### Group E: Numerical Features
*Handles preprocessing and mapping of numerical inputs.*
- **Scaling**: Recommends `StandardScaler` if scales differ.
- **Outliers**: If outliers exist, overrides standard scaling with robust methods (`RobustScaler`).
- **Distributions**: Bounded distributions $\rightarrow$ MinMax scaling. Skewed distributions $\rightarrow$ Logarithmic/Power transforms.

### Group F: Categorical Features
*Handles string values and mappings.*
- **Cardinality**: Low cardinality $\rightarrow$ One-Hot Encoding. High cardinality $\rightarrow$ Entity Embeddings.
- **Ordinal**: Suggests Ordinal/Integer encoding. Nominal $\rightarrow$ One-Hot or Target Encoding.

### Group G: Date-Time Features
- Handles cyclic features (e.g., month/day) using Sine/Cosine trigonometric mappings. Time-ordered sequences require temporal splitting instead of random shuffling.

### Group H: Text Features
- Recommends Bag-of-Words or TF-IDF for short text or small datasets, and pre-trained language embeddings or fine-tuned Transformer models for complex text tasks.

### Group I: Image Features
- Focuses on Transfer Learning (e.g. ResNet, MobileNet) for small datasets, and custom architectures + data augmentation for large datasets.

### Group J: Data Quality
- Recommends imputation methods for missing values, oversampling (SMOTE) or class weights for imbalance, and Label Smoothing for noisy labels.

### Group K: Evaluation Metrics
- Matches problem domains to appropriate validation metrics: Classification (F1-score, AUC-ROC), Regression (RMSE, MAE, R-squared), and Multi-label (Micro/Macro F1-scores).

### Group L: Optimizer Selection
- Recommends optimizers: **Adam** for standard tasks and small-to-medium datasets, and **SGD with Momentum** for large-scale training pipelines.

### Group M: Training Symptoms
*Handles iterative debugging recommendations.*
- **Overfitting**: Add dropout, weight decay (L2), early stopping, and data augmentation.
- **Underfitting**: Increase hidden layers, increase epochs, decrease dropout.
- **Unstable Training**: Reduce learning rate, add Gradient Clipping, check for NaN values, or add Batch Normalization.

### Group N: Final Checklist
- Final verification checks: comparison to a simple baseline, model checkpoint saving, and visual verification of learning curves.
