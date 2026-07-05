# Starter Code Generator

The **Starter Code Generator** (`codegen.py`) converts the selected questionnaire answers and final recommendations into executable Python scripts. It supports generating templates for two popular deep learning frameworks:
1. **PyTorch**
2. **Keras / TensorFlow**

Like other components, the code generator runs with a strict **No-Control-Flow** constraint. It uses dictionary-based dispatch and string mathematical operations to construct files dynamically.

---

## 🛠️ Code Generation Engine

### 1. Template Structures
The module defines two multi-line string templates:
* `_PYTORCH_TEMPLATE`: Sets up imports (`torch`, `nn`, `DataLoader`, `TensorDataset`), hyper-parameters, data loaders, custom `nn.Module` classes, loss functions, optimizers, and a functional epoch-based training loop.
* `_KERAS_TEMPLATE`: Sets up a `keras.Sequential` model, compiles it with the requested metrics/loss, and runs `model.fit()` with split validation.

### 2. Configuration Mappings (`_TARGET_CONFIG`)
A centralized lookup table maps each `target_kind` (`"binary"`, `"multiclass"`, `"continuous"`, `"multilabel"`) to specific model parameters:

```python
_TARGET_CONFIG = {
    "binary": (
        "nn.BCELoss()",                 # PyTorch Loss Function
        "        x = torch.sigmoid(x)\n",# PyTorch Output Activation
        "binary_crossentropy",          # Keras Loss Function
        "sigmoid",                      # Keras Output Activation
        "1",                            # Output Neurons count
        "float32",                      # PyTorch label tensor dtype
        ".squeeze(-1)",                 # Prediction squeeze format
        '["accuracy"]',                 # Keras metrics list
    ),
    "multiclass": (
        "nn.CrossEntropyLoss()",
        "",                             # PyTorch CrossEntropy applies Softmax internally
        "sparse_categorical_crossentropy",
        "softmax",
        "NUM_CLASSES",
        "long",
        "",
        '["accuracy"]',
    ),
    "continuous": (
        "nn.MSELoss()",
        "",                             # Linear activation (no activation)
        "mse",
        "linear",
        "1",
        "float32",
        ".squeeze(-1)",
        '["mae"]',
    ),
    "multilabel": (
        "nn.BCELoss()",
        "        x = torch.sigmoid(x)\n",
        "binary_crossentropy",
        "sigmoid",
        "NUM_LABELS",
        "float32",
        "",
        '["accuracy"]',
    )
}
```

### 3. Optimizer Configuration
Different optimizers are selected based on the dataset size:
- **Small / Medium**: Selects `Adam` with default learning rate.
- **Large**: Selects `SGD` with `momentum=0.9` for stable convergence under large sample batches.

---

## 🚫 Preprocessing & Regularization Dispatch

Rather than branching with standard `if` blocks to add preprocessing scalers or dropout layers, `codegen.py` casts Boolean parameters into integers ($0$ or $1$) and multiplies them against the template strings.

### Dynamic Feature Scaling Selection
Scalers are added based on numerical scale differences and outlier presence:
* If scales do not differ: Evaluates to empty strings.
* If scales differ: Imports `StandardScaler` from scikit-learn.
* If scales differ AND outliers are present: Overrides `StandardScaler` with `RobustScaler`.

```python
# Scaling indicator
scale_diff = int(params["numerical_scale"] == "different")
# Outlier indicator
outlier = int(params["numerical_outliers"] == "yes")

# Robust scaler block override via multiplication
robust_import = "from sklearn.preprocessing import RobustScaler\n" * outlier * scale_diff
robust_fit    = (
    "# Outliers detected — using RobustScaler instead of StandardScaler\n"
    "scaler   = RobustScaler()\n"
    "X_scaled = scaler.fit_transform(X_train)\n"
    "X_val_sc = scaler.transform(X_val)\n"
) * outlier * scale_diff
```

### Dynamic Dropout Layer Inclusion
Dropout layers are injected into the neural network architecture definition if the user selects `"yes"` for the `overfitting` symptom.
```python
# Overfitting indicator (0 or 1)
overfit = int(params["overfitting"] == "yes")

# Injected only when overfit == 1
dropout_init    = "        self.dropout = nn.Dropout(0.3)\n" * overfit
dropout_forward = "        x = self.dropout(x)\n"           * overfit
```

---

## 🚀 Public Entry Point

The UI triggers code generation by calling the unified `generate_code` dispatch function:
```python
def generate_code(framework: str, params: dict) -> str:
    """
    Dispatches parameters to the correct template builder
    depending on the framework key.
    """
    return GENERATORS[framework](params)
```
This returns a single code string, which can be previewed or directly copied using the "Copy Code" button in the application UI.
