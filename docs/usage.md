# User Guide & Walkthrough

This guide walks you through setting up, running, and using the Neural Network Design Expert System.

---

## ⚙️ Project Setup

### 1. Synchronize Dependencies
Ensure you have the virtual environment and package tools configured. The project is pre-configured to use the fast `uv` installer. Run the following command in the project root:
```bash
uv sync
```
*This installs CustomTkinter, Experta, Pandas, and other required packages in a `.venv` directory.*

### 2. Run the Expert System
Launch the custom-styled desktop UI:
```bash
uv run python main.py
```

---

## 🖥️ UI Workflow Walkthrough

```text
+--------------------------------------------------------------------------+
|  Neural Network Design Expert System                                     |
+--------------------------------------------------------------------------+
|  [ Load CSV Dataset ]  [ Reset Answers ]                                  |
+------------------------------------+-------------------------------------+
| LEFT PANEL: Questionnaire Tabs     | RIGHT PANEL: Output Report & Code   |
|                                    |                                     |
|  - Problem                         |  - Expert Recommendation Report     |
|  - Dataset                         |  - Rule Explanation Trace           |
|  - Features                        |  - PyTorch Code                     |
|  - Quality                         |  - Keras Code                       |
|  - Training Symptoms               |                                     |
|                                    |                                     |
|  [ Analyze Dataset Design ]        |  [ Copy Report ]  [ Copy Code ]     |
+------------------------------------+-------------------------------------+
```

### Step 1: Profile a Dataset
Rather than filling out all questionnaire fields manually, you can load one of the included sample datasets:
1. Click **Load CSV Dataset** at the top left of the UI.
2. Select one of the provided sample CSVs in the file browser:
   * **`sample_heart_disease.csv`**: Heart disease dataset (binary classification target).
   * **`sample_house_prices.csv`**: Real-estate pricing dataset (continuous/regression target).
   * **`sample_iris_extended.csv`**: Extended plant iris species (multiclass classification target).
3. The profiler runs in the background and populates relevant fields across the tabs. A `★ Auto` badge will appear next to the fields that were programmatically detected (e.g., target classification, dataset size, feature counts, presence of numerical fields, scaling requirement, outliers presence, and missing values presence).

### Step 2: Answer the Remaining Tabs
Review and manually adjust the questionnaire tabs for attributes the profiler cannot detect from raw tabular data (e.g., text/image attributes or training symptoms):
* **Problem**: Confirm the target kind and label format.
* **Dataset**: Check feature counts and sample ratio.
* **Features**: Specify if categorical, datetime, text, or image variables exist.
* **Quality**: Review detected missing values or flag class imbalance, noisy labels, or data leakage warnings.
* **Training Symptoms**: If you are troubleshooting an already-trained model, select observed behaviors like *overfitting*, *underfitting*, *unstable training*, or *slow training*.

### Step 3: Run the Inference Engine
1. Click the large blue button **Analyze Dataset Design** in the left panel.
2. The UI declares these inputs as Facts in the Experta engine, runs forward chaining, and compiles the recommendations into the right-hand panel.

### Step 4: Review Recommendations & Explanations
In the right-hand panel:
* **EXPERT RECOMMENDATION REPORT**: Read structured expert suggestions on preprocessing steps, optimal output activations, specific loss metrics, optimizer choice, regularization additions, and a final checklist.
* **RULE EXPLANATION TRACE**: View the behind-the-scenes rules log (such as `A-problem_regression`, `E-numerical_scale_different`) showing exactly which rules were triggered and why.

### Step 5: Get Code Starters
* Select the **PyTorch Code** or **Keras Code** tabs in the right-hand panel to preview ready-to-run code generated dynamically to match your dataset parameters.
* Click the **Copy Code** button at the bottom of the right panel to instantly copy the source code to your clipboard.
