# =============================================================================
# engine.py
# NeuralNetworkDesignEngine — Experta KnowledgeEngine with all production rules.
#
# RESTRICTION: No if / elif / else / for / while / match / case anywhere in
# this file. All decisions are made exclusively via @Rule(...) decorators.
# =============================================================================

from experta import KnowledgeEngine, Rule, AND

from facts import ProblemFact, DatasetFact, FeatureFact, QualityFact, TrainingFact


class NeuralNetworkDesignEngine(KnowledgeEngine):
    """
    The core rule engine for the Neural Network Design Expert System.

    Receives a ReportBuilder instance via the constructor.
    Each rule appends one or more recommendations to the report by calling
    the appropriate report.add_*() method.

    Rule groups:
        A  — Problem type             (4 rules)
        B  — Label format             (4 rules)
        C  — Architecture / data type (5 rules)
        D  — Dataset size & ratio     (5 rules)
        E  — Numerical features       (8 rules)
        F  — Categorical features     (7 rules)
        G  — Date-time features       (4 rules)
        H  — Text features            (6 rules)
        I  — Image features           (4 rules)
        J  — Data quality             (4 rules)
        K  — Evaluation metrics       (5 rules)
        L  — Optimizer                (3 rules)
        M  — Training symptoms        (4 rules)
        N  — Final checklist          (3 rules)
    """

    def __init__(self, report):
        super().__init__()
        self.report = report

    # =========================================================================
    # GROUP A — Problem Type Rules (4 rules)
    # =========================================================================

    @Rule(ProblemFact(target_kind="continuous"))
    def problem_regression(self):
        self.report.add_problem(
            "Detected a regression task because the target variable is continuous."
        )
        self.report.add_architecture(
            "Use one output neuron with linear activation (no activation function on the output)."
        )
        self.report.add_training(
            "Use Mean Squared Error (MSE) as the loss function when large errors should be "
            "penalized strongly. Use Mean Absolute Error (MAE) when robustness to outliers "
            "is more important."
        )

    @Rule(ProblemFact(target_kind="binary"))
    def problem_binary(self):
        self.report.add_problem(
            "Detected a binary classification task."
        )
        self.report.add_output_loss(
            "Use 1 output neuron with sigmoid activation. "
            "Use binary cross-entropy as the loss function."
        )

    @Rule(ProblemFact(target_kind="multiclass"))
    def problem_multiclass(self):
        self.report.add_problem(
            "Detected a multi-class classification task."
        )
        self.report.add_output_loss(
            "Use N output neurons (one per class) with softmax activation. "
            "Use categorical cross-entropy or sparse categorical cross-entropy "
            "depending on the label format."
        )

    @Rule(ProblemFact(target_kind="multilabel"))
    def problem_multilabel(self):
        self.report.add_problem(
            "Detected a multi-label classification task where each sample can "
            "belong to multiple classes simultaneously."
        )
        self.report.add_output_loss(
            "Use one output neuron per label with sigmoid activation. "
            "Use binary cross-entropy as the loss function. "
            "Evaluate with precision, recall, and F1-score."
        )

    # =========================================================================
    # GROUP B — Label Format Rules (4 rules)
    # =========================================================================

    @Rule(ProblemFact(label_format="one_hot"))
    def label_one_hot(self):
        self.report.add_output_loss(
            "Labels are one-hot encoded: use categorical cross-entropy as the loss function."
        )

    @Rule(ProblemFact(label_format="integer_class_ids"))
    def label_integer_ids(self):
        self.report.add_output_loss(
            "Labels are integer class IDs: use sparse categorical cross-entropy "
            "to avoid converting labels to one-hot vectors manually."
        )

    @Rule(ProblemFact(label_format="numeric_value"))
    def label_numeric_value(self):
        self.report.add_output_loss(
            "Labels are numeric values: use a regression loss such as MSE, MAE, or Huber loss."
        )

    @Rule(ProblemFact(label_format="multiple_binary_labels"))
    def label_multiple_binary(self):
        self.report.add_output_loss(
            "Labels are multiple independent binary flags: use sigmoid output units "
            "and binary cross-entropy for multi-label learning."
        )

    # =========================================================================
    # GROUP C — Architecture by Input Data Type (5 rules)
    # =========================================================================

    @Rule(DatasetFact(input_data_type="tabular"))
    def architecture_tabular(self):
        self.report.add_architecture(
            "Input is tabular data: use a feed-forward fully-connected (dense) neural network. "
            "Start with 2 to 4 hidden layers. Use ReLU activation in hidden layers."
        )

    @Rule(DatasetFact(input_data_type="image"))
    def architecture_image(self):
        self.report.add_architecture(
            "Input is image data: use a Convolutional Neural Network (CNN). "
            "Stack convolution layers, pooling layers, then flatten or use global average "
            "pooling, followed by dense output layers."
        )

    @Rule(DatasetFact(input_data_type="text"))
    def architecture_text(self):
        self.report.add_architecture(
            "Input is text data: use text vectorization followed by dense layers, "
            "or use embedding layers followed by LSTM, GRU, or a transformer-style encoder."
        )

    @Rule(DatasetFact(input_data_type="time_series"))
    def architecture_time_series(self):
        self.report.add_architecture(
            "Input is time-series data: use LSTM, GRU, temporal convolution, or 1D CNN. "
            "Preserve the temporal order during train-test splitting."
        )

    @Rule(DatasetFact(input_data_type="mixed"))
    def architecture_mixed(self):
        self.report.add_architecture(
            "Input is mixed data (multiple modalities): use a multi-input neural network. "
            "Create separate branches for tabular, text, image, or sequence inputs, "
            "then concatenate the learned representations before the output layer."
        )

    # =========================================================================
    # GROUP D — Dataset Size and Feature Ratio Rules (5 rules)
    # =========================================================================

    @Rule(DatasetFact(dataset_size="small"))
    def dataset_small(self):
        self.report.add_architecture(
            "Dataset is small (fewer than 1,000 samples): use a simple, shallow neural network "
            "to avoid overfitting. Avoid excessive depth."
        )
        self.report.add_regularization(
            "Dataset is small: apply early stopping, dropout, or L2 regularization. "
            "Prefer transfer learning for image or text tasks if possible."
        )

    @Rule(DatasetFact(dataset_size="medium"))
    def dataset_medium(self):
        self.report.add_architecture(
            "Dataset is medium size (1,000 to 100,000 samples): use a moderate architecture. "
            "Start with 2 to 4 hidden layers for tabular data."
        )
        self.report.add_training(
            "Medium dataset: Adam optimizer is a safe and effective default choice."
        )

    @Rule(DatasetFact(dataset_size="large"))
    def dataset_large(self):
        self.report.add_architecture(
            "Dataset is large (more than 100,000 samples): a deeper neural network can be "
            "considered. Mini-batch training is important for efficiency."
        )
        self.report.add_training(
            "Large dataset: SGD with momentum can be considered for large-scale training "
            "alongside Adam."
        )

    @Rule(DatasetFact(feature_count="high"))
    def feature_count_high(self):
        self.report.add_regularization(
            "Feature count is high (more than 100 features): apply regularization techniques. "
            "Consider dimensionality reduction or feature selection before training. "
            "Monitor overfitting carefully."
        )

    @Rule(DatasetFact(feature_sample_ratio="risky"))
    def feature_sample_risky(self):
        self.report.add_warning(
            "The number of features is high compared to the number of samples. "
            "This significantly increases the risk of overfitting."
        )
        self.report.add_regularization(
            "Risky feature-to-sample ratio: use stronger regularization, prefer a smaller "
            "model, and use a validation set carefully to detect overfitting early."
        )

    # =========================================================================
    # GROUP E — Numerical Feature Engineering Rules (8 rules)
    # =========================================================================

    @Rule(FeatureFact(has_numerical=1))
    def numerical_exists(self):
        self.report.add_preprocessing(
            "Numerical features detected: clean and scale them before training. "
            "Neural networks are sensitive to the magnitude of input values."
        )

    @Rule(
        AND(
            FeatureFact(has_numerical=1),
            FeatureFact(numerical_scale="different")
        )
    )
    def numerical_scale_different(self):
        self.report.add_feature_engineering(
            "Numerical features have different scales: apply feature scaling so that all "
            "inputs are on comparable ranges. Neural networks train faster and more "
            "reliably when features are normalized."
        )

    @Rule(
        AND(
            FeatureFact(has_numerical=1),
            FeatureFact(numerical_distribution="normal")
        )
    )
    def numerical_dist_normal(self):
        self.report.add_feature_engineering(
            "Numerical distribution is approximately normal: use Standardization "
            "(Z-score scaling) — subtract the mean and divide by the standard deviation."
        )

    @Rule(
        AND(
            FeatureFact(has_numerical=1),
            FeatureFact(numerical_distribution="bounded")
        )
    )
    def numerical_dist_bounded(self):
        self.report.add_feature_engineering(
            "Numerical distribution is bounded (has known min and max): use Min-Max scaling "
            "to map values into the [0, 1] range."
        )

    @Rule(
        AND(
            FeatureFact(has_numerical=1),
            FeatureFact(numerical_distribution="skewed")
        )
    )
    def numerical_dist_skewed(self):
        self.report.add_feature_engineering(
            "Numerical distribution is skewed: apply a log transformation or power "
            "transformation (e.g., Box-Cox, Yeo-Johnson) to reduce skewness before scaling."
        )

    @Rule(
        AND(
            FeatureFact(has_numerical=1),
            FeatureFact(numerical_distribution="unknown")
        )
    )
    def numerical_dist_unknown(self):
        self.report.add_feature_engineering(
            "The numerical distribution is unknown. Start with Standardization as a safe "
            "baseline, then compare with Min-Max or robust scaling during experiments."
        )

    @Rule(
        AND(
            FeatureFact(has_numerical=1),
            FeatureFact(numerical_outliers="yes")
        )
    )
    def numerical_outliers(self):
        self.report.add_feature_engineering(
            "Numerical outliers detected: use Robust Scaling (based on median and IQR) "
            "to reduce the influence of extreme values, or clip/remove outliers before training."
        )

    @Rule(
        AND(
            FeatureFact(has_numerical=1),
            FeatureFact(nonlinear_relationships="yes")
        )
    )
    def numerical_nonlinear(self):
        self.report.add_feature_engineering(
            "Nonlinear numerical relationships detected: consider adding polynomial features "
            "or interaction terms to expose these patterns to the model. "
            "Also consider binning continuous values into groups such as age ranges or "
            "income levels when a step-wise relationship is expected."
        )

    # =========================================================================
    # GROUP F — Categorical Feature Engineering Rules (7 rules)
    # =========================================================================

    @Rule(FeatureFact(has_categorical=1))
    def categorical_exists(self):
        self.report.add_preprocessing(
            "Categorical features detected: encode them into numerical representations "
            "before feeding them to a neural network."
        )

    @Rule(
        AND(
            FeatureFact(has_categorical=1),
            FeatureFact(categorical_cardinality="low")
        )
    )
    def categorical_low_cardinality(self):
        self.report.add_feature_engineering(
            "Categorical features have low cardinality (few unique values): "
            "use one-hot encoding to convert each category into a binary indicator column."
        )

    @Rule(
        AND(
            FeatureFact(has_categorical=1),
            FeatureFact(categorical_cardinality="high")
        )
    )
    def categorical_high_cardinality(self):
        self.report.add_feature_engineering(
            "Categorical features have high cardinality (many unique values): "
            "use target encoding or learnable embedding layers to avoid extremely "
            "wide one-hot vectors that can slow down training."
        )

    @Rule(
        AND(
            FeatureFact(has_categorical=1),
            FeatureFact(categorical_order="ordinal")
        )
    )
    def categorical_ordinal(self):
        self.report.add_feature_engineering(
            "Categorical features are ordinal (categories have a meaningful order): "
            "use ordinal encoding to preserve the rank information "
            "(e.g., low=1, medium=2, high=3)."
        )

    @Rule(
        AND(
            FeatureFact(has_categorical=1),
            FeatureFact(categorical_order="nominal")
        )
    )
    def categorical_nominal(self):
        self.report.add_feature_engineering(
            "Categorical features are nominal (no natural order between categories): "
            "use one-hot encoding or embedding layers. "
            "Do not use label encoding because it would imply a false ordinal relationship."
        )

    @Rule(
        AND(
            FeatureFact(has_categorical=1),
            FeatureFact(rare_categories="yes")
        )
    )
    def categorical_rare(self):
        self.report.add_feature_engineering(
            "Rare categories detected: group infrequent categories into a single "
            "'Other' category to reduce noise and prevent the model from learning "
            "spurious patterns from very few examples."
        )

    @Rule(
        AND(
            FeatureFact(has_categorical=1),
            FeatureFact(noisy_categories="yes")
        )
    )
    def categorical_noisy(self):
        self.report.add_preprocessing(
            "Noisy category names detected: standardize spelling, unify capitalization, "
            "and merge duplicates before encoding (e.g., 'male', 'Male', 'MALE' → 'male')."
        )

    # =========================================================================
    # GROUP G — Date-Time Feature Engineering Rules (4 rules)
    # =========================================================================

    @Rule(FeatureFact(has_datetime=1))
    def datetime_exists(self):
        self.report.add_feature_engineering(
            "Date-time features detected: extract components such as year, month, day, "
            "day of week, hour, minute, and weekend indicator as separate numerical features."
        )

    @Rule(
        AND(
            FeatureFact(has_datetime=1),
            FeatureFact(seasonality="yes")
        )
    )
    def datetime_seasonality(self):
        self.report.add_feature_engineering(
            "Seasonality detected in date-time data: create explicit seasonal features "
            "such as month, quarter, day of week, and public holiday indicator."
        )

    @Rule(
        AND(
            FeatureFact(has_datetime=1),
            FeatureFact(time_order="yes")
        )
    )
    def datetime_time_order(self):
        self.report.add_feature_engineering(
            "Time ordering matters: use time-aware train-validation splitting "
            "(never shuffle before splitting). Consider lag features or rolling window "
            "statistics as additional inputs."
        )
        self.report.add_checklist(
            "Ensure the train-validation split respects temporal order — "
            "all training data must come before validation data in time."
        )

    @Rule(
        AND(
            FeatureFact(has_datetime=1),
            FeatureFact(cyclic_time="yes")
        )
    )
    def datetime_cyclic(self):
        self.report.add_feature_engineering(
            "Cyclic time patterns detected: encode cyclic features using sine and cosine "
            "transformations so the model understands that hour 23 is close to hour 0. "
            "Example: hour_sin = sin(2π × hour / 24), hour_cos = cos(2π × hour / 24)."
        )

    # =========================================================================
    # GROUP H — Text Feature Engineering Rules (6 rules)
    # =========================================================================

    @Rule(FeatureFact(has_text=1))
    def text_exists(self):
        self.report.add_preprocessing(
            "Text features detected: clean text before processing — normalize case, "
            "remove punctuation and special characters, tokenize, and convert to a "
            "numerical representation suitable for the chosen model."
        )

    @Rule(
        AND(
            FeatureFact(has_text=1),
            FeatureFact(text_dataset_size="small")
        )
    )
    def text_small_dataset(self):
        self.report.add_feature_engineering(
            "Text dataset is small: use TF-IDF vectorization or simple pre-trained word "
            "embeddings (e.g., GloVe, FastText) with a compact dense network to avoid "
            "overfitting on limited data."
        )

    @Rule(
        AND(
            FeatureFact(has_text=1),
            FeatureFact(text_dataset_size="large")
        )
    )
    def text_large_dataset(self):
        self.report.add_feature_engineering(
            "Text dataset is large: use learned embedding layers combined with a neural "
            "architecture such as LSTM, GRU, 1D CNN, or a transformer-style encoder to "
            "capture complex linguistic patterns."
        )

    @Rule(
        AND(
            FeatureFact(has_text=1),
            FeatureFact(word_order="yes")
        )
    )
    def text_word_order(self):
        self.report.add_architecture(
            "Word order is important for the text task: use sequence models such as LSTM, "
            "GRU, 1D CNN, or a transformer-style encoder that preserve positional information."
        )

    @Rule(
        AND(
            FeatureFact(has_text=1),
            FeatureFact(text_length="short")
        )
    )
    def text_short(self):
        self.report.add_feature_engineering(
            "Text is short (e.g., tweets, titles, labels): TF-IDF, simple embeddings, "
            "or compact dense models often work well and train quickly."
        )

    @Rule(
        AND(
            FeatureFact(has_text=1),
            FeatureFact(text_length="long")
        )
    )
    def text_long(self):
        self.report.add_feature_engineering(
            "Text is long (e.g., articles, reports, documents): use models capable of "
            "capturing long-range context. Consider truncation, chunking, or hierarchical "
            "encoding strategies to handle length limitations."
        )

    # =========================================================================
    # GROUP I — Image Feature Engineering Rules (4 rules)
    # =========================================================================

    @Rule(FeatureFact(has_image=1))
    def image_exists(self):
        self.report.add_preprocessing(
            "Image features detected: normalize pixel values to the [0, 1] range by "
            "dividing by 255. Resize all images to a consistent spatial resolution "
            "before feeding them to the network."
        )

    @Rule(
        AND(
            FeatureFact(has_image=1),
            FeatureFact(image_dataset_size="small")
        )
    )
    def image_small_dataset(self):
        self.report.add_feature_engineering(
            "Image dataset is small: use transfer learning with a pretrained convolutional "
            "model (e.g., MobileNet, EfficientNet, ResNet). Freeze the base layers and "
            "fine-tune only the top layers on your data."
        )

    @Rule(
        AND(
            FeatureFact(has_image=1),
            FeatureFact(image_dataset_size="large")
        )
    )
    def image_large_dataset(self):
        self.report.add_feature_engineering(
            "Image dataset is large: train a CNN from scratch or fine-tune a pretrained "
            "CNN with a lower learning rate, depending on available computational resources."
        )

    @Rule(
        AND(
            FeatureFact(has_image=1),
            FeatureFact(image_augmentation="yes")
        )
    )
    def image_augmentation(self):
        self.report.add_feature_engineering(
            "Image augmentation is recommended: apply random transformations during training "
            "such as horizontal/vertical flipping, rotation, zoom, random cropping, and "
            "brightness or contrast adjustments to improve generalization."
        )

    # =========================================================================
    # GROUP J — Data Quality Rules (4 rules)
    # =========================================================================

    @Rule(QualityFact(missing_values="yes"))
    def quality_missing_values(self):
        self.report.add_preprocessing(
            "Missing values detected: handle them before training. Options include "
            "mean/median/mode imputation for numerical features, a dedicated 'missing' "
            "category for categorical features, or row removal when the proportion is small."
        )

    @Rule(QualityFact(class_imbalance="yes"))
    def quality_class_imbalance(self):
        self.report.add_warning(
            "Class imbalance detected: the dataset has significantly more samples of one "
            "class than others. Accuracy alone will be misleading."
        )
        self.report.add_training(
            "To handle class imbalance: use class weights in the loss function, apply "
            "oversampling (SMOTE) or undersampling, and monitor recall and F1-score "
            "rather than accuracy."
        )

    @Rule(QualityFact(noisy_labels="yes"))
    def quality_noisy_labels(self):
        self.report.add_warning(
            "Noisy labels detected: some training labels may be incorrect. "
            "Review and clean labels where possible. Use regularization and monitor "
            "validation loss carefully — a large gap between training and validation loss "
            "may indicate label noise rather than overfitting."
        )

    @Rule(QualityFact(data_leakage="yes"))
    def quality_data_leakage(self):
        self.report.add_warning(
            "Data leakage risk detected: ensure that preprocessing steps (scaling, "
            "imputation, encoding) are fitted only on training data and then applied "
            "to validation and test data. Also verify that future information is not "
            "included in any input features."
        )

    # =========================================================================
    # GROUP K — Evaluation Metrics Rules (5 rules)
    # =========================================================================

    @Rule(ProblemFact(target_kind="continuous"))
    def metrics_regression(self):
        self.report.add_metrics(
            "Regression task: use MAE (Mean Absolute Error), MSE (Mean Squared Error), "
            "RMSE (Root Mean Squared Error), and R-squared to evaluate model performance."
        )

    @Rule(ProblemFact(target_kind="binary"))
    def metrics_binary(self):
        self.report.add_metrics(
            "Binary classification: use Accuracy, Precision, Recall, F1-score, "
            "and ROC-AUC as primary evaluation metrics."
        )

    @Rule(ProblemFact(target_kind="multiclass"))
    def metrics_multiclass(self):
        self.report.add_metrics(
            "Multi-class classification: use Accuracy, Macro F1-score, "
            "Weighted F1-score, and the Confusion Matrix to evaluate performance "
            "across all classes."
        )

    @Rule(ProblemFact(target_kind="multilabel"))
    def metrics_multilabel(self):
        self.report.add_metrics(
            "Multi-label classification: use Micro F1-score, Macro F1-score, "
            "Precision, Recall, Hamming Loss, and Subset Accuracy to evaluate "
            "multi-label predictions."
        )

    @Rule(QualityFact(class_imbalance="yes"))
    def metrics_imbalanced(self):
        self.report.add_metrics(
            "Imbalanced classes: do not rely on accuracy alone. Prioritize Recall, "
            "Precision, F1-score, and PR-AUC (Precision-Recall Area Under Curve) "
            "as the main evaluation metrics."
        )

    # =========================================================================
    # GROUP L — Optimizer Rules (3 rules)
    # =========================================================================

    @Rule(DatasetFact(dataset_size="small"))
    def optimizer_small_dataset(self):
        self.report.add_training(
            "Small dataset: use the Adam optimizer as a strong default. "
            "It adapts the learning rate automatically and converges well with limited data."
        )

    @Rule(DatasetFact(dataset_size="large"))
    def optimizer_large_dataset(self):
        self.report.add_training(
            "Large dataset: Adam is a reliable baseline optimizer. For very large-scale "
            "training, SGD with momentum and a learning rate schedule can sometimes "
            "achieve better generalization."
        )

    @Rule(TrainingFact(unstable_training="yes"))
    def optimizer_unstable(self):
        self.report.add_training(
            "Unstable training detected: reduce the learning rate, add Batch Normalization "
            "layers, verify that input features are properly scaled, and check for "
            "exploding gradients using gradient clipping."
        )

    # =========================================================================
    # GROUP M — Training Symptom Rules (4 rules)
    # =========================================================================

    @Rule(TrainingFact(overfitting="yes"))
    def symptom_overfitting(self):
        self.report.add_regularization(
            "Overfitting detected: the model performs well on training data but poorly "
            "on validation data. Apply dropout layers, L2 weight regularization, "
            "and early stopping. Consider reducing model complexity."
        )
        self.report.add_training(
            "Overfitting remedies: add data augmentation where applicable, collect more "
            "training data if possible, and use a simpler architecture with fewer parameters."
        )

    @Rule(TrainingFact(underfitting="yes"))
    def symptom_underfitting(self):
        self.report.add_training(
            "Underfitting detected: the model is too simple or has not been trained long "
            "enough. Increase model capacity (more layers or neurons), train for more "
            "epochs, reduce excessive regularization, and improve feature representation."
        )

    @Rule(TrainingFact(unstable_training="yes"))
    def symptom_unstable(self):
        self.report.add_training(
            "Unstable training (loss oscillates or diverges): reduce the learning rate, "
            "use a learning rate scheduler, add Batch Normalization, and ensure all "
            "input features are properly scaled before training."
        )

    @Rule(TrainingFact(slow_training="yes"))
    def symptom_slow(self):
        self.report.add_training(
            "Slow training detected: increase the batch size if memory allows, use "
            "Batch Normalization to allow higher learning rates, reduce unnecessary "
            "features or input dimensions, and use GPU acceleration when available."
        )

    # =========================================================================
    # GROUP N — Final Checklist Rules (3 rules)
    # =========================================================================

    @Rule(DatasetFact(input_data_type="tabular"))
    def checklist_tabular_split(self):
        self.report.add_checklist(
            "Split tabular data into training, validation, and test sets before training. "
            "Use the test set only for final evaluation — never for hyperparameter tuning."
        )

    @Rule(FeatureFact(has_numerical=1))
    def checklist_scaling(self):
        self.report.add_checklist(
            "Fit all preprocessing transformers (scalers, encoders, imputers) exclusively "
            "on the training set, then apply the fitted transformer to the validation and "
            "test sets. Never fit on validation or test data."
        )

    @Rule(ProblemFact(target_kind="binary"))
    def checklist_binary_confusion(self):
        self.report.add_checklist(
            "After training a binary classifier, inspect the confusion matrix to understand "
            "the balance between false positives and false negatives for your use case."
        )
