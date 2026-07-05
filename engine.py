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
        self.report.add_explanation(
            "A-problem_regression",
            "target_kind='continuous' → the target is a real-valued number, so this is a "
            "regression problem requiring a linear output neuron and a magnitude-sensitive loss."
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
        self.report.add_explanation(
            "A-problem_binary",
            "target_kind='binary' → exactly two classes. Sigmoid maps logits to [0,1] "
            "as a single probability; binary cross-entropy penalises confident wrong answers."
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
        self.report.add_explanation(
            "A-problem_multiclass",
            "target_kind='multiclass' → more than two mutually exclusive classes. "
            "Softmax normalises logits to a valid probability distribution over all classes."
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
        self.report.add_explanation(
            "A-problem_multilabel",
            "target_kind='multilabel' → each sample can carry multiple independent labels. "
            "Independent sigmoids per label treat each label as a separate binary decision."
        )

    # =========================================================================
    # GROUP B — Label Format Rules (4 rules)
    # =========================================================================

    @Rule(ProblemFact(label_format="one_hot"))
    def label_one_hot(self):
        self.report.add_output_loss(
            "Labels are one-hot encoded: use categorical cross-entropy as the loss function."
        )
        self.report.add_explanation(
            "B-label_one_hot",
            "label_format='one_hot' → labels are already probability vectors; categorical "
            "cross-entropy compares them directly against the softmax output."
        )

    @Rule(ProblemFact(label_format="integer_class_ids"))
    def label_integer_ids(self):
        self.report.add_output_loss(
            "Labels are integer class IDs: use sparse categorical cross-entropy "
            "to avoid converting labels to one-hot vectors manually."
        )
        self.report.add_explanation(
            "B-label_integer_ids",
            "label_format='integer_class_ids' → raw integers (0, 1, 2 …) as labels. "
            "Sparse cross-entropy accepts integers directly, saving memory and a conversion step."
        )

    @Rule(ProblemFact(label_format="numeric_value"))
    def label_numeric_value(self):
        self.report.add_output_loss(
            "Labels are numeric values: use a regression loss such as MSE, MAE, or Huber loss."
        )
        self.report.add_explanation(
            "B-label_numeric_value",
            "label_format='numeric_value' → labels are real numbers; regression losses "
            "measure the distance between predicted and actual value."
        )

    @Rule(ProblemFact(label_format="multiple_binary_labels"))
    def label_multiple_binary(self):
        self.report.add_output_loss(
            "Labels are multiple independent binary flags: use sigmoid output units "
            "and binary cross-entropy for multi-label learning."
        )
        self.report.add_explanation(
            "B-label_multiple_binary",
            "label_format='multiple_binary_labels' → each position in the label vector is "
            "independently 0 or 1; sigmoid + BCE handles each position as its own binary task."
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
        self.report.add_explanation(
            "C-architecture_tabular",
            "input_data_type='tabular' → structured rows/columns have no spatial or temporal "
            "order, so a dense MLP is the natural and most effective architecture."
        )

    @Rule(DatasetFact(input_data_type="image"))
    def architecture_image(self):
        self.report.add_architecture(
            "Input is image data: use a Convolutional Neural Network (CNN). "
            "Stack convolution layers, pooling layers, then flatten or use global average "
            "pooling, followed by dense output layers."
        )
        self.report.add_explanation(
            "C-architecture_image",
            "input_data_type='image' → pixels have local spatial structure; CNNs exploit "
            "translation invariance through shared convolutional filters."
        )

    @Rule(DatasetFact(input_data_type="text"))
    def architecture_text(self):
        self.report.add_architecture(
            "Input is text data: use text vectorization followed by dense layers, "
            "or use embedding layers followed by LSTM, GRU, or a transformer-style encoder."
        )
        self.report.add_explanation(
            "C-architecture_text",
            "input_data_type='text' → words are discrete tokens with semantic relationships; "
            "embeddings project them into a continuous space that neural networks can process."
        )

    @Rule(DatasetFact(input_data_type="time_series"))
    def architecture_time_series(self):
        self.report.add_architecture(
            "Input is time-series data: use LSTM, GRU, temporal convolution, or 1D CNN. "
            "Preserve the temporal order during train-test splitting."
        )
        self.report.add_explanation(
            "C-architecture_time_series",
            "input_data_type='time_series' → observations are ordered in time; recurrent or "
            "1D convolutional architectures can capture temporal dependencies."
        )

    @Rule(DatasetFact(input_data_type="mixed"))
    def architecture_mixed(self):
        self.report.add_architecture(
            "Input is mixed data (multiple modalities): use a multi-input neural network. "
            "Create separate branches for tabular, text, image, or sequence inputs, "
            "then concatenate the learned representations before the output layer."
        )
        self.report.add_explanation(
            "C-architecture_mixed",
            "input_data_type='mixed' → different modalities require specialised sub-networks; "
            "their embeddings are merged before the shared output head."
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
        self.report.add_explanation(
            "D-dataset_small",
            "dataset_size='small' (<1 000 samples) → a deep model will memorise training "
            "examples; a shallow model + regularisation is more likely to generalise."
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
        self.report.add_explanation(
            "D-dataset_medium",
            "dataset_size='medium' (1k–100k samples) → enough data for a moderate architecture; "
            "Adam adapts learning rates per parameter and converges reliably."
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
        self.report.add_explanation(
            "D-dataset_large",
            "dataset_size='large' (>100k samples) → sufficient data for deeper models; "
            "SGD+momentum can generalise better than Adam on very large datasets."
        )

    @Rule(DatasetFact(feature_count="high"))
    def feature_count_high(self):
        self.report.add_regularization(
            "Feature count is high (more than 100 features): apply regularization techniques. "
            "Consider dimensionality reduction or feature selection before training. "
            "Monitor overfitting carefully."
        )
        self.report.add_explanation(
            "D-feature_count_high",
            "feature_count='high' (>100 features) → high-dimensional input increases the risk "
            "of the curse of dimensionality; regularisation and feature selection reduce noise."
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
        self.report.add_explanation(
            "D-feature_sample_risky",
            "feature_sample_ratio='risky' (features ≥10% of samples) → the model can memorise "
            "the training set because there is insufficient data per feature dimension."
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
        self.report.add_explanation(
            "E-numerical_exists",
            "has_numerical=1 → numerical columns are present; gradient-based optimisation "
            "converges much faster when inputs share a common scale."
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
        self.report.add_explanation(
            "E-numerical_scale_different",
            "has_numerical=1 AND numerical_scale='different' → features with vastly different "
            "magnitudes cause some weights to update much faster than others, destabilising training."
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
        self.report.add_explanation(
            "E-numerical_dist_normal",
            "numerical_distribution='normal' → Z-score standardisation centres the distribution "
            "at 0 with unit variance, which matches the natural assumptions of many activations."
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
        self.report.add_explanation(
            "E-numerical_dist_bounded",
            "numerical_distribution='bounded' → known min/max makes Min-Max scaling exact "
            "and safe; it preserves zero values and the relative spacing of data points."
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
        self.report.add_explanation(
            "E-numerical_dist_skewed",
            "numerical_distribution='skewed' → long tails produce extreme input values that "
            "dominate gradients; a power transform compresses the tail before scaling."
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
        self.report.add_explanation(
            "E-numerical_dist_unknown",
            "numerical_distribution='unknown' → with no prior knowledge of the distribution, "
            "standardisation is the safest starting point; compare against alternatives empirically."
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
        self.report.add_explanation(
            "E-numerical_outliers",
            "has_numerical=1 AND numerical_outliers='yes' → extreme values inflate the mean and "
            "standard deviation; RobustScaler uses the median/IQR which is resistant to outliers."
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
        self.report.add_explanation(
            "E-numerical_nonlinear",
            "nonlinear_relationships='yes' → shallow networks may not capture complex curves; "
            "polynomial features or binning make the non-linearity explicit for the model."
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
        self.report.add_explanation(
            "F-categorical_exists",
            "has_categorical=1 → neural networks operate on real-valued tensors; categorical "
            "columns must be converted to numbers before they can be processed."
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
        self.report.add_explanation(
            "F-categorical_low_cardinality",
            "categorical_cardinality='low' → few unique values means one-hot expansion is small "
            "and introduces no false ordinal relationship between categories."
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
        self.report.add_explanation(
            "F-categorical_high_cardinality",
            "categorical_cardinality='high' → one-hot with many categories creates sparse "
            "high-dimensional inputs; embeddings learn a dense, compact representation instead."
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
        self.report.add_explanation(
            "F-categorical_ordinal",
            "categorical_order='ordinal' → there is a meaningful rank between categories; "
            "ordinal encoding preserves this information as a numeric distance."
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
        self.report.add_explanation(
            "F-categorical_nominal",
            "categorical_order='nominal' → no rank exists between categories; label encoding "
            "would imply a numeric order that doesn't exist and mislead the model."
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
        self.report.add_explanation(
            "F-categorical_rare",
            "rare_categories='yes' → categories with very few samples give the model almost "
            "no signal but add parameters; grouping them reduces noise."
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
        self.report.add_explanation(
            "F-categorical_noisy",
            "noisy_categories='yes' → spelling variants are treated as separate categories, "
            "inflating cardinality and adding noise; normalisation merges duplicates."
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
        self.report.add_explanation(
            "G-datetime_exists",
            "has_datetime=1 → raw datetime strings carry no numeric meaning; extracted "
            "components expose calendar patterns the model can learn from."
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
        self.report.add_explanation(
            "G-datetime_seasonality",
            "seasonality='yes' → periodic patterns tied to the calendar (e.g., holiday peaks) "
            "need explicit seasonal indicators so the model can learn them."
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
        self.report.add_explanation(
            "G-datetime_time_order",
            "time_order='yes' → shuffling would leak future information into training; "
            "a chronological split ensures the model is evaluated on truly unseen future data."
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
        self.report.add_explanation(
            "G-datetime_cyclic",
            "cyclic_time='yes' → numeric encoding of hours treats 23 and 0 as far apart; "
            "sine/cosine encoding wraps the cycle so proximity is preserved."
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
        self.report.add_explanation(
            "H-text_exists",
            "has_text=1 → raw text strings must be tokenised and vectorised before a "
            "neural network can process them; cleaning reduces vocabulary noise."
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
        self.report.add_explanation(
            "H-text_small_dataset",
            "text_dataset_size='small' → training embeddings from scratch on little text leads "
            "to poor representations; pre-trained embeddings provide a head start."
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
        self.report.add_explanation(
            "H-text_large_dataset",
            "text_dataset_size='large' → enough text to train embeddings from scratch or "
            "fine-tune transformer models to capture domain-specific language patterns."
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
        self.report.add_explanation(
            "H-text_word_order",
            "word_order='yes' → meaning depends on word sequence (e.g., 'cat eats fish' ≠ "
            "'fish eats cat'); bag-of-words models discard this information."
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
        self.report.add_explanation(
            "H-text_short",
            "text_length='short' → short sequences have limited context; simple models "
            "avoid overfitting and are faster to train on short text."
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
        self.report.add_explanation(
            "H-text_long",
            "text_length='long' → RNNs struggle with long-range dependencies; transformers "
            "or hierarchical models better handle documents with many sentences."
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
        self.report.add_explanation(
            "I-image_exists",
            "has_image=1 → raw pixel values (0–255) have a large magnitude; dividing by 255 "
            "maps them to [0,1] which stabilises gradient flow through the network."
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
        self.report.add_explanation(
            "I-image_small_dataset",
            "image_dataset_size='small' → training a CNN from scratch needs tens of thousands "
            "of images; pretrained weights provide rich features that transfer across domains."
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
        self.report.add_explanation(
            "I-image_large_dataset",
            "image_dataset_size='large' → sufficient images to learn visual features from "
            "scratch; full fine-tuning of a pretrained model is also viable."
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
        self.report.add_explanation(
            "I-image_augmentation",
            "image_augmentation='yes' → artificially expanding the training set with random "
            "transforms reduces overfitting and improves the model's spatial invariance."
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
        self.report.add_explanation(
            "J-quality_missing_values",
            "missing_values='yes' → NaN inputs propagate through the network as NaN "
            "gradients, breaking training; imputation or removal resolves this."
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
        self.report.add_explanation(
            "J-quality_class_imbalance",
            "class_imbalance='yes' → a model predicting only the majority class achieves "
            "high accuracy while being useless; weighted loss forces it to learn minority patterns."
        )

    @Rule(QualityFact(noisy_labels="yes"))
    def quality_noisy_labels(self):
        self.report.add_warning(
            "Noisy labels detected: some training labels may be incorrect. "
            "Review and clean labels where possible. Use regularization and monitor "
            "validation loss carefully — a large gap between training and validation loss "
            "may indicate label noise rather than overfitting."
        )
        self.report.add_explanation(
            "J-quality_noisy_labels",
            "noisy_labels='yes' → wrong labels act as adversarial training signal; "
            "early stopping and regularisation prevent the model from memorising the incorrect labels."
        )

    @Rule(QualityFact(data_leakage="yes"))
    def quality_data_leakage(self):
        self.report.add_warning(
            "Data leakage risk detected: ensure that preprocessing steps (scaling, "
            "imputation, encoding) are fitted only on training data and then applied "
            "to validation and test data. Also verify that future information is not "
            "included in any input features."
        )
        self.report.add_explanation(
            "J-quality_data_leakage",
            "data_leakage='yes' → fitting scalers on the full dataset contaminates validation "
            "metrics with training information, causing overly optimistic evaluation."
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
        self.report.add_explanation(
            "K-metrics_regression",
            "target_kind='continuous' → regression metrics measure signed/unsigned distance "
            "between prediction and ground truth; R² indicates explained variance."
        )

    @Rule(ProblemFact(target_kind="binary"))
    def metrics_binary(self):
        self.report.add_metrics(
            "Binary classification: use Accuracy, Precision, Recall, F1-score, "
            "and ROC-AUC as primary evaluation metrics."
        )
        self.report.add_explanation(
            "K-metrics_binary",
            "target_kind='binary' → ROC-AUC measures the model's ability to rank positives "
            "above negatives regardless of threshold; F1 balances precision and recall."
        )

    @Rule(ProblemFact(target_kind="multiclass"))
    def metrics_multiclass(self):
        self.report.add_metrics(
            "Multi-class classification: use Accuracy, Macro F1-score, "
            "Weighted F1-score, and the Confusion Matrix to evaluate performance "
            "across all classes."
        )
        self.report.add_explanation(
            "K-metrics_multiclass",
            "target_kind='multiclass' → per-class F1 reveals which classes are hard; "
            "macro-average treats all classes equally regardless of frequency."
        )

    @Rule(ProblemFact(target_kind="multilabel"))
    def metrics_multilabel(self):
        self.report.add_metrics(
            "Multi-label classification: use Micro F1-score, Macro F1-score, "
            "Precision, Recall, Hamming Loss, and Subset Accuracy to evaluate "
            "multi-label predictions."
        )
        self.report.add_explanation(
            "K-metrics_multilabel",
            "target_kind='multilabel' → subset accuracy (all labels correct) is very strict; "
            "Hamming loss measures per-label error rate which is more informative."
        )

    @Rule(QualityFact(class_imbalance="yes"))
    def metrics_imbalanced(self):
        self.report.add_metrics(
            "Imbalanced classes: do not rely on accuracy alone. Prioritize Recall, "
            "Precision, F1-score, and PR-AUC (Precision-Recall Area Under Curve) "
            "as the main evaluation metrics."
        )
        self.report.add_explanation(
            "K-metrics_imbalanced",
            "class_imbalance='yes' → accuracy is dominated by the majority class; "
            "PR-AUC focuses on the minority class performance which is usually the class of interest."
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
        self.report.add_explanation(
            "L-optimizer_small_dataset",
            "dataset_size='small' → Adam's per-parameter adaptive rates help it converge "
            "even when gradients are noisy due to few samples."
        )

    @Rule(DatasetFact(dataset_size="large"))
    def optimizer_large_dataset(self):
        self.report.add_training(
            "Large dataset: Adam is a reliable baseline optimizer. For very large-scale "
            "training, SGD with momentum and a learning rate schedule can sometimes "
            "achieve better generalization."
        )
        self.report.add_explanation(
            "L-optimizer_large_dataset",
            "dataset_size='large' → with many mini-batches per epoch SGD+momentum can "
            "escape sharp minima and find flatter, more generalisable solutions."
        )

    @Rule(TrainingFact(unstable_training="yes"))
    def optimizer_unstable(self):
        self.report.add_training(
            "Unstable training detected: reduce the learning rate, add Batch Normalization "
            "layers, verify that input features are properly scaled, and check for "
            "exploding gradients using gradient clipping."
        )
        self.report.add_explanation(
            "L-optimizer_unstable",
            "unstable_training='yes' → large gradient norms cause loss spikes; clipping, "
            "lower LR, and BatchNorm normalise activations and stabilise the update magnitude."
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
        self.report.add_explanation(
            "M-symptom_overfitting",
            "overfitting='yes' → the model has memorised training examples; dropout randomly "
            "disables neurons during training, forcing the network to learn redundant representations."
        )

    @Rule(TrainingFact(underfitting="yes"))
    def symptom_underfitting(self):
        self.report.add_training(
            "Underfitting detected: the model is too simple or has not been trained long "
            "enough. Increase model capacity (more layers or neurons), train for more "
            "epochs, reduce excessive regularization, and improve feature representation."
        )
        self.report.add_explanation(
            "M-symptom_underfitting",
            "underfitting='yes' → both train and val loss are high; the model lacks "
            "capacity to fit the data — adding layers/neurons or reducing regularisation helps."
        )

    @Rule(TrainingFact(unstable_training="yes"))
    def symptom_unstable(self):
        self.report.add_training(
            "Unstable training (loss oscillates or diverges): reduce the learning rate, "
            "use a learning rate scheduler, add Batch Normalization, and ensure all "
            "input features are properly scaled before training."
        )
        self.report.add_explanation(
            "M-symptom_unstable",
            "unstable_training='yes' → oscillating loss often indicates too-high learning rate "
            "or unscaled inputs; a scheduler decays LR to settle into a stable minimum."
        )

    @Rule(TrainingFact(slow_training="yes"))
    def symptom_slow(self):
        self.report.add_training(
            "Slow training detected: increase the batch size if memory allows, use "
            "Batch Normalization to allow higher learning rates, reduce unnecessary "
            "features or input dimensions, and use GPU acceleration when available."
        )
        self.report.add_explanation(
            "M-symptom_slow",
            "slow_training='yes' → small batches under-utilise hardware; larger batches "
            "and higher LR (enabled by BatchNorm) speed up convergence per wall-clock second."
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
        self.report.add_explanation(
            "N-checklist_tabular_split",
            "input_data_type='tabular' → a held-out test set is essential to estimate "
            "real-world performance; using it for tuning creates an optimistic bias."
        )

    @Rule(FeatureFact(has_numerical=1))
    def checklist_scaling(self):
        self.report.add_checklist(
            "Fit all preprocessing transformers (scalers, encoders, imputers) exclusively "
            "on the training set, then apply the fitted transformer to the validation and "
            "test sets. Never fit on validation or test data."
        )
        self.report.add_explanation(
            "N-checklist_scaling",
            "has_numerical=1 → fitting a scaler on val/test data leaks their statistics into "
            "the model; fitting only on train ensures unbiased evaluation."
        )

    @Rule(ProblemFact(target_kind="binary"))
    def checklist_binary_confusion(self):
        self.report.add_checklist(
            "After training a binary classifier, inspect the confusion matrix to understand "
            "the balance between false positives and false negatives for your use case."
        )
        self.report.add_explanation(
            "N-checklist_binary_confusion",
            "target_kind='binary' → the relative cost of FP vs FN varies by application; "
            "the confusion matrix reveals which error type the model makes more often."
        )
