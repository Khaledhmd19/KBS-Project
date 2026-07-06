from experta import Fact


class ProblemFact(Fact):
    """
    Holds information about the problem type and label format.

    Fields:
        target_kind   : "continuous" | "binary" | "multiclass" | "multilabel"
        label_format  : "numeric_value" | "integer_class_ids" | "one_hot" | "multiple_binary_labels"
    """
    pass


class DatasetFact(Fact):
    """
    Holds information about the dataset properties.

    Fields:
        input_data_type     : "tabular" | "image" | "text" | "time_series" | "mixed"
        dataset_size        : "small" | "medium" | "large"
        feature_count       : "low" | "medium" | "high"
        feature_sample_ratio: "safe" | "risky"
    """
    pass


class FeatureFact(Fact):
    """
    Holds information about feature types and their properties.

    Numerical fields:
        has_numerical          : 1 | 0
        numerical_scale        : "similar" | "different"
        numerical_distribution : "normal" | "bounded" | "skewed" | "unknown"
        numerical_outliers     : "yes" | "no"
        nonlinear_relationships: "yes" | "no"

    Categorical fields:
        has_categorical       : 1 | 0
        categorical_cardinality: "low" | "high"
        categorical_order     : "nominal" | "ordinal"
        rare_categories       : "yes" | "no"
        noisy_categories      : "yes" | "no"

    Date-time fields:
        has_datetime  : 1 | 0
        seasonality   : "yes" | "no"
        time_order    : "yes" | "no"
        cyclic_time   : "yes" | "no"

    Text fields:
        has_text          : 1 | 0
        text_length       : "short" | "long"
        text_dataset_size : "small" | "large"
        word_order        : "yes" | "no"

    Image fields:
        has_image           : 1 | 0
        image_dataset_size  : "small" | "large"
        image_augmentation  : "yes" | "no"
    """
    pass


class QualityFact(Fact):
    """
    Holds information about data quality issues.

    Fields:
        missing_values : "yes" | "no"
        class_imbalance: "yes" | "no"
        noisy_labels   : "yes" | "no"
        data_leakage   : "yes" | "no"
    """
    pass


class TrainingFact(Fact):
    """
    Holds information about observed training symptoms.

    Fields:
        overfitting      : "yes" | "no"
        underfitting     : "yes" | "no"
        unstable_training: "yes" | "no"
        slow_training    : "yes" | "no"
    """
    pass


class FrameworkFact(Fact):
    """
    Holds the target framework name.

    Fields:
        name: "pytorch" | "keras"
    """
    pass

