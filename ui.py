# =============================================================================
# ui.py
# NeuralNetworkExpertApp — CustomTkinter UI for the expert system.
#
# RESTRICTION: No if / elif / else / for / while / match / case anywhere.
# No dynamic widget creation. Every widget and variable is declared manually.
# The UI collects answers and declares facts — it makes no decisions.
# =============================================================================

import customtkinter

from constants import (
    WINDOW_TITLE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    LEFT_PANEL_W,
    APP_BG,
    CARD_BG,
    PANEL_BG,
    ACCENT,
    TEXT_MAIN,
    TEXT_MUTED,
    FONT_FAMILY,
    FONT_TITLE_SIZE,
    FONT_SECTION_SIZE,
    FONT_NORMAL_SIZE,
    FONT_SMALL_SIZE,
    FONT_WEIGHT_BOLD,
    FONT_WEIGHT_NORMAL,
    PAD_OUTER,
    PAD_INNER,
    PAD_WIDGET,
    APP_SUBTITLE,
    APP_DESCRIPTION,
    REPORT_TITLE,
    REPORT_HINT,
    BTN_ANALYZE,
    BTN_RESET,
    BTN_COPY,
    TAB_PROBLEM,
    TAB_DATASET,
    TAB_FEATURES,
    TAB_QUALITY,
    TAB_TRAINING,
)
from report import ReportBuilder
from engine import NeuralNetworkDesignEngine
from facts import ProblemFact, DatasetFact, FeatureFact, QualityFact, TrainingFact
from profiler import profile_dataset

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")


class NeuralNetworkExpertApp(customtkinter.CTk):
    """
    Main application window.

    Responsibilities:
        - Display the input questionnaire across 5 tabs.
        - On Analyze: declare all facts into the Experta engine and display the report.
        - On Reset: restore all variables to defaults and clear the report.
        - On Copy Report: place the current report text in the clipboard.

    The UI makes zero decisions. All reasoning is done by the Experta engine.
    """

    def __init__(self):
        super().__init__()

        # --- Window setup ---
        self.title(WINDOW_TITLE)
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.configure(fg_color=APP_BG)
        self.resizable(True, True)

        # --- Root grid: 2 columns ---
        self.grid_columnconfigure(0, weight=0, minsize=LEFT_PANEL_W)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Fonts (created manually, no loop) ---
        self.font_title = customtkinter.CTkFont(
            family=FONT_FAMILY, size=FONT_TITLE_SIZE, weight=FONT_WEIGHT_BOLD
        )
        self.font_section = customtkinter.CTkFont(
            family=FONT_FAMILY, size=FONT_SECTION_SIZE, weight=FONT_WEIGHT_BOLD
        )
        self.font_normal = customtkinter.CTkFont(
            family=FONT_FAMILY, size=FONT_NORMAL_SIZE, weight=FONT_WEIGHT_NORMAL
        )
        self.font_small = customtkinter.CTkFont(
            family=FONT_FAMILY, size=FONT_SMALL_SIZE, weight=FONT_WEIGHT_NORMAL
        )
        self.font_bold = customtkinter.CTkFont(
            family=FONT_FAMILY, size=FONT_NORMAL_SIZE, weight=FONT_WEIGHT_BOLD
        )

        # --- Variables, report, engine ---
        self._create_variables()
        self.report = ReportBuilder()
        self.engine = NeuralNetworkDesignEngine(self.report)

        # --- Build UI panels ---
        self._build_left_panel()
        self._build_right_panel()

    # =========================================================================
    # Variable Declaration (35 variables, no loops)
    # =========================================================================

    def _create_variables(self):
        # Problem tab
        self.target_kind_var = customtkinter.StringVar(value="binary")
        self.label_format_var = customtkinter.StringVar(value="integer_class_ids")
        self.input_data_type_var = customtkinter.StringVar(value="tabular")

        # Dataset tab
        self.dataset_size_var = customtkinter.StringVar(value="medium")
        self.feature_count_var = customtkinter.StringVar(value="medium")
        self.feature_sample_ratio_var = customtkinter.StringVar(value="safe")

        # Features — Numerical
        self.has_numerical_var = customtkinter.IntVar(value=1)
        self.numerical_scale_var = customtkinter.StringVar(value="different")
        self.numerical_distribution_var = customtkinter.StringVar(value="unknown")
        self.numerical_outliers_var = customtkinter.StringVar(value="no")
        self.nonlinear_relationships_var = customtkinter.StringVar(value="no")

        # Features — Categorical
        self.has_categorical_var = customtkinter.IntVar(value=1)
        self.categorical_cardinality_var = customtkinter.StringVar(value="low")
        self.categorical_order_var = customtkinter.StringVar(value="nominal")
        self.rare_categories_var = customtkinter.StringVar(value="no")
        self.noisy_categories_var = customtkinter.StringVar(value="no")

        # Features — Date-Time
        self.has_datetime_var = customtkinter.IntVar(value=0)
        self.seasonality_var = customtkinter.StringVar(value="no")
        self.time_order_var = customtkinter.StringVar(value="no")
        self.cyclic_time_var = customtkinter.StringVar(value="no")

        # Features — Text
        self.has_text_var = customtkinter.IntVar(value=0)
        self.text_length_var = customtkinter.StringVar(value="short")
        self.text_dataset_size_var = customtkinter.StringVar(value="small")
        self.word_order_var = customtkinter.StringVar(value="no")

        # Features — Image
        self.has_image_var = customtkinter.IntVar(value=0)
        self.image_dataset_size_var = customtkinter.StringVar(value="small")
        self.image_augmentation_var = customtkinter.StringVar(value="yes")

        # Quality tab
        self.missing_values_var = customtkinter.StringVar(value="no")
        self.class_imbalance_var = customtkinter.StringVar(value="no")
        self.noisy_labels_var = customtkinter.StringVar(value="no")
        self.data_leakage_var = customtkinter.StringVar(value="no")

        # Training Symptoms tab
        self.overfitting_var = customtkinter.StringVar(value="no")
        self.underfitting_var = customtkinter.StringVar(value="no")
        self.unstable_training_var = customtkinter.StringVar(value="no")
        self.slow_training_var = customtkinter.StringVar(value="no")

        # ★ Auto badge text vars — one per profiler-fillable field
        self.badge_target_kind_var = customtkinter.StringVar(value="")
        self.badge_dataset_size_var = customtkinter.StringVar(value="")
        self.badge_feature_count_var = customtkinter.StringVar(value="")
        self.badge_fsr_var = customtkinter.StringVar(value="")
        self.badge_has_numerical_var = customtkinter.StringVar(value="")
        self.badge_num_scale_var = customtkinter.StringVar(value="")
        self.badge_num_outliers_var = customtkinter.StringVar(value="")
        self.badge_missing_values_var = customtkinter.StringVar(value="")

    # =========================================================================
    # Left Panel
    # =========================================================================

    def _build_left_panel(self):
        self.left_panel = customtkinter.CTkFrame(self, fg_color=APP_BG, corner_radius=0)
        self.left_panel.grid(row=0, column=0, sticky="nsew")
        self.left_panel.grid_columnconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_tabview()
        self._build_buttons()

    def _build_header(self):
        header_frame = customtkinter.CTkFrame(
            self.left_panel, fg_color=PANEL_BG, corner_radius=10
        )
        header_frame.grid(
            row=0, column=0, padx=PAD_OUTER, pady=(PAD_OUTER, 8), sticky="ew"
        )

        customtkinter.CTkLabel(
            header_frame, text=WINDOW_TITLE, font=self.font_title, text_color=ACCENT
        ).pack(padx=PAD_INNER, pady=(PAD_INNER, 2))

        customtkinter.CTkLabel(
            header_frame, text=APP_SUBTITLE, font=self.font_small, text_color=TEXT_MUTED
        ).pack(padx=PAD_INNER, pady=(0, 2))

        customtkinter.CTkLabel(
            header_frame,
            text=APP_DESCRIPTION,
            font=self.font_small,
            text_color=TEXT_MUTED,
            wraplength=LEFT_PANEL_W - 60,
        ).pack(padx=PAD_INNER, pady=(0, PAD_INNER))

    def _build_tabview(self):
        self.tabview = customtkinter.CTkTabview(
            self.left_panel,
            fg_color=CARD_BG,
            segmented_button_fg_color=PANEL_BG,
            segmented_button_selected_color=ACCENT,
            segmented_button_selected_hover_color="#2BA8D4",
            segmented_button_unselected_color=PANEL_BG,
            segmented_button_unselected_hover_color="#263B4A",
            text_color=TEXT_MAIN,
            corner_radius=10,
        )
        self.tabview.grid(row=1, column=0, padx=PAD_OUTER, pady=(0, 8), sticky="nsew")

        self.tabview.add(TAB_PROBLEM)
        self.tabview.add(TAB_DATASET)
        self.tabview.add(TAB_FEATURES)
        self.tabview.add(TAB_QUALITY)
        self.tabview.add(TAB_TRAINING)

        self._build_problem_tab(self.tabview.tab(TAB_PROBLEM))
        self._build_dataset_tab(self.tabview.tab(TAB_DATASET))
        self._build_features_tab(self.tabview.tab(TAB_FEATURES))
        self._build_quality_tab(self.tabview.tab(TAB_QUALITY))
        self._build_training_tab(self.tabview.tab(TAB_TRAINING))

    def _build_buttons(self):
        btn_frame = customtkinter.CTkFrame(
            self.left_panel, fg_color=APP_BG, corner_radius=0
        )
        btn_frame.grid(
            row=2, column=0, padx=PAD_OUTER, pady=(0, PAD_OUTER), sticky="ew"
        )
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)
        btn_frame.grid_columnconfigure(2, weight=1)
        btn_frame.grid_columnconfigure(3, weight=1)

        customtkinter.CTkButton(
            btn_frame,
            text="Load CSV",
            font=self.font_bold,
            fg_color="#4ADE80",
            hover_color="#22C55E",
            text_color="#0F172A",
            corner_radius=8,
            height=42,
            command=self.load_csv,
        ).grid(row=0, column=0, padx=(0, 5), sticky="ew")

        customtkinter.CTkButton(
            btn_frame,
            text=BTN_ANALYZE,
            font=self.font_bold,
            fg_color=ACCENT,
            hover_color="#2BA8D4",
            text_color="#0F172A",
            corner_radius=8,
            height=42,
            command=self.analyze,
        ).grid(row=0, column=1, padx=(0, 5), sticky="ew")

        customtkinter.CTkButton(
            btn_frame,
            text=BTN_RESET,
            font=self.font_bold,
            fg_color=PANEL_BG,
            hover_color="#2E3F50",
            text_color=TEXT_MAIN,
            corner_radius=8,
            height=42,
            command=self.reset,
        ).grid(row=0, column=2, padx=(0, 5), sticky="ew")

        customtkinter.CTkButton(
            btn_frame,
            text=BTN_COPY,
            font=self.font_bold,
            fg_color=PANEL_BG,
            hover_color="#2E3F50",
            text_color=TEXT_MAIN,
            corner_radius=8,
            height=42,
            command=self.copy_report,
        ).grid(row=0, column=3, sticky="ew")

    # =========================================================================
    # Problem Tab
    # =========================================================================

    def _build_problem_tab(self, tab):
        tab.configure(fg_color=CARD_BG)
        scroll = customtkinter.CTkScrollableFrame(
            tab, fg_color=CARD_BG, corner_radius=0
        )
        scroll.pack(fill="both", expand=True, padx=4, pady=4)

        # — Target Kind —
        customtkinter.CTkLabel(
            scroll,
            text="Target Variable Type",
            font=self.font_section,
            text_color=ACCENT,
        ).pack(anchor="w", padx=PAD_INNER, pady=(PAD_INNER, 2))
        customtkinter.CTkLabel(
            scroll,
            text="What kind of value does the model predict?",
            font=self.font_small,
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=PAD_INNER, pady=(0, 6))

        self.radio_continuous = customtkinter.CTkRadioButton(
            scroll,
            text="Continuous  (Regression — predicts a number)",
            variable=self.target_kind_var,
            value="continuous",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_continuous.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_binary = customtkinter.CTkRadioButton(
            scroll,
            text="Binary  (Two-class classification)",
            variable=self.target_kind_var,
            value="binary",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_binary.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_multiclass = customtkinter.CTkRadioButton(
            scroll,
            text="Multi-class  (More than two exclusive classes)",
            variable=self.target_kind_var,
            value="multiclass",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_multiclass.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_multilabel = customtkinter.CTkRadioButton(
            scroll,
            text="Multi-label  (Sample can belong to multiple classes)",
            variable=self.target_kind_var,
            value="multilabel",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_multilabel.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        customtkinter.CTkFrame(scroll, height=1, fg_color=PANEL_BG).pack(
            fill="x", padx=PAD_INNER, pady=PAD_INNER
        )

        # — Label Format —
        customtkinter.CTkLabel(
            scroll,
            text="Label / Target Format",
            font=self.font_section,
            text_color=ACCENT,
        ).pack(anchor="w", padx=PAD_INNER, pady=(4, 2))
        customtkinter.CTkLabel(
            scroll,
            text="How are the target labels stored in your dataset?",
            font=self.font_small,
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=PAD_INNER, pady=(0, 6))

        self.radio_numeric_value = customtkinter.CTkRadioButton(
            scroll,
            text="Numeric value  (e.g. 3.7, 150.0 — for regression)",
            variable=self.label_format_var,
            value="numeric_value",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_numeric_value.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_integer_ids = customtkinter.CTkRadioButton(
            scroll,
            text="Integer class IDs  (e.g. 0, 1, 2)",
            variable=self.label_format_var,
            value="integer_class_ids",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_integer_ids.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_one_hot = customtkinter.CTkRadioButton(
            scroll,
            text="One-hot encoded  (e.g. [0, 1, 0])",
            variable=self.label_format_var,
            value="one_hot",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_one_hot.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_multi_binary = customtkinter.CTkRadioButton(
            scroll,
            text="Multiple binary labels  (e.g. [1, 0, 1, 1])",
            variable=self.label_format_var,
            value="multiple_binary_labels",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_multi_binary.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        customtkinter.CTkFrame(scroll, height=1, fg_color=PANEL_BG).pack(
            fill="x", padx=PAD_INNER, pady=PAD_INNER
        )

        # — Input Data Type —
        customtkinter.CTkLabel(
            scroll,
            text="Main Input Data Type",
            font=self.font_section,
            text_color=ACCENT,
        ).pack(anchor="w", padx=PAD_INNER, pady=(4, 2))
        customtkinter.CTkLabel(
            scroll,
            text="What is the primary format of the model's input features?",
            font=self.font_small,
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=PAD_INNER, pady=(0, 6))

        self.radio_tabular = customtkinter.CTkRadioButton(
            scroll,
            text="Tabular  (structured rows and columns)",
            variable=self.input_data_type_var,
            value="tabular",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_tabular.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_image = customtkinter.CTkRadioButton(
            scroll,
            text="Image  (pixel arrays, photos)",
            variable=self.input_data_type_var,
            value="image",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_image.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_text_type = customtkinter.CTkRadioButton(
            scroll,
            text="Text  (natural language, documents)",
            variable=self.input_data_type_var,
            value="text",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_text_type.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_time_series = customtkinter.CTkRadioButton(
            scroll,
            text="Time Series  (sequential measurements over time)",
            variable=self.input_data_type_var,
            value="time_series",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_time_series.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_mixed = customtkinter.CTkRadioButton(
            scroll,
            text="Mixed  (multiple input modalities combined)",
            variable=self.input_data_type_var,
            value="mixed",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_mixed.pack(anchor="w", padx=PAD_INNER, pady=(PAD_WIDGET, PAD_INNER))

    # =========================================================================
    # Dataset Tab
    # =========================================================================

    def _build_dataset_tab(self, tab):
        tab.configure(fg_color=CARD_BG)
        scroll = customtkinter.CTkScrollableFrame(
            tab, fg_color=CARD_BG, corner_radius=0
        )
        scroll.pack(fill="both", expand=True, padx=4, pady=4)

        # — Dataset Size —
        ds_hdr = customtkinter.CTkFrame(scroll, fg_color="transparent")
        ds_hdr.pack(fill="x", padx=PAD_INNER, pady=(PAD_INNER, 0))
        customtkinter.CTkLabel(
            ds_hdr, text="Dataset Size", font=self.font_section, text_color=ACCENT
        ).pack(side="left")
        self.badge_dataset_size_lbl = customtkinter.CTkLabel(
            ds_hdr,
            textvariable=self.badge_dataset_size_var,
            font=self.font_small,
            text_color="#4ADE80",
        )
        self.badge_dataset_size_lbl.pack(side="left", padx=(8, 0))
        customtkinter.CTkLabel(
            scroll,
            text="How many total samples (rows) are in your dataset?",
            font=self.font_small,
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=PAD_INNER, pady=(0, 6))

        self.radio_ds_small = customtkinter.CTkRadioButton(
            scroll,
            text="Small  (fewer than 1,000 samples)",
            variable=self.dataset_size_var,
            value="small",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
            command=lambda: self.badge_dataset_size_var.set(""),
        )
        self.radio_ds_small.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_ds_medium = customtkinter.CTkRadioButton(
            scroll,
            text="Medium  (1,000 to 100,000 samples)",
            variable=self.dataset_size_var,
            value="medium",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
            command=lambda: self.badge_dataset_size_var.set(""),
        )
        self.radio_ds_medium.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_ds_large = customtkinter.CTkRadioButton(
            scroll,
            text="Large  (more than 100,000 samples)",
            variable=self.dataset_size_var,
            value="large",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
            command=lambda: self.badge_dataset_size_var.set(""),
        )
        self.radio_ds_large.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        customtkinter.CTkFrame(scroll, height=1, fg_color=PANEL_BG).pack(
            fill="x", padx=PAD_INNER, pady=PAD_INNER
        )

        # — Feature Count —
        fc_hdr = customtkinter.CTkFrame(scroll, fg_color="transparent")
        fc_hdr.pack(fill="x", padx=PAD_INNER, pady=(4, 0))
        customtkinter.CTkLabel(
            fc_hdr,
            text="Number of Input Features",
            font=self.font_section,
            text_color=ACCENT,
        ).pack(side="left")
        self.badge_feature_count_lbl = customtkinter.CTkLabel(
            fc_hdr,
            textvariable=self.badge_feature_count_var,
            font=self.font_small,
            text_color="#4ADE80",
        )
        self.badge_feature_count_lbl.pack(side="left", padx=(8, 0))
        customtkinter.CTkLabel(
            scroll,
            text="How many input columns / features does your dataset have?",
            font=self.font_small,
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=PAD_INNER, pady=(0, 6))

        self.radio_fc_low = customtkinter.CTkRadioButton(
            scroll,
            text="Low  (fewer than 20 features)",
            variable=self.feature_count_var,
            value="low",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
            command=lambda: self.badge_feature_count_var.set(""),
        )
        self.radio_fc_low.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_fc_medium = customtkinter.CTkRadioButton(
            scroll,
            text="Medium  (20 to 100 features)",
            variable=self.feature_count_var,
            value="medium",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
            command=lambda: self.badge_feature_count_var.set(""),
        )
        self.radio_fc_medium.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_fc_high = customtkinter.CTkRadioButton(
            scroll,
            text="High  (more than 100 features)",
            variable=self.feature_count_var,
            value="high",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
            command=lambda: self.badge_feature_count_var.set(""),
        )
        self.radio_fc_high.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        customtkinter.CTkFrame(scroll, height=1, fg_color=PANEL_BG).pack(
            fill="x", padx=PAD_INNER, pady=PAD_INNER
        )

        # — Feature-Sample Ratio —
        fsr_hdr = customtkinter.CTkFrame(scroll, fg_color="transparent")
        fsr_hdr.pack(fill="x", padx=PAD_INNER, pady=(4, 0))
        customtkinter.CTkLabel(
            fsr_hdr,
            text="Feature-to-Sample Risk",
            font=self.font_section,
            text_color=ACCENT,
        ).pack(side="left")
        self.badge_fsr_lbl = customtkinter.CTkLabel(
            fsr_hdr,
            textvariable=self.badge_fsr_var,
            font=self.font_small,
            text_color="#4ADE80",
        )
        self.badge_fsr_lbl.pack(side="left", padx=(8, 0))
        customtkinter.CTkLabel(
            scroll,
            text="Is the number of features high relative to the number of samples?",
            font=self.font_small,
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=PAD_INNER, pady=(0, 6))

        self.radio_fsr_safe = customtkinter.CTkRadioButton(
            scroll,
            text="Safe  (many more samples than features)",
            variable=self.feature_sample_ratio_var,
            value="safe",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
            command=lambda: self.badge_fsr_var.set(""),
        )
        self.radio_fsr_safe.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_fsr_risky = customtkinter.CTkRadioButton(
            scroll,
            text="Risky  (features are many compared to samples)",
            variable=self.feature_sample_ratio_var,
            value="risky",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
            command=lambda: self.badge_fsr_var.set(""),
        )
        self.radio_fsr_risky.pack(
            anchor="w", padx=PAD_INNER, pady=(PAD_WIDGET, PAD_INNER)
        )

    # =========================================================================
    # Features Tab
    # =========================================================================

    def _build_features_tab(self, tab):
        tab.configure(fg_color=CARD_BG)
        scroll = customtkinter.CTkScrollableFrame(
            tab, fg_color=CARD_BG, corner_radius=0
        )
        scroll.pack(fill="both", expand=True, padx=4, pady=4)

        # ---- Numerical Features ----
        num_hdr = customtkinter.CTkFrame(scroll, fg_color="transparent")
        num_hdr.pack(fill="x", padx=PAD_INNER, pady=(PAD_INNER, 4))
        customtkinter.CTkLabel(
            num_hdr,
            text="Numerical Features",
            font=self.font_section,
            text_color=ACCENT,
        ).pack(side="left")
        self.badge_has_numerical_lbl = customtkinter.CTkLabel(
            num_hdr,
            textvariable=self.badge_has_numerical_var,
            font=self.font_small,
            text_color="#4ADE80",
        )
        self.badge_has_numerical_lbl.pack(side="left", padx=(8, 0))

        self.check_numerical = customtkinter.CTkCheckBox(
            scroll,
            text="Dataset contains numerical features",
            variable=self.has_numerical_var,
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
            checkmark_color="#0F172A",
            command=lambda: self.badge_has_numerical_var.set(""),
        )
        self.check_numerical.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        scale_row = customtkinter.CTkFrame(scroll, fg_color="transparent")
        scale_row.pack(fill="x", padx=PAD_INNER, pady=(8, 2))
        customtkinter.CTkLabel(
            scale_row, text="Feature scales:", font=self.font_bold, text_color=TEXT_MAIN
        ).pack(side="left")
        self.badge_num_scale_lbl = customtkinter.CTkLabel(
            scale_row,
            textvariable=self.badge_num_scale_var,
            font=self.font_small,
            text_color="#4ADE80",
        )
        self.badge_num_scale_lbl.pack(side="left", padx=(8, 0))

        self.radio_scale_similar = customtkinter.CTkRadioButton(
            scroll,
            text="Similar scales  (all in the same range)",
            variable=self.numerical_scale_var,
            value="similar",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
            command=lambda: self.badge_num_scale_var.set(""),
        )
        self.radio_scale_similar.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        self.radio_scale_different = customtkinter.CTkRadioButton(
            scroll,
            text="Different scales  (very different magnitudes)",
            variable=self.numerical_scale_var,
            value="different",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
            command=lambda: self.badge_num_scale_var.set(""),
        )
        self.radio_scale_different.pack(
            anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET
        )

        customtkinter.CTkLabel(
            scroll,
            text="Distribution shape:",
            font=self.font_bold,
            text_color=TEXT_MAIN,
        ).pack(anchor="w", padx=PAD_INNER, pady=(8, 2))

        self.radio_dist_normal = customtkinter.CTkRadioButton(
            scroll,
            text="Normal  (bell-shaped / Gaussian)",
            variable=self.numerical_distribution_var,
            value="normal",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_dist_normal.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        self.radio_dist_bounded = customtkinter.CTkRadioButton(
            scroll,
            text="Bounded  (fixed min and max range)",
            variable=self.numerical_distribution_var,
            value="bounded",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_dist_bounded.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        self.radio_dist_skewed = customtkinter.CTkRadioButton(
            scroll,
            text="Skewed  (long tail on one side)",
            variable=self.numerical_distribution_var,
            value="skewed",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_dist_skewed.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        self.radio_dist_unknown = customtkinter.CTkRadioButton(
            scroll,
            text="Unknown  (not sure about the distribution)",
            variable=self.numerical_distribution_var,
            value="unknown",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_dist_unknown.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        out_row = customtkinter.CTkFrame(scroll, fg_color="transparent")
        out_row.pack(fill="x", padx=PAD_INNER, pady=(8, 2))
        customtkinter.CTkLabel(
            out_row, text="Outliers:", font=self.font_bold, text_color=TEXT_MAIN
        ).pack(side="left")
        self.badge_num_outliers_lbl = customtkinter.CTkLabel(
            out_row,
            textvariable=self.badge_num_outliers_var,
            font=self.font_small,
            text_color="#4ADE80",
        )
        self.badge_num_outliers_lbl.pack(side="left", padx=(8, 0))

        self.radio_outliers_yes = customtkinter.CTkRadioButton(
            scroll,
            text="Yes  (extreme values exist)",
            variable=self.numerical_outliers_var,
            value="yes",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
            command=lambda: self.badge_num_outliers_var.set(""),
        )
        self.radio_outliers_yes.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        self.radio_outliers_no = customtkinter.CTkRadioButton(
            scroll,
            text="No  (no significant outliers)",
            variable=self.numerical_outliers_var,
            value="no",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
            command=lambda: self.badge_num_outliers_var.set(""),
        )
        self.radio_outliers_no.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        customtkinter.CTkLabel(
            scroll,
            text="Nonlinear relationships:",
            font=self.font_bold,
            text_color=TEXT_MAIN,
        ).pack(anchor="w", padx=PAD_INNER, pady=(8, 2))

        self.radio_nonlinear_yes = customtkinter.CTkRadioButton(
            scroll,
            text="Yes  (features interact nonlinearly with the target)",
            variable=self.nonlinear_relationships_var,
            value="yes",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_nonlinear_yes.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        self.radio_nonlinear_no = customtkinter.CTkRadioButton(
            scroll,
            text="No  (relationships are roughly linear)",
            variable=self.nonlinear_relationships_var,
            value="no",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_nonlinear_no.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        customtkinter.CTkFrame(scroll, height=2, fg_color=PANEL_BG).pack(
            fill="x", padx=PAD_INNER, pady=PAD_INNER
        )

        # ---- Categorical Features ----
        customtkinter.CTkLabel(
            scroll,
            text="Categorical Features",
            font=self.font_section,
            text_color=ACCENT,
        ).pack(anchor="w", padx=PAD_INNER, pady=(4, 4))

        self.check_categorical = customtkinter.CTkCheckBox(
            scroll,
            text="Dataset contains categorical features",
            variable=self.has_categorical_var,
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
            checkmark_color="#0F172A",
        )
        self.check_categorical.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        customtkinter.CTkLabel(
            scroll, text="Cardinality:", font=self.font_bold, text_color=TEXT_MAIN
        ).pack(anchor="w", padx=PAD_INNER, pady=(8, 2))

        self.radio_card_low = customtkinter.CTkRadioButton(
            scroll,
            text="Low  (few unique categories per column)",
            variable=self.categorical_cardinality_var,
            value="low",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_card_low.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        self.radio_card_high = customtkinter.CTkRadioButton(
            scroll,
            text="High  (many unique categories per column)",
            variable=self.categorical_cardinality_var,
            value="high",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_card_high.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        customtkinter.CTkLabel(
            scroll, text="Order type:", font=self.font_bold, text_color=TEXT_MAIN
        ).pack(anchor="w", padx=PAD_INNER, pady=(8, 2))

        self.radio_nominal = customtkinter.CTkRadioButton(
            scroll,
            text="Nominal  (no natural order, e.g. city names, colors)",
            variable=self.categorical_order_var,
            value="nominal",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_nominal.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        self.radio_ordinal = customtkinter.CTkRadioButton(
            scroll,
            text="Ordinal  (meaningful order, e.g. low / medium / high)",
            variable=self.categorical_order_var,
            value="ordinal",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_ordinal.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        customtkinter.CTkLabel(
            scroll, text="Rare categories:", font=self.font_bold, text_color=TEXT_MAIN
        ).pack(anchor="w", padx=PAD_INNER, pady=(8, 2))

        self.radio_rare_yes = customtkinter.CTkRadioButton(
            scroll,
            text="Yes  (some categories appear very rarely)",
            variable=self.rare_categories_var,
            value="yes",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_rare_yes.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        self.radio_rare_no = customtkinter.CTkRadioButton(
            scroll,
            text="No  (all categories appear reasonably often)",
            variable=self.rare_categories_var,
            value="no",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_rare_no.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        customtkinter.CTkLabel(
            scroll,
            text="Noisy category names:",
            font=self.font_bold,
            text_color=TEXT_MAIN,
        ).pack(anchor="w", padx=PAD_INNER, pady=(8, 2))

        self.radio_noisy_cat_yes = customtkinter.CTkRadioButton(
            scroll,
            text="Yes  (typos, inconsistent capitalization, duplicates)",
            variable=self.noisy_categories_var,
            value="yes",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_noisy_cat_yes.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        self.radio_noisy_cat_no = customtkinter.CTkRadioButton(
            scroll,
            text="No  (category names are clean and consistent)",
            variable=self.noisy_categories_var,
            value="no",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_noisy_cat_no.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        customtkinter.CTkFrame(scroll, height=2, fg_color=PANEL_BG).pack(
            fill="x", padx=PAD_INNER, pady=PAD_INNER
        )

        # ---- Date-Time Features ----
        customtkinter.CTkLabel(
            scroll, text="Date-Time Features", font=self.font_section, text_color=ACCENT
        ).pack(anchor="w", padx=PAD_INNER, pady=(4, 4))

        self.check_datetime = customtkinter.CTkCheckBox(
            scroll,
            text="Dataset contains date / time columns",
            variable=self.has_datetime_var,
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
            checkmark_color="#0F172A",
        )
        self.check_datetime.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        customtkinter.CTkLabel(
            scroll, text="Seasonality:", font=self.font_bold, text_color=TEXT_MAIN
        ).pack(anchor="w", padx=PAD_INNER, pady=(8, 2))

        self.radio_season_yes = customtkinter.CTkRadioButton(
            scroll,
            text="Yes  (seasonal patterns exist)",
            variable=self.seasonality_var,
            value="yes",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_season_yes.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        self.radio_season_no = customtkinter.CTkRadioButton(
            scroll,
            text="No  (no significant seasonal patterns)",
            variable=self.seasonality_var,
            value="no",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_season_no.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        customtkinter.CTkLabel(
            scroll,
            text="Time order matters:",
            font=self.font_bold,
            text_color=TEXT_MAIN,
        ).pack(anchor="w", padx=PAD_INNER, pady=(8, 2))

        self.radio_timeord_yes = customtkinter.CTkRadioButton(
            scroll,
            text="Yes  (past → future ordering is meaningful)",
            variable=self.time_order_var,
            value="yes",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_timeord_yes.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        self.radio_timeord_no = customtkinter.CTkRadioButton(
            scroll,
            text="No  (temporal ordering is not critical)",
            variable=self.time_order_var,
            value="no",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_timeord_no.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        customtkinter.CTkLabel(
            scroll,
            text="Cyclic time patterns:",
            font=self.font_bold,
            text_color=TEXT_MAIN,
        ).pack(anchor="w", padx=PAD_INNER, pady=(8, 2))

        self.radio_cyclic_yes = customtkinter.CTkRadioButton(
            scroll,
            text="Yes  (e.g. hour of day, day of week, month)",
            variable=self.cyclic_time_var,
            value="yes",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_cyclic_yes.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        self.radio_cyclic_no = customtkinter.CTkRadioButton(
            scroll,
            text="No  (no cyclic time features)",
            variable=self.cyclic_time_var,
            value="no",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_cyclic_no.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        customtkinter.CTkFrame(scroll, height=2, fg_color=PANEL_BG).pack(
            fill="x", padx=PAD_INNER, pady=PAD_INNER
        )

        # ---- Text Features ----
        customtkinter.CTkLabel(
            scroll, text="Text Features", font=self.font_section, text_color=ACCENT
        ).pack(anchor="w", padx=PAD_INNER, pady=(4, 4))

        self.check_text = customtkinter.CTkCheckBox(
            scroll,
            text="Dataset contains text / natural language columns",
            variable=self.has_text_var,
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
            checkmark_color="#0F172A",
        )
        self.check_text.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        customtkinter.CTkLabel(
            scroll, text="Text length:", font=self.font_bold, text_color=TEXT_MAIN
        ).pack(anchor="w", padx=PAD_INNER, pady=(8, 2))

        self.radio_text_short = customtkinter.CTkRadioButton(
            scroll,
            text="Short  (tweets, titles, short sentences)",
            variable=self.text_length_var,
            value="short",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_text_short.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        self.radio_text_long = customtkinter.CTkRadioButton(
            scroll,
            text="Long  (paragraphs, articles, full documents)",
            variable=self.text_length_var,
            value="long",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_text_long.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        customtkinter.CTkLabel(
            scroll, text="Text dataset size:", font=self.font_bold, text_color=TEXT_MAIN
        ).pack(anchor="w", padx=PAD_INNER, pady=(8, 2))

        self.radio_tsize_small = customtkinter.CTkRadioButton(
            scroll,
            text="Small  (limited labelled text data)",
            variable=self.text_dataset_size_var,
            value="small",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_tsize_small.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        self.radio_tsize_large = customtkinter.CTkRadioButton(
            scroll,
            text="Large  (abundant labelled text data)",
            variable=self.text_dataset_size_var,
            value="large",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_tsize_large.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        customtkinter.CTkLabel(
            scroll,
            text="Word order importance:",
            font=self.font_bold,
            text_color=TEXT_MAIN,
        ).pack(anchor="w", padx=PAD_INNER, pady=(8, 2))

        self.radio_wordord_yes = customtkinter.CTkRadioButton(
            scroll,
            text="Yes  (sentence structure and word order matter)",
            variable=self.word_order_var,
            value="yes",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_wordord_yes.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        self.radio_wordord_no = customtkinter.CTkRadioButton(
            scroll,
            text="No  (bag-of-words approach is acceptable)",
            variable=self.word_order_var,
            value="no",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_wordord_no.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        customtkinter.CTkFrame(scroll, height=2, fg_color=PANEL_BG).pack(
            fill="x", padx=PAD_INNER, pady=PAD_INNER
        )

        # ---- Image Features ----
        customtkinter.CTkLabel(
            scroll, text="Image Features", font=self.font_section, text_color=ACCENT
        ).pack(anchor="w", padx=PAD_INNER, pady=(4, 4))

        self.check_image = customtkinter.CTkCheckBox(
            scroll,
            text="Dataset contains image data",
            variable=self.has_image_var,
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
            checkmark_color="#0F172A",
        )
        self.check_image.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        customtkinter.CTkLabel(
            scroll,
            text="Image dataset size:",
            font=self.font_bold,
            text_color=TEXT_MAIN,
        ).pack(anchor="w", padx=PAD_INNER, pady=(8, 2))

        self.radio_isize_small = customtkinter.CTkRadioButton(
            scroll,
            text="Small  (fewer than ~5,000 images)",
            variable=self.image_dataset_size_var,
            value="small",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_isize_small.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        self.radio_isize_large = customtkinter.CTkRadioButton(
            scroll,
            text="Large  (tens of thousands of images or more)",
            variable=self.image_dataset_size_var,
            value="large",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_isize_large.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        customtkinter.CTkLabel(
            scroll,
            text="Image augmentation needed:",
            font=self.font_bold,
            text_color=TEXT_MAIN,
        ).pack(anchor="w", padx=PAD_INNER, pady=(8, 2))

        self.radio_aug_yes = customtkinter.CTkRadioButton(
            scroll,
            text="Yes  (augmentation should be applied during training)",
            variable=self.image_augmentation_var,
            value="yes",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_aug_yes.pack(anchor="w", padx=PAD_INNER + 12, pady=PAD_WIDGET)

        self.radio_aug_no = customtkinter.CTkRadioButton(
            scroll,
            text="No  (use images as-is without augmentation)",
            variable=self.image_augmentation_var,
            value="no",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_aug_no.pack(
            anchor="w", padx=PAD_INNER + 12, pady=(PAD_WIDGET, PAD_INNER)
        )

    # =========================================================================
    # Quality Tab
    # =========================================================================

    def _build_quality_tab(self, tab):
        tab.configure(fg_color=CARD_BG)
        scroll = customtkinter.CTkScrollableFrame(
            tab, fg_color=CARD_BG, corner_radius=0
        )
        scroll.pack(fill="both", expand=True, padx=4, pady=4)

        # — Missing Values —
        customtkinter.CTkLabel(
            scroll, text="Missing Values", font=self.font_section, text_color=ACCENT
        ).pack(anchor="w", padx=PAD_INNER, pady=(PAD_INNER, 2))
        customtkinter.CTkLabel(
            scroll,
            text="Does your dataset contain NaN or null values?",
            font=self.font_small,
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=PAD_INNER, pady=(0, 6))

        self.radio_mv_yes = customtkinter.CTkRadioButton(
            scroll,
            text="Yes  (some cells are missing)",
            variable=self.missing_values_var,
            value="yes",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_mv_yes.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_mv_no = customtkinter.CTkRadioButton(
            scroll,
            text="No  (the dataset is complete)",
            variable=self.missing_values_var,
            value="no",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_mv_no.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        customtkinter.CTkFrame(scroll, height=1, fg_color=PANEL_BG).pack(
            fill="x", padx=PAD_INNER, pady=PAD_INNER
        )

        # — Class Imbalance —
        customtkinter.CTkLabel(
            scroll, text="Class Imbalance", font=self.font_section, text_color=ACCENT
        ).pack(anchor="w", padx=PAD_INNER, pady=(4, 2))
        customtkinter.CTkLabel(
            scroll,
            text="Is one class much more frequent than the others?",
            font=self.font_small,
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=PAD_INNER, pady=(0, 6))

        self.radio_ci_yes = customtkinter.CTkRadioButton(
            scroll,
            text="Yes  (significant class imbalance exists)",
            variable=self.class_imbalance_var,
            value="yes",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_ci_yes.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_ci_no = customtkinter.CTkRadioButton(
            scroll,
            text="No  (classes are roughly balanced)",
            variable=self.class_imbalance_var,
            value="no",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_ci_no.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        customtkinter.CTkFrame(scroll, height=1, fg_color=PANEL_BG).pack(
            fill="x", padx=PAD_INNER, pady=PAD_INNER
        )

        # — Noisy Labels —
        customtkinter.CTkLabel(
            scroll, text="Noisy Labels", font=self.font_section, text_color=ACCENT
        ).pack(anchor="w", padx=PAD_INNER, pady=(4, 2))
        customtkinter.CTkLabel(
            scroll,
            text="Do you suspect some training labels are incorrect?",
            font=self.font_small,
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=PAD_INNER, pady=(0, 6))

        self.radio_nl_yes = customtkinter.CTkRadioButton(
            scroll,
            text="Yes  (some labels may be wrong or noisy)",
            variable=self.noisy_labels_var,
            value="yes",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_nl_yes.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_nl_no = customtkinter.CTkRadioButton(
            scroll,
            text="No  (labels are reliable and accurate)",
            variable=self.noisy_labels_var,
            value="no",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_nl_no.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        customtkinter.CTkFrame(scroll, height=1, fg_color=PANEL_BG).pack(
            fill="x", padx=PAD_INNER, pady=PAD_INNER
        )

        # — Data Leakage —
        customtkinter.CTkLabel(
            scroll, text="Data Leakage Risk", font=self.font_section, text_color=ACCENT
        ).pack(anchor="w", padx=PAD_INNER, pady=(4, 2))
        customtkinter.CTkLabel(
            scroll,
            text="Could any feature contain information not available at prediction time?",
            font=self.font_small,
            text_color=TEXT_MUTED,
            wraplength=580,
        ).pack(anchor="w", padx=PAD_INNER, pady=(0, 6))

        self.radio_dl_yes = customtkinter.CTkRadioButton(
            scroll,
            text="Yes  (possible data leakage identified)",
            variable=self.data_leakage_var,
            value="yes",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_dl_yes.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_dl_no = customtkinter.CTkRadioButton(
            scroll,
            text="No  (features are safe and leak-free)",
            variable=self.data_leakage_var,
            value="no",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_dl_no.pack(anchor="w", padx=PAD_INNER, pady=(PAD_WIDGET, PAD_INNER))

    # =========================================================================
    # Training Symptoms Tab
    # =========================================================================

    def _build_training_tab(self, tab):
        tab.configure(fg_color=CARD_BG)
        scroll = customtkinter.CTkScrollableFrame(
            tab, fg_color=CARD_BG, corner_radius=0
        )
        scroll.pack(fill="both", expand=True, padx=4, pady=4)

        customtkinter.CTkLabel(
            scroll,
            text="Use this tab when redesigning or diagnosing an existing neural network.",
            font=self.font_small,
            text_color=TEXT_MUTED,
            wraplength=580,
        ).pack(anchor="w", padx=PAD_INNER, pady=(PAD_INNER, PAD_INNER))

        # — Overfitting —
        customtkinter.CTkLabel(
            scroll, text="Overfitting", font=self.font_section, text_color=ACCENT
        ).pack(anchor="w", padx=PAD_INNER, pady=(4, 2))
        customtkinter.CTkLabel(
            scroll,
            text="Training accuracy is high but validation accuracy is much lower.",
            font=self.font_small,
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=PAD_INNER, pady=(0, 6))

        self.radio_of_yes = customtkinter.CTkRadioButton(
            scroll,
            text="Yes  (model is overfitting)",
            variable=self.overfitting_var,
            value="yes",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_of_yes.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_of_no = customtkinter.CTkRadioButton(
            scroll,
            text="No  (not observed)",
            variable=self.overfitting_var,
            value="no",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_of_no.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        customtkinter.CTkFrame(scroll, height=1, fg_color=PANEL_BG).pack(
            fill="x", padx=PAD_INNER, pady=PAD_INNER
        )

        # — Underfitting —
        customtkinter.CTkLabel(
            scroll, text="Underfitting", font=self.font_section, text_color=ACCENT
        ).pack(anchor="w", padx=PAD_INNER, pady=(4, 2))
        customtkinter.CTkLabel(
            scroll,
            text="Both training and validation performance are poor.",
            font=self.font_small,
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=PAD_INNER, pady=(0, 6))

        self.radio_uf_yes = customtkinter.CTkRadioButton(
            scroll,
            text="Yes  (model is underfitting)",
            variable=self.underfitting_var,
            value="yes",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_uf_yes.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_uf_no = customtkinter.CTkRadioButton(
            scroll,
            text="No  (not observed)",
            variable=self.underfitting_var,
            value="no",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_uf_no.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        customtkinter.CTkFrame(scroll, height=1, fg_color=PANEL_BG).pack(
            fill="x", padx=PAD_INNER, pady=PAD_INNER
        )

        # — Unstable Training —
        customtkinter.CTkLabel(
            scroll, text="Unstable Training", font=self.font_section, text_color=ACCENT
        ).pack(anchor="w", padx=PAD_INNER, pady=(4, 2))
        customtkinter.CTkLabel(
            scroll,
            text="Loss oscillates wildly, spikes, or diverges during training.",
            font=self.font_small,
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=PAD_INNER, pady=(0, 6))

        self.radio_ut_yes = customtkinter.CTkRadioButton(
            scroll,
            text="Yes  (training is unstable)",
            variable=self.unstable_training_var,
            value="yes",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_ut_yes.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_ut_no = customtkinter.CTkRadioButton(
            scroll,
            text="No  (training is stable)",
            variable=self.unstable_training_var,
            value="no",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_ut_no.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        customtkinter.CTkFrame(scroll, height=1, fg_color=PANEL_BG).pack(
            fill="x", padx=PAD_INNER, pady=PAD_INNER
        )

        # — Slow Training —
        customtkinter.CTkLabel(
            scroll, text="Slow Training", font=self.font_section, text_color=ACCENT
        ).pack(anchor="w", padx=PAD_INNER, pady=(4, 2))
        customtkinter.CTkLabel(
            scroll,
            text="Training takes much longer than expected per epoch.",
            font=self.font_small,
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=PAD_INNER, pady=(0, 6))

        self.radio_st_yes = customtkinter.CTkRadioButton(
            scroll,
            text="Yes  (training is noticeably slow)",
            variable=self.slow_training_var,
            value="yes",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_st_yes.pack(anchor="w", padx=PAD_INNER, pady=PAD_WIDGET)

        self.radio_st_no = customtkinter.CTkRadioButton(
            scroll,
            text="No  (training speed is acceptable)",
            variable=self.slow_training_var,
            value="no",
            font=self.font_normal,
            text_color=TEXT_MAIN,
            fg_color=ACCENT,
        )
        self.radio_st_no.pack(anchor="w", padx=PAD_INNER, pady=(PAD_WIDGET, PAD_INNER))

    # =========================================================================
    # Right Panel — Report Output
    # =========================================================================

    def _build_right_panel(self):
        from codegen import generate_code as _gen

        self._gen = _gen

        self.right_panel = customtkinter.CTkFrame(
            self, fg_color=PANEL_BG, corner_radius=0
        )
        self.right_panel.grid(row=0, column=1, sticky="nsew")
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(1, weight=1)

        # ── Output tab view ────────────────────────────────────────────────────
        self.output_tabs = customtkinter.CTkTabview(
            self.right_panel,
            fg_color=CARD_BG,
            segmented_button_fg_color=PANEL_BG,
            segmented_button_selected_color=ACCENT,
            segmented_button_selected_hover_color="#2BA8D4",
            segmented_button_unselected_color=PANEL_BG,
            segmented_button_unselected_hover_color="#2E3F50",
            text_color=TEXT_MAIN,
            corner_radius=8,
        )
        self.output_tabs.grid(
            row=1, column=0, padx=PAD_INNER, pady=(PAD_INNER, 0), sticky="nsew"
        )

        self.output_tabs.add("Report")
        self.output_tabs.add("Explanation Trace")

        # ── Report tab ─────────────────────────────────────────────────────────
        report_tab = self.output_tabs.tab("Report")
        report_tab.grid_columnconfigure(0, weight=1)
        report_tab.grid_rowconfigure(0, weight=1)

        self.report_box = customtkinter.CTkTextbox(
            report_tab,
            font=self.font_small,
            fg_color=CARD_BG,
            text_color=TEXT_MAIN,
            corner_radius=0,
            wrap="word",
            activate_scrollbars=True,
        )
        self.report_box.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")
        self.report_box.insert("end", REPORT_HINT)
        self.report_box.configure(state="disabled")

        # ── Explanation Trace tab ──────────────────────────────────────────────
        expl_tab = self.output_tabs.tab("Explanation Trace")
        expl_tab.grid_columnconfigure(0, weight=1)
        expl_tab.grid_rowconfigure(0, weight=1)

        self.expl_box = customtkinter.CTkTextbox(
            expl_tab,
            font=self.font_small,
            fg_color=CARD_BG,
            text_color="#93C5FD",
            corner_radius=0,
            wrap="word",
            activate_scrollbars=True,
        )
        self.expl_box.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")
        self.expl_box.insert("end", "Run Analyze to see which rules fired and why.")
        self.expl_box.configure(state="disabled")

        # ── Code generation buttons ────────────────────────────────────────────
        codegen_frame = customtkinter.CTkFrame(
            self.right_panel, fg_color=PANEL_BG, corner_radius=0
        )
        codegen_frame.grid(
            row=2, column=0, padx=PAD_INNER, pady=(4, PAD_INNER), sticky="ew"
        )
        codegen_frame.grid_columnconfigure(0, weight=1)
        codegen_frame.grid_columnconfigure(1, weight=1)

        customtkinter.CTkLabel(
            codegen_frame,
            text="Generate Starter Code:",
            font=self.font_small,
            text_color=TEXT_MUTED,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 4))

        customtkinter.CTkButton(
            codegen_frame,
            text="⚡ PyTorch",
            font=self.font_bold,
            fg_color="#EF4444",
            hover_color="#DC2626",
            text_color="#FFFFFF",
            corner_radius=8,
            height=36,
            command=lambda: self._generate_code("pytorch"),
        ).grid(row=1, column=0, padx=(0, 4), sticky="ew")

        customtkinter.CTkButton(
            codegen_frame,
            text="🔷 Keras",
            font=self.font_bold,
            fg_color="#7C3AED",
            hover_color="#6D28D9",
            text_color="#FFFFFF",
            corner_radius=8,
            height=36,
            command=lambda: self._generate_code("keras"),
        ).grid(row=1, column=1, sticky="ew")

    # =========================================================================
    # Profiler Integration
    # =========================================================================

    def load_csv(self):
        """Open a file dialog, run the profiler, and auto-fill the questionnaire vars."""
        from tkinter import filedialog, messagebox

        path = filedialog.askopenfilename(
            title="Select a CSV dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        # filedialog returns "" when the user cancels — treat empty path as no-op
        # (uses 'and' short-circuit, not an if statement)
        path and self._apply_profile(path)

    def _apply_profile(self, path: str):
        """Run profiler on the given CSV path and populate questionnaire vars."""
        from tkinter import messagebox

        try:
            p = profile_dataset(path)
        except Exception as exc:
            from tkinter import messagebox

            messagebox.showerror("Profiler Error", str(exc))
            return

        # --- Fill Dataset tab vars and show ★ Auto badges ---
        self.dataset_size_var.set(p["dataset_size"])
        self.badge_dataset_size_var.set(f"★ Auto  ({p['_row_count']:,} rows)")

        self.feature_count_var.set(p["feature_count"])
        self.badge_feature_count_var.set(f"★ Auto  ({p['_col_count']} features)")

        self.feature_sample_ratio_var.set(p["feature_sample_ratio"])
        self.badge_fsr_var.set("★ Auto")

        # --- Fill Features tab vars and show ★ Auto badges ---
        self.has_numerical_var.set(p["has_numerical"])
        self.badge_has_numerical_var.set(
            f"★ Auto  ({p['_numeric_col_count']} numeric cols)"
        )

        self.numerical_scale_var.set(p["numerical_scale"])
        self.badge_num_scale_var.set("★ Auto")

        self.numerical_outliers_var.set(p["numerical_outliers"])
        self.badge_num_outliers_var.set("★ Auto")

        # --- Fill Quality tab vars ---
        self.missing_values_var.set(p["missing_values"])

        # --- Fill Problem tab: target kind ---
        self.target_kind_var.set(p["target_kind"])
        self.badge_target_kind_var.set(
            f"★ Auto  (target: {p['_target_col']}, {p['_unique_target_values']} unique values)"
        )

        # Jump to the Dataset tab to show the user what changed
        self.tabview.set("Dataset")

    # =========================================================================
    # Callbacks (no conditions, no loops)
    # =========================================================================

    def analyze(self):
        """Reset the engine, declare all facts one-by-one, run rules, display report."""
        self.report.clear()
        self.engine.reset()

        self.engine.declare(ProblemFact(target_kind=self.target_kind_var.get()))
        self.engine.declare(ProblemFact(label_format=self.label_format_var.get()))
        self.engine.declare(DatasetFact(input_data_type=self.input_data_type_var.get()))
        self.engine.declare(DatasetFact(dataset_size=self.dataset_size_var.get()))
        self.engine.declare(DatasetFact(feature_count=self.feature_count_var.get()))
        self.engine.declare(
            DatasetFact(feature_sample_ratio=self.feature_sample_ratio_var.get())
        )

        self.engine.declare(FeatureFact(has_numerical=self.has_numerical_var.get()))
        self.engine.declare(FeatureFact(numerical_scale=self.numerical_scale_var.get()))
        self.engine.declare(
            FeatureFact(numerical_distribution=self.numerical_distribution_var.get())
        )
        self.engine.declare(
            FeatureFact(numerical_outliers=self.numerical_outliers_var.get())
        )
        self.engine.declare(
            FeatureFact(nonlinear_relationships=self.nonlinear_relationships_var.get())
        )

        self.engine.declare(FeatureFact(has_categorical=self.has_categorical_var.get()))
        self.engine.declare(
            FeatureFact(categorical_cardinality=self.categorical_cardinality_var.get())
        )
        self.engine.declare(
            FeatureFact(categorical_order=self.categorical_order_var.get())
        )
        self.engine.declare(FeatureFact(rare_categories=self.rare_categories_var.get()))
        self.engine.declare(
            FeatureFact(noisy_categories=self.noisy_categories_var.get())
        )

        self.engine.declare(FeatureFact(has_datetime=self.has_datetime_var.get()))
        self.engine.declare(FeatureFact(seasonality=self.seasonality_var.get()))
        self.engine.declare(FeatureFact(time_order=self.time_order_var.get()))
        self.engine.declare(FeatureFact(cyclic_time=self.cyclic_time_var.get()))

        self.engine.declare(FeatureFact(has_text=self.has_text_var.get()))
        self.engine.declare(FeatureFact(text_length=self.text_length_var.get()))
        self.engine.declare(
            FeatureFact(text_dataset_size=self.text_dataset_size_var.get())
        )
        self.engine.declare(FeatureFact(word_order=self.word_order_var.get()))

        self.engine.declare(FeatureFact(has_image=self.has_image_var.get()))
        self.engine.declare(
            FeatureFact(image_dataset_size=self.image_dataset_size_var.get())
        )
        self.engine.declare(
            FeatureFact(image_augmentation=self.image_augmentation_var.get())
        )

        self.engine.declare(QualityFact(missing_values=self.missing_values_var.get()))
        self.engine.declare(QualityFact(class_imbalance=self.class_imbalance_var.get()))
        self.engine.declare(QualityFact(noisy_labels=self.noisy_labels_var.get()))
        self.engine.declare(QualityFact(data_leakage=self.data_leakage_var.get()))

        self.engine.declare(TrainingFact(overfitting=self.overfitting_var.get()))
        self.engine.declare(TrainingFact(underfitting=self.underfitting_var.get()))
        self.engine.declare(
            TrainingFact(unstable_training=self.unstable_training_var.get())
        )
        self.engine.declare(TrainingFact(slow_training=self.slow_training_var.get()))

        self.engine.run()

        # ── Populate Report tab ────────────────────────────────────────────────
        self.report_box.configure(state="normal")
        self.report_box.delete("1.0", "end")
        self.report_box.insert("end", self.report.get_text())
        self.report_box.configure(state="disabled")

        # ── Populate Explanation Trace tab ────────────────────────────────────
        self.expl_box.configure(state="normal")
        self.expl_box.delete("1.0", "end")
        self.expl_box.insert("end", self.report.get_explanation_text())
        self.expl_box.configure(state="disabled")

        # Switch to Report tab automatically
        self.output_tabs.set("Report")

    def _generate_code(self, framework: str):
        """Generate a starter script for the given framework and show it in a popup."""
        from tkinter import Toplevel, Text, Scrollbar, END, RIGHT, Y

        params = {
            "target_kind": self.target_kind_var.get(),
            "numerical_scale": self.numerical_scale_var.get(),
            "numerical_outliers": self.numerical_outliers_var.get(),
            "overfitting": self.overfitting_var.get(),
            "dataset_size": self.dataset_size_var.get(),
        }
        code = self._gen(framework, params)

        win = Toplevel(self)
        win.title(f"Generated {framework.capitalize()} Starter Code")
        win.geometry("820x640")
        win.configure(bg="#0F172A")

        txt = Text(
            win,
            wrap="none",
            font=("Courier New", 12),
            bg="#1E293B",
            fg="#E2E8F0",
            insertbackground="white",
            selectbackground="#334155",
        )
        sb_y = Scrollbar(win, orient="vertical", command=txt.yview)
        sb_x = Scrollbar(win, orient="horizontal", command=txt.xview)
        txt.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        sb_y.pack(side=RIGHT, fill=Y)
        sb_x.pack(side="bottom", fill="x")
        txt.pack(fill="both", expand=True)
        txt.insert(END, code)
        txt.configure(state="disabled")

    def reset(self):
        """Restore all variables to their default values and clear all badges. No loops."""
        self.target_kind_var.set("binary")
        self.label_format_var.set("integer_class_ids")
        self.input_data_type_var.set("tabular")

        self.dataset_size_var.set("medium")
        self.feature_count_var.set("medium")
        self.feature_sample_ratio_var.set("safe")

        self.has_numerical_var.set(1)
        self.numerical_scale_var.set("different")
        self.numerical_distribution_var.set("unknown")
        self.numerical_outliers_var.set("no")
        self.nonlinear_relationships_var.set("no")

        self.has_categorical_var.set(1)
        self.categorical_cardinality_var.set("low")
        self.categorical_order_var.set("nominal")
        self.rare_categories_var.set("no")
        self.noisy_categories_var.set("no")

        self.has_datetime_var.set(0)
        self.seasonality_var.set("no")
        self.time_order_var.set("no")
        self.cyclic_time_var.set("no")

        self.has_text_var.set(0)
        self.text_length_var.set("short")
        self.text_dataset_size_var.set("small")
        self.word_order_var.set("no")

        self.has_image_var.set(0)
        self.image_dataset_size_var.set("small")
        self.image_augmentation_var.set("yes")

        self.missing_values_var.set("no")
        self.class_imbalance_var.set("no")
        self.noisy_labels_var.set("no")
        self.data_leakage_var.set("no")

        self.overfitting_var.set("no")
        self.underfitting_var.set("no")
        self.unstable_training_var.set("no")
        self.slow_training_var.set("no")

        # Clear all ★ Auto badges
        self.badge_target_kind_var.set("")
        self.badge_dataset_size_var.set("")
        self.badge_feature_count_var.set("")
        self.badge_fsr_var.set("")
        self.badge_has_numerical_var.set("")
        self.badge_num_scale_var.set("")
        self.badge_num_outliers_var.set("")
        self.badge_missing_values_var.set("")

        self.report_box.configure(state="normal")
        self.report_box.delete("1.0", "end")
        self.report_box.insert("end", REPORT_HINT)
        self.report_box.configure(state="disabled")

        self.expl_box.configure(state="normal")
        self.expl_box.delete("1.0", "end")
        self.expl_box.insert("end", "Run Analyze to see which rules fired and why.")
        self.expl_box.configure(state="disabled")

    def copy_report(self):
        """Copy the current report text to the system clipboard."""
        self.clipboard_clear()
        self.clipboard_append(self.report.get_text())
