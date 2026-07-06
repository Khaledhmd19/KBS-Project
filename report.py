class ReportBuilder:
    """
    Section-based report builder for the Neural Network Design Expert System.

    Each section is stored as a plain string field.
    Rules append bullet lines using string concatenation only.
    get_text() assembles the final report manually without any loop or iteration.
    """

    def __init__(self):
        self.problem = ""
        self.preprocessing = ""
        self.feature_engineering = ""
        self.architecture = ""
        self.output_loss = ""
        self.training = ""
        self.metrics = ""
        self.regularization = ""
        self.warning = ""
        self.checklist = ""
        self.explanations = ""

    def clear(self):
        self.problem = ""
        self.preprocessing = ""
        self.feature_engineering = ""
        self.architecture = ""
        self.output_loss = ""
        self.training = ""
        self.metrics = ""
        self.regularization = ""
        self.warning = ""
        self.checklist = ""
        self.explanations = ""

    def add_problem(self, message):
        self.problem = self.problem + "- " + message + "\n\n"

    def add_preprocessing(self, message):
        self.preprocessing = self.preprocessing + "- " + message + "\n\n"

    def add_feature_engineering(self, message):
        self.feature_engineering = self.feature_engineering + "- " + message + "\n\n"

    def add_architecture(self, message):
        self.architecture = self.architecture + "- " + message + "\n\n"

    def add_output_loss(self, message):
        self.output_loss = self.output_loss + "- " + message + "\n\n"

    def add_training(self, message):
        self.training = self.training + "- " + message + "\n\n"

    def add_metrics(self, message):
        self.metrics = self.metrics + "- " + message + "\n\n"

    def add_regularization(self, message):
        self.regularization = self.regularization + "- " + message + "\n\n"

    def add_warning(self, message):
        self.warning = self.warning + "- " + message + "\n\n"

    def add_checklist(self, message):
        self.checklist = self.checklist + "- " + message + "\n\n"

    def add_explanation(self, rule_name: str, reason: str):
        self.explanations = self.explanations + f"- [{rule_name}]: {reason}\n\n"

    def get_text(self):
        return (
            "NEURAL NETWORK DESIGN EXPERT REPORT\n"
            "=====================================\n\n"
            "[Problem Analysis]\n\n" + self.problem + "\n"
            "[Preprocessing]\n\n" + self.preprocessing + "\n"
            "[Feature Engineering]\n\n" + self.feature_engineering + "\n"
            "[Recommended Architecture]\n\n" + self.architecture + "\n"
            "[Output Layer and Loss]\n\n" + self.output_loss + "\n"
            "[Optimizer and Training]\n\n" + self.training + "\n"
            "[Evaluation Metrics]\n\n" + self.metrics + "\n"
            "[Regularization]\n\n" + self.regularization + "\n"
            "[Warnings]\n\n" + self.warning + "\n"
            "[Final Checklist]\n\n" + self.checklist
        )

    def get_explanation_text(self):
        """Return only the explanation trace, formatted as a standalone section."""
        return "RULE EXPLANATION TRACE\n======================\n\n" + (
            self.explanations or "No rules were triggered yet.\n\n"
        )
