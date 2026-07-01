class ReportBuilder:
    """
    Section-based report builder for the Neural Network Design Expert System.

    Each section is stored as a plain string field.
    Rules append bullet lines using string concatenation only.
    get_text() assembles the final report manually without any loop or iteration.
    """

    def __init__(self):
        self.problem            = ""
        self.preprocessing      = ""
        self.feature_engineering = ""
        self.architecture       = ""
        self.output_loss        = ""
        self.training           = ""
        self.metrics            = ""
        self.regularization     = ""
        self.warning            = ""
        self.checklist          = ""

    def clear(self):
        self.problem            = ""
        self.preprocessing      = ""
        self.feature_engineering = ""
        self.architecture       = ""
        self.output_loss        = ""
        self.training           = ""
        self.metrics            = ""
        self.regularization     = ""
        self.warning            = ""
        self.checklist          = ""

    def add_problem(self, message):
        self.problem = self.problem + "- " + message + "\n"

    def add_preprocessing(self, message):
        self.preprocessing = self.preprocessing + "- " + message + "\n"

    def add_feature_engineering(self, message):
        self.feature_engineering = self.feature_engineering + "- " + message + "\n"

    def add_architecture(self, message):
        self.architecture = self.architecture + "- " + message + "\n"

    def add_output_loss(self, message):
        self.output_loss = self.output_loss + "- " + message + "\n"

    def add_training(self, message):
        self.training = self.training + "- " + message + "\n"

    def add_metrics(self, message):
        self.metrics = self.metrics + "- " + message + "\n"

    def add_regularization(self, message):
        self.regularization = self.regularization + "- " + message + "\n"

    def add_warning(self, message):
        self.warning = self.warning + "- " + message + "\n"

    def add_checklist(self, message):
        self.checklist = self.checklist + "- " + message + "\n"

    def get_text(self):
        return (
            "NEURAL NETWORK DESIGN EXPERT REPORT\n"
            "=====================================\n\n"
            "[Problem Analysis]\n" + self.problem + "\n"
            "[Preprocessing]\n" + self.preprocessing + "\n"
            "[Feature Engineering]\n" + self.feature_engineering + "\n"
            "[Recommended Architecture]\n" + self.architecture + "\n"
            "[Output Layer and Loss]\n" + self.output_loss + "\n"
            "[Optimizer and Training]\n" + self.training + "\n"
            "[Evaluation Metrics]\n" + self.metrics + "\n"
            "[Regularization]\n" + self.regularization + "\n"
            "[Warnings]\n" + self.warning + "\n"
            "[Final Checklist]\n" + self.checklist
        )
