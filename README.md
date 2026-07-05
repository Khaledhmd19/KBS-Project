# Neural Network Design Expert System

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Dependency Manager](https://img.shields.io/badge/dependency--manager-uv-blueviolet.svg)](https://github.com/astral-sh/uv)
[![GUI Library](https://img.shields.io/badge/GUI-CustomTkinter-orange.svg)](https://github.com/tomaszkiewicz/customtkinter)

An interactive, rule-based expert system that recommends optimal neural network configurations (layers, activations, loss functions, optimizers, regularization, and preprocessing steps) based on dataset traits, problem descriptions, and observed training symptoms.

---

## 🌟 Key Features

- **Automated Dataset Profiler**: Upload any CSV dataset to instantly extract metadata (dataset size, feature count, scales, outlier presence, missing values, and target classification) to auto-fill the expert system questionnaire.
- **Experta Rule Engine**: A declarative, forward-chaining reasoning engine implemented using `experta` to fire recommendations based on factual inputs.
- **Code Generator**: On-the-fly starter code generation for both **PyTorch** and **Keras / TensorFlow** matching the expert suggestions.
- **Beautiful Dark-Mode UI**: A modern desktop application built using `customtkinter` with real-time reporting, rule explanation tracing, and one-click code copy features.

---

## 🚫 The "No-Control-Flow" Architectural Constraint

A unique academic/design constraint was enforced throughout the entire codebase:
**No `if`, `elif`, `else`, `for`, `while`, `match`, or `case` statements are used.**

Decision-making and logic flows are achieved purely using:
1. **Declarative Production Rules**: Handled via the `experta` knowledge engine.
2. **Lookup Tables (Dictionary Dispatch)**: Selecting configurations and frameworks dynamically.
3. **Boolean Arithmetic and String Multiplication**: Generating optional code blocks (e.g., multiplying blocks by boolean flags).
4. **Vectorized Pandas Operations**: Performing automated profiling without standard loops.

---

## 📂 Project Structure

```text
├── assets/                    # Media assets and resources
├── codegen.py                 # PyTorch/Keras script generator
├── constants.py               # Application-wide styling variables
├── engine.py                  # Experta KnowledgeEngine containing all rules
├── facts.py                   # Experta Fact models definitions
├── main.py                    # Application launcher and entry point
├── profiler.py                # No-loop automated CSV analyzer
├── pyproject.toml             # Project config and package requirements
├── report.py                  # Structured recommendations builder
├── ui.py                      # CustomTkinter GUI layout and widget definitions
├── uv.lock                    # Locked dependencies file
└── README.md                  # Project overview (this file)
```

Detailed write-ups for each module can be found in the [docs/](file:///home/khaledhammoud/Desktop/projects/uni/KBS-Project/docs/) folder.

---

## 🚀 Quick Start

### Prerequisites
Make sure you have [uv](https://github.com/astral-sh/uv) installed (recommended Python 3.10+ package manager) or standard `pip`.

### 1. Installation
To install the project dependencies, run:
```bash
uv sync
```
Or with standard pip:
```bash
pip install -r pyproject.toml
```

### 2. Running the Expert System
Launch the CustomTkinter GUI:
```bash
uv run python main.py
```
*(Alternatively, run `python main.py` directly inside your virtual environment).*

---

## 📖 Deep Dive Documentation

For a detailed understanding of how the expert system works and is structured, please read the documentation files:

1. 🏛️ **[System Architecture](file:///home/khaledhammoud/Desktop/projects/uni/KBS-Project/docs/architecture.md)**: Design patterns, component interactions, and implementation of the "No-Control-Flow" constraint.
2. 🧠 **[Knowledge-Based Rules](file:///home/khaledhammoud/Desktop/projects/uni/KBS-Project/docs/rules.md)**: Detail of the Experta Facts and the 14 groups of production rules.
3. 📊 **[Automated CSV Profiler](file:///home/khaledhammoud/Desktop/projects/uni/KBS-Project/docs/profiler.md)**: How pandas profiles data without traditional conditional code.
4. 💻 **[Starter Code Generator](file:///home/khaledhammoud/Desktop/projects/uni/KBS-Project/docs/codegen.md)**: Lookups, template formats, and dynamic string-multiplication based template rendering.
5. 🛠️ **[User Guide & Examples](file:///home/khaledhammoud/Desktop/projects/uni/KBS-Project/docs/usage.md)**: Step-by-step usage guide and walkthrough utilizing sample datasets.
