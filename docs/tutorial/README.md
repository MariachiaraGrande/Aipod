# AIPOD – Tutorials and usage guide

This directory contains **step-by-step tutorials** that illustrate how to use AIPOD modules in practice.

The goal of these documents is to provide a **hands-on, operational understanding** of the framework, complementing the high-level overview presented in the main README.

Each tutorial focuses on a specific module or usage scenario and is written to be readable independently, while still following a logical progression.

---
## Table of Contents

- [How to read these tutorials](#how-to-read-these-tutorials)
- [Tutorials overview](#tutorials-overview)
  -  [01 – Design of Experiment](#01-–-design-of-experiment)
  -  [02 – Model training and performance evaluation](#02-–-model-training-and-performance-evaluation)
  -  [03 – Data visualization and interpretability](#03-–-data-visualization-and-interpretability)
  -  [04 – Full pipeline example](#04-–-full-pipeline-example)
- [Prerequisites](#prerequisites)


## How to read these tutorials

The recommended reading order is:

1. **Design of Experiment**  
   Introduction to the definition of input/output variables, constraints, and experimental strategies.

2. **Model training and performance evaluation**  
   Training surrogate models from experimental data and evaluating their performance.

3. **Data visualization and interpretability**  
   Visual exploration and interpretation of surrogate models.

4. **Full pipeline example**  
   End-to-end example showing how all modules interact in a complete workflow.

Readers already familiar with specific topics may jump directly to the relevant section.

---

## Tutorials overview

### 01 – Design of Experiment
**File:** `01_design_of_experiment.md`

Covers:
- definition of input and output variables
- configuration-driven Design of Experiment
- supported DoE strategies
- generation of structured experimental plans

---

### 02 – Model training and performance evaluation
**File:** `02_model_training.md`

Covers:
- data preprocessing
- surrogate model selection
- training and validation
- hyperparameter tuning

---

### 03 – Data visualization and interpretability
**File:** `03_visualization.md`

Covers:
- exploratory plots
- SHAP-based analysis
- biplot-like visualizations
- interpretation of surrogate models

---

### 04 – Full pipeline example
**File:** `04_full_pipeline_example.md`

Covers:
- configuration setup
- execution of all modules
- end-to-end workflow
- discussion of results

---

## Prerequisites

Before running the examples described in these tutorials, ensure that:

- the AIPOD package is installed in editable mode:
  ```bash
  pip install -e .
  ```
- configuration files are available in the `configs`/directory
- example datasets are available in the appropriate `data`/subfolders 
