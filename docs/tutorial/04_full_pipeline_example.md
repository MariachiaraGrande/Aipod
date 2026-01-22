# Full pipeline example – End-to-end workflow

This tutorial illustrates a **complete AIPOD workflow**, showing how the individual modules interact in a realistic usage scenario.

The goal is not to introduce new concepts, but to demonstrate **how Design of Experiment, model training, and visualization are combined** to support data-driven optimization.

---

## Problem statement

We consider an optimization problem characterized by:
- multiple controllable input parameters,
- one or more output quantities of interest,
- constraints on the admissible operating space.

The objective is to explore the design space efficiently and gain insight into the relationship between inputs and outputs, while minimizing experimental effort.

---

## Step 1 – Definition of the design space

The first step consists in defining:
- input variables and their admissible ranges,
- output variables of interest,
- variable types (continuous, categorical, ordered categorical).

This information is specified in the configuration file:

```text
configs/opt_datamodel_pydantic.yml
```

Using this configuration, a `DataModel` object is created to represent the structure of the process under study.

## Step 2 - Design of Experiment generation
Based on the defined design space, a Design of Experiment strategy is selected and configured in:
```text
configs/opt_doe_pydantic.yml
```
Typical choices include:
* full factorial or fractional designs for low-dimensional spaces,
* GSD or Latin hypercube sampling for higher-dimensional problems.

The DoE module generates a structured experimental worktable, which represents the set of experiments or simulations to be executed.

---
## Step 3 - Data acquisition and preparation

The experimental plan produced in the previous step is executed externally (e.g. laboratory experiments, numerical simulations).
The resulting dataset, containing both input variables and measured outputs, is stored and prepared for modeling.
The processed dataset is then referenced in:
```text
configs/opt_models_pydantic.yml
```
---
## Step 4 - Model training

Using the prepared dataset, surrogate models are trained to approximate the underlying objective function.

This step includes:
* preprocessing of input data,
* train/test splitting,
* model selection (e.g. random forests, Gaussian processes),
* optional hyperparameter tuning via grid search.

The trained model is used to generate predictions across the design space, producing an enriched dataset for further analysis.

---
## Step 5 - Visualization and interpretability

The predicted dataset is analyzed using the visualization module, configured through:
```text
configs/opt_visualization_pydantic.yml
```
This stage provides:
* scatter and conditional plots for exploratory analysis,
* SHAP-based feature attribution to assess variable influence,
* biplot-like visualizations to interpret multivariate relationships.

The emphasis is placed on understanding model behavior and trade-offs, rather than solely optimizing numerical performance.

---
## Outcome and insights
At the end of the pipeline, the user obtains:
* a structured and reproducible experimental workflow,
* models approximating the system response,
* interpretable visual tools to guide optimization decisions.
This end-to-end approach enables informed exploration of the design space while reducing experimental cost and complexity.

---
## Notes

* Each step of the pipeline is modular and can be adapted or replaced independently.
* Configuration files act as first-class components, enabling reproducibility and traceability.
* The same workflow can be applied to different processes with minimal changes to the configuration.