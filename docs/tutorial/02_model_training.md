# Module 2 – Model fitting and performance evaluation

This module is responsible for **data preprocessing, surrogate model training, and performance evaluation** based on the machine learning algorithm specified in the configuration files.

The workflow is entirely configuration-driven and supports both **regression and classification** tasks.

---
## Table of Contents
- [Step 1 – Definition of data and model configuration](#step-1--definition-of-data-and-model-configuration)
- [Step 2 - Model training and prediction](#step-2---model-training-and-prediction)

---

## Step 1 – Definition of data and model configuration

The input/output structure of the process and the machine learning configuration are defined in:

```text
configs/opt_models_pydantic.yml
```

The configuration is validated and parsered through Pydantic models.

## Relevant classes
```python
from aipod.models.pydantic_models_training import (
    FeatureTypeEnum,
    InputFeature,
    OutputFeature,
    DataModel,
    ModelTraining,
)
```
---
## Dataset specification
* `dataframe: dict` the key must be `classifier` or `regressor` according to the task and the value must be the dataframe path
    Example:
    ```python
    {"classifier": "../docs/data_processed/df_classifier.xlsx"}
    ```
---
## Input variables (`InputFeature`)
Each input feature is defined through an `InputFeature` object

### Main parameters

* `feature_name:str` factor name 
* `id: str` factor identification number 
* `feature_type: FeatureTypeEnum` the variable type:
    * `Float` (continuos variable)
    * `Cat`(discrete variable, categorical)
    * `OrdCat` (discrete variable, ordered categorical)
* `feature_description: str` Optional description of the feature
___
## Ouptpu variables (`OutputFeature`)
Output variables are defined similary and include:

* `feature_name:str` factor name 
* `id: str` factor identification number 
* `feature_type: FeatureTypeEnum` 
* `feature_description: str` 
___
## Model configuration

* `model_name: dict[str,str]` the key must be `classifier' or 'regressor` according to the task and the value must be the name of ML model available on sklearn modules. Supported modules include:
    * `sklearn.ensemble`
    * `sklearn.gaussian_process`
    * `sklearn.linear_model`
    * `sklearn.neural_network`
    
* `model_params: dict[str,dict]` Model-specific initialization parameters
* `model_params: dict[str,dict]` Hyperparameter grid for `GridSearchCV`
---

### Application example:
```python
    input0 = InputFeature(
    feature_name="A",
    id="id_0",
    feature_type=FeatureTypeEnum("OrdCat"),
    feature_description=None,
)

input1 = InputFeature(
    feature_name="B",
    id="id_1",
    feature_type=FeatureTypeEnum("OrdCat"),
    feature_description=None,
)

input2 = InputFeature(
    feature_name="C",
    id="id_2",
    feature_type=FeatureTypeEnum("OrdCat"),
    feature_description=None,
)

input3 = InputFeature(
    feature_name="D",
    id="id_3",
    feature_type=FeatureTypeEnum("OrdCat"),
    feature_description=None,
)

output0 = OutputFeature(
    feature_name="out1",
    id="out_0",
    feature_type=FeatureTypeEnum("Float"),
    feature_description="out1_description",
)

output1 = OutputFeature(
    feature_name="out2",
    id="out_1",
    feature_type=FeatureTypeEnum("Float"),
    feature_description="out2_description",
)

df = {"classifier": "../docs/data_processed/df_classifier.xlsx"}

model_name = {"classifier": "RandomForestClassifier"}
model_params = {"classifier": {}}
param_grid = {
    "classifier": {
        "n_estimators": [50, 80, 100],
        "criterion": ["entropy"],
        "max_depth": [8, 16, 32],
    }
}

datamodel = DataModel(
    dataframe=df,
    input_vars=[input0, input1, input2, input3],
    output_vars=[output0, output1],
)

model = ModelTraining(
    model_name=model_name,
    model_params=model_params,
    param_grid=param_grid,
)

```

## Step 2 - Model training and prediction
To execute the model fitting stage, the appropriate handler must be initialized depending on the task type.

## Relevant classes 
``` python
from aipod.optimization.handler_model import Classifier, Regressor
```

The handler requires the previously defined `DataModel` and `ModelTraining` objects.

---
## Application example (regression case)
```python
reg = Regressor(datamodel=datamodel, model_training=model)

reg.train_test_split(method="standard")
reg.model_prediction()
```
At this stage, the module:
* preprocess the input data
* trian the selected model
* evaluates its performance
* generate predictions to be used in subsequent visualization and analysis steps.