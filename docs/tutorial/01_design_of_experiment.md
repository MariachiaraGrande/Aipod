# Module 1 – Design of Experiment

This module enables the generation of structured **worktables for experimental campaigns** based on:
- the definition of input and output variables,
- the selected Design of Experiment (DoE) strategy,
- optional constraints and categorical handling rules.

The configuration is entirely **configuration-driven** and validated through **Pydantic models**.

---
## Table of Contents
- [Step 1 – Definition of input and output variables](#step-1-–-definition-of-input-and-output-variables)
- [Step 2 – Definition of DoE strategy and constraints ](#step-2--definition-of-doe-strategy-and-constraints)
- [Step 3 – Generation of the experimental worktable](#step-3--generation-of-the-experimental-worktable)

## Step 1 – Definition of input and output variables

The first step consists in defining the structure and type of the input and output variables to which the Design of Experiment is applied.

This information is specified in the configuration file:

```text
configs/opt_datamodel_pydantic.yml
```
and parsered through Pydantic models.

## Relevant classes
```python
from aipod.models.pydantic_models_data import (
    DataModel,
    DOEModel,
    InputFeature,
    OutputFeature,
    ValueRanges,
    FeatureTypeEnum,
    DoeParams,
    DoeMethod,
    DoeMethodEnum,
    CategoricalApproachEnum,
)
```
## Input variable (`InputFeature`)
Each input variable is defined through an `InputFeature` object
### Main parameters

* `feature_name:str` factor name 
* `id: str` factor identification number 
* `default_value: Union[float, int, str]` value assigned by default in case of optimize=False.
* `optimize: bool` True or False in order to select the optimized variables and the one for whichefault value is used
* `value_range: ValueRanges`. It's low (`lb`) and up (`ub`) boundaries inside which the solution can beound. \

    Example:
    ```
    value_range = ValueRanges(lb:Union[int,float]=3, ub:Union[int,float]=7)
    ``` 
* `value_list: list[str]` in case of catergorical variable, it's a list of all the
possible variable that should be selected (eg.['red','green','blue'])
* `value_type: FeatureTypeEnum` the variable type:
    * `Float` (continuos variable)
    * `Cat` (categorical variable) 
    * `OrdCat` (discrete variable, ordered categorical) \

    Example:
    ```
    value_type = FeatureTypeEnum(str ='Float')
    ``` 
* `transform`: transformation to be applied to the variable
* `transform_kwargs`: parameters associated wiith the transformation

`output_vars: OutputFeature`
* `feature_name:str` factor name 
* `id: str` factor identification number 
---
## Output variable (`OutputFeature`)
Output variables are defined through the `OutputFeature` class and include:
  * `feature_name:str` factor name 
  * `id: str` factor identification number 
---

## Application Example
```python
input0 = InputFeature(
    feature_name="A",
    id="id_01",
    default_value=100,
    optimize=True,
    value_range=ValueRanges(lb=50, ub=100),
    value_list=[25, 75, 90],
    value_type=FeatureTypeEnum("OrdCat"),
)

input1 = InputFeature(
    feature_name="B",
    id="id_02",
    default_value=60,
    optimize=True,
    value_range=ValueRanges(lb=60, ub=1008.10),
    value_list=[45, 60, 197.90],
    value_type=FeatureTypeEnum("Float"),
)

input2 = InputFeature(
    feature_name="C",
    id="id_03",
    default_value=240,
    optimize=True,
    value_range=ValueRanges(lb=240, ub=8000),
    value_list=[240, 3880],
    value_type=FeatureTypeEnum("OrdCat"),
)

input3 = InputFeature(
    feature_name="D",
    id="id_04",
    default_value=30,
    optimize=True,
    value_range=ValueRanges(lb=30, ub=110),
    value_list=[30, 50, 70],
    value_type=FeatureTypeEnum("Float"),
)

input4 = InputFeature(
    feature_name="E",
    id="id_05",
    default_value="dog",
    optimize=True,
    value_list=["sample_a", "sample_b", "sample_c"],
    value_type=FeatureTypeEnum("Cat"),
)

output0 = OutputFeature(feature_name="output1", id="id_06")
output1 = OutputFeature(feature_name="output2", id="id_07")

datamodel = DataModel(
    input_vars=[input0, input1, input2, input3, input4],
    output_vars=[output0, output1],
)

```
## Step 2 – Definition of DoE strategy and constraints
The Design of Experiment strategy and constraints are defined in:
```python
config/opt_doe_pydantic.yml
```
## DoE parameters
 * `method: DoeMethodEnum`
    * `name: str`  select DoE strategy. Available options include: `fullfact`, `ff2n`, `fracfact`, `gsd`, `lhs`, `pbdesign`, `bbdesign`, `ccdesign`. 
    * `kwargs: dict` method-specific parameters. 

    Example:
    ```
    method = DoeMethod(name='gsd', kwargs={'reduction':'2'})
    ```

* `categorical_approach: CategoricalApproachEnum`  Strategy for handling categorical variables:
    * `undersampling`, 
    * `random`
    * `oversampling`. \
  
  Example:
    ```
    categorical_approach = CategoricalApproachEnum(str='random')
    ```

* `mixture_design` Optional mixture design flag.
* `mixture_constraints: list[dict]`  in which you can specify the constraints to be placed on the design of     experiment. 
  
  Example:
    ```
    [{ 'formula': 'out = 0.7' , 'tol': .8 }]
    ```

### Application example:
```python
doe_param = DoeParams(
    method=DoeMethod(name=DoeMethodEnum("fullfact"), kwargs={}),
    categorical_approach=CategoricalApproachEnum("random"),
    mixture_design=None,
    mixture_constraints=[{"formula": "out = 0.5", "tol": 0.1}],
)

doemodel = DOEModel(doe_params=doe_param)

```

## Step 3 – Generation of the experimental worktable

To execute the Deisgn of Experiment, the `DoELeapHandler` must be initialized.

```python
from aipod.optimization.handler_data import DoELeapHandler
```
The handler requires the previously defined `DOEModel` and `DataModel`.
```python
doelab = DoELeapHandler(model=doemodel, datamodel=datamodel)
```

The experimental worktable (returned as a pandas `DataFrame`) can then be generated by specifying the desired number of points:

```python
points = doelab.ask(n_points=3)
```
---
At this stage, the Design of Experiment module outputs a structured set of experimental points ready to be executed or used as input for subsequent modeling steps.