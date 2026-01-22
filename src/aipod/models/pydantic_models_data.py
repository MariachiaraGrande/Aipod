
from pydantic import BaseModel, Field, root_validator
from typing import Optional, List, Union
from enum import Enum
from scipy import stats

### DATA MODEL
class FeatureTypeEnum(str, Enum):
    Float = 'Float'
    OrdCat = 'OrdCat'
    Cat = 'Cat'

class DoeMethodEnum(str, Enum):
    lhs = 'lhs'
    bbdesign = 'bbdesign'
    pbdesign = 'pbdesign'
    ccdesign = 'ccdesign'
    fullfact = 'fullfact'
    ff2n = 'ff2n'
    fracfact ='fracfact'
    gsd = 'gsd'

class CategoricalApproachEnum(str, Enum):
    random = 'random'
    oversampling = 'oversampling'
    undersampling = 'undersampling'

class TrasformationTypeEnumMeta(str, Enum):
    pass


# set OilTypesEnum values from local NOAA oil data (Tom wants this, Richi doesn't)
def get_cont_disrt():
    distr = {getattr(stats, d).name: getattr(stats, d).name
             for d in dir(stats) if isinstance(getattr(stats, d), stats.rv_continuous)}
    return distr


TrasformationTypeEnum = Enum(
    'TransformationTypeFilter',
    get_cont_disrt(),
    type=TrasformationTypeEnumMeta
)

class BaseFeature(BaseModel):
    feature_name: str = Field(
        title='Feature name'
    )
    id: str = Field(
        title='Feature id'
    )

class OutputFeature(BaseFeature):
    pass
class ValueRanges(BaseModel):
    lb: float = Field(
        title='Lower bound of the range'
    )
    ub: float = Field(
        title='Upper bound of the range'
    )
    # Check that lower bound is lower than upper bound
    @root_validator(pre=True)
    def check_value_range(cls, values):
        if values.get('lb') is not None and values.get('ub') is not None:
            if (
                    (values['lb'] > values['ub'])
            ):
                raise ValueError('Lower bound should be lower than upper bound')
            return values
        else:
            return values


class InputFeature(BaseFeature):
    default_value: Optional[Union[float, str, int]] = Field(
        title='Default value of the feature'
    )
    optimize: bool = Field(
        title='Whether to optimize the variables, otherwise default value is used'
    )
    value_range: Optional[ValueRanges] = Field(
        title='Range of values to be explored',
        default=None
    )
    value_list: Optional[List[Union[str, float]]] = Field(
        title='List of values to be explored',
        default=None
    )
    value_type: FeatureTypeEnum = Field(
        title='Type of variable'
    )
    transform: Optional[TrasformationTypeEnum] = Field(
        title='Type of transformation',
        default=None
    )
    transform_kwargs: Optional[dict] = Field(
        title='Transformation parameters',
        default={}
    )
    # Check
    # 1. If optimize == False then default_value is not None
    # 2. If optimize == True then value_range is not empty
    # 3. If value_type == Float then value_range is not None
    # 4. If value_type == Cat/OrdCat then value_list is not None
    @root_validator(pre=True)
    def check_value_space(cls, values):
        if not values.get('optimize'):
            if values['default_value'] is None:
                raise ValueError('Default value should be provided if optimize is False')
            return values
        else:
            if values.get('value_range') is None and values.get('value_list') is None:
                raise ValueError('Value range should be provided if optimize is True')
            return values
    @root_validator(pre=True)
    def check_value_type(cls, values):
        if values.get('value_type') == 'Float':
            if values['value_range'] is None:
                raise ValueError('Value range should be provided if value type is Float')
            return values
        elif values.get('value_type') == 'Cat' or values.get('value_type') == 'OrdCat':
            if values['value_list'] is None or len(values['value_list']) < 2:
                raise ValueError('Value list should be provided if value type is Cat or OrdCat and more than one value should be provided')
            print()
            return values
        else:
            return values

class DataModel(BaseModel):
    input_vars: List[InputFeature] = Field(
        title='List of input variables'
    )
    output_vars: List[OutputFeature] = Field(
        title='List of output variables'
    )


### OPTIMIZATION MODEL


### OPTIMIZER MODEL
# class OptimizerTypeMeta(str, Enum):
#     pass

# def get_list_of_optimizers():
#     opt_dict = {x:x for x in list(ng.optimizers.registry.keys())}
#     return opt_dict


# OptimizerTypeEnum = Enum(
#     'OptimizerTypeFilter',
#     get_list_of_optimizers(),
#     type=OptimizerTypeMeta
# )

#DESIGN OF EXPERIMENT

## DOE PARAMETERS

class DoeMethod(BaseModel):
    name: DoeMethodEnum = Field(
        title='Name of the method'
    )
    kwargs: dict = Field(
        title='Method parameters'
    )

class DoeParams(BaseModel):
    method: DoeMethod = Field(
        title="Design method"
    )
    categorical_approach: Optional[CategoricalApproachEnum] = Field(
        title="Categorical approach"
    )
    mixture_design: Optional[dict] = Field(
        title="Method chosen by user"
    )

    mixture_constraints: Optional[List[dict]] = Field(
        title="Mixture constraints"
    )


class DOEModel(BaseModel):
    doe_params: DoeParams = Field(
        title='Doe parameters'
    ) 

