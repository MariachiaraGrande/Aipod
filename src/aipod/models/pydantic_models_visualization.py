
from pydantic import BaseModel, Field
from typing import Optional, List, Union
from enum import Enum


### DATA MODEL

class FeatureTypeEnum(str, Enum):
    Float = 'Float'
    OrdCat = 'OrdCat'
    Cat = 'Cat'

class InputFeature(BaseModel):
    feature_name: str = Field(
        title='Feature name'
    )
    id: str = Field(
        title='Feature id'
    )
    value_list: Optional[List[Union[str, float]]] = Field(
        title='List of values to be explored',
        default=None
    )
    value_type: FeatureTypeEnum = Field(
        title = 'Type of variable'
    )
    feature_description: Optional[str]= Field(
        title = 'Description of the feature'
    )

class OutputFeature(BaseModel):
    feature_name: str = Field(
        title='Feature name'
    )
    id: str = Field(
        title='Feature id'
    )
    value_type: FeatureTypeEnum = Field(
        title = 'Type of variable'
    )
    feature_description: Optional[str] = Field(
        title = 'Description of the feature'
    )

class DataModel(BaseModel):
    dataframe: str = Field(
        title = 'Dataframe.xlx containing the input and output variables'
    )
    input_vars: List[InputFeature] = Field(
        title='List of input variables'
    )
    output_vars: List[OutputFeature] = Field(
        title='List of output variables'
    )