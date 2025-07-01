
from pydantic import BaseModel, Field
from typing import Optional, List
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
    feature_type: FeatureTypeEnum = Field(
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
    feature_type: FeatureTypeEnum = Field(
        title = 'Type of variable'
    )
    feature_description: Optional[str] = Field(
        title = 'Description of the feature'
    )

class DataModel(BaseModel):
    dataframe: dict = Field(
        title = 'Dataframe.xlx containing the input and output variables'
    )
    input_vars: List[InputFeature] = Field(
        title='List of input variables'
    )
    output_vars: List[OutputFeature] = Field(
        title='List of output variables'
    )


class ModelTraining(BaseModel):
    model_name: dict[str,str] = Field(
        title = 'Machine learning model to be trained'
    )
    model_params: Optional[dict[str,dict]] = Field(
        title = 'Parameters for the machine learning model',
        default = {}
    )  
    param_grid: Optional[dict[str,dict]] = Field(
        title = 'Parameter grid for hyperparameter tuning',
        default = {}
    )