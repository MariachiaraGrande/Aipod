from aidox.optimization.handler_data import DoELeapHandler
from aidox.models.pydantic_models_data import DataModel, DOEModel, InputFeature,ValueRanges, FeatureTypeEnum, DoeParams, DoeMethod, DoeMethodEnum,CategoricalApproachEnum,OutputFeature
import pandas as pd


def main():
    
    ## DOEmodel
    approach = CategoricalApproachEnum('oversampling')
    method = DoeMethodEnum('fullfact')
    method_doe = DoeMethod(name=method, kwargs={})
    doe_param = DoeParams(method=method_doe, categorical_approach=approach, mixture_design=None, mixture_constraints=[{'formula': 'out = 0.5','tol': .1}])
    doemodel = DOEModel(doe_params=doe_param)

    ## DATAmodel
    input0 = InputFeature(feature_name='A', id='id_0', default_value=100, optimize=True,
                          value_range=ValueRanges(lb=27, ub=122), value_list=[50, 75, 100],
                          value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
    input1 = InputFeature(feature_name='B', id='id_1', default_value=60, optimize=True,
                          value_range=ValueRanges(lb=1, ub=3), value_list=[60,120],
                          value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
    input2 = InputFeature(feature_name='C', id='id_2', default_value=240, optimize=True,
                          value_range=ValueRanges(lb=240, ub=8000), value_list=[240, 3880, 8000],
                          value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
    input3 = InputFeature(feature_name='D', id='id_3', default_value=30, optimize=True,
                          value_range=ValueRanges(lb=30, ub=110), value_list=[30,50,70,100],
                          value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
    input4 = InputFeature(feature_name='E', id='id_3', default_value=30, optimize=True,
                          value_range=ValueRanges(lb=30, ub=110), value_list=['sample_a', 'sample_b', 'sample_c'],
                          value_type=FeatureTypeEnum('Cat'), transform=None, transform_kwargs=None)

    output0 = OutputFeature(feature_name='Output1', id='id_01')
    output1 = OutputFeature(feature_name='Output2', id='id_02')

    datamodel = DataModel(input_vars=[input0, input1, input2, input3,input4], output_vars=[output0, output1])

    doelab = DoELeapHandler(model=doemodel, datamodel=datamodel)
    points = doelab.ask(n_points=3)
    df = pd.DataFrame.from_dict(points, orient='index')
    df.to_csv('docs/data/doe_df.csv', index=False)
    print(df)

if __name__=='__main__':
    main()

   