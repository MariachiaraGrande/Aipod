import unittest
from aidox.optimization.handler_data import DoELeapHandler
from aidox.models.pydantic_models_data import DataModel, DOEModel, InputFeature,ValueRanges, FeatureTypeEnum, DoeParams, DoeMethod, DoeMethodEnum,CategoricalApproachEnum,OutputFeature
import pandas as pd
from pandas.testing import assert_frame_equal


class TestDoELeapHandler(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        # Optional: Clean up resources after each test
        pass

    def test_doe_lhs_over(self):
        ## DOEmodel
        approach = CategoricalApproachEnum('oversampling')
        method = DoeMethodEnum('lhs')
        method_doe = DoeMethod(name=method, kwargs={})
        doe_param = DoeParams(method=method_doe, categorical_approach=approach, mixture_design=None,
                              mixture_constraints=[{
                                  'formula': 'out = 0.5',
                                  'tol': .1
                              }])
        doemodel = DOEModel(doe_params=doe_param)

        ## DATAmodel
        input0 = InputFeature(feature_name='Power', id='id_01', default_value=100, optimize=True,
                              value_range=ValueRanges(lb=50, ub=100), value_list=[25, 75, 90],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input1 = InputFeature(feature_name='Repetiotion Rate', id='id_02', default_value=60, optimize=True,
                              value_range=ValueRanges(lb=60, ub=1008.10), value_list=[45, 60, 197.90],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input2 = InputFeature(feature_name='tau', id='id_03', default_value=240, optimize=True,
                              value_range=ValueRanges(lb=240, ub=8000), value_list=[240, 3880],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input3 = InputFeature(feature_name='n_pulses', id='id_04', default_value=30, optimize=True,
                              value_range=ValueRanges(lb=30, ub=110), value_list=[30, 50,70],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input4 = InputFeature(feature_name='sample', id='id_05', default_value='dog', optimize=True,
                              value_list=['alluminio', 'acciaio', 'vetro'], value_type=FeatureTypeEnum('Cat'),
                              transform=None, transform_kwargs=None)
        output0 = OutputFeature(feature_name='d_mean', id='id_06')
        output1 = OutputFeature(feature_name='h_mean', id='id_07')

        datamodel = DataModel(input_vars=[input0, input1, input2, input3, input4], output_vars=[output0, output1])

        doelab = DoELeapHandler(model=doemodel, datamodel=datamodel)
        points = doelab.ask(n_points=3)
        df = pd.DataFrame.from_dict(points, orient='index')

        # Expected output
        df_exp = pd.read_csv('tests/assets/ok_test_doe_over_lhs.csv')

        assert_frame_equal(df, df_exp)

    def test_fail_doe_lhs_over(self):
        ## DOEmodel
        approach = CategoricalApproachEnum('oversampling')
        method = DoeMethodEnum('lhs')
        method_doe = DoeMethod(name=method, kwargs={})
        doe_param = DoeParams(method=method_doe, categorical_approach=approach, mixture_design=None,
                              mixture_constraints=[{
                                  'formula': 'out = 0.5',
                                  'tol': .1
                              }])
        doemodel = DOEModel(doe_params=doe_param)

        ## DATAmodel
        input0 = InputFeature(feature_name='Power', id='id_01', default_value=100, optimize=True,
                              value_range=ValueRanges(lb=50, ub=100), value_list=[25, 75, 90],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input1 = InputFeature(feature_name='Repetiotion Rate', id='id_02', default_value=60, optimize=True,
                              value_range=ValueRanges(lb=60, ub=1008.10), value_list=[45, 60, 197.90],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input2 = InputFeature(feature_name='tau', id='id_03', default_value=240, optimize=True,
                              value_range=ValueRanges(lb=240, ub=8000), value_list=[240, 3880],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input3 = InputFeature(feature_name='n_pulses', id='id_04', default_value=30, optimize=True,
                              value_range=ValueRanges(lb=30, ub=110), value_list=[30, 50,70],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input4 = InputFeature(feature_name='sample', id='id_05', default_value='dog', optimize=True,
                              value_list=['alluminio', 'acciaio', 'vetro'], value_type=FeatureTypeEnum('Cat'),
                              transform=None, transform_kwargs=None)
        output0 = OutputFeature(feature_name='d_mean', id='id_06')
        output1 = OutputFeature(feature_name='h_mean', id='id_07')

        datamodel = DataModel(input_vars=[input0, input1, input2, input3, input4], output_vars=[output0, output1])

        doelab = DoELeapHandler(model=doemodel, datamodel=datamodel)
        points = doelab.ask(n_points=3)
        df = pd.DataFrame.from_dict(points, orient='index')

        # Expected output
        df_exp = pd.read_csv('tests/assets/nok_test_doe_over_lhs.csv')

        # Assert not equal
        with self.assertRaises(AssertionError):
            assert_frame_equal(df, df_exp)

    def test_doe_res_lhs_under(self):
        ## DOEmodel
        approach = CategoricalApproachEnum('undersampling')
        method = DoeMethodEnum('lhs')
        method_doe = DoeMethod(name=method, kwargs={})
        doe_param = DoeParams(method=method_doe, categorical_approach=approach, mixture_design=None,
                              mixture_constraints=[{
                                  'formula': 'out = 0.5',
                                  'tol': .1
                              }])
        doemodel = DOEModel(doe_params=doe_param)

        ## DATAmodel
        input0 = InputFeature(feature_name='Power', id='id_01', default_value=100, optimize=True,
                              value_range=ValueRanges(lb=50, ub=100), value_list=[25, 75, 90],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input1 = InputFeature(feature_name='Repetiotion Rate', id='id_02', default_value=60, optimize=True,
                              value_range=ValueRanges(lb=60, ub=1008.10), value_list=[45, 60, 197.90],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input2 = InputFeature(feature_name='tau', id='id_03', default_value=240, optimize=True,
                              value_range=ValueRanges(lb=240, ub=8000), value_list=[240, 3880],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input3 = InputFeature(feature_name='n_pulses', id='id_04', default_value=30, optimize=True,
                              value_range=ValueRanges(lb=30, ub=110), value_list=[30, 50,70],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input4 = InputFeature(feature_name='sample', id='id_05', default_value='dog', optimize=True,
                              value_list=['alluminio', 'acciaio', 'vetro'], value_type=FeatureTypeEnum('Cat'),
                              transform=None, transform_kwargs=None)
        output0 = OutputFeature(feature_name='d_mean', id='id_06')
        output1 = OutputFeature(feature_name='h_mean', id='id_07')

        datamodel = DataModel(input_vars=[input0, input1, input2, input3, input4], output_vars=[output0, output1])

        doelab = DoELeapHandler(model=doemodel, datamodel=datamodel)
        points = doelab.ask(n_points=3)
        df = pd.DataFrame.from_dict(points, orient='index')

        # Expected output
        df_exp = pd.read_csv('tests/assets/ok_test_doe_under_lhs.csv')

        assert_frame_equal(df, df_exp)

    def test_fail_doe_lhs_under(self):
        ## DOEmodel
        approach = CategoricalApproachEnum('undersampling')
        method = DoeMethodEnum('lhs')
        method_doe = DoeMethod(name=method, kwargs={})
        doe_param = DoeParams(method=method_doe, categorical_approach=approach, mixture_design=None,
                              mixture_constraints=[{
                                  'formula': 'out = 0.5',
                                  'tol': .1
                              }])
        doemodel = DOEModel(doe_params=doe_param)

        ## DATAmodel
        input0 = InputFeature(feature_name='Power', id='id_01', default_value=100, optimize=True,
                              value_range=ValueRanges(lb=50, ub=100), value_list=[25, 75, 90],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input1 = InputFeature(feature_name='Repetiotion Rate', id='id_02', default_value=60, optimize=True,
                              value_range=ValueRanges(lb=60, ub=1008.10), value_list=[45, 60, 197.90],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input2 = InputFeature(feature_name='tau', id='id_03', default_value=240, optimize=True,
                              value_range=ValueRanges(lb=240, ub=8000), value_list=[240, 3880],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input3 = InputFeature(feature_name='n_pulses', id='id_04', default_value=30, optimize=True,
                              value_range=ValueRanges(lb=30, ub=110), value_list=[30, 50,70],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input4 = InputFeature(feature_name='sample', id='id_05', default_value='dog', optimize=True,
                              value_list=['alluminio', 'acciaio', 'vetro'], value_type=FeatureTypeEnum('Cat'),
                              transform=None, transform_kwargs=None)
        output0 = OutputFeature(feature_name='d_mean', id='id_06')
        output1 = OutputFeature(feature_name='h_mean', id='id_07')

        datamodel = DataModel(input_vars=[input0, input1, input2, input3, input4], output_vars=[output0, output1])

        doelab = DoELeapHandler(model=doemodel, datamodel=datamodel)
        points = doelab.ask(n_points=3)
        df = pd.DataFrame.from_dict(points, orient='index')

        # Expected output
        df_exp = pd.read_csv('tests/assets/nok_test_doe_under_lhs.csv')

        # Assert not equal
        with self.assertRaises(AssertionError):
            assert_frame_equal(df, df_exp)

    def test_doe_res_pbdesign_under(self):
        ## DOEmodel
        approach = CategoricalApproachEnum('undersampling')
        method = DoeMethodEnum('pbdesign')
        method_doe = DoeMethod(name=method, kwargs={})
        doe_param = DoeParams(method=method_doe, categorical_approach=approach, mixture_design=None,
                              mixture_constraints=[{
                                  'formula': 'out = 0.5',
                                  'tol': .1
                              }])
        doemodel = DOEModel(doe_params=doe_param)

        ## DATAmodel
        input0 = InputFeature(feature_name='Power', id='id_01', default_value=100, optimize=True,
                              value_range=ValueRanges(lb=50, ub=100), value_list=[25, 75, 90],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input1 = InputFeature(feature_name='Repetiotion Rate', id='id_02', default_value=60, optimize=True,
                              value_range=ValueRanges(lb=60, ub=1008.10), value_list=[45, 60, 197.90],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input2 = InputFeature(feature_name='tau', id='id_03', default_value=240, optimize=True,
                              value_range=ValueRanges(lb=240, ub=8000), value_list=[240, 3880],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input3 = InputFeature(feature_name='n_pulses', id='id_04', default_value=30, optimize=True,
                              value_range=ValueRanges(lb=30, ub=110), value_list=[30, 50,70],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input4 = InputFeature(feature_name='sample', id='id_05', default_value='dog', optimize=True,
                              value_list=['alluminio', 'acciaio', 'vetro'], value_type=FeatureTypeEnum('Cat'),
                              transform=None, transform_kwargs=None)
        output0 = OutputFeature(feature_name='d_mean', id='id_06')
        output1 = OutputFeature(feature_name='h_mean', id='id_07')

        datamodel = DataModel(input_vars=[input0, input1, input2, input3, input4], output_vars=[output0, output1])

        doelab = DoELeapHandler(model=doemodel, datamodel=datamodel)
        points = doelab.ask(n_points=3)
        df = pd.DataFrame.from_dict(points, orient='index')

        # Expected output
        df_exp = pd.read_csv('tests/assets/ok_test_doe_under_pbdesign.csv')

        assert_frame_equal(df, df_exp)

    def test_fail_doe_pbdesign_under(self):
        ## DOEmodel
        approach = CategoricalApproachEnum('undersampling')
        method = DoeMethodEnum('pbdesign')
        method_doe = DoeMethod(name=method, kwargs={})
        doe_param = DoeParams(method=method_doe, categorical_approach=approach, mixture_design=None,
                              mixture_constraints=[{
                                  'formula': 'out = 0.5',
                                  'tol': .1
                              }])
        doemodel = DOEModel(doe_params=doe_param)

        ## DATAmodel
        input0 = InputFeature(feature_name='Power', id='id_01', default_value=100, optimize=True,
                              value_range=ValueRanges(lb=50, ub=100), value_list=[25, 75, 90],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input1 = InputFeature(feature_name='Repetiotion Rate', id='id_02', default_value=60, optimize=True,
                              value_range=ValueRanges(lb=60, ub=1008.10), value_list=[45, 60, 197.90],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input2 = InputFeature(feature_name='tau', id='id_03', default_value=240, optimize=True,
                              value_range=ValueRanges(lb=240, ub=8000), value_list=[240, 3880],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input3 = InputFeature(feature_name='n_pulses', id='id_04', default_value=30, optimize=True,
                              value_range=ValueRanges(lb=30, ub=110), value_list=[30, 50,70],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input4 = InputFeature(feature_name='sample', id='id_05', default_value='dog', optimize=True,
                              value_list=['alluminio', 'acciaio', 'vetro'], value_type=FeatureTypeEnum('Cat'),
                              transform=None, transform_kwargs=None)
        output0 = OutputFeature(feature_name='d_mean', id='id_06')
        output1 = OutputFeature(feature_name='h_mean', id='id_07')

        datamodel = DataModel(input_vars=[input0, input1, input2, input3, input4], output_vars=[output0, output1])

        doelab = DoELeapHandler(model=doemodel, datamodel=datamodel)
        points = doelab.ask(n_points=3)
        df = pd.DataFrame.from_dict(points, orient='index')

        # Expected output
        df_exp = pd.read_csv('tests/assets/nok_test_doe_under_pbdesign.csv')

        # Assert not equal
        with self.assertRaises(AssertionError):
            assert_frame_equal(df, df_exp)
    
    def test_doe_res_bbdesign_under(self):
        ## DOEmodel
        approach = CategoricalApproachEnum('undersampling')
        method = DoeMethodEnum('bbdesign')
        method_doe = DoeMethod(name=method, kwargs={})
        doe_param = DoeParams(method=method_doe, categorical_approach=approach, mixture_design=None,
                              mixture_constraints=[{
                                  'formula': 'out = 0.5',
                                  'tol': .1
                              }])
        doemodel = DOEModel(doe_params=doe_param)

        ## DATAmodel
        input0 = InputFeature(feature_name='Power', id='id_01', default_value=100, optimize=True,
                              value_range=ValueRanges(lb=50, ub=100), value_list=[25, 75, 90],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input1 = InputFeature(feature_name='Repetiotion Rate', id='id_02', default_value=60, optimize=True,
                              value_range=ValueRanges(lb=60, ub=1008.10), value_list=[45, 60, 197.90],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input2 = InputFeature(feature_name='tau', id='id_03', default_value=240, optimize=True,
                              value_range=ValueRanges(lb=240, ub=8000), value_list=[240, 3880],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input3 = InputFeature(feature_name='n_pulses', id='id_04', default_value=30, optimize=True,
                              value_range=ValueRanges(lb=30, ub=110), value_list=[30, 50,70],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input4 = InputFeature(feature_name='sample', id='id_05', default_value='dog', optimize=True,
                              value_list=['alluminio', 'acciaio', 'vetro'], value_type=FeatureTypeEnum('Cat'),
                              transform=None, transform_kwargs=None)
        output0 = OutputFeature(feature_name='d_mean', id='id_06')
        output1 = OutputFeature(feature_name='h_mean', id='id_07')

        datamodel = DataModel(input_vars=[input0, input1, input2, input3, input4], output_vars=[output0, output1])

        doelab = DoELeapHandler(model=doemodel, datamodel=datamodel)
        points = doelab.ask(n_points=3)
        df = pd.DataFrame.from_dict(points, orient='index')

        # Expected output
        df_exp = pd.read_csv('tests/assets/ok_test_doe_under_bbdesign.csv')

        assert_frame_equal(df, df_exp)

    def test_fail_doe_bbdesign_under(self):
        ## DOEmodel
        approach = CategoricalApproachEnum('undersampling')
        method = DoeMethodEnum('bbdesign')
        method_doe = DoeMethod(name=method, kwargs={})
        doe_param = DoeParams(method=method_doe, categorical_approach=approach, mixture_design=None,
                              mixture_constraints=[{
                                  'formula': 'out = 0.5',
                                  'tol': .1
                              }])
        doemodel = DOEModel(doe_params=doe_param)

        ## DATAmodel
        input0 = InputFeature(feature_name='Power', id='id_01', default_value=100, optimize=True,
                              value_range=ValueRanges(lb=50, ub=100), value_list=[25, 75, 90],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input1 = InputFeature(feature_name='Repetiotion Rate', id='id_02', default_value=60, optimize=True,
                              value_range=ValueRanges(lb=60, ub=1008.10), value_list=[45, 60, 197.90],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input2 = InputFeature(feature_name='tau', id='id_03', default_value=240, optimize=True,
                              value_range=ValueRanges(lb=240, ub=8000), value_list=[240, 3880],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input3 = InputFeature(feature_name='n_pulses', id='id_04', default_value=30, optimize=True,
                              value_range=ValueRanges(lb=30, ub=110), value_list=[30, 50,70],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input4 = InputFeature(feature_name='sample', id='id_05', default_value='dog', optimize=True,
                              value_list=['alluminio', 'acciaio', 'vetro'], value_type=FeatureTypeEnum('Cat'),
                              transform=None, transform_kwargs=None)
        output0 = OutputFeature(feature_name='d_mean', id='id_06')
        output1 = OutputFeature(feature_name='h_mean', id='id_07')

        datamodel = DataModel(input_vars=[input0, input1, input2, input3, input4], output_vars=[output0, output1])

        doelab = DoELeapHandler(model=doemodel, datamodel=datamodel)
        points = doelab.ask(n_points=3)
        df = pd.DataFrame.from_dict(points, orient='index')

        # Expected output
        df_exp = pd.read_csv('tests/assets/nok_test_doe_under_bbdesign.csv')

        # Assert not equal
        with self.assertRaises(AssertionError):
            assert_frame_equal(df, df_exp)
    
    def test_doe_res_ccdesign_under(self):
        ## DOEmodel
        approach = CategoricalApproachEnum('undersampling')
        method = DoeMethodEnum('ccdesign')
        method_doe = DoeMethod(name=method, kwargs={})
        doe_param = DoeParams(method=method_doe, categorical_approach=approach, mixture_design=None,
                              mixture_constraints=[{
                                  'formula': 'out = 0.5',
                                  'tol': .1
                              }])
        doemodel = DOEModel(doe_params=doe_param)

        ## DATAmodel
        input0 = InputFeature(feature_name='Power', id='id_01', default_value=100, optimize=True,
                              value_range=ValueRanges(lb=50, ub=100), value_list=[25, 75, 90],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input1 = InputFeature(feature_name='Repetiotion Rate', id='id_02', default_value=60, optimize=True,
                              value_range=ValueRanges(lb=60, ub=1008.10), value_list=[45, 60, 197.90],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input2 = InputFeature(feature_name='tau', id='id_03', default_value=240, optimize=True,
                              value_range=ValueRanges(lb=240, ub=8000), value_list=[240, 3880],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input3 = InputFeature(feature_name='n_pulses', id='id_04', default_value=30, optimize=True,
                              value_range=ValueRanges(lb=30, ub=110), value_list=[30, 50,70],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input4 = InputFeature(feature_name='sample', id='id_05', default_value='dog', optimize=True,
                              value_list=['alluminio', 'acciaio', 'vetro'], value_type=FeatureTypeEnum('Cat'),
                              transform=None, transform_kwargs=None)
        output0 = OutputFeature(feature_name='d_mean', id='id_06')
        output1 = OutputFeature(feature_name='h_mean', id='id_07')

        datamodel = DataModel(input_vars=[input0, input1, input2, input3, input4], output_vars=[output0, output1])

        doelab = DoELeapHandler(model=doemodel, datamodel=datamodel)
        points = doelab.ask(n_points=3)
        df = pd.DataFrame.from_dict(points, orient='index')

        # Expected output
        df_exp = pd.read_csv('tests/assets/ok_test_doe_under_ccdesign.csv')

        assert_frame_equal(df, df_exp)

    def test_fail_doe_ccdesign_under(self):
        ## DOEmodel
        approach = CategoricalApproachEnum('undersampling')
        method = DoeMethodEnum('ccdesign')
        method_doe = DoeMethod(name=method, kwargs={})
        doe_param = DoeParams(method=method_doe, categorical_approach=approach, mixture_design=None,
                              mixture_constraints=[{
                                  'formula': 'out = 0.5',
                                  'tol': .1
                              }])
        doemodel = DOEModel(doe_params=doe_param)

        ## DATAmodel
        input0 = InputFeature(feature_name='Power', id='id_01', default_value=100, optimize=True,
                              value_range=ValueRanges(lb=50, ub=100), value_list=[25, 75, 90],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input1 = InputFeature(feature_name='Repetiotion Rate', id='id_02', default_value=60, optimize=True,
                              value_range=ValueRanges(lb=60, ub=1008.10), value_list=[45, 60, 197.90],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input2 = InputFeature(feature_name='tau', id='id_03', default_value=240, optimize=True,
                              value_range=ValueRanges(lb=240, ub=8000), value_list=[240, 3880],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input3 = InputFeature(feature_name='n_pulses', id='id_04', default_value=30, optimize=True,
                              value_range=ValueRanges(lb=30, ub=110), value_list=[30, 50,70],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input4 = InputFeature(feature_name='sample', id='id_05', default_value='dog', optimize=True,
                              value_list=['alluminio', 'acciaio', 'vetro'], value_type=FeatureTypeEnum('Cat'),
                              transform=None, transform_kwargs=None)
        output0 = OutputFeature(feature_name='d_mean', id='id_06')
        output1 = OutputFeature(feature_name='h_mean', id='id_07')

        datamodel = DataModel(input_vars=[input0, input1, input2, input3, input4], output_vars=[output0, output1])

        doelab = DoELeapHandler(model=doemodel, datamodel=datamodel)
        points = doelab.ask(n_points=3)
        df = pd.DataFrame.from_dict(points, orient='index')

        # Expected output
        df_exp = pd.read_csv('tests/assets/nok_test_doe_under_ccdesign.csv')

        # Assert not equal
        with self.assertRaises(AssertionError):
            assert_frame_equal(df, df_exp)

    def test_doe_res_fullfact_under(self):
        ## DOEmodel
        approach = CategoricalApproachEnum('undersampling')
        method = DoeMethodEnum('fullfact')
        method_doe = DoeMethod(name=method, kwargs={})
        doe_param = DoeParams(method=method_doe, categorical_approach=approach, mixture_design=None,
                              mixture_constraints=[{
                                  'formula': 'out = 0.5',
                                  'tol': .1
                              }])
        doemodel = DOEModel(doe_params=doe_param)

        ## DATAmodel
        input0 = InputFeature(feature_name='Power', id='id_01', default_value=100, optimize=True,
                              value_range=ValueRanges(lb=50, ub=100), value_list=[25, 75, 90],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input1 = InputFeature(feature_name='Repetiotion Rate', id='id_02', default_value=60, optimize=True,
                              value_range=ValueRanges(lb=60, ub=1008.10), value_list=[45, 60, 197.90],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input2 = InputFeature(feature_name='tau', id='id_03', default_value=240, optimize=True,
                              value_range=ValueRanges(lb=240, ub=8000), value_list=[240, 3880],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input3 = InputFeature(feature_name='n_pulses', id='id_04', default_value=30, optimize=True,
                              value_range=ValueRanges(lb=30, ub=110), value_list=[30, 50,70],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input4 = InputFeature(feature_name='sample', id='id_05', default_value='dog', optimize=True,
                              value_list=['alluminio', 'acciaio', 'vetro'], value_type=FeatureTypeEnum('Cat'),
                              transform=None, transform_kwargs=None)
        output0 = OutputFeature(feature_name='d_mean', id='id_06')
        output1 = OutputFeature(feature_name='h_mean', id='id_07')

        datamodel = DataModel(input_vars=[input0, input1, input2, input3, input4], output_vars=[output0, output1])

        doelab = DoELeapHandler(model=doemodel, datamodel=datamodel)
        points = doelab.ask(n_points=3)
        df = pd.DataFrame.from_dict(points, orient='index')

        # Expected output
        df_exp = pd.read_csv('tests/assets/ok_test_doe_under_fullfact.csv')

        assert_frame_equal(df, df_exp)

    def test_fail_doe_fullfact_under(self):
        ## DOEmodel
        approach = CategoricalApproachEnum('undersampling')
        method = DoeMethodEnum('fullfact')
        method_doe = DoeMethod(name=method, kwargs={})
        doe_param = DoeParams(method=method_doe, categorical_approach=approach, mixture_design=None,
                              mixture_constraints=[{
                                  'formula': 'out = 0.5',
                                  'tol': .1
                              }])
        doemodel = DOEModel(doe_params=doe_param)

        ## DATAmodel
        input0 = InputFeature(feature_name='Power', id='id_01', default_value=100, optimize=True,
                              value_range=ValueRanges(lb=50, ub=100), value_list=[25, 75, 90],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input1 = InputFeature(feature_name='Repetiotion Rate', id='id_02', default_value=60, optimize=True,
                              value_range=ValueRanges(lb=60, ub=1008.10), value_list=[45, 60, 197.90],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input2 = InputFeature(feature_name='tau', id='id_03', default_value=240, optimize=True,
                              value_range=ValueRanges(lb=240, ub=8000), value_list=[240, 3880],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input3 = InputFeature(feature_name='n_pulses', id='id_04', default_value=30, optimize=True,
                              value_range=ValueRanges(lb=30, ub=110), value_list=[30, 50,70],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input4 = InputFeature(feature_name='sample', id='id_05', default_value='dog', optimize=True,
                              value_list=['alluminio', 'acciaio', 'vetro'], value_type=FeatureTypeEnum('Cat'),
                              transform=None, transform_kwargs=None)
        output0 = OutputFeature(feature_name='d_mean', id='id_06')
        output1 = OutputFeature(feature_name='h_mean', id='id_07')

        datamodel = DataModel(input_vars=[input0, input1, input2, input3, input4], output_vars=[output0, output1])

        doelab = DoELeapHandler(model=doemodel, datamodel=datamodel)
        points = doelab.ask(n_points=3)
        df = pd.DataFrame.from_dict(points, orient='index')

        # Expected output
        df_exp = pd.read_csv('tests/assets/nok_test_doe_under_fullfact.csv')

        # Assert not equal
        with self.assertRaises(AssertionError):
            assert_frame_equal(df, df_exp)

    def test_doe_res_ff2n_under(self):
        ## DOEmodel
        approach = CategoricalApproachEnum('undersampling')
        method = DoeMethodEnum('ff2n')
        method_doe = DoeMethod(name=method, kwargs={})
        doe_param = DoeParams(method=method_doe, categorical_approach=approach, mixture_design=None,
                              mixture_constraints=[{
                                  'formula': 'out = 0.5',
                                  'tol': .1
                              }])
        doemodel = DOEModel(doe_params=doe_param)

        ## DATAmodel
        input0 = InputFeature(feature_name='Power', id='id_01', default_value=100, optimize=True,
                              value_range=ValueRanges(lb=50, ub=100), value_list=[25, 75, 90],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input1 = InputFeature(feature_name='Repetiotion Rate', id='id_02', default_value=60, optimize=True,
                              value_range=ValueRanges(lb=60, ub=1008.10), value_list=[45, 60, 197.90],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input2 = InputFeature(feature_name='tau', id='id_03', default_value=240, optimize=True,
                              value_range=ValueRanges(lb=240, ub=8000), value_list=[240, 3880],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input3 = InputFeature(feature_name='n_pulses', id='id_04', default_value=30, optimize=True,
                              value_range=ValueRanges(lb=30, ub=110), value_list=[30, 50,70],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input4 = InputFeature(feature_name='sample', id='id_05', default_value='dog', optimize=True,
                              value_list=['alluminio', 'acciaio', 'vetro'], value_type=FeatureTypeEnum('Cat'),
                              transform=None, transform_kwargs=None)
        output0 = OutputFeature(feature_name='d_mean', id='id_06')
        output1 = OutputFeature(feature_name='h_mean', id='id_07')

        datamodel = DataModel(input_vars=[input0, input1, input2, input3, input4], output_vars=[output0, output1])

        doelab = DoELeapHandler(model=doemodel, datamodel=datamodel)
        points = doelab.ask(n_points=3)
        df = pd.DataFrame.from_dict(points, orient='index')

        # Expected output
        df_exp = pd.read_csv('tests/assets/ok_test_doe_under_ff2n.csv')

        assert_frame_equal(df, df_exp)

    def test_fail_doe_ff2n_under(self):
        ## DOEmodel
        approach = CategoricalApproachEnum('undersampling')
        method = DoeMethodEnum('ff2n')
        method_doe = DoeMethod(name=method, kwargs={})
        doe_param = DoeParams(method=method_doe, categorical_approach=approach, mixture_design=None,
                              mixture_constraints=[{
                                  'formula': 'out = 0.5',
                                  'tol': .1
                              }])
        doemodel = DOEModel(doe_params=doe_param)

        ## DATAmodel
        input0 = InputFeature(feature_name='Power', id='id_01', default_value=100, optimize=True,
                              value_range=ValueRanges(lb=50, ub=100), value_list=[25, 75, 90],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input1 = InputFeature(feature_name='Repetiotion Rate', id='id_02', default_value=60, optimize=True,
                              value_range=ValueRanges(lb=60, ub=1008.10), value_list=[45, 60, 197.90],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input2 = InputFeature(feature_name='tau', id='id_03', default_value=240, optimize=True,
                              value_range=ValueRanges(lb=240, ub=8000), value_list=[240, 3880],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input3 = InputFeature(feature_name='n_pulses', id='id_04', default_value=30, optimize=True,
                              value_range=ValueRanges(lb=30, ub=110), value_list=[30, 50,70],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input4 = InputFeature(feature_name='sample', id='id_05', default_value='dog', optimize=True,
                              value_list=['alluminio', 'acciaio', 'vetro'], value_type=FeatureTypeEnum('Cat'),
                              transform=None, transform_kwargs=None)
        output0 = OutputFeature(feature_name='d_mean', id='id_06')
        output1 = OutputFeature(feature_name='h_mean', id='id_07')

        datamodel = DataModel(input_vars=[input0, input1, input2, input3, input4], output_vars=[output0, output1])

        doelab = DoELeapHandler(model=doemodel, datamodel=datamodel)
        points = doelab.ask(n_points=3)
        df = pd.DataFrame.from_dict(points, orient='index')

        # Expected output
        df_exp = pd.read_csv('tests/assets/nok_test_doe_under_ff2n.csv')

        # Assert not equal
        with self.assertRaises(AssertionError):
            assert_frame_equal(df, df_exp)

    def test_doe_res_fracfact_under(self):
        ## DOEmodel
        approach = CategoricalApproachEnum('undersampling')
        method = DoeMethodEnum('fracfact')
        method_doe = DoeMethod(name=method, kwargs={})
        doe_param = DoeParams(method=method_doe, categorical_approach=approach, mixture_design=None,
                              mixture_constraints=[{
                                  'formula': 'out = 0.5',
                                  'tol': .1
                              }])
        doemodel = DOEModel(doe_params=doe_param)

        ## DATAmodel
        input0 = InputFeature(feature_name='Power', id='id_01', default_value=100, optimize=True,
                              value_range=ValueRanges(lb=50, ub=100), value_list=[25, 75, 90],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input1 = InputFeature(feature_name='Repetiotion Rate', id='id_02', default_value=60, optimize=True,
                              value_range=ValueRanges(lb=60, ub=1008.10), value_list=[45, 60, 197.90],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input2 = InputFeature(feature_name='tau', id='id_03', default_value=240, optimize=True,
                              value_range=ValueRanges(lb=240, ub=8000), value_list=[240, 3880],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input3 = InputFeature(feature_name='n_pulses', id='id_04', default_value=30, optimize=True,
                              value_range=ValueRanges(lb=30, ub=110), value_list=[30, 50,70],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input4 = InputFeature(feature_name='sample', id='id_05', default_value='dog', optimize=True,
                              value_list=['alluminio', 'acciaio', 'vetro'], value_type=FeatureTypeEnum('Cat'),
                              transform=None, transform_kwargs=None)
        output0 = OutputFeature(feature_name='d_mean', id='id_06')
        output1 = OutputFeature(feature_name='h_mean', id='id_07')

        datamodel = DataModel(input_vars=[input0, input1, input2, input3, input4], output_vars=[output0, output1])

        doelab = DoELeapHandler(model=doemodel, datamodel=datamodel)
        points = doelab.ask(n_points=3)
        df = pd.DataFrame.from_dict(points, orient='index')

        # Expected output
        df_exp = pd.read_csv('tests/assets/ok_test_doe_under_fracfact.csv')

        assert_frame_equal(df, df_exp)

    def test_fail_doe_fracfact_under(self):
        ## DOEmodel
        approach = CategoricalApproachEnum('undersampling')
        method = DoeMethodEnum('fracfact')
        method_doe = DoeMethod(name=method, kwargs={})
        doe_param = DoeParams(method=method_doe, categorical_approach=approach, mixture_design=None,
                              mixture_constraints=[{
                                  'formula': 'out = 0.5',
                                  'tol': .1
                              }])
        doemodel = DOEModel(doe_params=doe_param)

        ## DATAmodel
        input0 = InputFeature(feature_name='Power', id='id_01', default_value=100, optimize=True,
                              value_range=ValueRanges(lb=50, ub=100), value_list=[25, 75, 90],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input1 = InputFeature(feature_name='Repetiotion Rate', id='id_02', default_value=60, optimize=True,
                              value_range=ValueRanges(lb=60, ub=1008.10), value_list=[45, 60, 197.90],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input2 = InputFeature(feature_name='tau', id='id_03', default_value=240, optimize=True,
                              value_range=ValueRanges(lb=240, ub=8000), value_list=[240, 3880],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input3 = InputFeature(feature_name='n_pulses', id='id_04', default_value=30, optimize=True,
                              value_range=ValueRanges(lb=30, ub=110), value_list=[30, 50,70],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input4 = InputFeature(feature_name='sample', id='id_05', default_value='dog', optimize=True,
                              value_list=['alluminio', 'acciaio', 'vetro'], value_type=FeatureTypeEnum('Cat'),
                              transform=None, transform_kwargs=None)
        output0 = OutputFeature(feature_name='d_mean', id='id_06')
        output1 = OutputFeature(feature_name='h_mean', id='id_07')

        datamodel = DataModel(input_vars=[input0, input1, input2, input3, input4], output_vars=[output0, output1])

        doelab = DoELeapHandler(model=doemodel, datamodel=datamodel)
        points = doelab.ask(n_points=3)
        df = pd.DataFrame.from_dict(points, orient='index')

        # Expected output
        df_exp = pd.read_csv('tests/assets/nok_test_doe_under_fracfact.csv')

        # Assert not equal
        with self.assertRaises(AssertionError):
            assert_frame_equal(df, df_exp)

    def test_doe_res_gsd_under(self):
        ## DOEmodel
        approach = CategoricalApproachEnum('undersampling')
        method = DoeMethodEnum('gsd')
        method_doe = DoeMethod(name=method, kwargs={})
        doe_param = DoeParams(method=method_doe, categorical_approach=approach, mixture_design=None,
                              mixture_constraints=[{
                                  'formula': 'out = 0.5',
                                  'tol': .1
                              }])
        doemodel = DOEModel(doe_params=doe_param)

        ## DATAmodel
        input0 = InputFeature(feature_name='Power', id='id_01', default_value=100, optimize=True,
                              value_range=ValueRanges(lb=50, ub=100), value_list=[25, 75, 90],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input1 = InputFeature(feature_name='Repetiotion Rate', id='id_02', default_value=60, optimize=True,
                              value_range=ValueRanges(lb=60, ub=1008.10), value_list=[45, 60, 197.90],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input2 = InputFeature(feature_name='tau', id='id_03', default_value=240, optimize=True,
                              value_range=ValueRanges(lb=240, ub=8000), value_list=[240, 3880],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input3 = InputFeature(feature_name='n_pulses', id='id_04', default_value=30, optimize=True,
                              value_range=ValueRanges(lb=30, ub=110), value_list=[30, 50,70],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input4 = InputFeature(feature_name='sample', id='id_05', default_value='dog', optimize=True,
                              value_list=['alluminio', 'acciaio', 'vetro'], value_type=FeatureTypeEnum('Cat'),
                              transform=None, transform_kwargs=None)
        output0 = OutputFeature(feature_name='d_mean', id='id_06')
        output1 = OutputFeature(feature_name='h_mean', id='id_07')

        datamodel = DataModel(input_vars=[input0, input1, input2, input3, input4], output_vars=[output0, output1])

        doelab = DoELeapHandler(model=doemodel, datamodel=datamodel)
        points = doelab.ask(n_points=3)
        df = pd.DataFrame.from_dict(points, orient='index')

        # Expected output
        df_exp = pd.read_csv('tests/assets/ok_test_doe_under_gsd.csv')

        assert_frame_equal(df, df_exp)

    def test_fail_doe_gsd_under(self):
        ## DOEmodel
        approach = CategoricalApproachEnum('undersampling')
        method = DoeMethodEnum('gsd')
        method_doe = DoeMethod(name=method, kwargs={})
        doe_param = DoeParams(method=method_doe, categorical_approach=approach, mixture_design=None,
                              mixture_constraints=[{
                                  'formula': 'out = 0.5',
                                  'tol': .1
                              }])
        doemodel = DOEModel(doe_params=doe_param)

        ## DATAmodel
        input0 = InputFeature(feature_name='Power', id='id_01', default_value=100, optimize=True,
                              value_range=ValueRanges(lb=50, ub=100), value_list=[25, 75, 90],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input1 = InputFeature(feature_name='Repetiotion Rate', id='id_02', default_value=60, optimize=True,
                              value_range=ValueRanges(lb=60, ub=1008.10), value_list=[45, 60, 197.90],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input2 = InputFeature(feature_name='tau', id='id_03', default_value=240, optimize=True,
                              value_range=ValueRanges(lb=240, ub=8000), value_list=[240, 3880],
                              value_type=FeatureTypeEnum('Float'), transform=None, transform_kwargs=None)
        input3 = InputFeature(feature_name='n_pulses', id='id_04', default_value=30, optimize=True,
                              value_range=ValueRanges(lb=30, ub=110), value_list=[30, 50,70],
                              value_type=FeatureTypeEnum('OrdCat'), transform=None, transform_kwargs=None)
        input4 = InputFeature(feature_name='sample', id='id_05', default_value='dog', optimize=True,
                              value_list=['alluminio', 'acciaio', 'vetro'], value_type=FeatureTypeEnum('Cat'),
                              transform=None, transform_kwargs=None)
        output0 = OutputFeature(feature_name='d_mean', id='id_06')
        output1 = OutputFeature(feature_name='h_mean', id='id_07')

        datamodel = DataModel(input_vars=[input0, input1, input2, input3, input4], output_vars=[output0, output1])

        doelab = DoELeapHandler(model=doemodel, datamodel=datamodel)
        points = doelab.ask(n_points=3)
        df = pd.DataFrame.from_dict(points, orient='index')

        # Expected output
        df_exp = pd.read_csv('tests/assets/nok_test_doe_under_gsd.csv')

        # Assert not equal
        with self.assertRaises(AssertionError):
            assert_frame_equal(df, df_exp)
