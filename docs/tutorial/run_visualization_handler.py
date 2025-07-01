from sklearn.ensemble import RandomForestRegressor
from aidox.optimization.handler_visualization import  Visualization
from aidox.models.pydantic_models_visualization import FeatureTypeEnum, InputFeature, OutputFeature, DataModel

import pandas as pd

def main():

    # DataModel
    file = 'docs/data_processed/regressor_predictions.xlsx'
    
    input0= InputFeature(feature_name='A',id= 'id_0',  value_list = [50,75,100] ,value_type = FeatureTypeEnum('OrdCat'), feature_description =None)  
    input1= InputFeature(feature_name='C',id='id_2', value_list = [240,3880,8000], value_type = FeatureTypeEnum('OrdCat'), feature_description = None) 
    input2= InputFeature(feature_name='D',id='id_3', value_list = [30,50,70,90,110] ,value_type = FeatureTypeEnum('OrdCat'),feature_description = None)
    output2= OutputFeature(feature_name='Output1', id='out_01', value_type = FeatureTypeEnum('Float'), feature_description = None)
    output3= OutputFeature(feature_name='Output2', id='out_02',value_type = FeatureTypeEnum('Float'), feature_description = None)
    datamodel = DataModel (dataframe = file, input_vars = [input0,input1,input2] , output_vars=[output2,output3])
    
    # Visualization
    graph = Visualization(data = datamodel)

    # Scatter plot data 
    graph.scatter_plot(x = 'D', y = ['Output1', 'Output1_pred'], col = 'A', row = 'C', orders = {'A': [50, 75, 100],'C': [240,3880,8000]}, title = 'Design Optimization')
    graph.scatter_plot(x = 'D', y = ['Output2', 'Output2_pred'], col = 'A', row = 'C', orders = {'A': [50, 75, 100],'C': [240,3880,8000]}, title = 'Design Optimization')
                    
    # Shap analysis and centroids
    graph.SHAP_analysis(target ='Output1', estimator = RandomForestRegressor(n_estimators=80, max_depth=8, criterion='squared_error',))
    shap_values_target_1 = graph.shap_values
    graph.biplot_centroids()
    feature_target1 = graph.features_target1    
    
    graph.SHAP_analysis(target ='Output2', estimator = RandomForestRegressor(n_estimators=80, max_depth=8, criterion='squared_error'))
    shap_values_target_2 = graph.shap_values
    graph.biplot_centroids()
    feature_target2 = graph.features_target1
   
    # Biplot graphs
    graph.biplot_features(shap_values_target_1 = shap_values_target_1, shap_values_target_2 = shap_values_target_2, features_target1 = feature_target1, features_target2 = feature_target2, title= 'Sample analysis', colors_x =['b','m','green'], colors_points = ['c','violet','y'])
    
if __name__=='__main__':
    main()