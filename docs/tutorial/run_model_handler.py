
from aidox.optimization.handler_model import Classifier, Regressor
from aidox.models.pydantic_models_training import FeatureTypeEnum, InputFeature, OutputFeature, DataModel, ModelTraining
import pandas as pd
import os

# ### REGRESSOR TEST EXAMPLE
def main():

    #Define the DataModel class
    input0= InputFeature(feature_name='A',id= 'id_0', feature_type = FeatureTypeEnum('OrdCat'), feature_description =None)  
    input1= InputFeature(feature_name='B', id='id_1', feature_type = FeatureTypeEnum('OrdCat'), feature_description = None) 
    input2= InputFeature(feature_name='C',id='id_2', feature_type = FeatureTypeEnum('OrdCat'), feature_description = None) 
    input3= InputFeature(feature_name='D',id='id_3', feature_type = FeatureTypeEnum('OrdCat'),feature_description = None)

    output2= OutputFeature(feature_name='Output1', id='out_0', feature_type = FeatureTypeEnum('Float'), feature_description = None)
    output3= OutputFeature(feature_name='Output2', id='out_1',feature_type = FeatureTypeEnum('Float'), feature_description = None)
   
    datamodel_reg = DataModel(dataframe = {'regressor':'docs/data_processed/processed_data.xlsx'}, input_vars = [input0,input1,input2,input3] , output_vars=[output2,output3])
    model_training_reg = ModelTraining(model_name = {'regressor':'RandomForestRegressor'}, model_params = {'regressor': {}}, param_grid = {'regressor':
                                        {"n_estimators":[50,80,100], "criterion":['squared_error','poisson', 'friedman_mse'],"max_depth":[8,16,32]}})
    regressor = Regressor(datamodel = datamodel_reg, model_training = model_training_reg)
    regressor.train_test_split(method= 'standard')
    regressor.model_prediction()

    df_conc = regressor.concat_dataframes(X_train = regressor.X_train, X_test = regressor.X_test, y_train = regressor.y_train, y_test = regressor.y_test, y_pred_train = regressor.y_pred_train, y_pred_test = regressor.y_pred_test, columns = ['Output1_pred', 'Output2_pred'])

    directory = 'docs/data_processed'
    file = 'regressor_predictions.xlsx'
    if not os.path.exists(directory):
        os.makedirs(directory)
    df_conc.to_excel(os.path.join(directory, file))

if __name__=='__main__':
    main()