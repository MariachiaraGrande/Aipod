import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import importlib as importlib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import  confusion_matrix, f1_score, accuracy_score, recall_score, root_mean_squared_error, mean_absolute_percentage_error,  mean_squared_error
from ..models.pydantic_models_training import DataModel, ModelTraining
from sklearn.model_selection import KFold


def rename_columns(filename: str, parameters: list, columns: list):
    ''' Rename columns in the given dataframe based on the provided parameters and
        columns list.'''
    data = pd.read_excel(filename)
    col = columns
    par = parameters
    for i, name in enumerate(par):
        data = data.rename(columns= {data.columns[i]:name})
    for j, name in enumerate(col):
        data = data.rename(columns={data.columns[len(par):][j]:name})
    return data

def evaluation_mean_std(filename: str):
    ''' Evaluate the mean and standard deviation of the design variables.'''
    data = pd.read_excel(filename)
    d_mean = data.iloc[:,lambda data:[4,6,8,10,12]].mean(axis=1)
    h_mean = data.iloc[:, lambda data:[5,7,9,11,13]].mean(axis=1)
    d_std = data.iloc[:, lambda data:[4,6,8,10,12]].std(axis=1)
    h_std = data.iloc[:, lambda data:[5,7,9,11,13]].std(axis=1)
    data['d_mean'] = d_mean
    data['d_std'] = d_std
    data['h_mean'] = h_mean
    data['h_std'] = h_std
    return data

def label_encoder(data: pd.DataFrame, target: list, encoded_target:list):
    ''' Encode the target variable astype int.'''
    for n,value in enumerate(target):
        data[encoded_target[n]] = (data[target[n]]!=0).astype('int')
    return data


class Classifier:
    def __init__(self, datamodel: DataModel, model_training: ModelTraining):
        
        super().__init__()
        self.data = datamodel
        self.model = model_training
        self.init_parameters()

    def init_parameters(self):
        ''' Initialize the classifier parameters.'''
        
        # This parameters are valid in presence of DataModel class
        self.df = pd.read_excel(self.data.dataframe['classifier'])
        self.features = self.data.input_vars
        self.target = self.data.output_vars
        self.model_name = self.model.model_name['classifier']
        self.model_params = self.model.model_params['classifier']
        self.param_grid = self.model.param_grid['classifier']

        

    def train_test_split(self):
        ''' Split the data into training and testing sets.'''
  
        features = []
        target = []
        for n,value in enumerate(self.features):
            features.append(self.features[n].feature_name)
        for n,value in enumerate(self.target):
            target.append(self.target[n].feature_name)

        X_features = self.df[features]
        target = self.df[target]
        stratSplit = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

        for i, (train_index, test_index) in enumerate(stratSplit.split(X_features,target)):
            X_train = X_features.iloc[train_index]
            X_test = X_features.iloc[test_index]
            y_train = target.iloc[train_index]
            y_test = target.iloc[test_index]
       
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def model_prediction(self):
        '''Returns the  predicted values of the target variable according to the chosen model.'''

        modules = ['sklearn.ensemble', 'sklearn.gaussian_process', 'sklearn.linear_model', 'sklearn.neural_network']
        for module in modules:
            skmodel = importlib.import_module(module)
            try:
                model = getattr(skmodel, self.model_name)
                break
            except:
                continue

        model = model()
        if self.param_grid is not None:
            np.random.seed(42)
            grid_search = GridSearchCV(model, self.param_grid, n_jobs=-1,
                         scoring=['f1','accuracy'], cv= 2, return_train_score=False, refit='f1')
            grid_search.fit(self.X_train, self.y_train)
            self.grid_best_params = grid_search.best_params_
            model.set_params(**self.grid_best_params)
            model.fit(self.X_train, self.y_train)
            y_pred_test = model.predict(self.X_test) 
            y_pred_train  = model.predict(self.X_train) 
            y_pred_test_prob = model.predict_proba(self.X_test)
            y_pred_train_prob = model.predict_proba(self.X_train)
            
        else:
            model = getattr(skmodel, self.model_name)()
            model.fit(self.X_train, self.y_train)
            y_pred_test = model.predict(self.X_test) 
            y_pred_train  = model.predict(self.X_train) 
            y_pred_test_prob = model.predict_proba(self.X_test)
            y_pred_train_prob = model.predict_proba(self.X_train)

        self.y_pred_test = y_pred_test
        self.y_pred_test_prob = y_pred_test_prob
        self.y_pred_train = y_pred_train
        self.y_pred_train_prob = y_pred_train_prob

    def evaluate_performance(self):
        '''Evaluate the performance of the model with suitable metrics .'''
       
        print("estimator: ", self.model_name )
        print("best_params: ", self.grid_best_params )
        print('======================================================================')

        print ('Recall_micro_Train:', recall_score(self.y_train,  self.y_pred_train, average= 'micro'))
        print ('Recall_micro_Test :', recall_score(self.y_test,  self.y_pred_test, average= 'micro'))
        print('======================================================================')

        print ('Accuracy_Training:', accuracy_score(self.y_train, self.y_pred_train))
        print ('Accuracy_Test:', accuracy_score(self.y_test,  self.y_pred_test))
        print('======================================================================')

        print ('f1_score_Training:', f1_score(self.y_train, self.y_pred_train, average= 'micro'))
        print ('f1_score_Test:', f1_score(self.y_test,  self.y_pred_test, average= 'micro'))
        print('======================================================================')
        
        cm_train= confusion_matrix(self.y_train.values.argmax(axis=1),  self.y_pred_train.argmax(axis=1), labels=[0,1], normalize ='all')
        cm_test= confusion_matrix(self.y_test.values.argmax(axis=1),  self.y_pred_test.argmax(axis=1),labels=[0,1], normalize ='all')
            
        print ('cm_train:', cm_train)
        print('cm_test:', cm_test)
        print('======================================================================')

    def check_classifier_probability(self, threshold:float):
        self.threshold = threshold
        self.X_train['predicted_d'] = self.y_pred_train[:,0]
        self.X_train['predicted_h'] = self.y_pred_train[:,1]

        self.X_train['prob_d'] = self.y_pred_train_prob[0][:,1]
        self.X_train['prob_h'] = self.y_pred_train_prob[1][:,1]

        self.X_test['predicted_d'] = self.y_pred_test[:,0]
        self.X_test['predicted_h'] = self.y_pred_test[:,1]

        self.X_test['prob_d'] = self.y_pred_test_prob[0][:,1]
        self.X_test['prob_h'] = self.y_pred_test_prob[1][:,1]

        data_train_win = self.X_train[(self.X_train['prob_d'] > threshold) & (self.X_train['prob_h'] > threshold)]
        data_test_win  = self.X_test[(self.X_test['prob_d'] > threshold) & (self.X_test['prob_h'] > threshold)]

        data_train_fail = self.X_train[(self.X_train['prob_d'] < threshold) & (self.X_train['prob_h'] < threshold)]
        data_test_fail  = self.X_test[(self.X_test['prob_d'] < threshold) & (self.X_test['prob_h'] < threshold)]

        print('threshold: ', self.threshold)
        print('=================================')

        return data_test_fail, data_test_win, data_train_fail, data_train_win

    def concat_dataframes (self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train:pd.DataFrame,
                        y_test: pd.DataFrame, y_pred_train: np.array, y_pred_test: np.array, columns: list):
        
        y_pred_train = pd.DataFrame(y_pred_train, index = self.X_train.index, columns = columns)
        y_pred_test = pd.DataFrame(y_pred_test, index = self.X_test.index, columns = columns)
        df_conc_train = pd.concat([X_train, y_train, y_pred_train], axis=1)
        df_conc_test = pd.concat([X_test, y_test, y_pred_test], axis=1)

        df_concat = pd.concat([df_conc_train, df_conc_test], axis=0)
        return df_concat

class Regressor:
    def __init__(self, datamodel: DataModel, model_training: ModelTraining):
        
        super().__init__()
        self.data = datamodel
        self.model = model_training
        self.init_parameters()

    def init_parameters(self):
        ''' Initialize the regressor parameters.'''
        
        # This parameters are valid in presence of DataModel and ModelTraining classes
        self.df = pd.read_excel(self.data.dataframe['regressor'])
        self.features = self.data.input_vars
        self.target = self.data.output_vars
        self.model_name = self.model.model_name['regressor']
        self.model_params = self.model.model_params['regressor']
        self.param_grid = self.model.param_grid['regressor']

    def train_test_split(self, method: str):
        ''' Split the data into training and testing sets.'''
            
        features = []
        target = []
        for n,value in enumerate(self.features):
            features.append(self.features[n].feature_name)
        for n,value in enumerate(self.target):
            target.append(self.target[n].feature_name)
        X_features = self.df[features]
        target = self.df[target]

        stratSplit = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)


        if method == 'KFold':
            k_fold = KFold(n_splits=3, shuffle=True, random_state=42)
            for i, (train_index, test_index) in enumerate(k_fold.split(X_features,target)):
                X_train = X_features.iloc[train_index]
                X_test = X_features.iloc[test_index]
                y_train = target.iloc[train_index]
                y_test= target.iloc[test_index]
        

        if method == 'StratSplit':
            for i, (train_index, test_index) in enumerate(stratSplit.split(X_features,target)):
                X_train = X_features.iloc[train_index]
                X_test = X_features.iloc[test_index]
                y_train = target.iloc[train_index]
                y_test = target.iloc[test_index]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_features, target, test_size=0.3, random_state=42)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def model_prediction(self):
        '''Returns the  predicted values of the target variable according to the chosen model.'''

        modules = ['sklearn.ensemble', 'sklearn.gaussian_process', 'sklearn.linear_model', 'sklearn.neural_network']
        for module in modules:
            skmodel = importlib.import_module(module)
            try:
                model = getattr(skmodel, self.model_name)
                break
            except:
                continue
        
        model = model()
        if self.param_grid is not None:
            np.random.seed(42)
            grid_search = GridSearchCV(model, self.param_grid,n_jobs=-1, scoring=['neg_mean_squared_error'], cv= 5, 
                            return_train_score=True, refit='neg_mean_squared_error') 
            grid_search.fit(self.X_train, self.y_train)
            self.grid_best_params = grid_search.best_params_
            model.set_params(**self.grid_best_params)
            model.fit(self.X_train, self.y_train)
            y_pred_test = model.predict(self.X_test)
            y_pred_train = model.predict(self.X_train)
            y_pred_test_score = model.score(self.X_test,self.y_test)
            y_pred_train_score = model.score(self.X_train, self.y_train)  
            root_mean_squared_error_train = root_mean_squared_error (self.y_train,y_pred_train)
            root_mean_squared_error_test = root_mean_squared_error (self.y_test,y_pred_test)
            mean_absolute_percentage_error_train = mean_absolute_percentage_error (self.y_train,y_pred_train)
            mean_absolute_percentage_error_test = mean_absolute_percentage_error (self.y_test,y_pred_test)
            mean_square_error_train = mean_squared_error (self.y_train,y_pred_train)
            mean_square_error_test = mean_squared_error (self.y_test,y_pred_test)

            print("estimator: ", self.model_name )
            print("best_params: ", self.grid_best_params )
            print('======================================================================')
            print ('score_train:', y_pred_train_score)
            print('score_test:', y_pred_test_score)
            print('======================================================================')
            print ('RMSE_train:', root_mean_squared_error_train)
            print('RMSE_test:', root_mean_squared_error_test)
            print('======================================================================')  
            print ('MAPE_train:', mean_absolute_percentage_error_train)
            print('MAPE_test:', mean_absolute_percentage_error_test)
            print('======================================================================')
            print ('MSE_train:', mean_square_error_train)
            print('MSE_test:', mean_square_error_test)
            print('======================================================================')
            
        else:
            model = getattr(skmodel, self.model_name)()
            model.fit(self.X_train, self.y_train)
            y_pred_test = model.predict(self.X_test)
            y_pred_train = model.predict(self.X_train)
            y_pred_test_score = model.score(self.X_test,self.y_test)
            y_pred_train_score = model.score(self.X_train, self.y_train)

            root_mean_squared_error_train = root_mean_squared_error (self.y_train,y_pred_train)
            root_mean_squared_error_test = root_mean_squared_error (self.y_test,y_pred_test)
            mean_absolute_percentage_error_train = mean_absolute_percentage_error (self.y_train,y_pred_train)
            mean_absolute_percentage_error_test = mean_absolute_percentage_error (self.y_test,y_pred_test)
            mean_square_error_train = mean_squared_error (self.y_train,y_pred_train)
            mean_square_error_test = mean_squared_error (self.y_test,y_pred_test)
        
            print("estimator: ", self.model_name )
            print('======================================================================')
            print ('score_train:', y_pred_train_score)
            print('score_test:', y_pred_test_score)
            print('======================================================================')
            print ('RMSE_train:', root_mean_squared_error_train)
            print('RMSE_test:', root_mean_squared_error_test)
            print('======================================================================')  
            print ('MAPE_train:', mean_absolute_percentage_error_train)
            print('MAPE_test:', mean_absolute_percentage_error_test)
            print('======================================================================')
            print ('MSE_train:', mean_square_error_train)
            print('MSE_test:', mean_square_error_test)
            print('======================================================================')
         
        self.y_pred_test = y_pred_test
        self.y_pred_test_prob = y_pred_test_score
        self.y_pred_train = y_pred_train
        self.y_pred_train_prob = y_pred_train_score  
    


    def concat_dataframes (self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train:pd.DataFrame,
                        y_test: pd.DataFrame, y_pred_train: np.array, y_pred_test: np.array, columns: list):
        
        y_pred_train = pd.DataFrame(y_pred_train, index = self.X_train.index, columns = columns)
        y_pred_test = pd.DataFrame(y_pred_test, index = self.X_test.index, columns = columns)
        df_conc_train = pd.concat([X_train, y_train, y_pred_train], axis=1)
        df_conc_test = pd.concat([X_test, y_test, y_pred_test], axis=1)

        df_concat = pd.concat([df_conc_train, df_conc_test], axis=0)
        return df_concat