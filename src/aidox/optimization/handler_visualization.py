
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import os
from ..models.pydantic_models_visualization import DataModel
import shap


class Visualization:

    def __init__(self, data: DataModel):
        
        super().__init__()
        self.data = data
        self.init_parameters()

    def init_parameters(self):

        self.df = pd.read_excel(self.data.dataframe)
        self.features = self.data.input_vars
        self.target = self.data.output_vars
        
        features = []
        target = []
        for n,value in enumerate(self.features):
            features.append(self.features[n].feature_name)
        for n,value in enumerate(self.target):
            target.append(self.target[n].feature_name)

        self.X = self.df[features]
        self.y = self.df[target]



    def scatter_plot(self, x, y, col, row, orders:None, title:None):
        '''  Scatter plot data 

        Args:
        df_concat : DataFrame
        x : str. The x coordinate
        y : str or list. The y coordinate
        col : str. Columns of global plot
        row: str. Row of global plot
        orders : dict. Order for col and row variables
        title : str. Title of the plot
        '''

        fig = px.scatter (self.df, x = self.df[x], y = y, facet_col = self.df[col], facet_col_spacing= 0.1, facet_row = self.df[row], 
                        facet_row_spacing = 0.06, category_orders = orders, width=1000, height = 1000,
                        title = title) 
        fig.update_layout(
            xaxis=dict(range=[0, 120]),  
            xaxis2=dict(range=[0, 120]), 
            xaxis3=dict(range=[0, 120]), 
            yaxis=dict(range=[0, 70])  
                )

        directory = 'docs/graphs'
        file = f'scatter_plot_{y}.png'
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.write_image(os.path.join(directory, file), format='png')
        fig.show()
   

    ## Perform sensitivity analysis using SHAP values

    def SHAP_analysis(self, target:str, estimator):

        ''' Show the SHAP graphs and values

        Args:
        target: str. Pay attention only one value!
        estimator: sklearn estimator i.e RandomForestRegressor
        '''

        explainer = shap.Explainer(estimator.fit(self.X, self.y[target]), self.X)
        shap_values = explainer(self.X)
        shap.plots.beeswarm(shap_values, color=plt.get_cmap("cool"), show = False)
    
        directory = 'docs/graphs'
        file = f'shap_plot_{target}.png'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, file), format='png')
        self.shap_values = shap_values
    def biplot_centroids(self):

        features_n1 = []
        for n,v in enumerate(self.features):
            features_n1.append([self.shap_values.values[self.shap_values.data[:,n] == v,n].mean() for v in self.features[n].value_list])
        
        self.features_target1 = features_n1
        
 

    def biplot_features(self, shap_values_target_1, shap_values_target_2, features_target1, features_target2, title, colors_x : list, colors_points : list):
        
        shap_values_target_1.values
        title = title
        fig, ax = plt.subplots()
        x_lims = []
        y_lims = []
        for idx, value in enumerate(features_target1):
            ax.plot(features_target1[idx], features_target2[idx], 'x:', label = str(self.features[idx].feature_name), color = colors_x[idx], linewidth= 0.4)
            for i, rannge in enumerate(self.features[idx].value_list):
                ax.scatter(shap_values_target_1.values[shap_values_target_1.data[:, idx] == self.features[idx].value_list[i],idx],
                    shap_values_target_2.values[shap_values_target_2.data[:, idx] == self.features[idx].value_list[i],idx], c = colors_points[idx], alpha= 0.4)
                
                ax.text(features_target1[idx][i], features_target2[idx][i], s= str(self.features[idx].value_list[i]), fontsize= 'x-small', ha = 'left', va= 'bottom')
                x_lims.append(shap_values_target_1.values[shap_values_target_1.data[:, idx] == self.features[idx].value_list[i],idx].max())
                y_lims.append(shap_values_target_2.values[shap_values_target_2.data[:, idx] == self.features[idx].value_list[i],idx].max())
       
        boundary_graph = max(max(x_lims), max(y_lims))
        ax.axhline(0, color='k', linestyle= ':', linewidth= 0.3)
        ax.axvline(0, color='k', linestyle= ':', linewidth= 0.3)
        ax.set_xlim((-boundary_graph-2), (boundary_graph+2))
        ax.set_ylim((-boundary_graph-2), (boundary_graph+2))
        ax.set_ylabel('Influence on '+str(self.target[1].feature_name))
        ax.set_xlabel('Influence on '+str(self.target[0].feature_name))
        ax.set_title(title)
        ax.legend()

        directory = 'docs/graphs'
        file = f'biplot_{title}.png'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, file), format='png')
        plt.show()
      




