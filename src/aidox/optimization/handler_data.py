import itertools
from ..opt_utils import build_constraints
from ..commons import LeapHandler, convert_to_float
from .._utils import flatlist
from ..models.pydantic_models_data import DataModel, DOEModel, InputFeature
import numpy as np
import sklearn.preprocessing as skpreprocessing
import scipy.stats.distributions as sp_disrt
from aidox import pyDOE2


class DoELeapHandler(LeapHandler):
    """
    This is an handler for Design Of Experiment
    pipeline. The pipeline can be employed to setup the DOE
    environment and retrieve the suggested experimetnal points

    """
    def __init__(self, model: DOEModel, datamodel: DataModel):
        """
        The init function init the DOE environment with the
        specified model and datamodel, which indicated the deisgn method and
        experimental input features 

        :param model: dictionary of setupt doe param and constraints
        :param datamodel: list of input and output variables
        """
        super().__init__()
        self.samples = None
        self.doe_mdl = model
        self.data_mdl = datamodel
        self.float_feature_transformed = {}
        self.cat_feature_transformed = {}
        self.ordcat_feature_transformed = {}

        self.init_parameters()

    def init_parameters(self):

        """ 
        The method init the DOE environment using the 
        optimization variable parameters and the optimizer 
        parameters defined in datamodel
        
        """

        # Set optimization/output vars
        self.opt_vars = self.data_mdl.input_vars
        self.out_vars = self.data_mdl.output_vars
        # Set DoE parameters
        self.opt_params = self.doe_mdl.doe_params
        constr_dict, _ = build_constraints(params=self.doe_mdl.doe_params.mixture_constraints)
        self.constraint_dict = constr_dict
        self.params, self.map_id, self.set_default_ = self.build_parametrization()

        self.float_vars_list = [x for x in range(len(self.opt_vars)) if self.opt_vars[x].value_type == 'Float']
        self.cat_vars_list = [x for x in range(len(self.opt_vars)) if self.opt_vars[x].value_type == 'Cat']
        self.ordcat_vars_list = [x for x in range(len(self.opt_vars)) if self.opt_vars[x].value_type == 'OrdCat']

        # Fit and transformed scalers
        for x in self.float_vars_list:
            value_lst_scaled, scaler = self.fit_trasform(input_feature=self.opt_vars[x])
            self.float_feature_transformed[self.opt_vars[x].feature_name] = {
                'range': value_lst_scaled,
                'scaler': scaler
            }

        for x in self.ordcat_vars_list:
            scaled_ordcat_ranges, scaler = self.fit_trasform_ordcat(input_feature=self.opt_vars[x])
            self.ordcat_feature_transformed[self.opt_vars[x].feature_name] = {
                'value_lst_ordcat_scaled': scaled_ordcat_ranges,
                'scaler': scaler
            }
        for x in self.cat_vars_list:
            scaled_ranges, scaler = self.fit_trasform_cat(input_feature=self.opt_vars[x])
            self.cat_feature_transformed[self.opt_vars[x].feature_name] = {
                'value_lst_cat_scaled': scaled_ranges,
                'scaler': scaler
            }

    def fit_trasform(self, input_feature: InputFeature):
        """
        Fit and trasform float input variables using MinMaxScaler
        
        :param input_feature: InputFeature class
        :return: list of scaled values and scaler

        """
        if input_feature.value_type == 'Float':
            ranges = np.array([input_feature.value_range.lb, input_feature.value_range.ub])
        # Standard is MinMaxScaler
            scaler = getattr(skpreprocessing, 'MinMaxScaler')()
            value_lst_scaled = scaler.fit_transform(ranges.reshape(-1, 1))
            return value_lst_scaled, scaler

    def fit_trasform_ordcat(self, input_feature: InputFeature):
        """
        Fit and trasform categorical input variables using OneHotEncoder
        
        :param input_feature: InputFeature class
        :return: list of scaled values and scaler

        """
        if input_feature.value_type == 'OrdCat':
            ranges = np.array(input_feature.value_list)
        # Standard is LabelEncoder
            scaler = getattr(skpreprocessing, 'LabelEncoder')()
            scaled_ordcat_ranges = scaler.fit_transform(ranges)
            return scaled_ordcat_ranges, scaler

    def fit_trasform_cat(self, input_feature: InputFeature):
        """
        Fit and trasform categorical input variables using OneHotEncoder
        
        :param input_feature: InputFeature class
        :return: list of scaled values and scaler

        """
        if input_feature.value_type == 'Cat':
            ranges = np.array(input_feature.value_list)
        # Standard is OneHotEncoder
            scaler = getattr(skpreprocessing, 'OneHotEncoder')()
            scaled_ranges = scaler.fit_transform(ranges.reshape(-1, 1))
            return scaled_ranges, scaler
    

    def build_parametrization(self):
        """
        Builds parametrization for nevergrad optimizer

        :return params: optimization params
        :return map_id: mapping dict to columns
        :return set_default_: default parameters

        """
        list_mixture_feat = set(flatlist([v['coeff'][1] for k, v in self.constraint_dict.items()]))
        list_mixture_feat = list(map(str, list_mixture_feat))
        dict_opt_vars = {x: self.opt_vars[x] for x in range(len(self.opt_vars))}
        dict_out_vars = {x: self.out_vars[x] for x in range(len(self.out_vars))}

        map_id = {k: v.id for k, v in
                  dict_opt_vars.items()}
        map_id.update({k: v.id for k, v in
                       dict_out_vars.items()})
        # split input vars
        floats = {k: {
            'range': v.value_range,
            'mixture': map_id[k] in list_mixture_feat
        } for k, v in
            dict_opt_vars.items() if v.optimize and v.value_type == 'Float'}
        cats = {k: {
            'range': set(v.value_list),
            'mixture': map_id[k] in list_mixture_feat
        } for k, v in dict_opt_vars.items() if v.value_type == 'Cat' and v.optimize}

        ordcats = {k: {
                'range': set(v.value_list),
                'mixture': map_id[k] in list_mixture_feat
         } for k, v in dict_opt_vars.items() if v.value_type == 'OrdCat' and v.optimize}

        set_default_ = {k: v.default for k, v in dict_opt_vars.items() if
                        not v.optimize and map_id[k] not in list_mixture_feat}

        params = {}

        # FLOAT
        params['float_input'] = floats

        # ORDCAT| CAT
        params['ordcat_input'] = ordcats
        params['cat_input'] = cats
        return params, map_id, set_default_

    def ask(self, n_points: int = None) -> dict:
        """
        Ask n_points to the DOE environment adopting user-defined method

        :param n_points: number of experimental points
        :return: dictionary with experimetnal points to be performed

        """
        method = self.doe_mdl.doe_params.method.name
        method_kwargs = self.doe_mdl.doe_params.method.kwargs
        self.samples = n_points if n_points is not None else self.doe_mdl.doe_params.n_doe_points
        if method == 'lhs':
            out_points = self.lhs(**method_kwargs)
            out_points = self.build_output_dict(out_points)
            return out_points

        elif method == 'pbdesign':
            out_points = self.pbdesign(**method_kwargs)
            out_points = self.build_output_dict(out_points)
            return out_points

        elif method == 'bbdesign':
            out_points = self.bbdesign(**method_kwargs)
            out_points = self.build_output_dict(out_points)
            return out_points

        elif method == 'ccdesign':
            out_points = self.ccdesign(**method_kwargs)
            out_points = self.build_output_dict(out_points)
            return out_points

        elif method == 'fullfact':
            out_points = self.fullfact(**method_kwargs)
            out_points = self.build_output_dict(out_points)
            return out_points

        elif method == 'ff2n':
            out_points = self.ff2n(**method_kwargs)
            out_points = self.build_output_dict(out_points)
            return out_points

        elif method == 'fracfact':
            out_points = self.fracfact(**method_kwargs)
            out_points = self.build_output_dict(out_points)
            return out_points

        elif method == 'gsd':
            out_points = self.gsd(**method_kwargs)
            out_points = self.build_output_dict(out_points)
            return out_points

        else:
            raise ValueError(
                f'Wrong method passed, received {method}, accepts lhs, meshgrid, fullfact, ff2n, fracfact, pbdesign, gsd, bbdesign, ccdesign, poi-d-optimal,poi-polytope ')

    def lhs(self, criterion: str = 'correlation', random_state = 42):
        cat_approach = self.doe_mdl.doe_params.categorical_approach
        out_values = pyDOE2.lhs((len(self.float_vars_list)+len(self.ordcat_vars_list)), samples=self.samples, criterion=criterion, random_state=42)
        
        listaf = [x for x in range(len(self.opt_vars)) if self.opt_vars[x].value_type == 'Float']
        if listaf != []:
            pbdesign_scaler = skpreprocessing.MinMaxScaler()
            out_values[:,listaf] = pbdesign_scaler.fit_transform(out_values[:,listaf])
            out_values[:,listaf] = self.rescale_output(out_values[:,listaf], listaf)
        
        listoc = [x for x in range(len(self.opt_vars)) if self.opt_vars[x].value_type == 'OrdCat']
        label_scaler = skpreprocessing.LabelEncoder()
        for idx, col in enumerate(listoc):
            out_values[:,col] = label_scaler.fit_transform(out_values[:,col])
        out_values[:,listoc] = self.rescale_output(out_values[:,listoc], listoc)

        out_values = self.add_categorical(out_values, cat_approach)

        return out_values


    def bbdesign(self, center=None):
        '''Number of variables must be at least 3'''
        cat_approach = self.doe_mdl.doe_params.categorical_approach
        out_values = pyDOE2.bbdesign((len(self.float_vars_list)+len(self.ordcat_vars_list)), center=center)
        
        listaf = [x for x in range(len(self.opt_vars)) if self.opt_vars[x].value_type == 'Float']
        if listaf != []:
            pbdesign_scaler = skpreprocessing.MinMaxScaler()
            out_values[:,listaf] = pbdesign_scaler.fit_transform(out_values[:,listaf])
            out_values[:,listaf] = self.rescale_output(out_values[:,listaf], listaf)
        
        listoc = [x for x in range(len(self.opt_vars)) if self.opt_vars[x].value_type == 'OrdCat']
        label_scaler = skpreprocessing.LabelEncoder()
        for idx, col in enumerate(listoc):
            out_values[:,col] = label_scaler.fit_transform(out_values[:,col])
        out_values[:,listoc] = self.rescale_output(out_values[:,listoc], listoc)

        out_values = self.add_categorical(out_values, cat_approach)

        return out_values

       
    def ccdesign(self, center=(4, 4), alpha='orthogonal', face='faced'):
        cat_approach = self.doe_mdl.doe_params.categorical_approach
        out_values = pyDOE2.ccdesign((len(self.float_vars_list)+len(self.ordcat_vars_list)), center=center, alpha=alpha, face=face)
        
        listaf = [x for x in range(len(self.opt_vars)) if self.opt_vars[x].value_type == 'Float']
        if listaf != []:
            pbdesign_scaler = skpreprocessing.MinMaxScaler()
            out_values[:,listaf] = pbdesign_scaler.fit_transform(out_values[:,listaf])
            out_values[:,listaf] = self.rescale_output(out_values[:,listaf], listaf)
        
        listoc = [x for x in range(len(self.opt_vars)) if self.opt_vars[x].value_type == 'OrdCat']
        label_scaler = skpreprocessing.LabelEncoder()
        for idx, col in enumerate(listoc):
            out_values[:,col] = label_scaler.fit_transform(out_values[:,col])
        out_values[:,listoc] = self.rescale_output(out_values[:,listoc], listoc)

        out_values = self.add_categorical(out_values, cat_approach)

        return out_values

    def pbdesign(self):
        cat_approach = self.doe_mdl.doe_params.categorical_approach
        out_values = pyDOE2.pbdesign((len(self.float_vars_list)+len(self.ordcat_vars_list)))
        
        listaf = [x for x in range(len(self.opt_vars)) if self.opt_vars[x].value_type == 'Float']
        if listaf != []:
            pbdesign_scaler = skpreprocessing.MinMaxScaler()
            out_values[:,listaf] = pbdesign_scaler.fit_transform(out_values[:,listaf])
            out_values[:,listaf] = self.rescale_output(out_values[:,listaf], listaf)
        
        listoc = [x for x in range(len(self.opt_vars)) if self.opt_vars[x].value_type == 'OrdCat']
        label_scaler = skpreprocessing.LabelEncoder()
        for idx, col in enumerate(listoc):
            out_values[:,col] = label_scaler.fit_transform(out_values[:,col])
        out_values[:,listoc] = self.rescale_output(out_values[:,listoc], listoc)

        out_values = self.add_categorical(out_values, cat_approach)

        return out_values

    def fullfact(self):
        cat_approach = self.doe_mdl.doe_params.categorical_approach
        ff_ = [] 
        if any(self.opt_vars[x].value_type == 'Float' for x in range(len(self.opt_vars))):
            factors = len(self.float_vars_list) 
            levels = self.samples
            ff_ = [levels for _ in range(factors)]

        if any(self.opt_vars[x].value_type == 'OrdCat' for x in range(len(self.opt_vars))):
            factors = len(self.ordcat_vars_list) 
            levels = []
            for i,j in enumerate(list(self.params['ordcat_input'].keys())):
                levels.append(len(self.params['ordcat_input'][j]['range']))
            
            if ff_!=[]:
                ff_.extend(levels)
            else:
                ff_ = levels

        out_values = pyDOE2.fullfact(ff_)

        listaf = [x for x in range(len(self.opt_vars)) if self.opt_vars[x].value_type == 'Float']
        if listaf != []:
            pbdesign_scaler = skpreprocessing.MinMaxScaler()
            out_values[:,listaf] = pbdesign_scaler.fit_transform(out_values[:,listaf])
            out_values[:,listaf] = self.rescale_output(out_values[:,listaf], listaf)
        
        listoc = [x for x in range(len(self.opt_vars)) if self.opt_vars[x].value_type == 'OrdCat']
        out_values[:,listoc] = self.rescale_output(out_values[:,listoc], listoc)

        out_values = self.add_categorical(out_values, cat_approach)

        return out_values

    def ff2n(self):
        cat_approach = self.doe_mdl.doe_params.categorical_approach
        out_values = pyDOE2.ff2n((len(self.float_vars_list)+len(self.ordcat_vars_list)))
        
        listaf = [x for x in range(len(self.opt_vars)) if self.opt_vars[x].value_type == 'Float']
        if listaf != []:
            pbdesign_scaler = skpreprocessing.MinMaxScaler()
            out_values[:,listaf] = pbdesign_scaler.fit_transform(out_values[:,listaf])
            out_values[:,listaf] = self.rescale_output(out_values[:,listaf], listaf)
        
        listoc = [x for x in range(len(self.opt_vars)) if self.opt_vars[x].value_type == 'OrdCat']
        label_scaler = skpreprocessing.LabelEncoder()
        for idx, col in enumerate(listoc):
            out_values[:,col] = label_scaler.fit_transform(out_values[:,col])
        out_values[:,listoc] = self.rescale_output(out_values[:,listoc], listoc)

        out_values = self.add_categorical(out_values, cat_approach)

        return out_values

    def fracfact(self):
        cat_approach = self.doe_mdl.doe_params.categorical_approach
        gen_ = []
        
        for n in self.float_vars_list:
            for n, name in enumerate(self.opt_vars[n].feature_name[:1]):
                gen_.append(name)
        for n in self.ordcat_vars_list:
            for n, name in enumerate(self.opt_vars[n].feature_name[:1]):
                gen_.append(name)
        gen = ' '.join(gen_)
        
        out_values = pyDOE2.fracfact(gen)   
        listaf = [x for x in range(len(self.opt_vars)) if self.opt_vars[x].value_type == 'Float']
        if listaf != []:
            pbdesign_scaler = skpreprocessing.MinMaxScaler()
            out_values[:,listaf] = pbdesign_scaler.fit_transform(out_values[:,listaf])
            out_values[:,listaf] = self.rescale_output(out_values[:,listaf], listaf)
        
        listoc = [x for x in range(len(self.opt_vars)) if self.opt_vars[x].value_type == 'OrdCat']
        label_scaler = skpreprocessing.LabelEncoder()
        for idx, col in enumerate(listoc):
            out_values[:,col] = label_scaler.fit_transform(out_values[:,col])
        out_values[:,listoc] = self.rescale_output(out_values[:,listoc], listoc)

        out_values = self.add_categorical(out_values, cat_approach)

        return out_values     


    def gsd(self, reduction: int = 2):
        cat_approach = self.doe_mdl.doe_params.categorical_approach
        # Attention! Levels and n_points are different concepts

        factors = len(self.float_vars_list)+len(self.ordcat_vars_list)
        n = self.samples
        levels = [n for _ in range(factors)]
        out_values = pyDOE2.gsd(levels=levels, reduction=reduction)
        
        listaf = [x for x in range(len(self.opt_vars)) if self.opt_vars[x].value_type == 'Float']
        if listaf != []:
            pbdesign_scaler = skpreprocessing.MinMaxScaler()
            out_values[:,listaf] = pbdesign_scaler.fit_transform(out_values[:,listaf])
            out_values[:,listaf] = self.rescale_output(out_values[:,listaf], listaf)
        
        listoc = [x for x in range(len(self.opt_vars)) if self.opt_vars[x].value_type == 'OrdCat']
        label_scaler = skpreprocessing.LabelEncoder()
        for idx, col in enumerate(listoc):
            out_values[:,col] = label_scaler.fit_transform(out_values[:,col])
        out_values[:,listoc] = self.rescale_output(out_values[:,listoc], listoc)

        out_values = self.add_categorical(out_values, cat_approach)

        return out_values

    def build_output_dict(self, out_points: np.ndarray) -> dict:
        
        """
        Build_output_dict transforms the out_points obtained by
        the predefined DOE method into a dictionary

        """
        feat_name = [x.feature_name for x in self.data_mdl.input_vars]
        res = {}
        for idx_test, values in enumerate(out_points.tolist()):
            # If conversion fails, keep as string
            values_cvt = convert_to_float(values)
            # Try to convert to float if possible
            res[idx_test] = {k: v for k, v in zip(feat_name, values_cvt)}
        return res



    def rescale_output(self, out_values, list_var):
        """
        
        """
        if self.float_vars_list != [] and list_var == self.float_vars_list:
        # Inverse transform for continuous variables
            for x, value in enumerate(self.float_vars_list):
                # Transform using ppf
                transform = self.opt_vars[value].transform
                transform_kwargs = self.opt_vars[value].transform_kwargs
                if transform is not None:
                    distribution = getattr(sp_disrt, transform)(**transform_kwargs)
                    out_values[:, x] = distribution.ppf(out_values[:, x])
                # Inverse transform for float features
                scaler = self.float_feature_transformed[self.opt_vars[value].feature_name][
                    'scaler']
                if scaler is not None:
                    out_values[:, x] = scaler.inverse_transform(
                        out_values[:, x].reshape(-1, 1)).flatten()

        # Inverse transform for discrete variable
        if self.ordcat_vars_list != [] and list_var == self.ordcat_vars_list:
            new_outvalues = np.zeros_like(out_values)
            for x, value in enumerate(self.ordcat_vars_list):
                scaler = self.ordcat_feature_transformed[self.opt_vars[value].feature_name][
                'scaler']
            # Inverse transform for OrdCat features from
                if scaler is not None:
                    col = self.opt_vars[value].value_list
                    for k,number in enumerate(col):
                        new_outvalues[out_values[:,x]==k,x] = np.full_like(new_outvalues[out_values[:,x]==k,x],number)
            out_values = new_outvalues
   
        return out_values

    def add_categorical(self, out_values, cat_approach):
        """
        Add categorical variables to output values considering the categorical
        approach defined in DOEModel
        
        :param out_values: ouput of fit_transform
        :param cat_approach: approach to be adopted (random or oversampling)
        :return: out_values
        
        """
        # Add categorical variables
        out_cat_values = []
        # Create combinations of categorical values
        for feat in self.cat_vars_list:
            out_cat_values.append(list(self.opt_vars[feat].value_list))
        # Create combination of all cat values
        combs = list(itertools.product(*out_cat_values))
        if cat_approach == 'random':
            # Get the random sample of combinations
            out_cat_arr = np.array(combs)[np.random.randint(len(combs), size=out_values.shape[0])]
            out_cat = out_cat_arr.T
            out_values = np.concatenate((out_values, out_cat.reshape(-1, 1)), axis=1)
        elif cat_approach == 'oversampling':
            # Get the oversampling of combinations
            out_values_1 = np.concatenate([out_values] * len(combs), axis=0)
            # out_values = np.concatenate([out_values, np.repeat(combs, self.samples).reshape(-1, 1)], axis=1)
            out_values = np.concatenate([out_values_1, np.repeat(combs, len(out_values)).reshape(-1, 1)], axis=1)
        return out_values

