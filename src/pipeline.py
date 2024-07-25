from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from processing import preprocessors as pp
import model
from config.core import config

import logging


_logger = logging.getLogger(__name__)


landscape_pipe = Pipeline(
    [
        (
            "numerical_imputer_1",
            pp.SklearnTransformerWrapper(variables=config.model_config.numerical_vars_1,
                                         transformer=SimpleImputer(strategy='median')),
        ),
        
        (
            "numerical_imputer_2",
            pp.SklearnTransformerWrapper(variables=config.model_config.numerical_vars_2,
                                         transformer=SimpleImputer(strategy='mean')),
        ),
        
        (
            "categorical_imputer_1",
            pp.SklearnTransformerWrapper(variables=config.model_config.categorical_vars_1,
                                         transformer=SimpleImputer(strategy='constant', fill_value=np.nan)),
        ),
        
        (
            "categorical_imputer_2",
            pp.SklearnTransformerWrapper(variables=config.model_config.categorical_vars_2,
                                         transformer=SimpleImputer(strategy='most_frequent')),
        ),
        
        (
            "dropna_in_all_variable",
            pp.DropNA(),
        ),
        
        (
            "apply_astype_encoder",
            pp.Astype_features(astype_features=config.model_config.astype_features),
        ),

        (
            "special_encoder",
            pp.Scarps_Special_Edit(variables=config.model_config.special_edit),
        ),
        
        (
            "negative_value_encoder",
            pp.NegativeValueEstimator(astype_features=config.model_config.negative_variables),
        ),
        
        (
            "drop_features",
            pp.DropUnecessaryFeatures(variables_to_drop=config.model_config.variables_to_drop),
        ),

        (
            "train_test_split",
            pp.Train_Test_Split(features=config.model_config.features, 
                                target=config.model_config.target, 
                                test_size=config.model_config.test_size, 
                                random_state=config.model_config.random_state),
        ),
        
        (
            "model_choice",
            model.Model_Choice(loss=config.model_config.loss,
                               random_state=config.model_config.random_state,
                               n_estimators=config.model_config.n_estimators,
                               modelchoice=config.model_config.modelchoice),
            
        ),
    ]
)
  