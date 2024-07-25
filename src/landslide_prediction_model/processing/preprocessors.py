import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


class SklearnTransformerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for Scikit-learn pre-processing transformers,
    like the SimpleImputer() or OrdinalEncoder(), to allow
    the use of the transformer on a selected group of variables.
    """

    def __init__(self, variables=None, transformer=None):

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        The `fit` method allows scikit-learn transformers to
        learn the required parameters from the training data set.
        """

        self.transformer.fit(X[self.variables])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""
        X = X.copy()
        X[self.variables] = self.transformer.transform(X[self.variables])
        return X


class DropNA(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.t = 1+1
        
    def fit(self, X, y=None):
        
        """
        The `fit` method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """
        
        return self

    def transform(self, X):
        # drop all rows of any features with nan from the data set
        X = X.copy()
        X = X.dropna(axis=0, how='any')

        return X


class Astype_features(BaseEstimator, TransformerMixin):
    def __init__(self, astype_features=None):
        self.variables = astype_features

    def fit(self, X, y=None):
        """
        The `fit` method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """

        return self

    def transform(self, X):
        # drop unnecesary / unused features from the data set
        X = X.copy()
        X = X.astype(self.variables)
        return X


class Scarps_Special_Edit(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        """
        The `fit` method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """
        return self

    def transform(self, X):
        # drop unnecesary / unused features from the data set
        X = X.copy()
        con0 = X[self.variables] == -1
        con1 = X[self.variables] == 23
        con2 = X[self.variables] == 17
        con3 = X[self.variables] == 25
        
        index = X[con0 | con1 | con2 | con3].index
        X.drop(axis=0, labels = index, inplace= True)

        return X

class NegativeValueEstimator(BaseEstimator, TransformerMixin):
    """dealing with negative values"""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
            
    def fit(self, X, y=None):
        """
        The `fit` method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """

        return self

    def transform(self, X):
        X = X.copy()
        for col in self.variables:
            X[col] = X[col] - X[col].min()

        return X
        
class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables = variables_to_drop

    def fit(self, X, y=None):
        """
        The `fit` method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """

        return self

    def transform(self, X):
        # drop unnecesary / unused features from the data set
        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X


class Train_Test_Split(BaseEstimator, TransformerMixin):
    """dealing with negative values"""

    def __init__(self, features=None, target=None, test_size=None, random_state=None):
        
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features
            
        if not isinstance(target, list):
            self.target = [target]
        else:
            self.target = target


        self.test_size = test_size
        self.random_state = random_state
        
    def fit(self, X, y=None):
        """
        The `fit` method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """

        return self

    def transform(self, X):
        X_train, X_test, y_train, y_test = train_test_split(X[self.features], X[self.target], test_size=self.test_size,
        random_state=self.random_state)

        test = pd.concat([X_test, y_test], axis=1)
        data_dir = DATASET_DIR / 'test1.csv'
        test.to_csv(data_dir)
        
        return [X_train, y_train]



