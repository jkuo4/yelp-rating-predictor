from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import dump_svmlight_file
import xlearn as xl


class FMRecommender(object):
    """
    Extension of xlearn's Factorization Machines
    """

    def __init__(self, X_train, X_test, y_train, y_test, categorical_cols):
        self.model_file = "data/fm/model.txt"
        self.weights_file = "data/fm/model_wt.out"
        self.x_train_file = "data/fm/x_train.txt"
        self.x_test_file = "data/fm/x_test.txt"
        self.categorical_cols = categorical_cols

        self.model = xl.create_fm()
        self.model.setTXTModel(self.model_file)
        
        X_train = self.one_hot_categorical_col(X_train)
        X_test = self.one_hot_categorical_col(X_test)
        dump_svmlight_file(X_train, y_train, self.x_train_file, zero_based=True, multilabel=False)
        dump_svmlight_file(X_test, y_test, self.x_test_file, zero_based=True, multilabel=False)
        

    def fit(self, param):
        """
        Fit the model.

        param: dict, parameters to fit the model with
        """
        self.model.setTrain(self.x_train_file)
        self.model.fit(param, self.weights_file)

        
    def predict(self):
        """
        Predict ratings.
        
        return: numpy.ndarray for the predicted values
        """
        self.model.setTest(self.x_test_file)
        return self.model.predict(self.weights_file)
    
    
    def one_hot_categorical_col(self, X):
        """
        Encode categorical variables as one-hot.

        param: dict, parameters to fit the model with
        return: scipy.sparse.coo.coo_matrix of feature variables
        """
        if not self.categorical_cols:
            return X
        
        output = X.drop(self.categorical_cols, axis=1).astype(float)
        for col in self.categorical_cols:
            enc = OneHotEncoder() # sparse format
            ohe_col = enc.fit_transform(X[col].values.reshape(-1, 1))
            output = hstack([output, ohe_col])            
        return output
