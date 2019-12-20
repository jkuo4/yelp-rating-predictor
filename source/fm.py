from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import dump_svmlight_file
import xlearn as xl

class FMRecommender(object):
    """
    Extension of xlearn's Factorization Machines
    """

    def __init__(self, X_train, y_train, categorical_cols, categorical_universe):
        self.model_file = "data/fm/model.txt"
        self.weights_file = "data/fm/model_wt.out"
        self.x_train_file = "data/fm/x_train.txt"
        self.x_test_file = "data/fm/x_test.txt"
        self.categorical_cols = categorical_cols
        self.categorical_universe = categorical_universe
        self.enc = OneHotEncoder().fit(self.categorical_universe.reshape(-1, 1))
        
        self.model = xl.create_fm()
        self.model.setTXTModel(self.model_file)
        
        self.transform_input(X_train, y_train, 'train')
        

    def fit(self, param):
        """
        Fit the model.

        param: dict, parameters to fit the model with
        """
        self.model.setTrain(self.x_train_file)
        self.model.fit(param, self.weights_file)
        
        
    def transform_input(self, X, y, input_type):
        """
        Transform the raw file into sparse file for model

        param: numpy.ndarray for the raw predictor values
        param: numpy.ndarray for the raw target values
        param: string, train/test
        """
        X_ohe_cat = self.one_hot_categorical_col(X)
        if input_type == 'train':
            file_path = self.x_train_file
        else:
            file_path = self.x_test_file
        dump_svmlight_file(X_ohe_cat, y, file_path, zero_based=True, multilabel=False)

        
    def predict(self, X_test, y_test):
        """
        Predict ratings.
        
        return: numpy.ndarray for the predicted values
        """
        self.transform_input(X_test, y_test, 'test')
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
            ohe_col = self.enc.transform(X[col].values.reshape(-1, 1))
            output = hstack([output, ohe_col])            
        return output
