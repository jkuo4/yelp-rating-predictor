import xlearn as xl


class FMRecommender(object):
    """
    Extension of xlearn's Factorization Machines
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        self.model_file = "./model.txt"
        self.weights_file = "./model_dm.out"

        self.model = xl.create_fm()
        self.model.setTXTModel(self.model_file)
        self.model.setTrain(xl.DMatrix(X_train, y_train))
        self.xdm_test = xl.DMatrix(X_test, y_test)

    def fit(self, param):
        """
        Fit the modeli.

        param: dict, parameters to fit the model with
        """
        self.model.fit(param, self.weights_file)

    def predict(self):
        """Predict ratings."""
        self.model.setTest(self.xdm_test)
        return self.model.predict(self.weights_file)
