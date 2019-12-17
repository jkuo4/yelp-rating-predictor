from surprise import BaselineOnly, accuracy


class BaseLineRecommender(object):
    """
    Use surprise's baselineonly algorithm as the baseline of prediction
    """

    def __init__(self):
        self.model = None

    def fit(self, train):
        """
        Fit the model
        """
        self.model = BaselineOnly(
            bsl_options={'method': 'sgd',
                         'n_epochs': 30, 
                         'reg': 0.01, 
                         'learning_rate': 0.01}
        )
        self.model.fit(train)

    def predict(self, user_id, item_id):
        """
        Predict ratings
        """
        return self.model.predict(user_id, item_id)

    def rmse(self, test):
        """
        Calculate RMSE for the predicted ratings
        """
        pred = self.model.test(test)
        return accuracy.rmse(pred)
    
    def mae(self, test):
        """
        Calculate MAE for the predicted ratings
        """
        pred = self.model.test(test)
        return accuracy.mae(pred)
