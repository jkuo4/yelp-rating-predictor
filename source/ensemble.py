"""
Ensemble several base models to create a mixed model to do prediction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression
from surprise import SVD, AlgoBase, BaselineOnly, KNNWithMeans, accuracy
from surprise.model_selection import split
from surprise.prediction_algorithms.co_clustering import CoClustering


class EnsembleRecommender(AlgoBase):
    """
        Ensemble Recommendation Class
    """
    def __init__(
        self, train_data, model_to_use=["baselineonly", "svd", "coClustering", "knn"]
    ):
        """initialize class with full dataset and a set of base models to use"""
        AlgoBase.__init__(self)
        self.available_models = {
            "baselineonly": BaselineOnly(
                bsl_options={"method": "sgd", "n_epochs": 30, "reg": 0.1, "learning_rate": 0.005}
            ),
            "svd": SVD(lr_all=0.005, n_factors=50, reg_all=0.1),
            "coClustering": CoClustering(n_epochs=3, n_cltr_u=3, n_cltr_i=3),
            "knn": KNNWithMeans(k=40, sim_options={"name": "cosine", "user_based": False}),
        }
        self.model_selection = []
        for model in model_to_use:
            self.model_selection.append([model, self.available_models[model]])
        self.model_rmse = {}
        self.model_mae = {}
        self.model_list = {}
        self.trainset = train_data.build_full_trainset()

    def fit(
        self, train_data, ensemble_method="LR:plain", retrain=True, retrain_split_num=2
    ):
        """fit the base models and ensemble with choices of weights"""
        kSplit = split.KFold(n_splits=retrain_split_num, shuffle=True)
        if retrain:
            print("**************** Start retraining models ******************")
            for model_name, model in self.model_selection:
                print(f"*************** Retraining: {model_name} *****************")
                for trainset, testset in kSplit.split(train_data):
                    model.fit(trainset)
                    model_prediction = model.test(testset)
                    if model_name not in self.model_rmse:
                        self.model_rmse[model_name] = [
                            accuracy.rmse(model_prediction, verbose=True)
                        ]
                        self.model_mae[model_name] = [
                            accuracy.mae(model_prediction, verbose=True)
                        ]
                    else:
                        self.model_rmse[model_name].append(
                            accuracy.rmse(model_prediction, verbose=True)
                        )
                        self.model_mae[model_name].append(
                            accuracy.mae(model_prediction, verbose=True)
                        )
                self.model_list[model_name] = model
            print("******************** Models retrained *********************")
        if ensemble_method[:2] == "LR":
            print("*** Starting tuning hyperparameter for stacking weights ***")
            train_data = train_data.build_full_trainset().build_testset()
            prediction_stack = []
            for model in self.model_list:
                pred = self.model_list[model].test(train_data)
                rating = pd.DataFrame(
                    pred,
                    columns=[
                        "uid",
                        "iid",
                        "rating",
                        f"predicted_rating_{model}",
                        "details",
                    ],
                )[["rating"]]
                pred_df = pd.DataFrame(
                    pred,
                    columns=[
                        "uid",
                        "iid",
                        "rating",
                        f"predicted_rating_{model}",
                        "details",
                    ],
                )[[f"predicted_rating_{model}"]]
                if not prediction_stack:
                    prediction_stack.append(rating)
                prediction_stack.append(pred_df)
            prediction_stacking = pd.concat(prediction_stack, axis=1)
            if ensemble_method == "LR:ElasticNet":
                reg = ElasticNet().fit(
                    prediction_stacking.iloc[:, 1:], prediction_stacking.iloc[:, 0]
                )
                self.intercept = reg.intercept_
                self.weights = np.array(reg.coef_)
            else:
                reg = LinearRegression().fit(
                    prediction_stacking.iloc[:, 1:], prediction_stacking.iloc[:, 0]
                )
                self.intercept = reg.intercept_
                self.weights = np.array(reg.coef_)
        else:
            self.intercept = 0
            self.weights = np.array(
                [
                    1.0 / len(self.model_selection)
                    for _ in range(len(self.model_selection))
                ]
            )

    def estimate(self, u, i):
        """estimate rating for given user and item"""
        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            algoResults = np.array(
                [
                    self.model_list[model].predict(
                        self.trainset.to_raw_uid(u), self.trainset.to_raw_iid(i)
                    )[3]
                    for model in self.model_list
                ]
            )
            new_pred = self.intercept + np.sum(np.dot(self.weights, algoResults))
            rounding_pred = new_pred
            # if new_pred >= 4.75:
            #     rounding_pred = 5
            # elif new_pred >= 4.25:
            #     rounding_pred = 4.5
            # elif new_pred > 3.75:
            #     rounding_pred = 4
            # elif new_pred >= 3.25:
            #     rounding_pred = 3.5
            # elif new_pred >= 3.25:
            #     rounding_pred = 3.5
            # elif new_pred > 2.75:
            #     rounding_pred = 3
            # elif new_pred >= 2.25:
            #     rounding_pred = 2.5
            # elif new_pred > 1.75:
            #     rounding_pred = 2
            # elif new_pred >= 1.25:
            #     rounding_pred = 1.5
            # else:
            #     rounding_pred = 1
            details = {
                "raw_predictions": algoResults,
                "weights": self.weights,
                "intercept": self.intercept,
                "new_prediction": new_pred,
            }
            return rounding_pred, details
        return None

    def rmse(self, testset):
        """Calculate the RMSE of given dataset"""
        pred = self.model.test(testset.build_full_trainset().build_testset())
        return accuracy.rmse(pred)

    def mae(self, testset):
        """Calculate the MAE of given dataset"""
        pred = self.model.test(testset.build_full_trainset().build_testset())
        return accuracy.mae(pred)
