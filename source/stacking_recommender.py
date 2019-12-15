# Stacking several recommendataion models to create hybrid model
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from surprise import SVD, AlgoBase, BaselineOnly, KNNBasic, accuracy
from surprise.model_selection import split
from surprise.prediction_algorithms.co_clustering import CoClustering


class StackingModel(AlgoBase):
    def __init__(self, train_data):
        AlgoBase.__init__(self)
        self.model_selection = [
            [
                "baselineonly",
                BaselineOnly(
                    bsl_options={
                        "method": "als",
                        "n_epochs": 25,
                        "reg_u": 5,
                        "reg_i": 3,
                    }
                ),
            ],
            ["svd", SVD(lr_all=0.01, n_epochs=25, reg_all=0.2)],
            ["coClustering", CoClustering(n_epochs=3, n_cltr_u=3, n_cltr_i=3)],
            [
                "knn",
                KNNBasic(k=40, sim_options={"name": "cosine", "user_based": False}),
            ],
        ]
        self.model_rmse = {}
        self.model_list = {}
        self.trainset = train_data.build_full_trainset()

    def fit(self, train_data, retrain=True, retrain_split_num=2):
        kSplit = split.KFold(n_splits=retrain_split_num, shuffle=True)
        if retrain:
            print("**************** Start retraining models ******************")
            for model_name, model in self.model_selection:
                print(f"*************** Retraining: {model_name} *****************")
                for trainset, testset in kSplit.split(train_data):
                    model.fit(trainset)
                    model_prediction = model.test(testset)
                    self.model_rmse[model_name] = accuracy.rmse(
                        model_prediction, verbose=True
                    )
                self.model_list[model_name] = model
            print("******************** Models retrained *********************")
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
        reg = LinearRegression(fit_intercept=False).fit(
            prediction_stacking.iloc[:, 1:], prediction_stacking.iloc[:, 0]
        )
        self.weights = np.array(reg.coef_)

    def estimate(self, u, i):
        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            algoResults = np.array(
                [
                    self.model_list[model].predict(
                        self.trainset.to_raw_uid(u), self.trainset.to_raw_iid(i)
                    )[3]
                    for model in self.model_list
                ]
            )
            details = {"raw_predictions": algoResults, "weights": self.weights}
            return np.sum(np.dot(self.weights, algoResults)), details
        return None
