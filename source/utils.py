# coding: utf-8

import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error


def train_test_split_feature(feature, sample_n_users=None):
    """Split feature to train and split to predict the last rating for each user"""

    if sample_n_users:
        sampled_users = pd.Series(feature.user_id.unique()).sample(sample_n_users)
        feature = feature[feature.user_id.isin(sampled_users)].copy()

    y_col = "review_stars"
    feature.review_date = pd.to_datetime(feature.review_date)

    last_review = feature.sort_values("review_date").groupby("user_id").tail(1)
    other_review = feature.drop(last_review.index)

    drop_cols = [y_col, "user_id", "review_date"]
    X_test = last_review.drop(columns=drop_cols)
    y_test = last_review[y_col]

    X_train = other_review.drop(columns=drop_cols)
    y_train = other_review[y_col]

    return X_train, X_test, y_train, y_test


def rmse(y_true, y_pred):
    """Calculate RMSE which is not implemented in sklearn"""
    return mean_squared_error(y_true, y_pred) ** 0.5


def rmse_scorer():
    """Make RMSE scorer"""
    return make_scorer(rmse, greater_is_better=False)
