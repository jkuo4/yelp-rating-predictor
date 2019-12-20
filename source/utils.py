# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def train_test_split_feature(feature, sample_n_users=None):
    """Split feature to train and split to predict the last rating for each user"""

    if sample_n_users:
        users = pd.Series(feature.user_id.unique())
        sampled_users = users.sample(sample_n_users, random_state=1)
        feature = feature[feature.user_id.isin(sampled_users)].copy()

    feature.review_date = pd.to_datetime(feature.review_date)
    last_review = feature.sort_values("review_date").groupby("user_id").tail(1)
    other_review = feature.drop(last_review.index)

    return other_review, last_review


def rmse(y_true, y_pred):
    """Calculate RMSE which is not implemented in sklearn"""
    return mean_squared_error(y_true, y_pred) ** 0.5


def rmse_scorer():
    """Make RMSE scorer"""
    return make_scorer(rmse, greater_is_better=False)


def plot_lines(x, y, title, x_lab, y_lab, legend_lab=None):
    """
    Output a line chart

    :param x: pandas Series of independant variable
    :param y: pandas Series of dependant variable
    :param title: string, chart title
    :param x_lab: string, chart x label
    :param y_lab: string, chart y label
    :param legend_lab: list, labels for groups used in the legend
    :return : pyplot line chart
    """

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y)
    step_size = (max(x) - min(x)) / (len(x) - 1)
    ax.set_xticks(np.arange(min(x), max(x) + step_size, step_size))
    ax.set_title(title)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    if step_size < 1:
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if legend_lab:
        ax.legend(legend_lab, loc="center left", bbox_to_anchor=(1, 0.5))


def create_quantile_bucket(ranking_measure, metric, n_tile, sort_asc=True):
    """
    Output the valuation metric distribution of the ranking measure

    :param ranking_measure: pandas Series containing associated ranking measure
    :param metric: pandas Series containing associated valuation metric per ID
    :param n_tile: int, number of buckets to split data
    :param sort_asc: bool, direction to rank the data in the Quantile
    :return : pyplot chart of the valuation metric per n_tile of data
    """

    measure_ranked = ranking_measure.rank(method="first", ascending=sort_asc)
    measure_quantile = pd.Series(pd.qcut(measure_ranked, n_tile, labels=False))
    measure_quantile.name = "Quantile"
    df = pd.concat([measure_quantile, metric], axis=1).reset_index()
    df_grouped = df.groupby("Quantile").mean().reset_index()
    return df_grouped.iloc[:, 2]


def calculate_catalog_coverage(y_train, y_test, y_pred):
    """
    Calculate the fraction of items that are in the top-k for at least 1 user

    :param y_train: Pandas dataframe containing business_id for training
    :param y_test: Pandas dataframe containing business_id for testing
    :param y_pred: Pandas dataframe containing business_id for prediction at k

    :return :  catalog coverage in float
    """
    num_recommended_items = y_pred.business_id.nunique()
    num_total_items = pd.concat([y_test, y_train]).business_id.nunique()
    return num_recommended_items / num_total_items
