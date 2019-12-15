# coding: utf-8

import pandas as pd


def add_prefix(cols, prefix):
    """Add prefix to columns except ID columns"""
    return [prefix + col if "id" not in col else col for col in cols]


def read_business():
    """Read US restaurants that are currently open and have at least n reviews"""
    min_n_reviews = 50
    min_per_cat = 0.01

    bus = pd.read_json("data/business.json", lines=True)

    mask = (
        (bus.postal_code.str.len() == 5)
        & bus.categories.str.contains("Restaurants")
        & (bus.review_count >= min_n_reviews)
        & (bus.is_open == 1)
    )
    bus = bus[mask]

    # Extract categories that appear at least min_per_cat and one hot encode
    # Remove categories such as Salvadoran, Hotels, Active Life
    cat_wide = bus.categories.str.get_dummies(sep=", ").drop(columns="Restaurants")
    cat_total = cat_wide.sum()
    selected_cat = cat_total[cat_total / len(cat_wide) > min_per_cat].index.to_list()

    bus = bus[["business_id", "stars", "review_count"]]
    bus.columns = add_prefix(bus.columns, "business_")

    bus_feat = pd.merge(bus, cat_wide[selected_cat], left_index=True, right_index=True)
    return bus_feat


def read_user():
    """Read active users"""
    min_n_reviews = 5

    user = pd.read_json("data/user.json", lines=True)
    user = user[user.review_count >= min_n_reviews]
    user["compliment"] = (
        user.compliment_hot
        + user.compliment_more
        + user.compliment_profile
        + user.compliment_cute
        + user.compliment_list
        + user.compliment_note
        + user.compliment_plain
        + user.compliment_cool
        + user.compliment_funny
        + user.compliment_writer
        + user.compliment_photos
    )
    # Calculate yelping_years
    user.yelping_since = pd.to_datetime(user.yelping_since)
    latest = user.yelping_since.max()
    user["yelping_years"] = (latest - user.yelping_since).dt.days / 360

    # Extract number of years users were on elite status
    user.elite = user.elite.str.split(",").apply(lambda x: len(x) if x[0] != "" else 0)

    cols = [
        "user_id",
        "review_count",
        "elite",
        "fans",
        "average_stars",
        "compliment",
        "yelping_years",
    ]
    user = user[cols]
    user.columns = add_prefix(user.columns, "user_")
    return user


def read_review():
    """Read reviews written by active users"""
    min_n_reviews = 5

    review = pd.read_json("data/review.json", lines=True)
    review["props"] = review.useful + review.funny + review.cool
    review = review.drop(columns=["review_id", "text", "useful", "funny", "cool"])
    review.columns = add_prefix(review.columns, "review_")

    user_counts = review.user_id.value_counts()
    active_users = user_counts[user_counts >= min_n_reviews].index

    review = review[review.user_id.isin(active_users)]
    return review


def create_features():
    """Create features by combining user, business, and review data"""
    review = read_review()
    user = read_user()
    bus = read_business()

    merged = pd.merge(review, user, on="user_id", how="inner")
    merged = pd.merge(merged, bus, on="business_id", how="inner")
    merged.to_csv("data/feature.csv", index=False)


if __name__ == "__main__":
    create_features()
