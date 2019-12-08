# Yelp Rating Predictor

### Overview
This project is to build a recommender to predict Yelp business ratings from active users.
- **Baseline-Bias**: One of our baselines to compare all the other recommenders against. It estimates the predicted rating using the biases of the user's and business's ratings.
- **Baseline-CF**: One of our baselines to compare all the other recommenders against. It uses a <model>-based collaborative filtering (<matrix factorization>) to predict ratings.


### Structure
This project is broken down to a few components. Final results are summarized in `final_recommendation_results.ipynb`.
All the training was performed in `model_recommendation.ipynb` using algorithms in `source/model_recommender.py` files.
Data is stored in `data` dir (dataset removed due to github limitations but can be downloaded at https://www.yelp.com/dataset/challenge) and results are generated in `data/results` dir.

```bash
.
├── README.md
├── baseline_recommendation.ipynb
├── final_recommendation_results.ipynb
├── data
│   ├── results
│   │   ├── temp.pkl
│   │   └── temp.pkl
├── environment.yml
├── images
│   ├── temp.png
│   └── temp.png
└── source
    ├── baseline_recommender.py
    ├── cf_recommender.py
    └── utils.py
```

### Setup
Install Spark version 2.4.4 and then set up conda env with `conda env create -f environment.yml`
