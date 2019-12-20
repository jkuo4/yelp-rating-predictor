# Yelp Rating Predictor

### Overview
This project is to build a recommender to predict Yelp business ratings from users.
- **Baseline-Bias**: One of our baselines to compare all the other recommenders against. It estimates the predicted rating using the biases of the user's and business's ratings.
- **Common Individual Models & Ensemble**: Common memory based and model based approaches including baselineonly, KNN(Basic and With_Means), SVD, co-colustering and a weighted ensemble approach.
- **Factorization Machines**: A more recent and novel linear model approach incorporating pairwise interaction effects and a quick implementation runtime on large datasets with sparse features. 


### Structure
This project is broken down to a few components. Final results and takeaways are summarized in `final_results.ipynb`.

All the training was performed in various `_train.ipynb` notebooks using algorithms in `source/<model>.py` files.
- Basline Model
    - Source code: .source/baseline.py
    - Training code: baseline_train.ipynb
- Individule Model and Ensembling:
    - Source code: .source/ensemble.py
    - Training code: individual_train.ipynb / ensemble_train.ipynb
- FM Model
    - Source code: .source/fm.py / .source/extract.py
    - Train code: fm_train.ipynb

Raw data is stored in `data` dir (dataset removed due to github limitations but can be downloaded at https://www.yelp.com/dataset/challenge)

Results are generated in `data/result` dir.

```bash
.
├── README.md
├── baseline_train.ipynb
├── individual_train.ipynb
├── ensemble_train.ipynb
├── fm_train.ipynb
├── final_results.ipynb
├── setup.cfg
├── result
│   └── <various hyperparameter tuning, prediction, top 20 coverage sets>
├── data
│   ├── fm
│   │   └── <empty>
├── image
│   └── <various notebook images>
└── source
    ├── baseline.py
    ├── ensemble.py
    ├── extract.py
    ├── fm.py
    └── utils.py
```

### Setup
Set up conda env with `conda env create -f environment.yml`
