from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def params_gridsearch(x_train, 
                      y_train,
                      max_sample_size = 20000,
                      param_grid={'n_estimators': [10, 25, 50, 75, 100, 150, 200],
                                'max_depth': [3, 5, 7, 9, 11, 15],
                                'learning_rate' : [0.01, 0.1, 0.2, 0.5, 1]}
                                ):

    if len(x_train)>max_sample_size:
        to_gridsearch_x = x_train.sample(max_sample_size)
        to_gridsearch_y = y_train.loc[to_gridsearch_x.index]
    else:
        to_gridsearch_x = x_train
        to_gridsearch_y = y_train

    param_grid = param_grid

    xgb = XGBRegressor(n_jobs=-1, importance_type='total_gain', learning_rate = 0.5)

    random_search = GridSearchCV(
        xgb,
        param_grid=param_grid,
        scoring='r2',
        cv=3,
        verbose=2,
        return_train_score=True,
    )

    random_search.fit(to_gridsearch_x, to_gridsearch_y)

    # Best model and parameters
    best_model = random_search.best_estimator_
    print("Best Parameters:", random_search.best_params_)

    return best_model

