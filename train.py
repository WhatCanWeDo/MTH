#!/usr/bin/env python3
from preprocessing import prepare_for_training
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split


def train():
    dataset, target = prepare_for_training()
    model_params = {
        'learning_rate': 0.5,
        'depth': 3,
        'iterations': 2000,
        'subsample': 0.5,
        'l2_leaf_reg': 3,
        'colsample_bylevel': 1,
        'random_state': 42,
        'eval_metric': 'RMSE',
        'loss_function': 'RMSE',
    }
    cbr = CatBoostRegressor(**model_params)
    train_size = 0.99
    X_train, X_val, Y_train, Y_val = train_test_split(dataset, target, train_size=train_size)
    cbr.fit(X_train, Y_train,
            cat_features=['user_id', 'event_id'],
            use_best_model=True,
            eval_set=[(X_val, Y_val)],
            verbose=True,
            )
    cbr.save_model('model_dump')


train()
