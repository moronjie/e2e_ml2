import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pickle 
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train,y_train,X_test,y_test, models,params):
    try:
        report = {}

        for model_name in models:
            model = models[model_name]
            param_grid = params[model_name]
            
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
            grid_search.fit(X_train, y_train)
            
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            best_estimator = grid_search.best_estimator_
            
            y_pred = best_estimator.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            
            report[model_name] = {
                'best_params': best_params,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2,
                'y_pred': y_pred,
                'best_estimator': best_estimator
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)