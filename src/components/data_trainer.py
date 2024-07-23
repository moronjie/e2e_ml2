import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logger
from src.exception import CustomException

from src.utils import save_object, evaluate_models


@dataclass
class Model_trainer_config:
    trained_model_file_path=os.path.join("data","model.pkl")

class Model_trainer:
    def __init__(self) -> None:
        self.data_trainer_config = Model_trainer_config()

    def initiate_model_trainer(self, train_data, test_data):
        try:
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            X_train,y_train,X_test,y_test=(
                train_data[:,:-1],
                train_data[:,-1],
                test_data[:,:-1],
                test_data[:,-1]
            )

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            results:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            

            for model_name in results:
                print(f"Model: {model_name}")
                print(f"Best Parameters: {results[model_name]['best_params']}")
                print(f"MAE: {results[model_name]['mae']}")
                print(f"MSE: {results[model_name]['mse']}")
                print(f"RMSE: {results[model_name]['rmse']}")
                print(f"R² Score: {results[model_name]['r2_score']}")
                print("-" * 30)

                # Plot residuals
                # residuals = y_test - results[model_name]['y_pred']
                # plt.figure(figsize=(10, 7))
                # sns.scatterplot(x=results[model_name]['y_pred'], y=residuals)
                # plt.axhline(y=0, color='r', linestyle='--')
                # plt.title(f"Residuals Plot for {model_name}")
                # plt.xlabel('Predicted Values')
                # plt.ylabel('Residuals')
                # plt.show()

            # Find the best model based on R² score
            best_model_name = max(results, key=lambda x: results[x]['r2_score'])
            best_model = results[best_model_name]['best_estimator']
            
            if results[best_model_name]['r2_score'] < 0.7:
                raise CustomException("The R² score of the best model is less than 0.7. The model is not good.")

            print(f"The best model is {best_model_name} with R² score of {results[best_model_name]['r2_score']}")

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            
            return best_model_name, best_model

        except Exception as e:
            CustomException(e)