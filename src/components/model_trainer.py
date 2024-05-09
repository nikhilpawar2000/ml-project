import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
   trained_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('splitting train and test input data')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model={
                'RandomForest':RandomForestRegressor(),
                'DecisionTree': DecisionTreeRegressor(),
                'GradientBoosting':GradientBoostingRegressor(),
                'LinearRegression':LinearRegression(),
                'AdaBoostRegressor':AdaBoostRegressor(),
                'KNeighborsRegressor':KNeighborsRegressor(),
                'Xgboost':XGBRegressor(),
                'catboost':CatBoostRegressor(verbose=False),
                }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=model)
            best_model_score=max(sorted(model_report.values()))  # to get best model score from dict

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]  # to get best model name from dict
            best_model=model[best_model_name]

            if best_model_score<0.6:
                raise CustomException('best model not found')
            logging.info('best found model on both train and test dataset')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)

