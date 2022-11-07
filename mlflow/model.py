### Imports

import mlflow
from mlflow.models.signature import infer_signature

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score


if __name__ == "__main__":
    # Set the MLflow tracking environment
   
    # Loading dataset

    getaround_price = pd.read_csv('get_around_pricing_project.csv')



    # Remove outliers

    getaround_price = getaround_price[getaround_price['mileage'] > 0]

    getaround_price = getaround_price[getaround_price['engine_power'] > 0]


    # Remove useless columns


    # Paint_color and Unnamed

    getaround_price = getaround_price.drop(columns=['paint_color'])

    getaround_price = getaround_price.drop(columns=['Unnamed: 0']).reset_index(drop=True)



    # Preprocessing


    # Separate Numerical and categorical features

    features = getaround_price.columns.to_list()
    target = features.pop(features.index('rental_price_per_day'))
    cat_features = features.copy()
    num_features = [cat_features.pop(cat_features.index('mileage'))]
    num_features.append(cat_features.pop(cat_features.index('engine_power')))

    # Identify target and other features

    X = getaround_price.loc[:, features]
    Y = getaround_price.loc[:, target]
    print('X = ',X.head())
    print('Y = ',Y.head())

    # Train set and Test set 

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Pipeline for the numerical features

    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Pipeline for the categorical features

    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first'))
    ])

    #  ColumnTransformer (Numerical transformer and categorical tranformer)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ]
    )

    # Preprocess data

    X_train_prepro = preprocessor.fit_transform(X_train)
    X_test_prepro = preprocessor.transform(X_test)



    # Models
    
    # Linear regression model

    lin_model = LinearRegression()
    lin_model.fit(X_train_prepro, Y_train)



    # Ridge regression model 


    # Grid search
    params_ridge = {'alpha': [0.1, 1, 10, 100, 1000]}
    grid_ridge = GridSearchCV(Ridge(), params_ridge, cv=3)
    grid_ridge.fit(X_train_prepro, Y_train)
    print('Best parameters for lin: ', grid_ridge.best_params_)


    # Random forest regression model


    # Grid search
    params_rf = {'n_estimators': [10, 100, 1000],
                'max_depth': [2, 4, 6, 8, 10],
                'max_features': [1.0, 'sqrt', 'log2'],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],}
    grid_rf = GridSearchCV(RandomForestRegressor(), params_rf, cv=3)
    grid_rf.fit(X_train_prepro, Y_train)
    print('Best parameters for rf: ', grid_rf.best_params_)

    # Gradient boosting regression model


    # Grid search
    params_gbr = {'n_estimators': [10, 100, 1000],
                'max_depth': [2, 4, 6, 8, 10],
                'max_features': [1.0, 'sqrt', 'log2'],
                'subsample': [0.5, 0.75, 1],
                'learning_rate': [0.1, 0.5, 1]
                }
    grid_gbr = GridSearchCV(GradientBoostingRegressor(), params_gbr, cv=3)
    grid_gbr.fit(X_train_prepro, Y_train)
    print('Best parameters for gbr: ', grid_gbr.best_params_)

    # AdaBoost regression model


    # Grid search
    params_ada = {'n_estimators': [10, 100, 1000],
                'learning_rate': [0.1, 0.5, 1],
                'loss' : ['linear']
                }
    grid_ada = GridSearchCV(AdaBoostRegressor(), params_ada, cv=3)
    grid_ada.fit(X_train_prepro, Y_train)
    print('Best parameters for ada: ', grid_ada.best_params_)

    # Voting regression model with a pipeline


    voting_regressor = Pipeline(steps=[
                                        ('preprocessor', preprocessor),
                                        ('vote', VotingRegressor(estimators=[('lin', lin_model),
                                                        ('ridge', grid_ridge.best_estimator_),
                                                        ('rf', grid_rf.best_estimator_),
                                                        ('gbr', grid_gbr.best_estimator_),
                                                        ('ada', grid_ada.best_estimator_)
                                                        ]))
                                    ]
    )




    # Log Voting Regressor model to MLflow


    # Set your variables for your environment


    EXPERIMENT_NAME="voting-regressor-experiment"

    # Set experiment's info 

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Get our experiment info

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)


    client = mlflow.tracking.MlflowClient()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    run = client.create_run(experiment.experiment_id) # Creates a new run for a given experiment

    # Call mlflow autolog
    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run(run_id=run.info.run_id) as run:
    
        # Fit the model

        voting_regressor.fit(X_train, Y_train)  # Fit the model
        predictions = voting_regressor.predict(X_train)  # Predict on the training set

        # Store metrics 
        
        # Log model seperately to have more flexibility on setup 

        mlflow.sklearn.log_model(
            sk_model= voting_regressor,
            artifact_path="pricing-model",
            registered_model_name="voting-regressor-model",
            signature=infer_signature(X_train, predictions) # Infer signature to tell what should be as model's inputs and outputs
        )

        # Print results 
        
        print("Voting Regressor Model")