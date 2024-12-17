import pandas as pd
import mlflow
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")


def MLPRegressorModel(data_url, version, df, X_train, X_test, y_train, y_test):
    # disable autologging
    mlflow.sklearn.autolog(disable=True)
    with mlflow.start_run(run_name=f'MLPRegressor'):
        mlflow.log_param("data_url", data_url)
        mlflow.log_param("data_version", version)
        mlflow.log_param("input_rows", df.shape[0])
        mlflow.log_param("input_cols", df.shape[1])
        # model fitting and training
        mlpr = MLPRegressor(hidden_layer_sizes=(10, 4), activation='relu', solver='adam')
        mlflow.set_tag(key="model", value="MLPRegressor")
        # params = var.get_params()
        # mlflow.log_params(params)
        mlpr.fit(X_train, y_train)
        y_pred = mlpr.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric("mse_score_test", mse)
        mlflow.sklearn.log_model(mlpr, artifact_path="ML_models")


def XGBoostModel(data_url, version, df, X_train, X_test, y_train, y_test):
    # disable autologging
    mlflow.xgboost.autolog(disable=True)
    with mlflow.start_run(run_name=f'XGBoost'):
        mlflow.log_param("data_url", data_url)
        mlflow.log_param("data_version", version)
        mlflow.log_param("input_rows", df.shape[0])
        mlflow.log_param("input_cols", df.shape[1])
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01)
        # params = xgb.get_params()
        mlflow.set_tag(key="model", value="XGBClassifier")
        # mlflow.log_params(params)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric("mse_score_test", mse)
        mlflow.xgboost.log_model(xgb_model, artifact_path="ML_models")
