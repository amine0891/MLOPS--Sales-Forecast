import pandas as pd
import mlflow
import xgboost as xgb
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

def VARModel(product_name,data_url,version,df,train_split,test_split):

    # disable autologging
    mlflow.sklearn.autolog(disable=True)
    with mlflow.start_run(run_name=f'VAR-{product_name}'):
        mlflow.log_param("data_url",data_url)
        mlflow.log_param("data_version",version)
        mlflow.log_param("input_rows",df.shape[0])
        mlflow.log_param("input_cols",df.shape[1])
        #model fitting and training
        var = VAR(train_split)
        mlflow.set_tag(key= "model",value="VAR")
        #params = var.get_params()
        #mlflow.log_params(params)
        var = var.fit(maxlags=3, ic='aic')
        forecast = var.forecast(train_split.values[-var.k_ar:], steps=10)
        forecast_dates = pd.date_range(start=train_split.index[-1] + pd.Timedelta(days=1), periods=10)
        forecast_df = pd.DataFrame(forecast, index=forecast_dates, columns=train_split.columns)
        mse = mean_squared_error(test_split[:10], forecast_df)
        mlflow.log_metric("mse_score_test",mse)
        mlflow.sklearn.log_model(var,artifact_path="ML_models")


def XGBoostModel(product_name,data_url,version,df,X_train,y_train,X_test,y_test):

    # disable autologging
    mlflow.xgboost.autolog(disable=True)
    with mlflow.start_run(run_name=f'XGBoost-{product_name}'):
        mlflow.log_param("data_url",data_url)
        mlflow.log_param("data_version",version)
        mlflow.log_param("input_rows",df.shape[0])
        mlflow.log_param("input_cols",df.shape[1])
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01)
        #params = xgb.get_params()
        mlflow.set_tag(key= "model", value="XGBClassifier")
        #mlflow.log_params(params)
        xgb_model.fit(X_train, y_train)
        y_pred=xgb_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric("mse_score_test",mse)
        mlflow.xgboost.log_model(xgb_model,artifact_path="ML_models")

