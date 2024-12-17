# from dotenv import load_dotenv
import os

from data_preprocessing_training import process_and_split_data
import pandas as pd
from dotenv import load_dotenv
import mlflow
from treatement import XGBoostModel, MLPRegressorModel

# Affect Daghsub credentials

load_dotenv("src/.env")
DagsHub_username = os.getenv("DagsHub_username")
DagsHub_token = os.getenv("DagsHub_token")
os.environ['MLFLOW_TRACKING_USERNAME'] = DagsHub_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = DagsHub_token

# setup mlflow
mlflow.set_tracking_uri('https://dagshub.com/mohamedhassine33/MLOPS.mlflow')
mlflow.set_experiment("sales-forecast-experiment1")

# Data Url and version
version = "v1.0"
data_url = "../data/Cleaned_Sales_Data.xlsx"

# read the data
df = pd.read_excel(data_url)
# cleaning and preprocessing
X_train, X_test, y_train, y_test = process_and_split_data(df)
XGBoostModel(data_url, version, df, X_train, X_test, y_train, y_test)
MLPRegressorModel(data_url, version, df, X_train, X_test, y_train, y_test)

