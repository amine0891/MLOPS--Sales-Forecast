import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score

import mlflow

import warnings
from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments
import sys 
from data_preprocessing_monitoring import transform_data,preprocess_data
from dotenv import load_dotenv
import os
import mlflow.pyfunc
import uuid
warnings.filterwarnings("ignore")

version = "v2.0"
data_url = "../data/Cleaned_Sales_Data.csv"

 
sys.path.insert(0, '../main/src')


load_dotenv("../main/src/.env")

DagsHub_username = os.getenv("DagsHub_username")
DagsHub_token=os.getenv("DagsHub_token")

os.environ['MLFLOW_TRACKING_USERNAME']= DagsHub_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = DagsHub_token

#setup mlflow
mlflow.set_tracking_uri('https://dagshub.com/mohamedhassine33/MLOPS.mlflow') #your mlfow tracking uri
mlflow.set_experiment("fraud-detector-experiment")

#read the data
raw_train = pd.read_excel(data_url)

#Reading Pandas Dataframe from mlflow
all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
df_mlflow = mlflow.search_runs(experiment_ids=all_experiments,filter_string="metrics.F1_score_test <1")
run_id = df_mlflow.loc[df_mlflow['metrics.F1_score_test'].idxmax()]['run_id']

#let's call the model from the model registry ( in production stage)


logged_model = f'runs:/{run_id}/ML_models'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
print(loaded_model)


baseline = raw_train


baseline["Date"] = pd.to_datetime(baseline['Date'], format='%d-%m-%Y')
baseline = baseline.set_index("Date")

baseline= transform_data(baseline)
baseline = preprocess_data(baseline)
X = baseline.iloc[:-20].drop('sales_number')

preds = loaded_model.predict(X)

baseline['prediction_label'] = preds


# Prediction ID is required for all datasets
def generate_prediction_ids(X):
    return pd.Series((str(uuid.uuid4()) for _ in range(len(X))), index=X.index)

baseline["prediction_id"]=generate_prediction_ids(baseline)


baseline["prediction_id"]=generate_prediction_ids(baseline)


load_dotenv(".env")
SPACE_KEY =os.getenv("SPACE_KEY")
API_KEY = os.getenv("API_KEY")


arize_client = Client(space_key=SPACE_KEY, api_key=API_KEY)

model_id = (
    "Sales-Forecast-detector-model"  # This is the model name that will show up in Arize
)
model_version = "v2"  # Version of model - can be any string

if SPACE_KEY == "SPACE_KEY" or API_KEY == "API_KEY":
    raise ValueError("❌ NEED TO CHANGE SPACE AND/OR API_KEY")
else:
    print("✅ Arize setup complete!")

    features = feature_column_names=list(baseline.columns.drop(
        ["prediction_id", "prediction_label", "actual_label"]))
    
    # Define a Schema() object for Arize to pick up data from the correct columns for logging
training_schema = Schema(
    prediction_id_column_name="prediction_id",
    prediction_label_column_name="prediction_label",
    actual_label_column_name="actual_label",
    feature_column_names=features)

# Logging Training DataFrame
training_response = arize_client.log(
    dataframe=baseline,
    model_id=model_id,
    model_version=model_version,
    model_type=ModelTypes.SCORE_CATEGORICAL,
    environment=Environments.TRAINING,
    schema=training_schema,
)

# If successful, the server will return a status_code of 200
if training_response.status_code != 200:
    print(
        f"logging failed with response code {training_response.status_code}, {training_response.text}"
    )
else:
    print(f"✅ You have successfully logged training set to Arize")