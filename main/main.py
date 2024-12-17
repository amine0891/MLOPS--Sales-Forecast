import pandas as pd
from fastapi import FastAPI, Body
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from src.data_preprocessing_training import  predict_for_future_date, fill_missing_dates
import mlflow.pyfunc
from dotenv import load_dotenv
import os
from pydantic import BaseModel

load_dotenv("src/.env")
DagsHub_username = os.getenv("DagsHub_username")
DagsHub_token = os.getenv("DagsHub_token")
os.environ['MLFLOW_TRACKING_USERNAME'] = DagsHub_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = DagsHub_token

# setup mlflow
mlflow.set_tracking_uri('https://dagshub.com/mohamedhassine33/MLOPS.mlflow')  # your mlfow tracking uri
mlflow.set_experiment("sales-forecast-experiment1")



class PredictionInput(BaseModel):
    date: str
    product_code: str
    country: str


# let's call the model from the model registry ( in production stage)

df_mlflow = mlflow.search_runs(filter_string="metrics.mse_score_test < 1")
run_id = df_mlflow.loc[df_mlflow['metrics.mse_score_test'].idxmin()]['run_id']

logged_model = f'runs:/{run_id}/ML_models'

# Load model as a PyFuncModel.
model = mlflow.pyfunc.load_model(logged_model)
data_url = "historical_data/complete-data.csv"
historical_data = pd.read_csv(data_url,index_col="Date")

app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "to sales forecast app version 1"}


# this endpoint predicts sales number for a specific date , country and product
@app.post("/predict")
def return_predictions(
        prediction_input: PredictionInput = Body(...)
):
    # Add predict fields
    target_date = pd.to_datetime(prediction_input.date)
    product_code =  prediction_input.product_code
    country = prediction_input.country
    
    # Predictions
    forecast = predict_for_future_date(historical_data, product_code, country, target_date, model)

    return {"prediction": round(forecast[-1].get("predicted_sales"))}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=9000)
