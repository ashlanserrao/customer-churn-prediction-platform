from fastapi import FastAPI
import pandas as pd
import logging
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import CustomerData
from api.model_loader import model, scaler, feature_columns
from src.data_preprocessing import preprocess_data, preprocess_for_inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer is likely to churn",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    # convert input to DataFrame
    input_df = pd.DataFrame([customer.dict()])
    '''
    debugging
    # Apply preprocessing (inference mode)
    processed_df, _ = preprocess_data(input_df, scaler=scaler)

    # Align columns with training features
    processed_df = processed_df.reindex(columns=feature_columns, fill_value=0)
    '''
    processed_df = preprocess_for_inference(
        input_df, scaler=scaler, feature_columns=feature_columns
    )

    # Do guard against corrupt rows
    if processed_df.isnull().any().any():
        return {
            "error": "Invalid input after preprocessing"
        }

    # Prediction
    churn_prob = model.predict_proba(processed_df)[0][1]
    churn_pred = int(churn_prob >= 0.5)

    # Threshold can be tuned based on business preference
    # e.g. , lower threshold to increase recall

    logger.info(f"Prediction made | Probability: {churn_prob:.3f}")

    return {
        "churn_prediction": churn_pred,
        "churn_probability": round(float(churn_prob), 3)
    }

