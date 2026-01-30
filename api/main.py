from fastapi import FastAPI
import pandas as pd

from api.schemas import CustomerData
from api.model_loader import model, scaler, feature_columns
from src.data_preprocessing import preprocess_data, preprocess_for_inference

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer is likely to churn",
    version="1.0.0"
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

    # debugging
    print("Processed input:")
    print(processed_df.head())

    print("\nNon-zero features:")
    print(processed_df.loc[:, (processed_df != 0).any(axis=0)])


    # Prediction
    churn_prob = model.predict_proba(processed_df)[0][1]
    churn_pred = int(churn_prob >= 0.5)

    return {
        "churn_prediction": churn_pred,
        "churn_probability": round(churn_prob, 3)
    }

