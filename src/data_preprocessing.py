import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import PROCESSED_DATA_PATH
import joblib

def preprocess_data(df: pd.DataFrame, scaler: StandardScaler = None):
    """
    Clean and preprocess the Telco churn dataset
    
    If scaler is a None we fit new scaler (training)
    If scaler exists we reuse scaler (inference)
    """

    df = df.copy()

    # Converting TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors = "coerce")

    # Dropping any rows with missing totalcharges
    df = df.dropna(subset=["TotalCharges"])

    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})


    # Drop customerID (its not a feature so we drop)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Now we seperate numerical and categorical columns
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = [col for col in df.columns if col not in numerical_cols + ["Churn"]]

    # One hot encoding categorical variables
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Scale numerical features
    if scaler is None:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    else:
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, scaler

def save_processed_data(df: pd.DataFrame, filename: str = "processed_churn.csv") -> None:
    """
    Save processed dataset to disk
    """
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_PATH / filename
    df.to_csv(output_path, index=False)
    
def save_artifact(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def preprocess_for_inference(df: pd.DataFrame, scaler, feature_columns):
    """
    Preprocess single input for inference ensuring feature alignment.
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = [c for c in df.columns if c not in numerical_cols]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    #  CRITICAL STEP: align columns BEFORE scaling
    df = df.reindex(columns=feature_columns, fill_value=0)

    df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df


