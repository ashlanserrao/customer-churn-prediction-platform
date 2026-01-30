import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import PROCESSED_DATA_PATH

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the Telco churn dataset
    
    Steps:
    - Convert TotalCharges to numeric
    - Handle missing values
    - Encode categorical variables
    - Scale numerical features
    """

    df = df.copy()

    # Converting TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors = "coerce")

    # Dropping any rows with missing totalcharges
    df = df.dropna(subset=["TotalCharges"])

    # Convert the target variable to binary (0 or 1)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop customerID (its not a feature so we drop)
    df = df.drop(columns=["customerID"])

    # Now we seperate numerical and categorical columns
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = [col for col in df.columns if col not in numerical_cols + ["Churn"]]

    # One hot encoding categorical variables
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

def save_processed_data(df: pd.DataFrame, filename: str = "processed_churn.csv") -> None:
    """
    Save processed dataset to disk
    """
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_PATH / filename
    df.to_csv(output_path, index=False)
    