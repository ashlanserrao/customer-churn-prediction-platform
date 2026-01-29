import pandas as pd
from src.config import RAW_DATA_PATH

def load_raw_data() -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn dataset
    returns:
    pd.DataFrame: Raw churn dataset
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_DATA_PATH}")
    
    df = pd.read_csv(RAW_DATA_PATH)
    return df