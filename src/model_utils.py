import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE

def split_features_target(df: pd.DataFrame, target: str = "Churn"):
    """
    Split dataframe into features and target
    """
    X = df.drop(columns=[target])
    y = df[target]

    return X, y

def train_test_split_data(X, y, test_size: float = 0.2):
    """
    Perform Stratified train-test split
    """

    return train_test_split(X, y, test_size=test_size, 
                            random_state=RANDOM_STATE,
                            stratify = y)

