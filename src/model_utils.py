import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE
from sklearn.metrics import (
    accuracy_score, 
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score, 
    classification_report
)

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

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate model using multiple metrics
    """

    y_pred = model.predict(X_test)

    # some models support predict_proba
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc
    }
