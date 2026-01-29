from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data paths
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

# Random seed for reproducibility
RANDOM_STATE = 42