"""
Preprocessing Utilities for Indian Agricultural Advisory System
Handles data loading, categorical encoding, scaling, imputing, and feature validation.
"""

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Canonical Feature Schema
CATEGORICAL_FEATURES = ["State", "District or Region", "Soil Type", "Season"]
NUMERICAL_FEATURES = [
    "Temperature",
    "Humidity",
    "Rainfall",
    "Soil pH",
    "Nitrogen (N)",
    "Phosphorus (P)",
    "Potassium (K)"
]
TARGET_FEATURE = "Crop Name"

# Typical default NPK values based on Indian agro-climatic averages when unmeasured
DEFAULT_NPK_BY_SOIL = {
    "Black soil": {"Nitrogen (N)": 60.0, "Phosphorus (P)": 45.0, "Potassium (K)": 50.0},
    "Red soil": {"Nitrogen (N)": 50.0, "Phosphorus (P)": 40.0, "Potassium (K)": 40.0},
    "Alluvial soil": {"Nitrogen (N)": 80.0, "Phosphorus (P)": 55.0, "Potassium (K)": 45.0},
    "Laterite soil": {"Nitrogen (N)": 55.0, "Phosphorus (P)": 35.0, "Potassium (K)": 45.0},
    "Sandy soil": {"Nitrogen (N)": 40.0, "Phosphorus (P)": 30.0, "Potassium (K)": 30.0},
    "Clay soil": {"Nitrogen (N)": 75.0, "Phosphorus (P)": 50.0, "Potassium (K)": 55.0},
    "Loamy soil": {"Nitrogen (N)": 70.0, "Phosphorus (P)": 50.0, "Potassium (K)": 50.0}
}

def load_crop_dataset(dataset_path=None):
    """
    Loads and validates the main agricultural dataset.
    """
    if dataset_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = os.path.join(base_dir, "datasets", "indian_crop_dataset.csv")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Crop dataset not found at {dataset_path}")
        
    df = pd.read_csv(dataset_path)
    # Strip whitespace in string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        
    return df

def build_preprocessor():
    """
    Builds a scikit-learn ColumnTransformer for categorical and numerical features.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, NUMERICAL_FEATURES),
        ("cat", cat_pipeline, CATEGORICAL_FEATURES)
    ])
    
    return preprocessor

def format_user_input(state, district, soil_type, season, temp, humidity, rainfall, ph, n=None, p=None, k=None):
    """
    Transforms individual farmer inputs into a DataFrame matching training schema.
    Applies standard defaults for missing or unmeasured NPK if necessary.
    """
    soil_key = soil_type if soil_type in DEFAULT_NPK_BY_SOIL else "Loamy soil"
    defaults = DEFAULT_NPK_BY_SOIL.get(soil_key, {"Nitrogen (N)": 60.0, "Phosphorus (P)": 45.0, "Potassium (K)": 45.0})
    
    n_val = float(n) if n is not None and not np.isnan(float(n)) else defaults["Nitrogen (N)"]
    p_val = float(p) if p is not None and not np.isnan(float(p)) else defaults["Phosphorus (P)"]
    k_val = float(k) if k is not None and not np.isnan(float(k)) else defaults["Potassium (K)"]
    
    data = {
        "State": [str(state).strip()],
        "District or Region": [str(district).strip()],
        "Soil Type": [str(soil_type).strip()],
        "Season": [str(season).strip()],
        "Temperature": [float(temp)],
        "Humidity": [float(humidity)],
        "Rainfall": [float(rainfall)],
        "Soil pH": [float(ph)],
        "Nitrogen (N)": [n_val],
        "Phosphorus (P)": [p_val],
        "Potassium (K)": [k_val]
    }
    
    return pd.DataFrame(data)
