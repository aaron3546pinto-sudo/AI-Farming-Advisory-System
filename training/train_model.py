"""
Model Training Script for Indian Agricultural Crop Recommendation
Trains Random Forest Classifier and Decision Tree Classifier pipelines.
Evaluates metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
Saves model pipeline to models/crop_recommendation_model.pkl and metrics to models/model_metrics.json.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Add parent directory to path so utils can be imported
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils.preprocessing import (
    load_crop_dataset,
    build_preprocessor,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_FEATURE
)

def train_and_evaluate_models():
    print("=" * 60)
    print("AI FARMER ADVISORY SYSTEM - MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    # 1. Load dataset
    df = load_crop_dataset()
    print(f"Loaded dataset containing {len(df)} records.")
    print(f"Target classes (Crops): {df[TARGET_FEATURE].nunique()} unique crops.")
    
    # 2. Features and target split
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET_FEATURE]
    
    # Train-test split (80-20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} samples | Testing set: {len(X_test)} samples")
    
    # 3. Model Pipelines
    preprocessor = build_preprocessor()
    
    # Primary Model: Random Forest Classifier
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=180,
            max_depth=22,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Alternative Model: Decision Tree Classifier
    dt_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", DecisionTreeClassifier(
            max_depth=18,
            min_samples_split=4,
            random_state=42
        ))
    ])
    
    print("\nTraining Random Forest Classifier...")
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    
    print("Training Decision Tree Classifier (Alternative)...")
    dt_pipeline.fit(X_train, y_train)
    y_pred_dt = dt_pipeline.predict(X_test)
    
    # 4. Evaluation Metrics
    rf_metrics = {
        "model_name": "Random Forest Classifier",
        "accuracy": round(float(accuracy_score(y_test, y_pred_rf)), 4),
        "precision_weighted": round(float(precision_score(y_test, y_pred_rf, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_test, y_pred_rf, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred_rf, average="weighted", zero_division=0)), 4),
        "precision_macro": round(float(precision_score(y_test, y_pred_rf, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred_rf, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred_rf, average="macro", zero_division=0)), 4),
    }
    
    dt_metrics = {
        "model_name": "Decision Tree Classifier",
        "accuracy": round(float(accuracy_score(y_test, y_pred_dt)), 4),
        "precision_weighted": round(float(precision_score(y_test, y_pred_dt, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_test, y_pred_dt, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred_dt, average="weighted", zero_division=0)), 4),
        "precision_macro": round(float(precision_score(y_test, y_pred_dt, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred_dt, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred_dt, average="macro", zero_division=0)), 4),
    }
    
    # Confusion matrix for Random Forest
    labels = sorted(list(y.unique()))
    cm = confusion_matrix(y_test, y_pred_rf, labels=labels)
    
    print("\n--- MODEL PERFORMANCE COMPARISON ---")
    print(f"Random Forest  -> Accuracy: {rf_metrics['accuracy']:.2%}, F1 (weighted): {rf_metrics['f1_weighted']:.4f}")
    print(f"Decision Tree  -> Accuracy: {dt_metrics['accuracy']:.2%}, F1 (weighted): {dt_metrics['f1_weighted']:.4f}")
    
    print("\nDetailed Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf, zero_division=0))
    
    # 5. Save artifacts
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_save_path = os.path.join(models_dir, "crop_recommendation_model.pkl")
    joblib.dump(rf_pipeline, model_save_path)
    print(f"Successfully saved trained Random Forest model to: {model_save_path}")
    
    metrics_save_path = os.path.join(models_dir, "model_metrics.json")
    all_metrics = {
        "random_forest": rf_metrics,
        "decision_tree": dt_metrics,
        "classes": labels,
        "features": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES
        },
        "sample_counts": {
            "total": len(df),
            "train": len(X_train),
            "test": len(X_test)
        },
        "confusion_matrix": cm.tolist()
    }
    
    with open(metrics_save_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Saved model metrics and confusion matrix to: {metrics_save_path}")
    
    return all_metrics

if __name__ == "__main__":
    train_and_evaluate_models()
