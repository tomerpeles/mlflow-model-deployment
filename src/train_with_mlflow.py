"""
Modified model training script with MLflow integration
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import os
from datetime import datetime
import argparse
import json

# MLflow configuration
EXPERIMENT_NAME = "wine-classification"
REGISTERED_MODEL_NAME = "wine-classifier"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train wine classifier with MLflow")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-split", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    return parser.parse_args()


def load_and_preprocess_data(test_size, random_state):
    """Load wine dataset and preprocess it"""
    print("Loading wine dataset...")
    data = load_wine()

    # Create DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Log dataset info to MLflow
    mlflow.log_param("dataset_name", "wine")
    mlflow.log_param("n_samples", len(df))
    mlflow.log_param("n_features", len(data.feature_names))
    mlflow.log_param("n_classes", len(data.target_names))

    # Check for missing values
    missing_values = df.isnull().sum().sum()
    mlflow.log_metric("missing_values", missing_values)

    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Log data split info
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, data.feature_names, X_train


def train_model(X_train, y_train, args):
    """Train Random Forest classifier"""
    print("\nTraining Random Forest model...")

    # Log hyperparameters
    mlflow.log_params({
        "model_type": "RandomForestClassifier",
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": args.random_state
    })

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=-1
    )

    # Train the model
    model.fit(X_train, y_train)

    # Get feature importance
    feature_importance = model.feature_importances_

    return model, feature_importance


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    print("\nEvaluating model...")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Log metrics to MLflow
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    # Log confusion matrix as artifact
    conf_matrix_df = pd.DataFrame(conf_matrix)
    conf_matrix_df.to_csv("confusion_matrix.csv", index=False)
    mlflow.log_artifact("confusion_matrix.csv")
    os.remove("confusion_matrix.csv")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return accuracy, y_pred_proba


def main():
    """Main training pipeline with MLflow integration"""
    args = parse_args()

    print("Wine Quality Classification Model Training with MLflow")
    print("=" * 50)

    # Set MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Start MLflow run
    with mlflow.start_run() as run:
        # Log code version
        mlflow.log_param("code_version", "1.0")

        # Load and preprocess data
        X_train, X_test, y_train, y_test, scaler, feature_names, X_train_original = load_and_preprocess_data(
            args.test_size, args.random_state
        )

        # Train model
        model, feature_importance = train_model(X_train, y_train, args)

        # Evaluate model
        accuracy, y_pred_proba = evaluate_model(model, X_test, y_test)

        # Create model signature
        signature = infer_signature(X_train_original, model.predict(X_train_original))

        # Log the model with signature
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME,
            await_registration_for=0
        )

        # Log the scaler
        mlflow.sklearn.log_model(
            sk_model=scaler,
            artifact_path="scaler"
        )

        run_id = run.info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        print(f"MLflow Experiment: {EXPERIMENT_NAME}")
        print(f"Model registered as: {REGISTERED_MODEL_NAME}")

        # Set tags
        mlflow.set_tag("model_type", "classification")
        mlflow.set_tag("dataset", "wine")
        mlflow.set_tag("author", "mlflow-exercise")

    print("\nTraining completed successfully!")
    print(f"View results in MLflow UI: mlflow ui")


if __name__ == "__main__":
    main()