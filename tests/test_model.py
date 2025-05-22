"""
Tests for MLflow model training and serving
"""

import pytest
import numpy as np
import mlflow
import pandas as pd
from sklearn.datasets import load_wine
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_with_mlflow import load_and_preprocess_data, train_model, parse_args


class TestModelTraining:
    """Test model training functionality"""

    def test_model_training_completes(self):
        """Test that model training completes successfully"""
        # Set up MLflow experiment
        mlflow.set_experiment("test-experiment")

        with mlflow.start_run():
            # Parse default arguments
            args = parse_args()

            # Load data
            X_train, X_test, y_train, y_test, scaler, feature_names, X_train_original = load_and_preprocess_data(
                args.test_size, args.random_state
            )

            # Train model
            model, feature_importance = train_model(X_train, y_train, args)

            # Check that model is trained
            assert model is not None
            assert hasattr(model, 'predict')
            assert len(feature_importance) == X_train.shape[1]

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Check that run was created
            run_id = mlflow.active_run().info.run_id
            assert run_id is not None

    def test_model_prediction(self):
        """Test that model predictions work correctly"""
        # Set up MLflow experiment
        mlflow.set_experiment("test-experiment")

        with mlflow.start_run() as run:
            # Train a simple model
            args = parse_args()
            X_train, X_test, y_train, y_test, scaler, feature_names, X_train_original = load_and_preprocess_data(
                args.test_size, args.random_state
            )
            model, _ = train_model(X_train, y_train, args)

            # Log model
            mlflow.sklearn.log_model(model, "model")
            run_id = run.info.run_id

        # Load model from MLflow
        model_uri = f"runs:/{run_id}/model"
        loaded_model = mlflow.sklearn.load_model(model_uri)

        # Make predictions
        predictions = loaded_model.predict(X_test[:5])

        # Check predictions
        assert predictions is not None
        assert len(predictions) == 5
        assert all(pred in [0, 1, 2] for pred in predictions)  # Wine dataset has 3 classes

        # Test prediction probabilities
        predict_proba = loaded_model.predict_proba(X_test[:5])
        assert predict_proba.shape == (5, 3)
        assert np.allclose(predict_proba.sum(axis=1), 1.0)  # Probabilities should sum to 1


if __name__ == "__main__":
    pytest.main([__file__])