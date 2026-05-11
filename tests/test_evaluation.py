import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path

# Add the project root to the path so we can import from models
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))

# Import the evaluation function we want to test
from models.ensemble_models import evaluate_model

@pytest.fixture
def sample_data():
    """Create a synthetic dataset for testing."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@pytest.fixture
def sample_model():
    """Provide a simple Logistic Regression model for testing."""
    return LogisticRegression(max_iter=100, random_state=42)

def test_evaluate_model_returns_correct_types(sample_data, sample_model):
    """Test that evaluate_model returns the expected data types."""
    X_train, X_test, y_train, y_test = sample_data
    
    metrics, cm, report, fitted_model = evaluate_model(
        sample_model, X_train, y_train, X_test, y_test, "TestModel"
    )
    
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    assert isinstance(cm, np.ndarray), "Confusion matrix should be a numpy array"
    assert isinstance(report, dict), "Classification report should be a dictionary"

def test_evaluate_model_metrics_range(sample_data, sample_model):
    """Test that evaluation metrics are within the valid [0, 1] range."""
    X_train, X_test, y_train, y_test = sample_data
    
    metrics, _, _, _ = evaluate_model(
        sample_model, X_train, y_train, X_test, y_test, "TestModel"
    )
    
    # Check bounds for standard metrics
    for key in ["Accuracy", "F1", "Recall", "Precision", "ROC-AUC", "PR-AUC"]:
        assert 0.0 <= metrics[key] <= 1.0, f"Metric {key} out of bounds: {metrics[key]}"
        
    # MCC can range from -1 to 1
    assert -1.0 <= metrics["MCC"] <= 1.0, f"Metric MCC out of bounds: {metrics['MCC']}"

def test_evaluate_model_features_count(sample_data, sample_model):
    """Test that the number of features is correctly logged."""
    X_train, X_test, y_train, y_test = sample_data
    
    metrics, _, _, _ = evaluate_model(
        sample_model, X_train, y_train, X_test, y_test, "TestModel"
    )
    
    assert metrics["Features"] == X_train.shape[1], "Feature count mismatch"
