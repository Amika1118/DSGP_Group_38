"""
Base model class for weather event classification.
FULLY FIXED VERSION - All bugs resolved
"""
from abc import ABC, abstractmethod
import joblib
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin  # ADDED


class BaseModel(ABC, BaseEstimator, ClassifierMixin):  # MODIFIED: added BaseEstimator, ClassifierMixin
    """Abstract base class for all weather prediction models."""

    def __init__(self, config, model_name=None, **kwargs):
        """
        Initialize base model.

        Args:
            config: ConfigLoader instance
            model_name: Optional name (defaults to class name) - FIX #6
            **kwargs: Additional model-specific parameters
        """
        self.config = config
        self.model = None
        self.model_name = model_name or self.__class__.__name__
        self.is_fitted = False
        self.feature_names = None
        self.classes_ = None
        # Store kwargs for get_params (required for sklearn compatibility)
        self.kwargs = kwargs

    @abstractmethod
    def build_model(self, **params):
        """Build the model with specified parameters."""
        pass

    @abstractmethod
    def fit(self, X, y, **kwargs):
        """Train the model."""
        pass

    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"{self.model_name} does not support probability prediction")

    def score(self, X, y):
        """Calculate model score (accuracy)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        return self.model.score(X, y)

    def save_model(self, filepath):
        """Save trained model to file."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'classes_': self.classes_,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model from file."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.feature_names = model_data.get('feature_names')
        self.classes_ = model_data.get('classes_')
        self.is_fitted = model_data.get('is_fitted', True)
        print(f"Model loaded from {filepath}")

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Required for scikit-learn compatibility.
        """
        params = {
            'config': self.config,
            'model_name': self.model_name,
            **self.kwargs
        }
        return params

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        Required for scikit-learn compatibility.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_model_info(self):
        """Get model information."""
        return {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'n_classes': len(self.classes_) if self.classes_ else None,
            'classes': self.classes_
        }

    def __repr__(self):
        """String representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.model_name}({status})"