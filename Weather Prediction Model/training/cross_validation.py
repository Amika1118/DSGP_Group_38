"""Time series cross-validation for weather prediction with proper gap handling."""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


class TimeSeriesCV:
    """
    Time series cross-validation with expanding window and gap.

    Ensures no data leakage by:
    - Using only past data for training
    - Adding gap between train and validation
    - Expanding training window over time
    """

    def __init__(self, n_splits=5, gap=0, config=None):
        """
        Initialize TimeSeriesCV.

        Args:
            n_splits: Number of splits
            gap: Number of samples to skip between train and val (prevents leakage)
            config: Optional config object
        """
        self.n_splits = n_splits
        self.gap = gap
        self.config = config
        self.cv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    def split(self, X, y=None, groups=None):
        """
        Generate train/val indices for time series CV.

        Compatible with sklearn's GridSearchCV/RandomizedSearchCV.

        Args:
            X: Features (array or DataFrame)
            y: Labels (optional, not used but required by sklearn interface)
            groups: Groups (optional, not used)

        Yields:
            train_idx, val_idx: Arrays of indices for train and validation sets
        """
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]

        for train_idx, val_idx in self.cv.split(X):
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits (required by sklearn interface)."""
        return self.n_splits

    def evaluate_model(self, model, X, y):
        """
        Evaluate model using time series CV.

        Args:
            model: Model instance with fit/score methods
            X: Features
            y: Labels

        Returns:
            Dict with mean score, std score, and individual scores
        """
        scores = []

        for train_idx, val_idx in self.split(X):
            # Handle both DataFrame and numpy arrays
            if isinstance(X, pd.DataFrame):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            scores.append(score)

        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores
        }