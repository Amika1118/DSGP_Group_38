"""
Hyperparameter tuning for weather prediction models.
FIXED: Extracts the underlying sklearn estimator for tuning,
       avoiding issues with wrapper classes.
"""
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np

class HyperparameterTuner:
    """Hyperparameter optimization for models."""

    def __init__(self, config):
        self.config = config
        self.best_params_ = {}

    def tune_model(self, model, X, y, param_grid=None, search_type='grid', cv=5, scoring='f1_macro', n_iter=20):
        """
        Tune hyperparameters and return the fitted search object.

        Args:
            model: Model instance (wrapper) – must already have `model.model` built.
            X: Features
            y: Labels
            param_grid: Parameter grid (if None, uses config hyperparameters)
            search_type: 'grid' or 'random'
            cv: Cross-validation splitter or number of folds
            scoring: Scoring metric (default 'f1_macro')
            n_iter: Number of iterations for random search

        Returns:
            Fitted GridSearchCV or RandomizedSearchCV object
        """
        if param_grid is None:
            # Get the model's hyperparameters from config
            model_config = self.config.get_model_config(model.model_name)
            param_grid = model_config.get('hyperparameters', {})
            if not param_grid:
                raise ValueError(f"No hyperparameters defined for {model.model_name} in config.")

        print(f"\nTuning {model.model_name} using {search_type} search...")
        print(f"Scoring metric: {scoring}")

        # Ensure all values in param_grid are lists (GridSearchCV requirement)
        for key, value in param_grid.items():
            if not isinstance(value, (list, np.ndarray)):
                param_grid[key] = [value]

        # Extract the underlying sklearn estimator
        if not hasattr(model, 'model') or model.model is None:
            raise ValueError(f"Model {model.model_name} does not have a built underlying estimator. Call build_model() first.")
        estimator = model.model

        if search_type == 'grid':
            search = GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        else:
            search = RandomizedSearchCV(estimator, param_grid, n_iter=n_iter, cv=cv,
                                        scoring=scoring, n_jobs=-1, random_state=42)

        search.fit(X, y)

        self.best_params_[model.model_name] = search.best_params_
        print(f"Best parameters: {search.best_params_}")
        print(f"Best score: {search.best_score_:.4f}")

        return search