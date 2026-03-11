"""
XGBoost model for weather event classification.
ENHANCED: Early stopping set via constructor (compatible with older versions).
Stores original class names for correct prediction output.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost classifier for weather event prediction."""

    _DEFAULT_PARAMS = dict(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1,
        objective='multi:softprob',
        eval_metric='mlogloss',
        n_jobs=-1,
        tree_method='hist',
        enable_categorical=False,
    )

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.feature_importances_ = None
        self.evals_result_        = {}
        self.original_classes     = None   # will store original string class names
        self.random_state = kwargs.get(
            'random_state', config.get('training.random_state', 42)
        )
        for k, v in self._DEFAULT_PARAMS.items():
            setattr(self, k, kwargs.get(k, v))

    # ── sklearn compatibility ──────────────────────────────────────────────

    def get_params(self, deep=True):
        params = {k: getattr(self, k) for k in self._DEFAULT_PARAMS}
        params['random_state'] = self.random_state
        return params

    def set_params(self, **params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self

    # ── Model construction ─────────────────────────────────────────────────

    def build_model(self, **params):
        """
        Build the underlying XGBClassifier.
        If early_stopping_rounds is provided, include it in the constructor.
        """
        resolved = {k: getattr(self, k) for k in self._DEFAULT_PARAMS}
        resolved['random_state'] = self.random_state
        resolved.update(params)

        for k, v in resolved.items():
            if hasattr(self, k):
                setattr(self, k, v)

        self.model = xgb.XGBClassifier(**resolved)

        print("XGBoost built with parameters:")
        for k, v in resolved.items():
            print(f"  {k}: {v}")

    # ── Training ───────────────────────────────────────────────────────────

    def fit(self, X, y, **kwargs):
        if self.model is None:
            self.build_model()

        # Store original class names if provided
        if 'original_classes' in kwargs:
            self.original_classes = kwargs['original_classes']

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_arr = X.values
        else:
            X_arr = X

        y_arr = y if isinstance(y, np.ndarray) else np.array(y)
        self.classes_ = np.unique(y_arr)   # these are the encoded integers

        # Prepare fit parameters (only eval_set and verbose)
        fit_params = {}
        eval_set = kwargs.get('eval_set', None)
        if eval_set is not None:
            processed = []
            for Xv, yv in eval_set:
                if isinstance(Xv, pd.DataFrame):
                    Xv = Xv.values
                if isinstance(yv, pd.Series):
                    yv = yv.values
                processed.append((Xv, yv))
            fit_params['eval_set'] = processed
            fit_params['verbose'] = kwargs.get('verbose', False)

            # early_stopping_rounds is already in the model's constructor,
            # so we do NOT pass it again.

        print(f"\nTraining XGBoost …")
        print(f"  Samples: {X_arr.shape[0]}  |  Features: {X_arr.shape[1]}"
              f"  |  Classes: {len(self.classes_)}")
        if hasattr(self.model, 'early_stopping_rounds') and self.model.early_stopping_rounds:
            print(f"  Early stopping: {self.model.early_stopping_rounds} rounds")

        self.model.fit(X_arr, y_arr, **fit_params)
        self.is_fitted = True
        self.feature_importances_ = self.model.feature_importances_

        if hasattr(self.model, 'evals_result_'):
            self.evals_result_ = self.model.evals_result_

        train_preds = self.model.predict(X_arr)
        print(f"  Training accuracy: {(train_preds == y_arr).mean():.4f}")
        return self

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)

    def score(self, X, y):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        y_pred = self.predict(X)
        return (y_pred == y).mean()

    # ── Feature importance ─────────────────────────────────────────────────

    def get_feature_importance(self, importance_type='weight', top_n=20):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        names = self.feature_names or [f"f{i}" for i in range(len(self.feature_importances_))]
        booster = self.model.get_booster()
        imp_dict = booster.get_score(importance_type=importance_type)
        rows = [{'feature': n, 'importance': imp_dict.get(f'f{i}', 0.0)}
                for i, n in enumerate(names)]
        df = pd.DataFrame(rows).sort_values('importance', ascending=False)
        return df.head(top_n) if top_n else df

    def plot_feature_importance(self, importance_type='weight', top_n=20,
                                figsize=(10, 8), save_path=None):
        imp = self.get_feature_importance(importance_type=importance_type, top_n=top_n)
        plt.figure(figsize=figsize)
        sns.barplot(data=imp, x='importance', y='feature', palette='rocket')
        plt.title(f'Top {top_n} Feature Importances – XGBoost ({importance_type})',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def get_model_complexity(self):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return {
            'n_estimators':    self.model.n_estimators,
            'max_depth':       self.model.max_depth,
            'learning_rate':   self.model.learning_rate,
            'reg_lambda':      self.model.reg_lambda,
            'reg_alpha':       self.model.reg_alpha,
            'min_child_weight': self.model.min_child_weight,
        }

    # ── Persistence ────────────────────────────────────────────────────────

    def save_model(self, path):
        """Save model including original_classes."""
        import joblib
        data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'classes_': self.classes_,
            'original_classes': self.original_classes,
            'params': self.get_params()
        }
        joblib.dump(data, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model and restore original_classes."""
        import joblib
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']
        self.classes_ = data['classes_']
        self.original_classes = data.get('original_classes', None)
        for k, v in data.get('params', {}).items():
            if hasattr(self, k):
                setattr(self, k, v)
        print(f"Model loaded from {path}")