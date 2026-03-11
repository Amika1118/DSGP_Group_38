"""
SVM model with RBF kernel for weather event classification.
PRODUCTION FIX: build_model() updates instance variables so get_params()
always reflects the actual model parameters (sklearn GridSearchCV compat).
Assumes integer labels.
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from base_model import BaseModel


class SVMModel(BaseModel):
    """SVM classifier with RBF kernel for weather event prediction."""

    _DEFAULT_PARAMS = dict(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        class_weight='balanced',
        probability=True,
        cache_size=1000,
        max_iter=-1,
    )

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.scaler                = StandardScaler()
        self.is_scaled             = False
        self.support_vectors_count = None
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
        Build the underlying SVC.
        FIX: writes resolved params back to instance attrs.
        """
        resolved = {k: getattr(self, k) for k in self._DEFAULT_PARAMS}
        resolved['random_state'] = self.random_state
        resolved.update(params)

        # FIX: sync instance attributes
        for k, v in resolved.items():
            if hasattr(self, k):
                setattr(self, k, v)

        self.model = SVC(**resolved)

        print("SVM built with parameters:")
        for k, v in resolved.items():
            print(f"  {k}: {v}")

    # ── Training ───────────────────────────────────────────────────────────

    def fit(self, X, y, **kwargs):
        if self.model is None:
            self.build_model()

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_arr = X.values
        else:
            X_arr = X

        if kwargs.get('scale', True):
            print("\nScaling features for SVM …")
            X_arr         = self.scaler.fit_transform(X_arr)
            self.is_scaled = True

        self.classes_ = np.unique(y)
        print(f"\nTraining SVM …")
        print(f"  Samples: {X_arr.shape[0]}  |  Features: {X_arr.shape[1]}"
              f"  |  Classes: {len(self.classes_)}")

        self.model.fit(X_arr, y)
        self.is_fitted             = True
        self.support_vectors_count = self.model.n_support_

        print(f"  Training accuracy: {self.model.score(X_arr, y):.4f}")
        print(f"  Support vectors  : {sum(self.support_vectors_count)}")
        return self

    # ── Inference ──────────────────────────────────────────────────────────

    def _to_numpy_scaled(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.is_scaled:
            X = self.scaler.transform(X)
        return X

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(self._to_numpy_scaled(X))

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(self._to_numpy_scaled(X))

    def score(self, X, y):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        return self.model.score(self._to_numpy_scaled(X), y)

    # ── Feature importance (N/A for RBF SVM) ──────────────────────────────

    def get_feature_importance(self, top_n=20):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        names = (self.feature_names[:top_n] if self.feature_names
                 else [f"f{i}" for i in range(top_n or 0)])
        print("Note: RBF-SVM has no native feature importance. "
              "Use permutation importance if needed.")
        return pd.DataFrame({'feature': names,
                             'importance': [0.0] * len(names)})

    def get_model_complexity(self):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        total_sv = sum(self.support_vectors_count)
        return {
            'C':                   self.model.C,
            'gamma':               self.model.gamma,
            'kernel':              self.model.kernel,
            'n_support_vectors':   total_sv,
            'support_vector_ratio': total_sv / self.model.shape_fit_[0],
            'n_features':          self.model.shape_fit_[1],
        }

    # ── Persistence (overrides base to include scaler) ─────────────────────

    def save_model(self, filepath):
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump({
            'model':                  self.model,
            'model_name':             self.model_name,
            'feature_names':          self.feature_names,
            'classes_':               self.classes_,
            'is_fitted':              self.is_fitted,
            'scaler':                 self.scaler,
            'is_scaled':              self.is_scaled,
            'support_vectors_count':  self.support_vectors_count,
        }, filepath)
        print(f"SVM model saved → {filepath}")

    def load_model(self, filepath):
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        import joblib
        d = joblib.load(filepath)
        self.model                = d['model']
        self.model_name           = d['model_name']
        self.feature_names        = d.get('feature_names')
        self.classes_             = d.get('classes_')
        self.is_fitted            = d.get('is_fitted', True)
        self.scaler               = d.get('scaler', StandardScaler())
        self.is_scaled            = d.get('is_scaled', False)
        self.support_vectors_count = d.get('support_vectors_count')
        print(f"SVM model loaded ← {filepath}")


if __name__ == '__main__':
    print("SVMModel defined successfully")