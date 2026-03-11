"""
Random Forest model for weather event classification.
PRODUCTION FIX: build_model() updates instance variables so get_params()
always reflects the actual model parameters (sklearn GridSearchCV compat).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest classifier for weather event prediction."""

    # Default parameter values – also serves as the canonical param registry
    _DEFAULT_PARAMS = dict(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        verbose=0,
    )

    def __init__(self, config, model_name=None, **kwargs):
        super().__init__(config, model_name=model_name or 'random_forest', **kwargs)
        self.feature_importances_ = None
        self.random_state = kwargs.get(
            'random_state', config.get('training.random_state', 42)
        )
        # Initialise every param from defaults, then apply kwargs overrides
        for k, v in self._DEFAULT_PARAMS.items():
            setattr(self, k, kwargs.get(k, v))

    # ── sklearn compatibility ──────────────────────────────────────────────

    def get_params(self, deep=True):
        """Return current parameter values (required by sklearn clone/CV)."""
        params = {k: getattr(self, k) for k in self._DEFAULT_PARAMS}
        params.update({'random_state': self.random_state,
                       'config': self.config,
                       'model_name': self.model_name})
        return params

    def set_params(self, **params):
        """Set parameters (required by sklearn GridSearchCV)."""
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self

    # ── Model construction ─────────────────────────────────────────────────

    def build_model(self, **params):
        """
        Build (or re-build) the underlying RandomForestClassifier.
        FIX: updates instance attributes so get_params() stays consistent.
        """
        # Resolve final parameter values: instance attrs → override with params
        resolved = {k: getattr(self, k) for k in self._DEFAULT_PARAMS}
        resolved['random_state'] = self.random_state
        resolved.update(params)

        # Normalise string-encoded special values (from YAML / grid search)
        if resolved['max_depth'] in (None, 'None', 'null', 'none'):
            resolved['max_depth'] = None
        elif isinstance(resolved['max_depth'], str):
            try:
                resolved['max_depth'] = int(resolved['max_depth'])
            except ValueError:
                resolved['max_depth'] = None

        for int_param in ('min_samples_split', 'min_samples_leaf', 'n_estimators'):
            if isinstance(resolved[int_param], str):
                try:
                    resolved[int_param] = int(resolved[int_param])
                except ValueError:
                    pass

        # FIX: write resolved values back so get_params() reflects reality
        for k, v in resolved.items():
            if hasattr(self, k):
                setattr(self, k, v)

        self.model = RandomForestClassifier(**resolved)

        print("Random Forest built with parameters:")
        for k, v in resolved.items():
            print(f"  {k}: {v}")

    # ── Training ───────────────────────────────────────────────────────────

    def fit(self, X, y, **kwargs):
        if self.model is None:
            self.build_model()

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()

        self.classes_ = np.unique(y)
        sample_weight = kwargs.get('sample_weight', None)

        print(f"\nTraining Random Forest …")
        print(f"  Samples: {X.shape[0]}  |  Features: {X.shape[1]}"
              f"  |  Classes: {len(self.classes_)}")

        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True
        self.feature_importances_ = self.model.feature_importances_

        print(f"  Training accuracy: {self.model.score(X, y):.4f}")
        return self

    # ── Feature importance ─────────────────────────────────────────────────

    def get_feature_importance(self, top_n=20):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        names = self.feature_names or [f"f{i}" for i in range(len(self.feature_importances_))]
        df = pd.DataFrame({'feature': names,
                           'importance': self.feature_importances_}
                          ).sort_values('importance', ascending=False)
        return df.head(top_n) if top_n else df

    def plot_feature_importance(self, top_n=20, figsize=(10, 8), save_path=None):
        imp = self.get_feature_importance(top_n=top_n)
        plt.figure(figsize=figsize)
        sns.barplot(data=imp, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances – Random Forest',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def get_model_complexity(self):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        total_nodes  = sum(t.tree_.node_count for t in self.model.estimators_)
        total_leaves = sum(t.tree_.n_leaves  for t in self.model.estimators_)
        return {
            'n_estimators':        self.model.n_estimators,
            'total_nodes':         total_nodes,
            'total_leaves':        total_leaves,
            'avg_nodes_per_tree':  total_nodes  / self.model.n_estimators,
            'avg_leaves_per_tree': total_leaves / self.model.n_estimators,
        }


if __name__ == '__main__':
    print("RandomForestModel defined successfully")