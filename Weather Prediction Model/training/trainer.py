"""
Trainer module for weather prediction models.
ENHANCED: Passes early_stopping_rounds to model constructor (not fit).
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.model_factory import ModelFactory
from training.hyperparameter_tuning import HyperparameterTuner
from training.cross_validation import TimeSeriesCV


class Trainer:
    """Orchestrate model training for weather prediction."""

    def __init__(self, config):
        self.config = config
        self.trained_models = {}
        self.training_history = {}
        self.tuning_results = {}
        self.feature_engineer = None
        self.target_creator = None

    def train_model(self, model_name, X_train, y_train, X_val=None, y_val=None,
                    class_names=None, **kwargs):
        """
        Train a single model.
        Args:
            class_names: list of original string class names (for saving)
        """
        print(f"\n{'='*70}")
        print(f"TRAINING {model_name.upper()}")
        print(f"{'='*70}")

        model_params = kwargs.get('model_params', {}).copy()

        # If early stopping is enabled and we have validation data,
        # add it to model_params (so it's in the constructor)
        if model_name == 'xgboost' and X_val is not None and y_val is not None:
            early_stopping_rounds = self.config.get('training.early_stopping_rounds', None)
            if early_stopping_rounds is not None:
                model_params['early_stopping_rounds'] = early_stopping_rounds
                print(f"  Early stopping enabled: {early_stopping_rounds} rounds")

        model = ModelFactory.create_model(model_name, self.config)
        model.build_model(**model_params)

        fit_kwargs = {}
        if model_name == 'xgboost' and X_val is not None and y_val is not None:
            fit_kwargs['eval_set'] = [(X_val, y_val)]
            fit_kwargs['verbose'] = kwargs.get('verbose', False)
            # Do NOT add early_stopping_rounds here; it's already in the model

        # Pass original class names if provided
        if class_names is not None:
            fit_kwargs['original_classes'] = class_names

        model.fit(X_train, y_train, **fit_kwargs)

        self.trained_models[model_name] = model

        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            print(f"  Validation accuracy: {val_score:.4f}")

            self.training_history[model_name] = {
                'train_score': model.score(X_train, y_train),
                'val_score': val_score
            }
        else:
            self.training_history[model_name] = {
                'train_score': model.score(X_train, y_train)
            }

        return model

    def train_with_tuning(self, model_name, X_train, y_train, X_val=None, y_val=None,
                          class_names=None, **kwargs):
        """Train with hyperparameter tuning."""
        print(f"\n{'='*70}")
        print(f"TRAINING {model_name.upper()} WITH HYPERPARAMETER TUNING")
        print(f"{'='*70}")

        tuning_config = self.config.get('hyperparameter_tuning', {})
        method = tuning_config.get('method', 'grid')
        n_splits = tuning_config.get('n_splits', 3)
        scoring = tuning_config.get('scoring', 'f1_macro')
        n_iter = tuning_config.get('n_iter', 20)
        cv_gap = tuning_config.get('cv_gap', 0)

        print(f"  Tuning method: {method}")
        print(f"  CV splits: {n_splits}")
        print(f"  Scoring: {scoring}")

        # Create base model (without any fit-time arguments)
        model = ModelFactory.create_model(model_name, self.config)
        model.build_model()  # builds with default parameters (no early stopping)

        tuner = HyperparameterTuner(self.config)
        cv = TimeSeriesCV(n_splits=n_splits, gap=cv_gap)

        try:
            search = tuner.tune_model(
                model,
                X_train,
                y_train,
                search_type=method,
                cv=cv,
                scoring=scoring,
                n_iter=n_iter
            )

            best_params = search.best_params_
            best_score = search.best_score_

            print(f"\n✓ Best parameters found:")
            for param, value in best_params.items():
                print(f"    {param}: {value}")

            self.tuning_results[model_name] = {
                'best_params': best_params,
                'best_score': best_score,
                'cv_results': search.cv_results_,
                'search': search
            }

            print(f"  Best CV score: {best_score:.4f}")

        except Exception as e:
            print(f"\n⚠️  Tuning failed: {e}")
            print(f"  Falling back to default parameters")
            best_params = {}

        # Now train the final model with the best parameters
        # (early stopping will be added in train_model if configured)
        final_model = self.train_model(
            model_name,
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            class_names=class_names,
            model_params=best_params,
            **kwargs
        )

        return final_model

    def train_all_models(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train all available models (not used in XGBoost-only mode)."""
        model_names = ModelFactory.get_available_models()
        tuning_enabled = self.config.get('hyperparameter_tuning.enabled', False)

        for model_name in model_names:
            try:
                if tuning_enabled:
                    self.train_with_tuning(model_name, X_train, y_train, X_val, y_val, **kwargs)
                else:
                    self.train_model(model_name, X_train, y_train, X_val, y_val, **kwargs)
            except Exception as e:
                print(f"\nError training {model_name}: {str(e)}")
                continue

        return self.trained_models

    def save_models(self, output_dir='models/saved'):
        """Save models and transformers."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for model_name, model in self.trained_models.items():
            model_file = output_path / f"{model_name}.pkl"
            model.save_model(str(model_file))

        if self.feature_engineer is not None:
            feat_file = output_path / "feature_engineer.pkl"
            self.feature_engineer.save(str(feat_file))
            print(f"Feature engineer saved to {feat_file}")

        if self.target_creator is not None:
            target_file = output_path / "target_creator.pkl"
            self.target_creator.save(str(target_file))
            print(f"Target creator saved to {target_file}")

    def load_models(self, input_dir='models/saved'):
        """Load models and transformers."""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Model directory not found: {input_dir}")

        feat_file = input_path / "feature_engineer.pkl"
        if feat_file.exists():
            from utils.feature_engineer import FeatureEngineer
            self.feature_engineer = FeatureEngineer(self.config)
            self.feature_engineer.load(str(feat_file))
            print(f"Loaded feature engineer from {feat_file}")

        target_file = input_path / "target_creator.pkl"
        if target_file.exists():
            from utils.target_creator import TargetCreator
            self.target_creator = TargetCreator(self.config)
            self.target_creator.load(str(target_file))
            print(f"Loaded target creator from {target_file}")

        for model_name in ModelFactory.get_available_models():
            model_file = input_path / f"{model_name}.pkl"
            if model_file.exists():
                model = ModelFactory.create_model(model_name, self.config)
                model.load_model(str(model_file))
                self.trained_models[model_name] = model
                print(f"Loaded {model_name}")

    def get_training_summary(self):
        """Get summary of training results."""
        summary_data = []
        for model_name, history in self.training_history.items():
            summary_data.append({
                'model': model_name,
                'train_accuracy': history.get('train_score', np.nan),
                'val_accuracy': history.get('val_score', np.nan)
            })
        return pd.DataFrame(summary_data).sort_values('val_accuracy', ascending=False)