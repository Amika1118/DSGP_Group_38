"""
Model comparison module for weather event classification.
FULLY FIXED VERSION - All bugs resolved
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score


class ModelComparator:
    """Compare multiple models and select the best with overfitting detection."""

    def __init__(self, config):
        """
        Initialize model comparator.

        Args:
            config: ConfigLoader instance
        """
        self.config = config
        self.comparison_results = None
        self.overfitting_analysis = None

    def detect_overfitting(self, train_score, val_score, threshold=0.1):
        """
        Detect overfitting or underfitting based on train/val score gap.

        Args:
            train_score: Training score (accuracy/F1)
            val_score: Validation score
            threshold: Maximum acceptable gap (default 0.1 = 10%)

        Returns:
            fit_status: 'good_fit', 'overfitting', or 'underfitting'
            gap: Train - validation score difference
        """
        gap = train_score - val_score

        if val_score < 0.6:
            # Poor validation performance regardless of gap
            return 'underfitting', gap
        elif gap > threshold:
            # Large gap indicates overfitting
            return 'overfitting', gap
        elif gap < -0.02:
            # Validation better than training (unusual, might indicate issues)
            return 'suspicious', gap
        else:
            # Good generalization
            return 'good_fit', gap

    def compare_models_with_diagnostics(self, models_dict, X_train, y_train, X_val, y_val):
        """
        Compare models with overfitting/underfitting diagnostics.

        Args:
            models_dict: Dictionary of {model_name: model_instance}
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            DataFrame with comparison results including diagnostics
        """
        results = []

        print("\n" + "="*70)
        print("MODEL COMPARISON WITH OVERFITTING DIAGNOSTICS")
        print("="*70)

        for name, model in models_dict.items():
            try:
                print(f"\nEvaluating {name}...")

                # Training metrics
                y_train_pred = model.predict(X_train)
                train_acc = accuracy_score(y_train, y_train_pred)
                train_f1 = f1_score(y_train, y_train_pred, average='macro', zero_division=0)

                # Validation metrics
                y_val_pred = model.predict(X_val)
                val_acc = accuracy_score(y_val, y_val_pred)
                val_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
                val_kappa = cohen_kappa_score(y_val, y_val_pred)

                # Detect overfitting
                fit_status, gap = self.detect_overfitting(train_f1, val_f1)

                results.append({
                    'model': name,
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'train_f1': train_f1,
                    'val_f1': val_f1,
                    'cohen_kappa': val_kappa,
                    'gap': gap,
                    'fit_status': fit_status
                })

                # Print diagnostics
                print(f"  Train F1: {train_f1:.4f}")
                print(f"  Val F1:   {val_f1:.4f}")
                print(f"  Gap:      {gap:+.4f}")

                if fit_status == 'overfitting':
                    print(f"  ⚠️  WARNING: Model is OVERFITTING (gap > 10%)")
                elif fit_status == 'underfitting':
                    print(f"  ⚠️  WARNING: Model is UNDERFITTING (val F1 < 60%)")
                elif fit_status == 'suspicious':
                    print(f"  ⚠️  SUSPICIOUS: Val score > train score")
                else:
                    print(f"  ✓ Good generalization")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue

        if not results:
            raise ValueError("No models could be evaluated")

        self.overfitting_analysis = pd.DataFrame(results).sort_values('val_f1', ascending=False)

        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("\n" + self.overfitting_analysis.to_string(index=False))

        return self.overfitting_analysis

    def compare_models(self, models_dict, X, y):
        """
        Compare all models on the same dataset.

        Args:
            models_dict: Dictionary of {model_name: model_instance}
            X: Features
            y: True labels

        Returns:
            DataFrame with comparison results
        """
        results = []
        predictions = {}

        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)

        # FIX #3: Better error handling for each model evaluation
        for name, model in models_dict.items():
            try:
                print(f"\nEvaluating {name}...")
                y_pred = model.predict(X)
                predictions[name] = y_pred

                # Calculate metrics
                f1_macro = f1_score(y, y_pred, average='macro')
                kappa = cohen_kappa_score(y, y_pred)
                acc = accuracy_score(y, y_pred)

                results.append({
                    'model': name,
                    'f1_macro': f1_macro,
                    'kappa': kappa,
                    'accuracy': acc
                })

                print(f"  F1-Macro: {f1_macro:.4f}")
                print(f"  Kappa: {kappa:.4f}")
                print(f"  Accuracy: {acc:.4f}")

            except Exception as e:
                print(f"  ❌ Error evaluating {name}: {e}")
                continue

        # FIX #3: Check if we have any results before creating DataFrame
        if not results:
            raise ValueError("No models successfully evaluated. Check model training and data.")

        # Create comparison DataFrame
        self.comparison_results = pd.DataFrame(results).sort_values('f1_macro', ascending=False)

        print("\n" + "="*70)
        print("COMPARISON RESULTS (sorted by F1-Macro)")
        print("="*70)
        print(self.comparison_results.to_string(index=False))
        print("="*70)

        return self.comparison_results

    def get_best_model(self, criterion='f1_macro'):
        """
        Get the name of the best performing model.

        Args:
            criterion: Metric to use for selection ('f1_macro', 'kappa', 'accuracy')

        Returns:
            String name of best model
        """
        # FIX #3: Better validation of comparison results
        if self.comparison_results is None or len(self.comparison_results) == 0:
            raise ValueError("No comparison results available. Run compare_models() first.")

        if criterion not in self.comparison_results.columns:
            available = list(self.comparison_results.columns)
            raise ValueError(
                f"Criterion '{criterion}' not found in results. "
                f"Available columns: {available}"
            )

        # Get best model based on criterion
        best_idx = self.comparison_results[criterion].idxmax()
        best_model_name = self.comparison_results.loc[best_idx, 'model']

        return best_model_name

    def mcnemar_test(self, y_true, y_pred1, y_pred2):
        """
        Perform McNemar's test for statistical significance between two models.

        Args:
            y_true: True labels
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2

        Returns:
            p-value (lower = more significant difference)
        """
        # Create contingency table
        correct1 = (y_true == y_pred1)
        correct2 = (y_true == y_pred2)

        # Count disagreements
        n01 = np.sum(~correct1 & correct2)  # Model 1 wrong, Model 2 correct
        n10 = np.sum(correct1 & ~correct2)  # Model 1 correct, Model 2 wrong

        # McNemar statistic with continuity correction
        if (n01 + n10) == 0:
            return 1.0  # No difference

        chi2_stat = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)

        # Calculate p-value
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2_stat, df=1)

        return p_value

    def pairwise_mcnemar_test(self, models_dict, X, y, alpha=0.05):
        """
        Perform pairwise McNemar tests between all models.

        Args:
            models_dict: Dictionary of models
            X: Features
            y: True labels
            alpha: Significance level

        Returns:
            DataFrame with pairwise test results
        """
        model_names = list(models_dict.keys())
        predictions = {}

        # Get predictions from all models
        for name, model in models_dict.items():
            try:
                predictions[name] = model.predict(X)
            except Exception as e:
                print(f"Error getting predictions from {name}: {e}")
                continue

        # Perform pairwise tests
        results = []
        for i, name1 in enumerate(model_names):
            if name1 not in predictions:
                continue
            for j, name2 in enumerate(model_names):
                if name2 not in predictions:
                    continue
                if i < j:  # Only test each pair once
                    p_value = self.mcnemar_test(y, predictions[name1], predictions[name2])
                    significant = p_value < alpha

                    results.append({
                        'model_1': name1,
                        'model_2': name2,
                        'p_value': p_value,
                        'significant': 'Yes' if significant else 'No'
                    })

        return pd.DataFrame(results)

    def get_comparison_summary(self):
        """
        Get a summary of the comparison results.

        Returns:
            Dictionary with summary statistics
        """
        if self.comparison_results is None:
            raise ValueError("No comparison results available.")

        summary = {
            'n_models': len(self.comparison_results),
            'best_model': self.get_best_model(),
            'best_f1_macro': self.comparison_results['f1_macro'].max(),
            'mean_f1_macro': self.comparison_results['f1_macro'].mean(),
            'std_f1_macro': self.comparison_results['f1_macro'].std()
        }

        return summary


if __name__ == '__main__':
    print("ModelComparator class defined successfully")