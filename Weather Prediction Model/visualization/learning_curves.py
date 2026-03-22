"""Learning curve and validation curve visualization."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.model_selection import learning_curve, validation_curve


def plot_learning_curve(model, X, y, cv=5, train_sizes=None, scoring='f1_macro',
                        save_path=None, figsize=(10, 6)):
    """
    Plot learning curve to diagnose bias/variance.

    Shows how training and validation scores change with training set size.
    Helps determine if more data would improve performance.

    Args:
        model: Model instance or sklearn estimator
        X: Features
        y: Labels
        cv: Cross-validation strategy or number of folds
        train_sizes: Array of training set sizes to evaluate
        scoring: Scoring metric
        save_path: Path to save figure (optional)
        figsize: Figure size tuple

    Returns:
        Path to saved figure if save_path provided, else None
    """
    try:
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        print(f"Computing learning curve for {model.__class__.__name__}...")

        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X

        # Compute learning curve
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_array, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )

        # Calculate mean and std
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot training scores
        ax.plot(train_sizes_abs, train_mean, 'o-', color='steelblue',
                label='Training score', linewidth=2, markersize=6)
        ax.fill_between(train_sizes_abs,
                        train_mean - train_std,
                        train_mean + train_std,
                        alpha=0.2, color='steelblue')

        # Plot validation scores
        ax.plot(train_sizes_abs, val_mean, 'o-', color='coral',
                label='Validation score', linewidth=2, markersize=6)
        ax.fill_between(train_sizes_abs,
                        val_mean - val_std,
                        val_mean + val_std,
                        alpha=0.2, color='coral')

        ax.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{scoring.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_title(f'Learning Curve - {model.__class__.__name__}',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.05)

        # Add interpretation text
        final_gap = train_mean[-1] - val_mean[-1]
        if final_gap > 0.1:
            interpretation = "⚠️ High variance (overfitting) - Consider:\n• Regularization\n• More data\n• Feature reduction"
            color = 'red'
        elif val_mean[-1] < 0.7:
            interpretation = "⚠️ High bias (underfitting) - Consider:\n• More features\n• More complex model\n• Less regularization"
            color = 'orange'
        else:
            interpretation = "✓ Good fit - Model generalizes well"
            color = 'green'

        ax.text(0.02, 0.98, interpretation,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.2),
                fontsize=9)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Learning curve saved to {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None

    except Exception as e:
        print(f"Error plotting learning curve: {e}")
        return None


def plot_validation_curve(model, X, y, param_name, param_range, cv=5,
                          scoring='f1_macro', save_path=None, figsize=(10, 6)):
    """
    Plot validation curve for a hyperparameter.

    Shows how training and validation scores change with hyperparameter values.
    Helps identify optimal hyperparameter range.

    Args:
        model: Model instance or sklearn estimator
        X: Features
        y: Labels
        param_name: Name of hyperparameter to vary
        param_range: Array of parameter values to test
        cv: Cross-validation strategy or number of folds
        scoring: Scoring metric
        save_path: Path to save figure (optional)
        figsize: Figure size tuple

    Returns:
        Path to saved figure if save_path provided, else None
    """
    try:
        print(f"Computing validation curve for {param_name}...")

        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X

        # Compute validation curve
        train_scores, val_scores = validation_curve(
            model, X_array, y,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )

        # Calculate mean and std
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot training scores
        ax.plot(param_range, train_mean, 'o-', color='steelblue',
                label='Training score', linewidth=2, markersize=6)
        ax.fill_between(param_range,
                        train_mean - train_std,
                        train_mean + train_std,
                        alpha=0.2, color='steelblue')

        # Plot validation scores
        ax.plot(param_range, val_mean, 'o-', color='coral',
                label='Validation score', linewidth=2, markersize=6)
        ax.fill_between(param_range,
                        val_mean - val_std,
                        val_mean + val_std,
                        alpha=0.2, color='coral')

        # Mark best parameter
        best_idx = np.argmax(val_mean)
        best_param = param_range[best_idx]
        best_score = val_mean[best_idx]

        ax.axvline(x=best_param, color='green', linestyle='--',
                   linewidth=2, alpha=0.5, label=f'Best: {best_param}')
        ax.plot(best_param, best_score, 'g*', markersize=15)

        ax.set_xlabel(param_name.replace('_', ' ').title(),
                      fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{scoring.replace("_", " ").title()}',
                      fontsize=12, fontweight='bold')
        ax.set_title(f'Validation Curve - {model.__class__.__name__}',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.05)

        # Use log scale for x-axis if param values span multiple orders of magnitude
        if param_range[-1] / param_range[0] > 100:
            ax.set_xscale('log')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Validation curve saved to {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None

    except Exception as e:
        print(f"Error plotting validation curve: {e}")
        return None


def plot_cv_scores(cv_results, save_path=None, figsize=(10, 6)):
    """
    Plot cross-validation scores from hyperparameter tuning.

    Args:
        cv_results: cv_results_ from GridSearchCV or RandomizedSearchCV
        save_path: Path to save figure (optional)
        figsize: Figure size tuple

    Returns:
        Path to saved figure if save_path provided, else None
    """
    try:
        # Extract scores
        mean_scores = cv_results['mean_test_score']
        std_scores = cv_results['std_test_score']
        params = cv_results['params']

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(mean_scores))

        ax.errorbar(x, mean_scores, yerr=std_scores,
                    fmt='o-', capsize=5, capthick=2,
                    color='steelblue', ecolor='gray',
                    linewidth=2, markersize=6)

        # Mark best score
        best_idx = np.argmax(mean_scores)
        ax.plot(best_idx, mean_scores[best_idx], 'r*', markersize=15,
                label=f'Best score: {mean_scores[best_idx]:.4f}')

        ax.set_xlabel('Parameter Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('CV Score', fontsize=12, fontweight='bold')
        ax.set_title('Cross-Validation Scores Across Hyperparameters',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.05)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ CV scores plot saved to {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None

    except Exception as e:
        print(f"Error plotting CV scores: {e}")
        return None