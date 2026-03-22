"""Feature importance and correlation visualization."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def plot_feature_importance(model, top_n=20, save_path=None, figsize=(12, 8)):
    """
    Plot feature importance for tree-based models.

    Args:
        model: Trained model with get_feature_importance() method
        top_n: Number of top features to show
        save_path: Path to save figure (optional)
        figsize: Figure size tuple

    Returns:
        Path to saved figure if save_path provided, else None
    """
    try:
        # Get feature importance
        importance_df = model.get_feature_importance()

        if importance_df is None or len(importance_df) == 0:
            print(f"No feature importance available for {model.model_name}")
            return None

        # Get top N features
        top_features = importance_df.head(top_n)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        colors = sns.color_palette("viridis", len(top_features))
        bars = ax.barh(range(len(top_features)), top_features['importance'], color=colors)

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Features - {model.model_name}',
                     fontsize=14, fontweight='bold', pad=20)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
            ax.text(value, i, f' {value:.4f}',
                    va='center', fontsize=9)

        ax.invert_yaxis()  # Highest importance at top
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Feature importance plot saved to {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None

    except Exception as e:
        print(f"Error plotting feature importance: {e}")
        return None


def plot_feature_correlations(data, features=None, save_path=None, figsize=(14, 12)):
    """
    Plot correlation heatmap of features.

    Args:
        data: DataFrame with features
        features: List of features to include (uses all numeric if None)
        save_path: Path to save figure (optional)
        figsize: Figure size tuple

    Returns:
        Path to saved figure if save_path provided, else None
    """
    try:
        # Select numeric columns
        if features is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            # Remove target if present
            numeric_cols = [col for col in numeric_cols if col not in ['target', 'weather_event']]
        else:
            numeric_cols = [col for col in features if col in data.columns]

        if len(numeric_cols) == 0:
            print("No numeric features found for correlation plot")
            return None

        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr()

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Plot heatmap
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                    vmin=-1, vmax=1, annot=False, ax=ax)

        ax.set_title('Feature Correlation Matrix',
                     fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Correlation plot saved to {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None

    except Exception as e:
        print(f"Error plotting correlations: {e}")
        return None


def plot_top_correlated_features(data, target_feature, top_n=15, save_path=None, figsize=(10, 8)):
    """
    Plot features most correlated with a target feature.

    Args:
        data: DataFrame with features
        target_feature: Name of target feature
        top_n: Number of top correlated features to show
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    """
    try:
        if target_feature not in data.columns:
            print(f"Target feature '{target_feature}' not found in data")
            return None

        # Calculate correlations
        correlations = data.corr()[target_feature].sort_values(ascending=False)

        # Remove self-correlation
        correlations = correlations[correlations.index != target_feature]

        # Get top N positive and negative correlations
        top_corr = pd.concat([
            correlations.head(top_n // 2),
            correlations.tail(top_n // 2)
        ]).sort_values()

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        colors = ['red' if x < 0 else 'green' for x in top_corr]
        bars = ax.barh(range(len(top_corr)), top_corr, color=colors, alpha=0.7)

        ax.set_yticks(range(len(top_corr)))
        ax.set_yticklabels(top_corr.index)
        ax.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Features Correlated with {target_feature}',
                     fontsize=14, fontweight='bold', pad=20)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_corr)):
            label_x = value + 0.01 if value > 0 else value - 0.01
            ax.text(label_x, i, f'{value:.3f}',
                    va='center', ha='left' if value > 0 else 'right',
                    fontsize=9)

        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Correlation plot saved to {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None

    except Exception as e:
        print(f"Error plotting correlated features: {e}")
        return None