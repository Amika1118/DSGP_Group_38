"""Model performance comparison and confusion matrix visualization."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion_matrix(y_true, y_pred, classes=None, model_name='Model',
                          save_path=None, figsize=(10, 8), normalize=False):
    """
    Plot confusion matrix for classification results.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names (auto-detected if None)
        model_name: Name of model for title
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
        normalize: Whether to normalize counts to percentages

    Returns:
        Path to saved figure if save_path provided, else None
    """
    try:
        # Get class names if not provided
        if classes is None:
            classes = sorted(list(set(y_true) | set(y_pred)))

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title_suffix = ' (Normalized)'
        else:
            fmt = 'd'
            title_suffix = ''

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=classes, yticklabels=classes,
                    square=True, cbar_kws={"shrink": 0.8}, ax=ax)

        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix - {model_name}{title_suffix}',
                     fontsize=14, fontweight='bold', pad=20)

        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None

    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")
        return None


def plot_model_comparison(comparison_df, metric='f1_macro', save_path=None, figsize=(12, 6)):
    """
    Plot comparison of multiple models across metrics.

    Args:
        comparison_df: DataFrame with model comparison results
        metric: Metric to highlight (default 'f1_macro')
        save_path: Path to save figure (optional)
        figsize: Figure size tuple

    Returns:
        Path to saved figure if save_path provided, else None
    """
    try:
        if 'model' not in comparison_df.columns:
            print("DataFrame must contain 'model' column")
            return None

        # Determine metrics to plot
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns.tolist()
        metrics_to_plot = [col for col in numeric_cols if col not in ['gap']]

        if len(metrics_to_plot) == 0:
            print("No numeric metrics found to plot")
            return None

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(comparison_df))
        width = 0.8 / len(metrics_to_plot)

        colors = sns.color_palette("husl", len(metrics_to_plot))

        for i, metric_name in enumerate(metrics_to_plot):
            offset = (i - len(metrics_to_plot) / 2 + 0.5) * width
            bars = ax.bar(x + offset, comparison_df[metric_name],
                          width, label=metric_name.replace('_', ' ').title(),
                          color=colors[i], alpha=0.8)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['model'], rotation=45, ha='right')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.1)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Model comparison plot saved to {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None

    except Exception as e:
        print(f"Error plotting model comparison: {e}")
        return None


def plot_overfitting_analysis(overfitting_df, save_path=None, figsize=(12, 6)):
    """
    Plot train vs validation performance to visualize overfitting.

    Args:
        overfitting_df: DataFrame with train/val scores and fit_status
        save_path: Path to save figure (optional)
        figsize: Figure size tuple

    Returns:
        Path to saved figure if save_path provided, else None
    """
    try:
        required_cols = ['model', 'train_f1', 'val_f1', 'fit_status']
        if not all(col in overfitting_df.columns for col in required_cols):
            print(f"DataFrame must contain columns: {required_cols}")
            return None

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Train vs Val F1 scores
        x = np.arange(len(overfitting_df))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, overfitting_df['train_f1'], width,
                        label='Train F1', color='steelblue', alpha=0.8)
        bars2 = ax1.bar(x + width / 2, overfitting_df['val_f1'], width,
                        label='Val F1', color='coral', alpha=0.8)

        ax1.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax1.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
        ax1.set_title('Train vs Validation Performance',
                      fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(overfitting_df['model'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(0, 1.1)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.3f}',
                         ha='center', va='bottom', fontsize=8)

        # Plot 2: Gap (Train - Val)
        colors = []
        for status in overfitting_df['fit_status']:
            if status == 'overfitting':
                colors.append('red')
            elif status == 'underfitting':
                colors.append('orange')
            elif status == 'suspicious':
                colors.append('yellow')
            else:
                colors.append('green')

        bars3 = ax2.bar(x, overfitting_df['gap'], color=colors, alpha=0.7)

        ax2.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Gap (Train - Val)', fontsize=11, fontweight='bold')
        ax2.set_title('Generalization Gap',
                      fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(overfitting_df['model'], rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.axhline(y=0.1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Overfitting threshold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels and status markers
        for i, (bar, status) in enumerate(zip(bars3, overfitting_df['fit_status'])):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:+.3f}',
                     ha='center', va='bottom' if height > 0 else 'top',
                     fontsize=8)

            # Add status emoji
            emoji = '⚠️' if status in ['overfitting', 'underfitting'] else '✓'
            ax2.text(i, -0.05, emoji, ha='center', fontsize=12)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Overfitting analysis plot saved to {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None

    except Exception as e:
        print(f"Error plotting overfitting analysis: {e}")
        return None