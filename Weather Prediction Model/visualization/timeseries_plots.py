"""Time series visualization for predictions and events."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def plot_predictions_timeline(predictions_df, cities=None, date_range=None,
                              save_path=None, figsize=(15, 8)):
    """
    Plot predicted vs actual events over time.

    Args:
        predictions_df: DataFrame with date, city, predicted_class, actual_class
        cities: List of cities to plot (plots all if None)
        date_range: Tuple of (start_date, end_date) to filter
        save_path: Path to save figure (optional)
        figsize: Figure size tuple

    Returns:
        Path to saved figure if save_path provided, else None
    """
    try:
        # Validate required columns
        required_cols = ['date', 'city', 'predicted_class']
        if not all(col in predictions_df.columns for col in required_cols):
            print(f"DataFrame must contain columns: {required_cols}")
            return None

        df = predictions_df.copy()

        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Filter by date range
        if date_range:
            df = df[(df['date'] >= date_range[0]) & (df['date'] <= date_range[1])]

        # Filter by cities
        if cities:
            df = df[df['city'].isin(cities)]
        else:
            cities = df['city'].unique()[:4]  # Max 4 cities for readability
            df = df[df['city'].isin(cities)]

        # Map events to numeric values for plotting
        event_mapping = {'drought': 2, 'flood_risk': 1, 'normal': 0}
        df['pred_numeric'] = df['predicted_class'].map(event_mapping)

        if 'actual_class' in df.columns:
            df['actual_numeric'] = df['actual_class'].map(event_mapping)
            has_actual = True
        else:
            has_actual = False

        # Create subplots for each city
        n_cities = len(cities)
        fig, axes = plt.subplots(n_cities, 1, figsize=figsize, sharex=True)

        if n_cities == 1:
            axes = [axes]

        colors = {'drought': '#d62728', 'flood_risk': '#1f77b4', 'normal': '#2ca02c'}

        for i, city in enumerate(cities):
            ax = axes[i]
            city_data = df[df['city'] == city].sort_values('date')

            if len(city_data) == 0:
                continue

            # Plot predicted events
            for event, color in colors.items():
                mask = city_data['predicted_class'] == event
                if mask.any():
                    ax.scatter(city_data[mask]['date'],
                               city_data[mask]['pred_numeric'],
                               c=color, label=f'Pred: {event}',
                               alpha=0.6, s=30, marker='o')

            # Plot actual events if available
            if has_actual:
                ax.plot(city_data['date'], city_data['actual_numeric'],
                        color='black', alpha=0.3, linewidth=1,
                        label='Actual', linestyle='-')

            ax.set_ylabel(city, fontsize=11, fontweight='bold')
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['Normal', 'Flood', 'Drought'])
            ax.grid(alpha=0.3, linestyle='--', axis='x')
            ax.set_ylim(-0.5, 2.5)

            if i == 0:
                ax.legend(loc='upper right', fontsize=8, ncol=4)

            # Add horizontal lines for reference
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.axhline(y=1, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.axhline(y=2, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
        fig.suptitle('Weather Events Timeline - Predictions vs Actual',
                     fontsize=14, fontweight='bold', y=0.995)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Timeline plot saved to {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None

    except Exception as e:
        print(f"Error plotting predictions timeline: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_events_by_city(predictions_df, event_type='drought',
                        save_path=None, figsize=(12, 6)):
    """
    Plot event frequency by city.

    Args:
        predictions_df: DataFrame with city and predicted_class columns
        event_type: Type of event to plot ('drought', 'flood_risk', or 'all')
        save_path: Path to save figure (optional)
        figsize: Figure size tuple

    Returns:
        Path to saved figure if save_path provided, else None
    """
    try:
        df = predictions_df.copy()

        if event_type != 'all':
            # Count specific event type
            event_counts = df[df['predicted_class'] == event_type].groupby('city').size()
            title = f'{event_type.replace("_", " ").title()} Events by City'
            colors = ['#d62728'] if event_type == 'drought' else ['#1f77b4']
        else:
            # Count all event types
            event_counts = df.groupby(['city', 'predicted_class']).size().unstack(fill_value=0)
            title = 'Weather Events by City'
            colors = ['#2ca02c', '#d62728', '#1f77b4']  # normal, drought, flood

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        if event_type != 'all':
            event_counts.sort_values(ascending=True).plot(
                kind='barh', ax=ax, color=colors[0], alpha=0.7
            )
            ax.set_xlabel('Number of Events', fontsize=12, fontweight='bold')
        else:
            event_counts.plot(kind='bar', ax=ax, color=colors, alpha=0.7)
            ax.set_xlabel('City', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Events', fontsize=12, fontweight='bold')
            ax.legend(title='Event Type', labels=['Normal', 'Drought', 'Flood Risk'])
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x' if event_type != 'all' else 'y', alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Events by city plot saved to {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None

    except Exception as e:
        print(f"Error plotting events by city: {e}")
        return None


def plot_monthly_event_distribution(predictions_df, cities=None,
                                    save_path=None, figsize=(14, 8)):
    """
    Plot event distribution by month for each city.

    Args:
        predictions_df: DataFrame with date, city, predicted_class
        cities: List of cities to include (all if None)
        save_path: Path to save figure (optional)
        figsize: Figure size tuple

    Returns:
        Path to saved figure if save_path provided, else None
    """
    try:
        df = predictions_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month

        if cities:
            df = df[df['city'].isin(cities)]
        else:
            cities = df['city'].unique()

        # Create heatmap for each event type
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        event_types = ['drought', 'flood_risk', 'normal']
        titles = ['Drought Events', 'Flood Risk Events', 'Normal Days']
        cmaps = ['Reds', 'Blues', 'Greens']

        for idx, (event, title, cmap) in enumerate(zip(event_types, titles, cmaps)):
            ax = axes[idx]

            # Count events by city and month
            event_data = df[df['predicted_class'] == event].groupby(['city', 'month']).size().unstack(fill_value=0)

            # Reindex to ensure all months are present
            event_data = event_data.reindex(columns=range(1, 13), fill_value=0)

            # Create heatmap
            sns.heatmap(event_data, annot=True, fmt='d', cmap=cmap,
                        cbar_kws={'label': 'Count'}, ax=ax,
                        vmin=0, linewidths=0.5, linecolor='gray')

            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel('Month', fontsize=10, fontweight='bold')
            ax.set_ylabel('City' if idx == 0 else '', fontsize=10, fontweight='bold')
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

        plt.suptitle('Monthly Event Distribution by City',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Monthly distribution plot saved to {save_path}")
            plt.close()
            return save_path
        else:
            plt.show()
            return None

    except Exception as e:
        print(f"Error plotting monthly distribution: {e}")
        return None