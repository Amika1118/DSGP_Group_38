"""Visualization module for weather predictions."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


class Visualizer:
    """Create visualizations for weather predictions."""

    def __init__(self, config):
        self.config = config
        sns.set_style('whitegrid')

    def plot_spi_timeseries(self, data, city, save_path=None):
        """Plot SPI-3 time series with drought/flood events."""
        city_data = data[data['city'] == city].copy()

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot SPI-3
        ax.plot(city_data['date'], city_data['spi_3'], label='SPI-3', color='navy', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Highlight drought periods
        drought_mask = city_data['target'] == 'drought'
        ax.fill_between(city_data['date'], -3, 3, where=drought_mask, alpha=0.3, color='red', label='Drought')

        # Highlight flood periods
        flood_mask = city_data['target'] == 'flood_risk'
        ax.fill_between(city_data['date'], -3, 3, where=flood_mask, alpha=0.3, color='blue', label='Flood Risk')

        ax.set_xlabel('Date')
        ax.set_ylabel('SPI-3')
        ax.set_title(f'SPI-3 Time Series with Weather Events - {city}')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()

    def plot_prediction_probabilities(self, probabilities, classes, save_path=None):
        """Plot distribution of prediction probabilities."""
        fig, axes = plt.subplots(1, len(classes), figsize=(15, 4))

        for i, class_name in enumerate(classes):
            axes[i].hist(probabilities[:, i], bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(f'{class_name} Probabilities')
            axes[i].set_xlabel('Probability')
            axes[i].set_ylabel('Frequency')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()
