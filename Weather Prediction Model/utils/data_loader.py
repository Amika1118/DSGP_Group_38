import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.data_config = config.get_data_config()

    def load_data(self, file_path=None):
        if file_path is None:
            file_path = self.data_config['file_path']
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def split_by_date(self, df):
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        train = df[(df['date'] >= self.data_config['train_start']) &
                   (df['date'] <= self.data_config['train_end'])]
        val = df[(df['date'] >= self.data_config['val_start']) &
                 (df['date'] <= self.data_config['val_end'])]
        test = df[(df['date'] >= self.data_config['test_start']) &
                  (df['date'] <= self.data_config['test_end'])]
        return train, val, test

    def get_X_y(self, df):
        target_col = self.data_config.get('target_column', 'weather_event')
        exclude = ['date', target_col]
        X_cols = [c for c in df.columns if c not in exclude]
        return df[X_cols], df[target_col]

    # Added for compatibility with main.py - alias for split_by_date
    def get_temporal_splits(self, df=None):
        """Wrapper method for compatibility with main.py"""
        # If df is provided, split it
        if df is not None:
            return self.split_by_date(df)
        # Otherwise, load data and split (for backward compatibility)
        else:
            df = self.load_data()
            return self.split_by_date(df)