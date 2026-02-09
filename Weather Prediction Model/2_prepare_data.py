"""
2_prepare_data.py - Data Preprocessing and Feature Engineering
Comprehensive preprocessing pipeline similar to professional ML projects
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

BASE_DIR = "weather_prediction_project"


def load_raw_data():
    """Load raw datasets"""
    print("\n" + "="*70)
    print("📂 LOADING RAW DATA")
    print("="*70)

    # Load climate data
    climate_path = f"{BASE_DIR}/data/raw/global_climate_indicators.csv"
    df_climate = pd.read_csv(climate_path)
    print(f"✓ Climate data: {len(df_climate)} records")

    # Load weather data
    weather_path = f"{BASE_DIR}/data/raw/all_cities_weather_raw.csv"
    df_weather = pd.read_csv(weather_path)
    print(f"✓ Weather data: {len(df_weather)} records")

    return df_climate, df_weather


def preprocess_weather_data(df):
    """
    Clean and engineer features from weather data
    """
    print("\n" + "="*70)
    print("🧹 PREPROCESSING WEATHER DATA")
    print("="*70)

    # Convert time to datetime
    df['date'] = pd.to_datetime(df['time'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter

    # Add season
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df['season'] = df['month'].apply(get_season)

    # Cyclical encoding for temporal features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Rename columns for clarity
    df = df.rename(columns={
        'temperature_2m_mean': 'temp_mean',
        'temperature_2m_max': 'temp_max',
        'temperature_2m_min': 'temp_min',
        'precipitation_sum': 'precipitation',
        'windspeed_10m_max': 'wind_speed',
        'relative_humidity_2m_mean': 'humidity',
        'pressure_msl_mean': 'pressure'
    })

    # Handle missing values
    numeric_cols = ['temp_mean', 'temp_max', 'temp_min', 'precipitation',
                    'wind_speed', 'humidity', 'pressure']

    for col in numeric_cols:
        if col in df.columns:
            # Fill with forward fill, then backward fill, then median
            df[col] = df[col].ffill().bfill()
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

    # Create derived features
    if 'temp_max' in df.columns and 'temp_min' in df.columns:
        df['temp_range'] = df['temp_max'] - df['temp_min']

    # Create lag features (previous day's weather)
    df = df.sort_values(['city', 'date'])
    for col in ['temp_mean', 'precipitation', 'wind_speed']:
        if col in df.columns:
            df[f'{col}_lag1'] = df.groupby('city')[col].shift(1)
            df[f'{col}_lag3'] = df.groupby('city')[col].shift(3)
            df[f'{col}_lag7'] = df.groupby('city')[col].shift(7)

    # Rolling statistics (7-day window)
    for col in ['temp_mean', 'precipitation']:
        if col in df.columns:
            df[f'{col}_rolling_mean_7'] = df.groupby('city')[col].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()
            )
            df[f'{col}_rolling_std_7'] = df.groupby('city')[col].transform(
                lambda x: x.rolling(window=7, min_periods=1).std()
            )

    print(f"✓ Weather preprocessing complete")
    print(f"  Records: {len(df)}")
    print(f"  Features: {len(df.columns)}")

    return df


def merge_climate_and_weather(df_weather, df_climate):
    """
    Merge weather data with global climate indicators
    """
    print("\n" + "="*70)
    print("🔗 MERGING CLIMATE AND WEATHER DATA")
    print("="*70)

    # Merge on year and month
    df_merged = pd.merge(
        df_weather,
        df_climate,
        on=['year', 'month'],
        how='left'
    )

    # Create climate lag features
    df_merged = df_merged.sort_values(['city', 'date'])

    climate_features = ['global_temp_anomaly_c', 'co2_ppm', 'enso_index']
    for col in climate_features:
        if col in df_merged.columns:
            df_merged[f'{col}_lag1'] = df_merged.groupby('city')[col].shift(1)
            df_merged[f'{col}_lag3'] = df_merged.groupby('city')[col].shift(3)

    # Create interaction features
    if 'temp_mean' in df_merged.columns and 'global_temp_anomaly_c' in df_merged.columns:
        df_merged['temp_climate_interaction'] = (
            df_merged['temp_mean'] * df_merged['global_temp_anomaly_c']
        )

    # Fill any remaining NaN values
    df_merged = df_merged.bfill().ffill()

    # Fill any still-remaining NaN with median
    for col in df_merged.select_dtypes(include=[np.number]).columns:
        if df_merged[col].isnull().any():
            df_merged[col] = df_merged[col].fillna(df_merged[col].median())

    print(f"✓ Merge complete")
    print(f"  Combined records: {len(df_merged)}")
    print(f"  Total features: {len(df_merged.columns)}")

    return df_merged


def create_target_variable(df):
    """
    Create prediction target: next day's temperature
    """
    print("\n" + "="*70)
    print("🎯 CREATING TARGET VARIABLE")
    print("="*70)

    df = df.sort_values(['city', 'date'])

    # Target: predict tomorrow's mean temperature
    df['target_temp_next_day'] = df.groupby('city')['temp_mean'].shift(-1)

    # Remove rows where we don't have target (last day for each city)
    initial_len = len(df)
    df = df.dropna(subset=['target_temp_next_day'])

    print(f"✓ Target variable created")
    print(f"  Target: next_day_temperature")
    print(f"  Records with target: {len(df)} (removed {initial_len - len(df)} rows)")
    print(f"  Target range: {df['target_temp_next_day'].min():.2f}°C to "
          f"{df['target_temp_next_day'].max():.2f}°C")

    return df


def prepare_for_modeling(df):
    """
    Prepare final dataset for modeling with train-test split
    """
    print("\n" + "="*70)
    print("🔧 PREPARING FOR MODELING")
    print("="*70)

    # Select features (exclude metadata and target)
    exclude_cols = [
        'time', 'date', 'latitude', 'longitude',
        'target_temp_next_day', 'city'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Separate features and target
    X = df[feature_cols].copy()
    y = df['target_temp_next_day'].copy()

    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Also encode city for stratification
    city_encoder = LabelEncoder()
    city_encoded = city_encoder.fit_transform(df['city'])

    # Time-based split (80% train, 20% test)
    # Sort by date to ensure temporal ordering
    df_sorted = df.sort_values('date').reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)

    # Reset index on X and y to match df_sorted
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Now split using integer positions
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # Create preprocessing pipeline
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features)
    ])

    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save preprocessor
    joblib.dump(preprocessor, f'{BASE_DIR}/preprocessing/preprocessor.pkl')
    joblib.dump(label_encoders, f'{BASE_DIR}/preprocessing/label_encoders.pkl')
    joblib.dump(city_encoder, f'{BASE_DIR}/preprocessing/city_encoder.pkl')

    # Save feature names
    feature_names = numeric_features
    with open(f'{BASE_DIR}/preprocessing/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))

    print(f"✓ Data prepared for modeling")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    print(f"  Number of features: {len(feature_names)}")
    print(f"  Feature names saved to: preprocessing/feature_names.txt")

    # Save processed data
    np.save(f'{BASE_DIR}/data/processed/X_train.npy', X_train_processed)
    np.save(f'{BASE_DIR}/data/processed/X_test.npy', X_test_processed)
    np.save(f'{BASE_DIR}/data/processed/y_train.npy', y_train.values)
    np.save(f'{BASE_DIR}/data/processed/y_test.npy', y_test.values)

    # Save metadata
    metadata = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(feature_names),
        'target_mean': float(y_train.mean()),
        'target_std': float(y_train.std()),
        'split_date': str(df_sorted.loc[split_idx, 'date'])
    }

    import json
    with open(f'{BASE_DIR}/data/processed/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    return X_train_processed, X_test_processed, y_train.values, y_test.values


def save_eda_summary(df):
    """Save EDA insights"""
    print("\n" + "="*70)
    print("📊 GENERATING EDA SUMMARY")
    print("="*70)

    insights = []
    insights.append(f"Dataset Shape: {df.shape}")
    insights.append(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    insights.append(f"Number of Cities: {df['city'].nunique()}")
    insights.append(f"\nTarget Statistics:")
    insights.append(f"  Mean: {df['target_temp_next_day'].mean():.2f}°C")
    insights.append(f"  Std: {df['target_temp_next_day'].std():.2f}°C")
    insights.append(f"  Min: {df['target_temp_next_day'].min():.2f}°C")
    insights.append(f"  Max: {df['target_temp_next_day'].max():.2f}°C")

    insights.append(f"\nMissing Values: {df.isnull().sum().sum()}")

    # Save
    with open(f'{BASE_DIR}/reports/eda_summary.txt', 'w') as f:
        f.write('\n'.join(insights))

    print(f"✓ EDA summary saved to: reports/eda_summary.txt")


def main():
    """
    Main preprocessing pipeline
    """
    print("="*70)
    print("🚀 WEATHER PREDICTION - DATA PREPROCESSING PIPELINE")
    print("="*70)

    # Load raw data
    df_climate, df_weather = load_raw_data()

    # Preprocess weather
    df_weather = preprocess_weather_data(df_weather)

    # Merge with climate
    df_merged = merge_climate_and_weather(df_weather, df_climate)

    # Create target
    df_final = create_target_variable(df_merged)

    # Save complete dataset
    df_final.to_csv(f'{BASE_DIR}/data/processed/final_dataset.csv', index=False)
    print(f"\n✓ Complete dataset saved to: data/processed/final_dataset.csv")

    # Save EDA summary
    save_eda_summary(df_final)

    # Prepare for modeling
    X_train, X_test, y_train, y_test = prepare_for_modeling(df_final)

    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\n📁 Output files:")
    print(f"  • data/processed/final_dataset.csv")
    print(f"  • data/processed/X_train.npy, X_test.npy")
    print(f"  • data/processed/y_train.npy, y_test.npy")
    print(f"  • preprocessing/preprocessor.pkl")
    print(f"  • reports/eda_summary.txt")


if __name__ == "__main__":
    main()
