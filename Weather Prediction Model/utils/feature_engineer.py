"""
Feature engineering for weather prediction pipeline.
PRODUCTION-READY VERSION – All critical issues resolved:
  - Configurable feature list (full vs. forecastable-only)
  - Lag features actually generated
  - Rolling windows use proper min_periods
  - Anomaly NaN fallback via city/global stats
  - Climatology column name uses 'day_of_year' (consistent with api_client)
  - FIX: anomaly features are now used during training (climatology passed in)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ---------------------------------------------------------------------------
# Features that are available from a live Open-Meteo forecast.
# Used when training the dedicated future-prediction model.
# ---------------------------------------------------------------------------
FORECASTABLE_FEATURES = [
    # Raw forecast variables
    'tmax', 'tmin', 'temp_range', 'temp_mean',
    'precipitation', 'rh_mean', 'wind_speed', 'solar_radi',

    # Rolling windows (built from the sliding window of forecast + recent history)
    'precip_7day_avg', 'precip_14day_avg', 'precip_30day_avg',
    'precip_60day_avg', 'precip_intensity',
    'temp_14day_avg', 'temp_60day_avg',
    'precip_30day_std',

    # Atmospheric memory (derived from pressure/humidity sequences)
    'pressure_trend_3d', 'pressure_momentum', 'humidity_momentum',

    # Cyclical time features
    'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
    'monsoon_active',

    # Geographical
    'latitude', 'longitude', 'elevation',

    # Categorical
    'city', 'climate_zone', 'season',

    # Anomaly features (z-scores against climatology)
    'tmax_anomaly', 'tmin_anomaly', 'precip_anomaly', 'rh_anomaly',
    'pressure_anomaly',

    # Interaction / ratio features (fully derivable from forecast vars)
    'rh_temp_interaction', 'wind_precip_interaction',
    'precip_per_temp', 'precip_recent_vs_longterm', 'temp_anomaly',
    'precip_cv',
]

# Full feature list (includes SPI-based and lag features for historical training)
FULL_FEATURES = FORECASTABLE_FEATURES + [
    # Drought indices (available in historical data)
    'spi_3', 'spi_severity', 'spi_change',

    # Lag features
    'precipitation_lag1', 'precipitation_lag2',
    'precipitation_lag3', 'precipitation_lag7',
    'temp_mean_lag1', 'temp_mean_lag2',
    'temp_mean_lag3', 'temp_mean_lag7',
    'spi_3_lag1', 'spi_3_lag2', 'spi_3_lag3', 'spi_3_lag7',
    'tmax_lag1', 'tmin_lag1',

    # Heat / cumulative indices
    'heat_index', 'cdd', 'gdd', 'thermal_balance',

    # Extra time granularity
    'month', 'day_of_year',

    # Persistence / variability (require SPI)
    'drought_persistence',
]


class FeatureEngineer:
    """Create and transform features for weather prediction.

    Parameters
    ----------
    config : ConfigLoader
    feature_set : str
        'full'        – all features including SPI and lags (for historical training).
        'forecastable' – only features derivable from live API forecasts.
    feature_list : list | None
        Override the auto-selected list entirely.
    climatology : pd.DataFrame | None
        Precomputed climatology (means/stds per city and day_of_year) for anomaly features.
    """

    def __init__(self, config, feature_set: str = 'full', feature_list=None, climatology=None):
        self.config = config
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None          # populated after first fit
        self.scaled_columns = None
        self.feature_medians = {}
        self.climatology = climatology     # NEW: store climatology

        # Resolve feature list
        if feature_list is not None:
            self._candidate_features = list(feature_list)
        elif feature_set == 'forecastable':
            self._candidate_features = list(FORECASTABLE_FEATURES)
        else:
            self._candidate_features = list(FULL_FEATURES)

    # ------------------------------------------------------------------
    # Climatology helpers
    # ------------------------------------------------------------------

    def set_climatology(self, climatology_df):
        """Set precomputed climatology DataFrame (from OpenMeteoClient)."""
        self.climatology = climatology_df

    # ------------------------------------------------------------------
    # Feature creation
    # ------------------------------------------------------------------

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build all engineered features from raw data."""
        data = data.copy()

        # Ensure temporal order within each city
        if 'date' in data.columns and 'city' in data.columns:
            data = data.sort_values(['city', 'date']).reset_index(drop=True)

        # ── Basic temperature derivations ──────────────────────────────
        data['temp_range'] = data['tmax'] - data['tmin']
        data['temp_mean']  = (data['tmax'] + data['tmin']) / 2

        # ── Cyclical seasonality ───────────────────────────────────────
        if 'date' in data.columns:
            dt = pd.to_datetime(data['date'])
            data['month']       = dt.dt.month
            data['day_of_year'] = dt.dt.dayofyear
            data['month_sin']   = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos']   = np.cos(2 * np.pi * data['month'] / 12)
            data['doy_sin']     = np.sin(2 * np.pi * data['day_of_year'] / 365)
            data['doy_cos']     = np.cos(2 * np.pi * data['day_of_year'] / 365)
            data['season']      = data['month'].map({
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring',  4: 'spring', 5: 'spring',
                6: 'summer',  7: 'summer', 8: 'summer',
                9: 'fall',   10: 'fall',  11: 'fall',
            })

        # ── Atmospheric memory: pressure trend & momentum ──────────────
        if 'surface_pressure' in data.columns:
            data['pressure_trend_3d'] = (
                data.groupby('city')['surface_pressure'].diff(3)
            )
            data['pressure_momentum'] = (
                data.groupby('city')['surface_pressure'].diff(1)
            )

        # ── Humidity momentum ──────────────────────────────────────────
        if 'rh_mean' in data.columns:
            data['humidity_momentum'] = (
                data.groupby('city')['rh_mean'].diff(1)
            )

        # ── Anomaly features (if climatology is attached) ──────────────
        if self.climatology is not None and 'day_of_year' in data.columns:
            data = self._add_anomaly_features(data)

        # ── Rolling features ───────────────────────────────────────────
        grouped = data.groupby('city')
        rolling_specs = {
            'precip_7day_avg':   (7,  'precipitation', 'mean',  7),
            'precip_14day_avg':  (14, 'precipitation', 'mean', 14),
            'precip_30day_avg':  (30, 'precipitation', 'mean', 30),
            'precip_60day_avg':  (60, 'precipitation', 'mean', 60),
            'temp_14day_avg':    (14, 'temp_mean',      'mean', 14),
            'temp_60day_avg':    (60, 'temp_mean',      'mean', 60),
            'precip_30day_std':  (30, 'precipitation', 'std',   7),
        }
        for new_col, (window, src_col, stat, min_p) in rolling_specs.items():
            if src_col in data.columns:
                if stat == 'mean':
                    data[new_col] = grouped[src_col].transform(
                        lambda x, w=window, mp=min_p:
                            x.rolling(window=w, min_periods=mp).mean()
                    )
                else:  # std
                    data[new_col] = grouped[src_col].transform(
                        lambda x, w=window, mp=min_p:
                            x.rolling(window=w, min_periods=mp).std()
                    )

        # ── Precipitation intensity ────────────────────────────────────
        if 'precip_30day_avg' in data.columns:
            data['precip_intensity'] = (
                data['precipitation'] / (data['precip_30day_avg'] + 1e-6)
            )

        # ── SPI-derived features (historical data only) ────────────────
        if 'spi_3' in data.columns:
            data['spi_severity'] = np.abs(data['spi_3'])
            data['spi_change']   = (
                data.groupby('city')['spi_3'].diff().fillna(0)
            )
        if 'spi_6' in data.columns and 'spi_3' in data.columns:
            data['spi_divergence'] = data['spi_6'] - data['spi_3']

        # ── Monsoon active flag ────────────────────────────────────────
        if 'month' in data.columns:
            data['monsoon_active'] = (
                data['month'].apply(lambda m: 1 if m in [5,6,7,8,9,10] else 0)
            )

        # ── Heat stress flag ───────────────────────────────────────────
        if 'heat_index' in data.columns:
            data['heat_stress'] = (data['heat_index'] > 35).astype(int)

        # ── Thermal balance (requires cdd & gdd from collector) ───────
        if 'cdd' in data.columns and 'gdd' in data.columns:
            data['thermal_balance'] = data['gdd'] - data['cdd']

        # ── Interaction features ───────────────────────────────────────
        if 'rh_mean' in data.columns and 'temp_mean' in data.columns:
            data['rh_temp_interaction'] = data['rh_mean'] * data['temp_mean']
        if 'wind_speed' in data.columns and 'precipitation' in data.columns:
            data['wind_precip_interaction'] = (
                data['wind_speed'] * data['precipitation']
            )

        # ── Ratio / composite features ─────────────────────────────────
        if 'precipitation' in data.columns and 'temp_mean' in data.columns:
            data['precip_per_temp'] = (
                data['precipitation'] / (data['temp_mean'] + 1)
            )
        if 'precip_7day_avg' in data.columns and 'precip_60day_avg' in data.columns:
            data['precip_recent_vs_longterm'] = (
                data['precip_7day_avg'] / (data['precip_60day_avg'] + 1e-6)
            )
        if 'temp_mean' in data.columns and 'temp_60day_avg' in data.columns:
            data['temp_anomaly'] = data['temp_mean'] - data['temp_60day_avg']
        if 'precip_30day_std' in data.columns and 'precip_30day_avg' in data.columns:
            data['precip_cv'] = (
                data['precip_30day_std'] / (data['precip_30day_avg'] + 1e-6)
            )

        # ── Drought persistence (requires SPI) ────────────────────────
        if 'spi_3' in data.columns:
            data['drought_persistence'] = data.groupby('city')['spi_3'].transform(
                lambda x: (x < -1).astype(int).groupby(
                    ((x < -1).astype(int) != (x < -1).astype(int).shift()).cumsum()
                ).cumsum()
            )

        # ── Lag features ───────────────────────────────────────
        lag_config = {
            'precipitation': [1, 2, 3, 7],
            'temp_mean':     [1, 2, 3, 7],
            'spi_3':         [1, 2, 3, 7],
            'tmax':          [1],
            'tmin':          [1],
        }
        for col, lags in lag_config.items():
            if col in data.columns:
                for lag in lags:
                    data[f'{col}_lag{lag}'] = (
                        data.groupby('city')[col].shift(lag)
                    )

        return data

    def _add_anomaly_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge climatology and compute z-score anomaly columns.
        FIX: fills missing climatology values with city-level and then
        global fallbacks so anomaly columns never contain NaN from missing
        day-of-year entries.
        """
        df = data.merge(self.climatology, on=['city', 'day_of_year'], how='left')

        var_triples = [
            ('tmax',             'tmax_mean',             'tmax_std'),
            ('tmin',             'tmin_mean',             'tmin_std'),
            ('precipitation',    'precipitation_mean',    'precipitation_std'),
            ('rh_mean',          'rh_mean_mean',          'rh_mean_std'),
            ('surface_pressure', 'surface_pressure_mean', 'surface_pressure_std'),
        ]

        for var, mean_col, std_col in var_triples:
            if var not in df.columns or mean_col not in df.columns:
                continue

            # Fill missing climatology mean/std with city-level aggregates
            # then global fallback
            if df[mean_col].isna().any():
                df[mean_col] = df.groupby('city')[mean_col].transform(
                    lambda s: s.fillna(s.mean())
                )
                global_mean = (
                    self.climatology[mean_col].mean()
                    if mean_col in self.climatology.columns
                    else df[var].mean()
                )
                df[mean_col] = df[mean_col].fillna(global_mean)

            if df[std_col].isna().any():
                df[std_col] = df.groupby('city')[std_col].transform(
                    lambda s: s.fillna(s.mean())
                )
                global_std = (
                    self.climatology[std_col].mean()
                    if std_col in self.climatology.columns
                    else df[var].std()
                )
                df[std_col] = df[std_col].fillna(max(global_std, 0.1))

            df[f'{var}_anomaly'] = (
                (df[var] - df[mean_col]) / df[std_col].replace(0, 1)
            )

        # Drop all helper climatology columns
        drop_cols = [c for c in df.columns if c.endswith('_mean') or c.endswith('_std')]
        df = df.drop(columns=drop_cols, errors='ignore')
        return df

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    def select_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Select the feature columns for modelling.

        Training (fit=True): intersect candidate list with available columns,
            then store the final list.
        Inference (fit=False): replicate the stored training list, padding
            any missing column with its training median (or 0).
        """
        # Drop columns that must never be features
        drop_always = ['date', 'target', 'monsoon_season', 'monsoon_type',
                       'data_source', 'weather_event']
        data = data.drop(
            columns=[c for c in drop_always if c in data.columns], errors='ignore'
        )

        if fit:
            available = set(data.columns)
            feature_list = [f for f in self._candidate_features if f in available]
            self.feature_names = feature_list
        else:
            if self.feature_names is None:
                raise ValueError(
                    "FeatureEngineer has not been fitted yet (feature_names is None)."
                )
            feature_list = self.feature_names

        # Pad missing columns with training medians (or 0 / -1 for categoricals)
        for col in feature_list:
            if col not in data.columns:
                if col in self.label_encoders:
                    data[col] = -1
                else:
                    data[col] = self.feature_medians.get(col, 0.0)

        return data[feature_list].copy()

    # ------------------------------------------------------------------
    # Encoding & scaling
    # ------------------------------------------------------------------

    def encode_categorical(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Label-encode all object/category columns."""
        data = data.copy()
        cat_cols = data.select_dtypes(include=['object', 'category']).columns

        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                data[col] = data[col].astype(str)
                data[col] = le.fit_transform(data[col])
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    data[col] = data[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        return data

    def scale_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """StandardScaler on numeric columns; stores medians for NaN filling."""
        data = data.copy()
        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if fit:
            all_nan = [c for c in num_cols if data[c].isnull().all()]
            if all_nan:
                raise ValueError(f"All-NaN columns detected: {all_nan}")

            self.feature_medians = {}
            for col in num_cols:
                if data[col].isnull().any():
                    med = data[col].median()
                    self.feature_medians[col] = med if not pd.isna(med) else 0.0
                    data[col] = data[col].fillna(self.feature_medians[col])

            self.scaler        = StandardScaler()
            self.scaled_columns = num_cols
            data[num_cols]     = self.scaler.fit_transform(data[num_cols])

        else:
            if self.scaler is not None and self.scaled_columns is not None:
                cols = [c for c in self.scaled_columns if c in num_cols]
                for col in cols:
                    if data[col].isnull().any():
                        data[col] = data[col].fillna(
                            self.feature_medians.get(col, 0.0)
                        )
                if cols:
                    data[cols] = self.scaler.transform(data[cols])

        return data

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def prepare_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Run the complete feature-engineering pipeline."""
        data = self.create_features(data)
        data = self.select_features(data, fit=fit)
        data = self.encode_categorical(data, fit=fit)
        data = self.scale_features(data, fit=fit)
        return data

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def get_feature_importance_names(self):
        return self.feature_names if self.feature_names is not None else []

    def save(self, filepath: str):
        import joblib
        joblib.dump({
            'scaler':              self.scaler,
            'label_encoders':      self.label_encoders,
            'feature_names':       self.feature_names,
            'scaled_columns':      self.scaled_columns,
            'feature_medians':     self.feature_medians,
            '_candidate_features': self._candidate_features,
            'climatology':         self.climatology,   # NEW: save climatology
        }, filepath)
        print(f"FeatureEngineer saved → {filepath}")

    def load(self, filepath: str):
        import joblib
        d = joblib.load(filepath)
        self.scaler              = d['scaler']
        self.label_encoders      = d['label_encoders']
        self.feature_names       = d['feature_names']
        self.scaled_columns      = d['scaled_columns']
        self.feature_medians     = d['feature_medians']
        self._candidate_features = d.get('_candidate_features', list(FULL_FEATURES))
        self.climatology         = d.get('climatology', None)   # NEW: load climatology


if __name__ == '__main__':
    print("FeatureEngineer class defined successfully")