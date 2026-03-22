"""
Open-Meteo API client – fetches only forecast.
Historical data is loaded from the collector's CSV.
"""
import pandas as pd
import requests
from datetime import datetime
import time


class OpenMeteoClient:
    """Fetch live weather forecasts from Open-Meteo."""

    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    CITY_COORDS = {
        'Nuwara_Eliya': {'lat': 6.97,  'lon': 80.77},
        'Badulla':      {'lat': 6.98,  'lon': 81.06},
        'Matale':       {'lat': 7.47,  'lon': 80.62},
        'Ratnapura':    {'lat': 6.68,  'lon': 80.40},
        'Jaffna':       {'lat': 9.67,  'lon': 80.00},
        'Hambantota':   {'lat': 6.12,  'lon': 81.12},
        'Kurunegala':   {'lat': 7.48,  'lon': 80.36},
    }

    def __init__(self, config, historical_data=None):
        """
        Args:
            config          : ConfigLoader instance.
            historical_data : Optional DataFrame (used only to build climatology).
        """
        self.config          = config
        self.historical_data = historical_data
        self.climatology     = None
        if historical_data is not None:
            self._build_climatology()

    def _build_climatology(self):
        """Pre‑compute daily climatology from historical data."""
        df = self.historical_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_year'] = df['date'].dt.dayofyear
        latest_year = df['date'].dt.year.max()
        baseline_start = latest_year - 20
        df_base = df[df['date'].dt.year >= baseline_start]
        vars_to_agg = ['tmax', 'tmin', 'precipitation', 'rh_mean']
        if 'surface_pressure' in df_base.columns:
            vars_to_agg.append('surface_pressure')
        agg_funcs = {v: ['mean', 'std'] for v in vars_to_agg}
        clim = (
            df_base.groupby(['city', 'day_of_year'])
                   .agg(agg_funcs)
        )
        clim.columns = ['_'.join(col) for col in clim.columns]
        clim = clim.reset_index()
        for col in clim.columns:
            if col.endswith('_std'):
                clim[col] = clim[col].fillna(0.1).replace(0, 0.1)
        self.climatology = clim

    def get_forecast(self, cities, start_date=None, days: int = 14) -> pd.DataFrame:
        """
        Fetch daily forecast for the given cities (no historical data).
        """
        if start_date is None:
            start_date = datetime.now().date()
        else:
            start_date = pd.to_datetime(start_date).date()

        all_data = []

        for city in cities:
            if city not in self.CITY_COORDS:
                print(f"⚠️  No coordinates for {city}, skipping.")
                continue
            lat = self.CITY_COORDS[city]['lat']
            lon = self.CITY_COORDS[city]['lon']

            params = {
                'latitude':  lat,
                'longitude': lon,
                'daily': [
                    'temperature_2m_max', 'temperature_2m_min',
                    'precipitation_sum', 'relative_humidity_2m_mean',
                    'surface_pressure_mean', 'windspeed_10m_max',
                    'shortwave_radiation_sum'
                ],
                'timezone':     'auto',
                'forecast_days': min(days, 16),
            }

            try:
                resp = requests.get(self.FORECAST_URL, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"❌ Error fetching {city}: {e}")
                continue

            daily   = data.get('daily', {})
            dates   = daily.get('time', [])
            n       = len(dates)
            _get    = lambda key: daily.get(key, [None] * n)

            tmax     = _get('temperature_2m_max')
            tmin     = _get('temperature_2m_min')
            precip   = _get('precipitation_sum')
            rh       = _get('relative_humidity_2m_mean')
            pressure = _get('surface_pressure_mean')
            wind     = _get('windspeed_10m_max')
            solar    = _get('shortwave_radiation_sum')

            for i, d in enumerate(dates):
                all_data.append({
                    'date':             d,
                    'city':             city,
                    'tmax':             tmax[i],
                    'tmin':             tmin[i],
                    'precipitation':    precip[i],
                    'rh_mean':          rh[i],
                    'surface_pressure': pressure[i],
                    'wind_speed':       wind[i],
                    'solar_radi':       solar[i],
                })

            time.sleep(0.5)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def add_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add anomaly columns using climatology (unchanged)."""
        if self.climatology is None:
            raise ValueError("Climatology not built. Pass historical_data to the constructor.")
        df = df.copy()
        df['day_of_year'] = df['date'].dt.dayofyear
        df = df.merge(self.climatology, on=['city', 'day_of_year'], how='left')
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
            if df[mean_col].isna().any():
                df[mean_col] = df.groupby('city')[mean_col].transform(lambda s: s.fillna(s.mean()))
                global_mean = self.climatology[mean_col].mean()
                df[mean_col] = df[mean_col].fillna(global_mean)
            if df[std_col].isna().any():
                df[std_col] = df.groupby('city')[std_col].transform(lambda s: s.fillna(s.mean()))
                global_std = max(self.climatology[std_col].mean(), 0.1)
                df[std_col] = df[std_col].fillna(global_std)
            df[f'{var}_anomaly'] = (df[var] - df[mean_col]) / df[std_col].replace(0, 1)
        drop_cols = [c for c in df.columns if c.endswith('_mean') or c.endswith('_std')]
        df = df.drop(columns=drop_cols, errors='ignore')
        return df