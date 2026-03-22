"""
Target variable creation for weather event classification.
ENHANCED: Flood now requires 3‑day cumulative precipitation above threshold.
FIXED: Added global drought SPI default fallback.
"""
import pandas as pd
import numpy as np


class TargetCreator:
    def __init__(self, config):
        self.config = config
        self.thresholds = config.get_thresholds_config()
        self.cities = config.get_cities_config()
        self.min_drought_days = self.thresholds.get('min_drought_days', 7)
        self.min_flood_days = self.thresholds.get('min_flood_days', 1)
        self.flood_saturation_percentile = self.thresholds.get('flood_saturation_percentile', 0.90)
        self.drought_spi_default = self.thresholds.get('drought_spi_default', -1.0)   # ADDED global default

        self.fitted_flood_thresholds = None
        self.fitted_saturation_thresholds = None
        self.global_flood_threshold = None
        self.global_saturation_threshold = None

    def get_climate_zone(self, city):
        for zone, cities in self.cities.items():
            if zone != 'all_cities' and city in cities:
                return zone
        return 'intermediate'

    def fit_thresholds(self, df):
        """
        Fit flood thresholds (daily and 3‑day cumulative) on training data.
        """
        flood_thresholds = {}
        saturation_thresholds = {}
        all_precip = []
        all_cum3 = []

        for city in df['city'].unique():
            city_data = df[df['city'] == city].sort_values('date')
            daily = city_data['precipitation'].dropna()
            if len(daily) > 0:
                flood_thresholds[city] = daily.quantile(self.thresholds['flood_percentile'])
                all_precip.extend(daily)

                # Compute 3‑day cumulative precipitation
                cum3 = city_data['precipitation'].rolling(3, min_periods=3).sum().dropna()
                if len(cum3) > 0:
                    saturation_thresholds[city] = cum3.quantile(self.flood_saturation_percentile)
                    all_cum3.extend(cum3)
                else:
                    saturation_thresholds[city] = 150.0  # fallback
            else:
                flood_thresholds[city] = 100.0
                saturation_thresholds[city] = 150.0

        self.fitted_flood_thresholds = flood_thresholds
        self.fitted_saturation_thresholds = saturation_thresholds

        # Global fallbacks
        if all_precip:
            self.global_flood_threshold = np.percentile(all_precip, self.thresholds['flood_percentile'] * 100)
        else:
            self.global_flood_threshold = 100.0

        if all_cum3:
            self.global_saturation_threshold = np.percentile(all_cum3, self.flood_saturation_percentile * 100)
        else:
            self.global_saturation_threshold = 150.0

        print(f"✓ Fitted flood thresholds on training data:")
        for city, thresh in flood_thresholds.items():
            sat = saturation_thresholds.get(city, self.global_saturation_threshold)
            print(f"  {city}: daily {thresh:.2f} mm, 3‑day sat {sat:.2f} mm")
        print(f"  Global fallback: daily {self.global_flood_threshold:.2f} mm, "
              f"3‑day sat {self.global_saturation_threshold:.2f} mm")

        return self

    def create_target(self, df, fit=True):
        """
        Create target variable with flood saturation logic.
        """
        df = df.copy()
        df = df.sort_values(['city', 'date']).reset_index(drop=True)

        if fit:
            self.fit_thresholds(df)
        else:
            if self.fitted_flood_thresholds is None:
                raise ValueError("Must fit on training data first.")

        # Prepare thresholds with fallback
        flood_thresh = self.fitted_flood_thresholds.copy()
        sat_thresh = self.fitted_saturation_thresholds.copy()
        for city in df['city'].unique():
            if city not in flood_thresh:
                print(f"⚠️ City '{city}' not in training, using global thresholds.")
                flood_thresh[city] = self.global_flood_threshold
                sat_thresh[city] = self.global_saturation_threshold

        # Compute 3‑day cumulative precipitation
        df['precip_3day_sum'] = df.groupby('city')['precipitation'].transform(
            lambda x: x.rolling(3, min_periods=3).sum()
        )

        # Initial classification
        def classify_point(row):
            zone = self.get_climate_zone(row['city'])
            drought_thresh = self.thresholds.get(f'drought_spi_{zone}', self.drought_spi_default)   # FIXED: use global default

            # Drought candidate
            if pd.notna(row['spi_3']) and row['spi_3'] < drought_thresh:
                return 'drought_candidate'

            # Flood candidate: daily exceed AND 3‑day cumulative exceed
            flood_daily_ok = pd.notna(row['precipitation']) and row['precipitation'] > flood_thresh[row['city']]
            flood_sat_ok = pd.notna(row['precip_3day_sum']) and row['precip_3day_sum'] > sat_thresh[row['city']]
            if flood_daily_ok and flood_sat_ok:
                return 'flood_candidate'

            return 'normal'

        df['temp_classification'] = df.apply(classify_point, axis=1)

        # Apply duration filters
        df['weather_event'] = 'normal'

        for city in df['city'].unique():
            city_mask = df['city'] == city
            city_data = df[city_mask]

            # Drought: consecutive drought_candidate >= min_drought_days
            drought_mask = city_data['temp_classification'] == 'drought_candidate'
            if drought_mask.any():
                drought_groups = (drought_mask != drought_mask.shift()).cumsum()
                for group_id in city_data[drought_mask].groupby(drought_groups[drought_mask]).groups.keys():
                    idx = city_data[drought_mask][drought_groups[drought_mask] == group_id].index
                    if len(idx) >= self.min_drought_days:
                        df.loc[idx, 'weather_event'] = 'drought'

            # Flood: consecutive flood_candidate >= min_flood_days
            flood_mask = city_data['temp_classification'] == 'flood_candidate'
            if flood_mask.any():
                flood_groups = (flood_mask != flood_mask.shift()).cumsum()
                for group_id in city_data[flood_mask].groupby(flood_groups[flood_mask]).groups.keys():
                    idx = city_data[flood_mask][flood_groups[flood_mask] == group_id].index
                    if len(idx) >= self.min_flood_days:
                        df.loc[idx, 'weather_event'] = 'flood_risk'

        df = df.drop(columns=['temp_classification', 'precip_3day_sum'])
        return df

    def save(self, filepath):
        """Save fitted thresholds to file."""
        import joblib
        data = {
            'fitted_flood_thresholds': self.fitted_flood_thresholds,
            'fitted_saturation_thresholds': self.fitted_saturation_thresholds,
            'global_flood_threshold': self.global_flood_threshold,
            'global_saturation_threshold': self.global_saturation_threshold,
            'min_drought_days': self.min_drought_days,
            'min_flood_days': self.min_flood_days,
            'thresholds': self.thresholds,
            'cities': self.cities,
            'drought_spi_default': self.drought_spi_default
        }
        joblib.dump(data, filepath)

    def load(self, filepath):
        """Load fitted thresholds from file."""
        import joblib
        data = joblib.load(filepath)
        self.fitted_flood_thresholds = data['fitted_flood_thresholds']
        self.fitted_saturation_thresholds = data['fitted_saturation_thresholds']
        self.global_flood_threshold = data['global_flood_threshold']
        self.global_saturation_threshold = data['global_saturation_threshold']
        self.min_drought_days = data['min_drought_days']
        self.min_flood_days = data['min_flood_days']
        self.thresholds = data['thresholds']
        self.cities = data['cities']
        self.drought_spi_default = data.get('drought_spi_default', -1.0)