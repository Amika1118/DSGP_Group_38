"""
Historical Weather Data Collector for Sri Lankan Cities - v2 HYBRID VERSION
NASA POWER API (2000 → ~7 days ago) + Open-Meteo API (recent gap → yesterday)

PRODUCTION FIXES:
  - NASA_END is now dynamic (today - 10 days) so the collector never develops
    a growing blind spot after the hard-coded 2025-12-25 date passes.
  - Master CSV is written as both 'sri_lanka_weather_master.csv' AND
    'processed_data.csv' so config_future.yaml doesn't have to change.
  - Year-by-year NASA fallback is always attempted when a 422 is received.
"""

import os
import json
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy import stats
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

CITIES = {
    'Nuwara_Eliya': (6.970, 80.782),
    'Badulla':      (6.993, 81.055),
    'Matale':       (7.467, 80.623),
    'Ratnapura':    (6.706, 80.385),
    'Jaffna':       (9.661, 80.025),
    'Hambantota':   (6.124, 81.118),
    'Kurunegala':   (7.483, 80.367)
}

CLIMATE_ZONES = {
    'Nuwara_Eliya': {'zone': 'Upcountry Wet', 'elevation': 1895, 'monsoon': 'Both'},
    'Badulla':      {'zone': 'Uva Basin',      'elevation': 680,  'monsoon': 'Both'},
    'Matale':       {'zone': 'Intermediate',   'elevation': 364,  'monsoon': 'Both'},
    'Ratnapura':    {'zone': 'Wet',            'elevation': 130,  'monsoon': 'SW'},
    'Jaffna':       {'zone': 'Dry',            'elevation': 5,    'monsoon': 'NE'},
    'Hambantota':   {'zone': 'Arid',           'elevation': 10,   'monsoon': 'SW'},
    'Kurunegala':   {'zone': 'Intermediate',   'elevation': 116,  'monsoon': 'Both'}\
}

# ── FIX: dynamic NASA end date (always 10 days behind today to allow for
#    NASA POWER latency).  No more hard-coded 2025-12-25 expiry. ──────────────
YESTERDAY    = datetime.now() - timedelta(days=1)
# NASA typically lags real-time by 7–10 days; use 10 to be safe.
NASA_END     = datetime.now() - timedelta(days=10)
NASA_END_STR = NASA_END.strftime("%Y%m%d")

START_DATE      = "20000101"
OPENMETEO_START = (NASA_END + timedelta(days=1)).strftime("%Y-%m-%d")
OPENMETEO_END   = YESTERDAY.strftime("%Y-%m-%d")
FULL_END_STR    = YESTERDAY.strftime("%Y%m%d")

# API base URLs
NASA_BASE_URL      = "https://power.larc.nasa.gov/api/temporal/daily/point"
OPENMETEO_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# NASA POWER parameters
NASA_PARAMS = {
    "parameters":    "T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,WS2M,ALLSKY_SFC_SW_DWN",
    "community":     "AG",
    "format":        "JSON",
    "time-standard": "UTC"
}

# Open-Meteo daily variables
OPENMETEO_VARS = (
    "temperature_2m_max,"
    "temperature_2m_min,"
    "precipitation_sum,"
    "relative_humidity_2m_mean,"
    "wind_speed_10m_mean,"
    "shortwave_radiation_sum"
)

SPI_SCALES        = [3, 6]
DRY_DAY_THRESHOLD = 1.0

VALIDATION_RANGES = {
    'precipitation':   (0, 500),
    'tmax':            (15, 45),
    'tmin':            (10, 35),
    'rh_mean':         (20, 100),
    'wind_speed':      (0, 20),
    'solar_radiation': (0, 35)
}

CITY_ADJUSTMENTS = {
    'Nuwara_Eliya': {'tmax': -6.5, 'tmin': -6.5},
    'Badulla':      {'tmax': -2.0, 'tmin': -2.0},
}


# ==================== API CLIENT ====================

class WeatherAPIClient:
    """Client for NASA POWER + Open-Meteo with retry logic."""

    def __init__(self, max_retries=3, backoff_factor=1):
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://",  adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({'User-Agent': 'SriLankaCropResearch/2.0'})

    # ---------- NASA POWER ----------

    def fetch_nasa_data(self, lat: float, lon: float,
                        start_date: str, end_date: str) -> Optional[Dict]:
        """Fetch from NASA POWER; falls back to year-by-year on 422."""
        params = {
            **NASA_PARAMS,
            "latitude":  lat,
            "longitude": lon,
            "start":     start_date,
            "end":       end_date,
        }
        try:
            print(f"  [NASA] Fetching ({lat},{lon}) {start_date}→{end_date}")
            time.sleep(1)
            r = self.session.get(NASA_BASE_URL, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            if 'properties' not in data or 'parameter' not in data['properties']:
                print("  [NASA] Invalid response structure")
                return None
            return data
        except requests.exceptions.HTTPError as e:
            if r.status_code == 422:
                print("  [NASA] 422 received – switching to year-by-year")
                return self._fetch_nasa_yearly(lat, lon, start_date, end_date)
            print(f"  [NASA] HTTP error: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"  [NASA] Request failed: {e}")
            return None

    def _fetch_nasa_yearly(self, lat, lon, start_date, end_date) -> Optional[Dict]:
        """Fallback: fetch NASA POWER one year at a time."""
        start_year = int(start_date[:4])
        end_year   = int(end_date[:4])
        merged     = {}

        for year in range(start_year, end_year + 1):
            y_start = start_date if year == start_year else f"{year}0101"
            y_end   = end_date   if year == end_year   else f"{year}1231"
            print(f"  [NASA] Year {year} ({lat},{lon})")
            params = {
                **NASA_PARAMS,
                "latitude":  lat,
                "longitude": lon,
                "start":     y_start,
                "end":       y_end,
            }
            try:
                time.sleep(2)
                r = self.session.get(NASA_BASE_URL, params=params, timeout=60)
                r.raise_for_status()
                data = r.json()
                if 'properties' in data and 'parameter' in data['properties']:
                    if not merged:
                        merged = data
                    else:
                        for param, vals in data['properties']['parameter'].items():
                            if param in merged['properties']['parameter']:
                                merged['properties']['parameter'][param].update(vals)
                            else:
                                merged['properties']['parameter'][param] = vals
            except Exception as e:
                print(f"  [NASA] Year {year} failed: {e}")

        return merged if merged else None

    # ---------- Open-Meteo ----------

    def fetch_openmeteo_data(self, lat: float, lon: float,
                             start_date: str, end_date: str) -> Optional[Dict]:
        """Fetch recent data from Open-Meteo ERA5 historical archive."""
        params = {
            "latitude":   lat,
            "longitude":  lon,
            "daily":      OPENMETEO_VARS,
            "start_date": start_date,
            "end_date":   end_date,
            "timezone":   "UTC",
        }
        try:
            print(f"  [Open-Meteo] Fetching ({lat},{lon}) {start_date}→{end_date}")
            time.sleep(0.5)
            r = self.session.get(OPENMETEO_BASE_URL, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            if 'daily' not in data or 'time' not in data['daily']:
                print("  [Open-Meteo] Invalid response structure")
                return None
            return data
        except requests.exceptions.RequestException as e:
            print(f"  [Open-Meteo] Request failed: {e}")
            return None


# ==================== DATA PROCESSING ====================

class WeatherDataProcessor:
    """Parse, validate, fill, and enrich weather data."""

    @staticmethod
    def parse_nasa_response(data: Dict, city_name: str) -> pd.DataFrame:
        if not data:
            return pd.DataFrame()

        parameters = data['properties']['parameter']
        if not parameters:
            return pd.DataFrame()

        first_param = list(parameters.keys())[0]
        dates = list(parameters[first_param].keys())
        df_dict = {p: [v.get(d) for d in dates] for p, v in parameters.items()}

        try:
            df = pd.DataFrame(df_dict, index=pd.to_datetime(dates, format='%Y%m%d'))
        except Exception:
            df = pd.DataFrame(df_dict, index=pd.to_datetime(dates))

        df.index.name = 'date'

        column_map = {
            'PRECTOTCORR':       'precipitation',
            'T2M_MAX':           'tmax',
            'T2M_MIN':           'tmin',
            'RH2M':              'rh_mean',
            'WS2M':              'wind_speed',
            'ALLSKY_SFC_SW_DWN': 'solar_radiation',
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        df = df.replace(-999, np.nan).replace(-999.0, np.nan)

        if city_name in CITY_ADJUSTMENTS:
            for param, adj in CITY_ADJUSTMENTS[city_name].items():
                if param in df.columns:
                    df[param] = df[param] + adj

        df['data_source'] = 'NASA_POWER'
        return df

    @staticmethod
    def parse_openmeteo_response(data: Dict, city_name: str) -> pd.DataFrame:
        if not data or 'daily' not in data:
            return pd.DataFrame()

        daily = data['daily']
        dates = pd.to_datetime(daily['time'])

        col_map = {
            'temperature_2m_max':        'tmax',
            'temperature_2m_min':        'tmin',
            'precipitation_sum':         'precipitation',
            'relative_humidity_2m_mean': 'rh_mean',
            'wind_speed_10m_mean':       'wind_speed',
            'shortwave_radiation_sum':   'solar_radiation',
        }

        df_dict = {}
        for om_col, std_col in col_map.items():
            if om_col in daily:
                df_dict[std_col] = daily[om_col]

        df = pd.DataFrame(df_dict, index=dates)
        df.index.name = 'date'

        if city_name in CITY_ADJUSTMENTS:
            for param, adj in CITY_ADJUSTMENTS[city_name].items():
                if param in df.columns:
                    df[param] = df[param] + adj

        df['data_source'] = 'Open-Meteo'
        return df

    @staticmethod
    def merge_sources(nasa_df: pd.DataFrame, om_df: pd.DataFrame) -> pd.DataFrame:
        if nasa_df.empty and om_df.empty:
            return pd.DataFrame()
        if nasa_df.empty:
            return om_df
        if om_df.empty:
            return nasa_df

        combined = pd.concat([nasa_df, om_df])
        combined = combined[~combined.index.duplicated(keep='first')]
        combined.sort_index(inplace=True)
        return combined

    @staticmethod
    def validate_data(df: pd.DataFrame, city_name: str) -> Dict:
        report = {
            'city': city_name,
            'total_records': len(df),
            'missing_values': {},
            'out_of_range': {},
            'quality_flags': [],
            'date_range': {
                'start': str(df.index.min()) if not df.empty else None,
                'end':   str(df.index.max()) if not df.empty else None,
            }
        }
        if df.empty:
            report['quality_flags'].append('EMPTY_DATASET')
            return report

        for col in df.columns:
            if col in ('monsoon_season', 'data_source'):
                continue
            missing = int(df[col].isna().sum())
            if missing:
                report['missing_values'][col] = {
                    'count': missing,
                    'percentage': round(missing / len(df) * 100, 2)
                }

        for col, (lo, hi) in VALIDATION_RANGES.items():
            if col in df.columns:
                oor = int(((df[col] < lo) | (df[col] > hi)).sum())
                if oor:
                    report['out_of_range'][col] = {
                        'count': oor,
                        'min_actual': float(df[col].min()),
                        'max_actual': float(df[col].max()),
                    }

        if city_name == 'Nuwara_Eliya' and 'tmax' in df.columns:
            extreme = int((df['tmax'] > 30).sum())
            if extreme:
                report['quality_flags'].append(f'HIGH_ELEVATION_HEAT: {extreme} days >30°C')

        if len(df) > 0:
            expected = (df.index.max() - df.index.min()).days + 1
            actual   = len(df)
            if actual < expected * 0.95:
                report['quality_flags'].append(f'DATA_GAPS: {actual}/{expected} days')

        return report

    @staticmethod
    def fill_missing_values(df: pd.DataFrame, city_name: str) -> pd.DataFrame:
        result = df.copy()
        strategies = {
            'precipitation':   0,
            'tmax':            'ffill',
            'tmin':            'ffill',
            'rh_mean':         'ffill',
            'wind_speed':      'ffill',
            'solar_radiation': 'ffill',
        }
        for col, strategy in strategies.items():
            if col in result.columns and result[col].isna().any():
                if strategy == 0:
                    result[col] = result[col].fillna(0)
                else:
                    result[col] = result[col].ffill().bfill()

        for col in result.columns:
            if col not in ('monsoon_season', 'data_source') and result[col].isna().any():
                try:
                    result[col] = result[col].interpolate(
                        method='linear', limit_direction='both'
                    )
                except Exception:
                    result[col] = result[col].fillna(result[col].mean())
        return result

    @staticmethod
    def calculate_derived_indices(df: pd.DataFrame, city_name: str) -> pd.DataFrame:
        if df.empty:
            return df
        result = df.copy()

        # SPI
        if 'precipitation' in result.columns:
            for scale in SPI_SCALES:
                col_name = f'spi_{scale}'
                try:
                    monthly  = result['precipitation'].resample('ME').sum()
                    spi_vals = []
                    for i in range(len(monthly)):
                        if i >= scale - 1:
                            window = monthly.iloc[i - scale + 1: i + 1]
                            if len(window) == scale and not window.isna().any():
                                try:
                                    sh, loc, sc = stats.gamma.fit(window, floc=0)
                                    cdf = stats.gamma.cdf(window.iloc[-1], sh, loc=loc, scale=sc)
                                    spi = stats.norm.ppf(cdf)
                                    spi_vals.append(spi if not np.isinf(spi) else np.nan)
                                except Exception:
                                    spi_vals.append(np.nan)
                            else:
                                spi_vals.append(np.nan)
                        else:
                            spi_vals.append(np.nan)
                    if spi_vals:
                        spi_series = pd.Series(spi_vals, index=monthly.index)
                        spi_series.index = spi_series.index.to_period('M')
                        result[col_name] = result.index.to_period('M').map(spi_series)
                except Exception as e:
                    print(f"  SPI-{scale} error for {city_name}: {e}")
                    result[col_name] = np.nan

        # Heat index
        if 'tmax' in result.columns and 'rh_mean' in result.columns:
            try:
                tf  = result['tmax'] * 9/5 + 32
                h   = result['rh_mean']
                hi_f = (-42.379 + 2.04901523*tf + 10.14333127*h
                        - 0.22475541*tf*h - 6.83783e-3*tf**2
                        - 5.481717e-2*h**2 + 1.22874e-3*tf**2*h
                        + 8.5282e-4*tf*h**2 - 1.99e-6*tf**2*h**2)
                result['heat_index'] = (hi_f - 32) * 5/9
            except Exception as e:
                print(f"  Heat index error for {city_name}: {e}")

        # Consecutive dry days
        if 'precipitation' in result.columns:
            try:
                dry  = result['precipitation'] < DRY_DAY_THRESHOLD
                grps = (dry != dry.shift()).cumsum()
                result['consecutive_dry_days'] = dry.groupby(grps).cumsum()
            except Exception as e:
                print(f"  CDD error for {city_name}: {e}")

        # Growing degree days
        if 'tmax' in result.columns and 'tmin' in result.columns:
            try:
                tmean = (result['tmax'] + result['tmin']) / 2
                result['gdd'] = np.where(tmean > 30, 20, np.maximum(tmean - 10, 0))
            except Exception as e:
                print(f"  GDD error for {city_name}: {e}")

        # Monsoon season
        monsoon_type = CLIMATE_ZONES[city_name]['monsoon']
        def _monsoon(date):
            m = date.month
            if monsoon_type == 'SW':
                return 'SW Monsoon' if 5 <= m <= 9 else 'Dry Season'
            elif monsoon_type == 'NE':
                return 'NE Monsoon' if m in (12, 1, 2) else 'Dry Season'
            else:
                if 5 <= m <= 9:    return 'SW Monsoon'
                if m in (12,1,2):  return 'NE Monsoon'
                return 'Inter-monsoon'
        try:
            result['monsoon_season'] = result.index.map(_monsoon)
        except Exception as e:
            print(f"  Monsoon season error for {city_name}: {e}")

        return result


# ==================== MAIN COLLECTOR ====================

class SriLankaWeatherCollector:
    """Orchestrates data collection: NASA POWER + Open-Meteo gap-fill."""

    def __init__(self, output_dir="../data"):
        self.output_dir    = output_dir
        self.junk_dir      = os.path.join(output_dir, "junk")
        self.raw_dir       = os.path.join(self.junk_dir, "raw")
        self.processed_dir = os.path.join(self.junk_dir, "processed")
        self.metadata_dir  = os.path.join(self.junk_dir, "metadata")
        self.report_dir    = os.path.join(self.junk_dir, "reports")
        self.dirs          = {'raw': self.raw_dir, 'processed': self.processed_dir}

        for d in [self.junk_dir, self.raw_dir, self.processed_dir,
                  self.metadata_dir, self.report_dir]:
            os.makedirs(d, exist_ok=True)

        self.metadata = {
            'collection_date': datetime.now().isoformat(),
            'cities':          CITIES,
            'climate_zones':   CLIMATE_ZONES,
            'date_range':      {'start': START_DATE, 'end': FULL_END_STR},
            'nasa_end':        NASA_END_STR,
            'openmeteo_start': OPENMETEO_START,
            'openmeteo_end':   OPENMETEO_END,
            'data_sources':    {},
            'validation_reports': {},
        }

    # ---------- Per-city collection ----------

    def collect_city_data(self, city_name: str,
                          lat: float, lon: float) -> Optional[pd.DataFrame]:
        print(f"\n{'='*50}")
        print(f"Collecting: {city_name}")
        print(f"{'='*50}")

        nasa_raw = self.api_client.fetch_nasa_data(lat, lon, START_DATE, NASA_END_STR)
        nasa_df  = (self.processor.parse_nasa_response(nasa_raw, city_name)
                    if nasa_raw else pd.DataFrame())

        if not nasa_df.empty:
            print(f"  NASA  → {len(nasa_df)} days "
                  f"({nasa_df.index.min().date()} → {nasa_df.index.max().date()})")
        else:
            print("  NASA  → no data returned")

        om_raw = self.api_client.fetch_openmeteo_data(
            lat, lon, OPENMETEO_START, OPENMETEO_END)
        om_df  = (self.processor.parse_openmeteo_response(om_raw, city_name)
                  if om_raw else pd.DataFrame())

        if not om_df.empty:
            print(f"  Open-Meteo → {len(om_df)} days "
                  f"({om_df.index.min().date()} → {om_df.index.max().date()})")
        else:
            print("  Open-Meteo → no data returned")

        df = self.processor.merge_sources(nasa_df, om_df)

        if df.empty:
            print(f"  ✗ No data at all for {city_name}")
            return None

        sources_used = []
        if not nasa_df.empty: sources_used.append('NASA_POWER')
        if not om_df.empty:   sources_used.append('Open-Meteo')
        self.metadata['data_sources'][city_name] = ' + '.join(sources_used)

        print(f"  Merged → {len(df)} days "
              f"({df.index.min().date()} → {df.index.max().date()})  "
              f"[{self.metadata['data_sources'][city_name]}]")

        val_report = self.processor.validate_data(df, city_name)
        self.metadata['validation_reports'][city_name] = val_report
        if val_report['missing_values']:
            print(f"  Missing: {val_report['missing_values']}")
        if val_report['out_of_range']:
            print(f"  Out of range: {val_report['out_of_range']}")

        try:
            df = self.processor.fill_missing_values(df, city_name)
        except Exception as e:
            print(f"  Gap-fill error: {e}")

        try:
            df = self.processor.calculate_derived_indices(df, city_name)
        except Exception as e:
            print(f"  Derived indices error: {e}")

        print(f"  Final: {len(df)} records, {len(df.columns)} variables")
        return df

    # ---------- Save ----------

    def save_data(self, df: pd.DataFrame, city_name: str):
        if df is None or df.empty:
            print(f"  Nothing to save for {city_name}")
            return
        try:
            raw_cols = [c for c in
                        ['precipitation','tmax','tmin','rh_mean',
                         'wind_speed','solar_radiation','data_source']
                        if c in df.columns]
            if raw_cols:
                df[raw_cols].to_csv(
                    os.path.join(self.dirs['raw'], f"{city_name}_raw.csv"),
                    float_format='%.2f')

            df_save = df.copy()
            for col in df_save.select_dtypes(include=['float64']).columns:
                df_save[col] = df_save[col].round(3)
            df_save.to_csv(
                os.path.join(self.dirs['processed'], f"{city_name}_processed.csv"))

            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                agg = {c: ('sum' if c == 'precipitation' else 'mean') for c in num_cols}
                df.resample('ME').agg(agg).to_csv(
                    os.path.join(self.dirs['processed'], f"{city_name}_monthly.csv"),
                    float_format='%.2f')
                df.resample('YE').agg(agg).to_csv(
                    os.path.join(self.dirs['processed'], f"{city_name}_annual.csv"),
                    float_format='%.2f')

            print(f"  Saved all files for {city_name}")
        except Exception as e:
            print(f"  Save error for {city_name}: {e}")

    # ---------- Sequential / parallel ----------

    def collect_all_cities_sequential(self) -> Dict[str, pd.DataFrame]:
        all_data = {}
        for city_name, (lat, lon) in tqdm(CITIES.items(), desc="Cities"):
            try:
                df = self.collect_city_data(city_name, lat, lon)
                if df is not None and not df.empty:
                    all_data[city_name] = df
                    self.save_data(df, city_name)
            except Exception as e:
                print(f"  Collection failed for {city_name}: {e}")
            time.sleep(2)
        return all_data

    def collect_all_cities_parallel(self, max_workers=3) -> Dict[str, pd.DataFrame]:
        all_data = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.collect_city_data, n, lat, lon): n
                for n, (lat, lon) in CITIES.items()
            }
            for future in tqdm(as_completed(futures), total=len(CITIES), desc="Cities"):
                city_name = futures[future]
                try:
                    df = future.result(timeout=600)
                    if df is not None and not df.empty:
                        all_data[city_name] = df
                        self.save_data(df, city_name)
                except Exception as e:
                    print(f"  {city_name} parallel error: {e}")
        return all_data

    # ---------- Master file ----------

    def create_master_file(self, all_data: Dict[str, pd.DataFrame]):
        """
        Write master CSV.
        FIX: writes BOTH 'sri_lanka_weather_master.csv' (descriptive name)
        AND 'processed_data.csv' (name expected by config_future.yaml) so
        neither the config nor this function needs to be kept in sync.
        """
        try:
            frames = []
            for city_name, df in all_data.items():
                if df is not None and not df.empty:
                    tmp = df.copy()
                    tmp['city']         = city_name
                    tmp['latitude']     = CITIES[city_name][0]
                    tmp['longitude']    = CITIES[city_name][1]
                    tmp['climate_zone'] = CLIMATE_ZONES[city_name]['zone']
                    tmp['elevation']    = CLIMATE_ZONES[city_name]['elevation']
                    tmp['monsoon_type'] = CLIMATE_ZONES[city_name]['monsoon']
                    frames.append(tmp)

            if frames:
                master = pd.concat(frames).reset_index()

                # Primary descriptive name
                master_path = os.path.join(self.output_dir, 'sri_lanka_weather_master.csv')
                master.to_csv(master_path, index=False, float_format='%.3f')
                master.to_csv(master_path.replace('.csv', '.zip'),
                              index=False, compression='zip')
                print(f"\nMaster file saved → {master_path}")

                # FIX: also write as 'processed_data.csv' (config_future.yaml default)
                alias_path = os.path.join(self.output_dir, 'processed_data.csv')
                master.to_csv(alias_path, index=False, float_format='%.3f')
                print(f"Alias saved        → {alias_path}")

                numeric = master.select_dtypes(include=[np.number]).columns
                if len(numeric):
                    master.groupby('city')[numeric].agg(
                        ['mean','std','min','max']
                    ).to_csv(
                        os.path.join(self.metadata_dir, 'summary_statistics.csv')
                    )
        except Exception as e:
            print(f"Master file error: {e}")

    # ---------- Metadata ----------

    def save_metadata(self):
        try:
            def _serial(obj):
                if isinstance(obj, (np.integer, np.floating)): return float(obj)
                if isinstance(obj, np.ndarray):                return obj.tolist()
                if isinstance(obj, pd.Timestamp):              return obj.isoformat()
                if isinstance(obj, dict):  return {k: _serial(v) for k, v in obj.items()}
                if isinstance(obj, list):  return [_serial(i) for i in obj]
                return obj

            path = os.path.join(self.metadata_dir, 'collection_metadata.json')
            with open(path, 'w') as f:
                json.dump(_serial(self.metadata), f, indent=2, default=str)
            print(f"Metadata saved → {path}")
        except Exception as e:
            print(f"Metadata save error: {e}")

    # ---------- Summary report ----------

    def generate_summary_report(self, all_data: Dict[str, pd.DataFrame]):
        try:
            path = os.path.join(self.report_dir, 'collection_report.txt')
            with open(path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("SRI LANKA WEATHER DATA COLLECTION REPORT (v2 Hybrid)\n")
                f.write("="*80 + "\n")
                f.write(f"Generated    : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
                f.write(f"NASA period  : {START_DATE} → {NASA_END_STR}\n")
                f.write(f"Open-Meteo   : {OPENMETEO_START} → {OPENMETEO_END}\n")
                f.write(f"Full range   : {START_DATE} → {FULL_END_STR}\n")
                f.write(f"Cities       : {len(CITIES)}\n")
                f.write(f"Successful   : {len(all_data)}\n\n")

                for city_name in sorted(CITIES):
                    f.write(f"\n{city_name}:\n" + "-"*40 + "\n")
                    if city_name in all_data and all_data[city_name] is not None:
                        df  = all_data[city_name]
                        src = self.metadata['data_sources'].get(city_name, '?')
                        f.write(f"  Source  : {src}\n")
                        f.write(f"  Records : {len(df)}\n")
                        f.write(f"  Range   : {df.index.min().date()} → {df.index.max().date()}\n")
                        f.write(f"  Zone    : {CLIMATE_ZONES[city_name]['zone']}\n")
                        if 'precipitation' in df.columns:
                            f.write(f"  Ann. Precip (mean): "
                                    f"{df['precipitation'].resample('YE').sum().mean():.1f} mm\n")
                        if 'tmax' in df.columns:
                            f.write(f"  Mean Tmax : {df['tmax'].mean():.1f}°C\n")
                        if 'tmin' in df.columns:
                            f.write(f"  Mean Tmin : {df['tmin'].mean():.1f}°C\n")
                    else:
                        f.write("  Status: FAILED\n")

            print(f"Report saved → {path}")
        except Exception as e:
            print(f"Report error: {e}")

    # ---------- Entry point ----------

    def run_collection(self, parallel=False, create_viz=False) -> Dict[str, pd.DataFrame]:
        print("="*60)
        print("SRI LANKA WEATHER DATA COLLECTOR  (v2 Hybrid)")
        print("="*60)
        print(f"NASA POWER   : {START_DATE} → {NASA_END_STR}  [dynamic]")
        print(f"Open-Meteo   : {OPENMETEO_START} → {OPENMETEO_END}")
        print(f"Full coverage: {START_DATE} → {FULL_END_STR}")
        print(f"Cities       : {', '.join(CITIES)}")
        print(f"Output dir   : {os.path.abspath(self.output_dir)}")

        t0 = time.time()

        all_data = (self.collect_all_cities_parallel(max_workers=3)
                    if parallel
                    else self.collect_all_cities_sequential())

        if all_data:
            self.create_master_file(all_data)

        self.save_metadata()
        self.generate_summary_report(all_data)

        elapsed = time.time() - t0
        print("\n" + "="*60)
        print("COLLECTION COMPLETE")
        print(f"Elapsed      : {elapsed:.1f}s")
        print(f"Successful   : {len(all_data)}/{len(CITIES)} cities")
        print("="*60)
        return all_data


# ==================== MAIN ====================

def main():
    collector            = SriLankaWeatherCollector(output_dir="../data")
    collector.api_client = WeatherAPIClient()
    collector.processor  = WeatherDataProcessor()

    all_data = collector.run_collection(parallel=False)

    if all_data:
        print("\n" + "="*60 + "\nCOLLECTION SUMMARY\n" + "="*60)
        for city_name, df in all_data.items():
            src = collector.metadata['data_sources'].get(city_name, '?')
            print(f"\n{city_name}  [{src}]")
            print(f"  Records   : {len(df)}")
            print(f"  Date range: {df.index.min().date()} → {df.index.max().date()}")
            print(f"  Variables : {', '.join(sorted(df.columns))}")
            if 'precipitation' in df.columns:
                print(f"  Ann. rain : {df['precipitation'].resample('YE').sum().mean():.1f} mm/yr")
            if 'tmax' in df.columns:
                print(f"  Mean Tmax : {df['tmax'].mean():.1f}°C")

    return collector, all_data


if __name__ == "__main__":
    collector, all_data = main()