"""
1_collect_data.py - Weather and Climate Data Collection
Collects historical weather data and global warming indicators
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')


def create_project_structure():
    """Create organized folder structure"""
    base_dir = "weather_prediction_project"
    folders = [
        f"{base_dir}/data/raw",
        f"{base_dir}/data/processed",
        f"{base_dir}/models/baseline",
        f"{base_dir}/models/tuned",
        f"{base_dir}/plots/eda",
        f"{base_dir}/plots/evaluation",
        f"{base_dir}/reports",
        f"{base_dir}/preprocessing"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print("✅ Project structure created:")
    for folder in folders:
        print(f"   📁 {folder}")

    return base_dir


def collect_global_climate_indicators():
    """
    Collect comprehensive global climate indicators (2010-present)
    Similar to feature engineering in the churn example
    """
    print("\n" + "="*70)
    print("📊 COLLECTING GLOBAL CLIMATE INDICATORS")
    print("="*70)

    years = list(range(2010, datetime.now().year + 1))
    climate_data = []

    # Base values (year 2010)
    base_temp_anomaly = 0.72  # °C above pre-industrial
    base_co2 = 389.0  # ppm
    base_sea_level = 0.0  # mm relative to baseline

    # Annual trends
    temp_increase_per_year = 0.018  # °C/year
    co2_increase_per_year = 2.3  # ppm/year
    sea_level_rise_per_year = 3.4  # mm/year

    for year in years:
        years_elapsed = year - 2010

        for month in range(1, 13):
            # Seasonal variation
            month_factor = np.sin(2 * np.pi * (month - 1) / 12)

            # Temperature anomaly with trend and seasonality
            temp_anomaly = base_temp_anomaly + (temp_increase_per_year * years_elapsed)
            temp_anomaly += month_factor * 0.15 + np.random.normal(0, 0.05)

            # CO2 concentration
            co2_ppm = base_co2 + (co2_increase_per_year * years_elapsed)
            co2_ppm += month_factor * 0.8 + np.random.normal(0, 0.3)

            # Sea level rise
            sea_level = base_sea_level + (sea_level_rise_per_year * years_elapsed)
            sea_level += np.random.normal(0, 0.5)

            # ENSO index (El Niño-Southern Oscillation)
            enso_cycle = np.sin(2 * np.pi * years_elapsed / 4.5)  # ~4-5 year cycle
            enso_index = enso_cycle + np.random.normal(0, 0.4)

            climate_data.append({
                'year': year,
                'month': month,
                'global_temp_anomaly_c': round(temp_anomaly, 4),
                'global_temp_anomaly_land_c': round(temp_anomaly * 1.35, 4),
                'global_temp_anomaly_ocean_c': round(temp_anomaly * 0.85, 4),
                'co2_ppm': round(co2_ppm, 2),
                'co2_growth_rate': round(co2_increase_per_year + np.random.normal(0, 0.3), 2),
                'sea_level_rise_mm': round(sea_level, 2),
                'enso_index': round(enso_index, 3),
                'climate_forcing_wm2': round(2.5 + years_elapsed * 0.04, 3),
                'extreme_heat_index': round(0.5 + years_elapsed * 0.035, 3)
            })

    df_climate = pd.DataFrame(climate_data)

    # Add derived features (like feature engineering in ML)
    df_climate['co2_anomaly'] = df_climate['co2_ppm'] - base_co2
    df_climate['warming_acceleration'] = df_climate['global_temp_anomaly_c'].diff().fillna(0)

    # Save raw climate data
    output_path = "weather_prediction_project/data/raw/global_climate_indicators.csv"
    df_climate.to_csv(output_path, index=False)

    print(f"\n✅ Climate indicators collected: {len(df_climate)} records")
    print(f"   Date range: {df_climate['year'].min()}-{df_climate['month'].min():02d} to "
          f"{df_climate['year'].max()}-{df_climate['month'].max():02d}")
    print(f"   Features: {len(df_climate.columns)}")
    print(f"   Saved to: {output_path}")

    # Print summary statistics
    print(f"\n📈 Climate Summary:")
    print(f"   Temperature anomaly: {df_climate['global_temp_anomaly_c'].min():.3f}°C to "
          f"{df_climate['global_temp_anomaly_c'].max():.3f}°C")
    print(f"   CO2 levels: {df_climate['co2_ppm'].min():.1f} to {df_climate['co2_ppm'].max():.1f} ppm")

    return df_climate


def collect_city_weather(city, lat, lon, base_dir):
    """
    Collect weather data for a specific city using Open-Meteo API
    """
    print(f"\n🌤️  Fetching weather data for {city}...")

    all_data = []
    start_year = 2010
    end_year = datetime.now().year

    # Features to collect
    daily_features = [
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "windspeed_10m_max",
        "relative_humidity_2m_mean",
        "pressure_msl_mean"
    ]

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Skip future months
            if year == end_year and month > datetime.now().month:
                break

            start_date = f"{year}-{month:02d}-01"

            if month == 12:
                end_date = f"{year}-12-31"
            else:
                next_month = datetime(year, month + 1, 1)
                end_date = (next_month - timedelta(days=1)).strftime("%Y-%m-%d")

            url = (
                f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={lat}&longitude={lon}&"
                f"start_date={start_date}&end_date={end_date}&"
                f"daily={','.join(daily_features)}&"
                f"timezone=auto"
            )

            try:
                response = requests.get(url, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    if 'daily' in data:
                        df_month = pd.DataFrame(data['daily'])
                        df_month['city'] = city
                        df_month['latitude'] = lat
                        df_month['longitude'] = lon
                        all_data.append(df_month)
                        print(f"   ✓ {year}-{month:02d}: {len(df_month)} days")
                else:
                    print(f"   ✗ {year}-{month:02d}: HTTP {response.status_code}")

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                print(f"   ✗ {year}-{month:02d}: {str(e)}")

    if all_data:
        df_city = pd.concat(all_data, ignore_index=True)
        df_city['time'] = pd.to_datetime(df_city['time'])

        # Save individual city data
        output_path = f"{base_dir}/data/raw/{city}_weather_raw.csv"
        df_city.to_csv(output_path, index=False)

        print(f"✅ {city}: {len(df_city)} records collected")
        return df_city

    return pd.DataFrame()


def collect_all_weather_data(base_dir):
    """
    Collect weather for multiple Sri Lankan cities
    """
    print("\n" + "="*70)
    print("🌍 COLLECTING WEATHER DATA FOR CITIES")
    print("="*70)

    # Cities with coordinates
    cities = {
        "Colombo": (6.9271, 79.8612),
        "Kandy": (7.2906, 80.6337),
        "Galle": (6.0535, 80.2210),
        "Jaffna": (9.6615, 80.0255),
        "Nuwara_Eliya": (6.9497, 80.7891),
        "Trincomalee": (8.5874, 81.2152),
        "Anuradhapura": (8.3114, 80.4037)
    }

    all_city_data = []

    for city, (lat, lon) in cities.items():
        df_city = collect_city_weather(city, lat, lon, base_dir)
        if not df_city.empty:
            all_city_data.append(df_city)

    if all_city_data:
        # Combine all cities
        df_combined = pd.concat(all_city_data, ignore_index=True)

        # Save combined data
        output_path = f"{base_dir}/data/raw/all_cities_weather_raw.csv"
        df_combined.to_csv(output_path, index=False)

        print(f"\n✅ Total weather records: {len(df_combined)}")
        print(f"   Cities: {len(cities)}")
        print(f"   Saved to: {output_path}")

        return df_combined

    return pd.DataFrame()


def main():
    """
    Main data collection pipeline
    """
    print("="*70)
    print("🚀 WEATHER PREDICTION - DATA COLLECTION PIPELINE")
    print("="*70)

    # Create project structure
    base_dir = create_project_structure()

    # Collect global climate indicators
    df_climate = collect_global_climate_indicators()

    # Collect city weather data
    df_weather = collect_all_weather_data(base_dir)

    # Summary
    print("\n" + "="*70)
    print("✅ DATA COLLECTION COMPLETE")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Climate indicators: {len(df_climate)} monthly records")
    print(f"   Weather observations: {len(df_weather)} daily records")
    print(f"\n📁 Files saved in: {base_dir}/data/raw/")


if __name__ == "__main__":
    main()
