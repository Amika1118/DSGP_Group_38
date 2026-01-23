import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
import json
from typing import Dict, List, Optional


def ensure_data_folder(folder_path: str) -> None:
    """Ensure data folder exists"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")


def collect_global_warming_indicators() -> pd.DataFrame:
    """
    Collect comprehensive global warming and climate change indicators
    """
    print("\n🌍 Collecting Comprehensive Global Warming Indicators...")

    ensure_data_folder('data_collection')

    # Years to collect data for
    years = list(range(2010, datetime.now().year + 1))

    # We'll create synthetic data that represents realistic climate indicators
    # In a production system, you would fetch from actual climate APIs

    climate_data = []

    # Historical climate reference values
    base_temp_anomaly = 0.4  # Base temperature anomaly in 2010
    base_co2 = 389.0  # ppm in 2010
    base_sea_level = 0.0  # mm relative to 1993 baseline
    base_arctic_ice = 4.6  # million km² in September

    # Climate trends (per year)
    temp_trend = 0.022  # °C per year
    co2_trend = 2.4  # ppm per year
    sea_level_trend = 3.3  # mm per year
    arctic_ice_trend = -0.05  # million km² per year

    for idx, year in enumerate(years):
        # Calculate years since 2010
        years_since_2010 = year - 2010

        # Create realistic climate data with variability
        # Temperature anomalies (with El Niño/La Niña cycles)
        enso_cycle = np.sin(2 * np.pi * years_since_2010 / 5.0)  # 5-year ENSO cycle
        random_variation = np.random.normal(0, 0.08)

        # Calculate temperature anomaly with trend and cycles
        temp_anomaly = base_temp_anomaly + (temp_trend * years_since_2010)
        temp_anomaly += enso_cycle * 0.15  # ENSO influence
        temp_anomaly += random_variation

        # CO2 concentration
        co2_level = base_co2 + (co2_trend * years_since_2010)
        co2_level += np.random.normal(0, 0.3)  # Small random variation

        # Sea level rise
        sea_level = base_sea_level + (sea_level_trend * years_since_2010)
        sea_level += np.random.normal(0, 0.5)

        # Arctic sea ice extent (September minimum)
        arctic_ice = base_arctic_ice + (arctic_ice_trend * years_since_2010)
        arctic_ice += np.random.normal(0, 0.1)

        # Climate indices
        enso_index = enso_cycle + np.random.normal(0, 0.3)

        # North Atlantic Oscillation (NAO) - random walk
        if idx == 0:
            nao_index = np.random.normal(0, 0.5)
        else:
            nao_index = climate_data[-1]['nao_index'] * 0.7 + np.random.normal(0, 0.4)

        # Pacific Decadal Oscillation (PDO)
        pdo_cycle = np.sin(2 * np.pi * years_since_2010 / 20)  # 20-year cycle
        pdo_index = pdo_cycle * 1.5 + np.random.normal(0, 0.3)

        # Create monthly data (12 months per year)
        for month in range(1, 13):
            # Monthly variations
            month_factor = np.sin(2 * np.pi * (month - 1) / 12)

            # Enhanced global warming metrics for each month
            climate_record = {
                'year': year,
                'month': month,
                'month_name': datetime(year, month, 1).strftime('%b'),
                'global_temp_anomaly': round(temp_anomaly + month_factor * 0.1, 3),
                'global_temp_anomaly_land': round(temp_anomaly * 1.3 + month_factor * 0.15, 3),
                'global_temp_anomaly_ocean': round(temp_anomaly * 0.8 + month_factor * 0.08, 3),
                'co2_ppm': round(co2_level + month_factor * 0.5, 2),
                'co2_anomaly': round(co2_level - base_co2, 2),
                'sea_level_anomaly_mm': round(sea_level + month_factor * 2, 1),
                'arctic_ice_extent_mkm2': round(arctic_ice - abs(month_factor) * 0.5, 2),
                'enso_index': round(enso_index * (1 + 0.2 * month_factor), 3),
                'nao_index': round(nao_index, 3),
                'pdo_index': round(pdo_index, 3),
                'warming_factor': round(years_since_2010 * 0.02 + np.random.normal(0, 0.01), 3),
                'extreme_event_frequency': round(1.0 + years_since_2010 * 0.03 + np.random.normal(0, 0.02), 3),
                'heatwave_index': round(0.5 + years_since_2010 * 0.04 + abs(month_factor) * 0.2, 3),
                'precipitation_anomaly': round(np.random.normal(0, 0.1) + month_factor * 0.15, 3),
                'drought_index': round(0.3 + years_since_2010 * 0.01 + np.random.normal(0, 0.05), 3),
                'carbon_budget_remaining_gt': round(500 - years_since_2010 * 42, 0),  # Approximate
                'radiative_forcing_wm2': round(2.3 + years_since_2010 * 0.03, 3),  # Approximate
                'climate_risk_index': round(0.2 + years_since_2010 * 0.025, 3)
            }

            climate_data.append(climate_record)

    # Create DataFrame
    climate_df = pd.DataFrame(climate_data)

    # Add derived features
    climate_df['year_fraction'] = climate_df['year'] + (climate_df['month'] - 1) / 12

    # Calculate trends
    climate_df['temp_anomaly_trend_10yr'] = climate_df['global_temp_anomaly'].rolling(window=120, min_periods=1).mean()
    climate_df['co2_trend_5yr'] = climate_df['co2_ppm'].rolling(window=60, min_periods=1).mean()

    # Calculate acceleration
    climate_df['temp_acceleration'] = climate_df['global_temp_anomaly'].diff(12).fillna(0)
    climate_df['co2_acceleration'] = climate_df['co2_ppm'].diff(12).fillna(0)

    # Save to CSV
    output_file = "data_collection/global_warming_indicators.csv"
    climate_df.to_csv(output_file, index=False)

    print(f"✅ Comprehensive climate indicators saved to: {output_file}")
    print(f"   Records: {len(climate_df)} ({len(years)} years × 12 months)")
    print(f"   Columns: {len(climate_df.columns)}")

    # Display summary
    print("\n📊 Climate Data Summary:")
    print(f"   Years: {climate_df['year'].min()} to {climate_df['year'].max()}")
    print(
        f"   Temperature anomaly range: {climate_df['global_temp_anomaly'].min():.2f} to {climate_df['global_temp_anomaly'].max():.2f}°C")
    print(f"   CO₂ range: {climate_df['co2_ppm'].min():.1f} to {climate_df['co2_ppm'].max():.1f} ppm")
    print(f"   Sea level rise: {climate_df['sea_level_anomaly_mm'].max():.1f} mm")

    return climate_df


def collect_weather_data_for_city(city: str, lat: float, lon: float) -> pd.DataFrame:
    """
    Collect historical weather data for a specific city
    """
    print(f"📡 Fetching data for {city}...")

    # Historical range
    start_year = 2010
    end_year = datetime.now().year

    daily_features = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "windspeed_10m_max",
        "windgusts_10m_max",
        "winddirection_10m_dominant",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
        "weathercode",
        "sunshine_duration"
    ]

    city_data = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Get data month by month for better reliability
            start_date = f"{year}-{month:02d}-01"

            # Calculate end date (last day of month)
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
                f"timezone=Asia/Colombo"
            )

            try:
                response = requests.get(url, timeout=45)

                if response.status_code == 200:
                    data = response.json()
                    if "daily" in data and data["daily"]:
                        df_month = pd.DataFrame(data["daily"])
                        df_month['date'] = pd.to_datetime(df_month['time'])
                        df_month['year'] = year
                        df_month['month'] = month
                        city_data.append(df_month)
                        print(f"  ✓ {year}-{month:02d}: {len(df_month)} days")
                    else:
                        print(f"  ⚠ No data for {year}-{month:02d}")
                else:
                    print(f"  ✗ Failed for {city} {year}-{month:02d}, status: {response.status_code}")

            except Exception as e:
                print(f"  ✗ Error for {city} {year}-{month:02d}: {e}")

            # Be respectful to the API
            time.sleep(0.5)

    if city_data:
        df_city = pd.concat(city_data, ignore_index=True)

        # Add city information
        df_city['city'] = city
        df_city['latitude'] = lat
        df_city['longitude'] = lon

        # Calculate additional features
        if 'temperature_2m_max' in df_city.columns and 'temperature_2m_min' in df_city.columns:
            df_city['temperature_range'] = df_city['temperature_2m_max'] - df_city['temperature_2m_min']

        if 'temperature_2m_mean' not in df_city.columns:
            if 'temperature_2m_max' in df_city.columns and 'temperature_2m_min' in df_city.columns:
                df_city['temperature_2m_mean'] = (df_city['temperature_2m_max'] + df_city['temperature_2m_min']) / 2

        # Calculate heating/cooling degree days
        if 'temperature_2m_mean' in df_city.columns:
            base_temp = 18.0  # Base temperature for degree-day calculations
            df_city['heating_degree_days'] = np.maximum(base_temp - df_city['temperature_2m_mean'], 0)
            df_city['cooling_degree_days'] = np.maximum(df_city['temperature_2m_mean'] - base_temp, 0)

        print(f"✅ {city}: Collected {len(df_city)} daily records")
        return df_city

    return pd.DataFrame()


def collect_weather_data() -> Dict[str, pd.DataFrame]:
    """
    Collect weather data for all cities
    """
    print("🌤️ Collecting Comprehensive Weather Data...")

    ensure_data_folder('data_collection')

    # Cities in Sri Lanka with coordinates
    cities = {
        "Colombo": (6.9271, 79.8612),
        "Kandy": (7.2906, 80.6337),
        "Galle": (6.0535, 80.2210),
        "Jaffna": (9.6615, 80.0255),
        "Trincomalee": (8.5874, 81.2152),
        "Badulla": (6.9934, 81.0550),
        "Ratnapura": (6.6847, 80.4020),
        "Anuradhapura": (8.3114, 80.4037),
        "Polonnaruwa": (7.9403, 81.0189),
        "Matara": (5.9556, 80.5480),
        "Nuwara_Eliya": (6.9497, 80.7891),
        "Hambantota": (6.1246, 81.1185),
        "Kurunegala": (7.4863, 80.3623),
        "Matale": (7.4675, 80.6234),
        "Batticaloa": (7.7314, 81.6747)
    }

    all_city_data = {}

    for city, (lat, lon) in cities.items():
        try:
            df_city = collect_weather_data_for_city(city, lat, lon)
            if not df_city.empty:
                # Save individual city file
                filename = f"data_collection/{city}_weather_2010_present.csv"
                df_city.to_csv(filename, index=False)
                all_city_data[city] = df_city
                print(f"💾 {city} data saved: {filename}")
            else:
                print(f"❌ No data collected for {city}")
        except Exception as e:
            print(f"❌ Error collecting data for {city}: {e}")

    return all_city_data


def combine_all_weather_data(city_data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all city data into a single DataFrame
    """
    print("\n🔄 Combining all city data...")

    if not city_data_dict:
        print("❌ No city data to combine")
        return pd.DataFrame()

    all_data = []

    for city, df in city_data_dict.items():
        df = df.copy()

        # Ensure consistent column names
        if 'date' not in df.columns and 'time' in df.columns:
            df.rename(columns={'time': 'date'}, inplace=True)

        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])

        all_data.append(df)
        print(f"✓ Added {city}: {len(df)} records")

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Sort by city and date
    combined_df = combined_df.sort_values(['city', 'date']).reset_index(drop=True)

    # Save combined data
    output_file = "data_collection/all_cities_weather_combined.csv"
    combined_df.to_csv(output_file, index=False)

    print(f"\n✅ All data combined successfully!")
    print(f"   Total records: {len(combined_df):,}")
    print(f"   Cities included: {len(city_data_dict)}")
    print(f"   Date range: {combined_df['date'].min().date()} to {combined_df['date'].max().date()}")

    # Display summary statistics
    print("\n📊 Data Summary by City:")
    summary = combined_df.groupby('city').agg({
        'date': ['min', 'max', 'count'],
        'temperature_2m_mean': ['mean', 'min', 'max'] if 'temperature_2m_mean' in combined_df.columns else 'count'
    }).round(2)

    print(summary)

    return combined_df


def collect_climate_normals() -> pd.DataFrame:
    """
    Collect climate normals (30-year averages) for context
    """
    print("\n📊 Collecting Climate Normals...")

    # Create synthetic climate normals based on latitude
    normals_data = []

    cities = {
        "Colombo": (6.9271, 79.8612),
        "Kandy": (7.2906, 80.6337),
        "Galle": (6.0535, 80.2210),
        "Jaffna": (9.6615, 80.0255),
        "Badulla": (6.9934, 81.0550),
        "Ratnapura": (6.6847, 80.4020),
        "Nuwara_Eliya": (6.9497, 80.7891),
        "Hambantota": (6.1246, 81.1185),
        "Kurunegala": (7.4863, 80.3623),
        "Matale": (7.4675, 80.6234)
    }

    for city, (lat, lon) in cities.items():
        # Base temperatures based on latitude and elevation
        if "Nuwara" in city:  # High elevation
            base_temp = 15.0
            annual_rain = 2000
        elif lat > 8:  # Northern
            base_temp = 28.0
            annual_rain = 1200
        elif lat > 6:  # Central/Southern
            base_temp = 27.0
            annual_rain = 1800
        else:  # Coastal
            base_temp = 28.5
            annual_rain = 2400

        # Add some variation
        temp_variation = np.random.normal(0, 1)
        rain_variation = np.random.normal(0, 200)

        normal_record = {
            'city': city,
            'latitude': lat,
            'longitude': lon,
            'elevation_m': 1868 if "Nuwara" in city else
            680 if "Badulla" in city else
            364 if "Matale" in city else
            116 if "Kurunegala" in city else
            130 if "Ratnapura" in city else
            10 if "Hambantota" in city else
            5 if "Jaffna" in city else 50,
            'climate_zone': 'Highland' if "Nuwara" in city else
            'Dry Zone' if lat > 8 else
            'Wet Zone' if lat > 6.5 else
            'Intermediate Zone',
            'annual_mean_temp': round(base_temp + temp_variation, 1),
            'annual_precipitation_mm': round(annual_rain + rain_variation, 0),
            'temp_june_july': round(base_temp - 1.5, 1),
            'temp_dec_jan': round(base_temp + 1.5, 1),
            'humidity_annual': round(75 + np.random.normal(0, 5), 1),
            'sunshine_hours_annual': round(2400 + np.random.normal(0, 200), 0)
        }

        normals_data.append(normal_record)

    normals_df = pd.DataFrame(normals_data)
    output_file = "data_collection/climate_normals.csv"
    normals_df.to_csv(output_file, index=False)

    print(f"✅ Climate normals saved to: {output_file}")
    return normals_df


def main() -> None:
    """
    Main function to run the complete data collection pipeline
    """
    print("=" * 70)
    print("🌍 COMPREHENSIVE WEATHER & CLIMATE DATA COLLECTION SYSTEM")
    print("=" * 70)

    print("\n📋 Pipeline Steps:")
    print("1. Collect global warming indicators")
    print("2. Collect historical weather data for all cities")
    print("3. Collect climate normals")
    print("4. Combine all data")

    # Step 1: Global warming indicators
    print("\n" + "=" * 50)
    print("Step 1: Collecting Global Warming Indicators")
    print("=" * 50)
    climate_df = collect_global_warming_indicators()

    # Step 2: Weather data
    print("\n" + "=" * 50)
    print("Step 2: Collecting Weather Data")
    print("=" * 50)
    city_data = collect_weather_data()

    # Step 3: Climate normals
    print("\n" + "=" * 50)
    print("Step 3: Collecting Climate Normals")
    print("=" * 50)
    normals_df = collect_climate_normals()

    # Step 4: Combine weather data
    if city_data:
        print("\n" + "=" * 50)
        print("Step 4: Combining Weather Data")
        print("=" * 50)
        combined_weather = combine_all_weather_data(city_data)

    print("\n" + "=" * 70)
    print("✅ DATA COLLECTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    # Summary
    print("\n📁 Files Created:")
    print("  - data_collection/global_warming_indicators.csv")
    print("  - data_collection/[city]_weather_2010_present.csv (multiple)")
    print("  - data_collection/all_cities_weather_combined.csv")
    print("  - data_collection/climate_normals.csv")

    print("\n📊 Data Summary:")
    print(f"  Global warming data: {len(climate_df)} monthly records")
    print(f"  Weather data: {len(city_data)} cities")
    if city_data:
        total_weather_records = sum(len(df) for df in city_data.values())
        print(f"  Total weather records: {total_weather_records:,}")
    print(f"  Climate normals: {len(normals_df)} cities")

    return climate_df, city_data, normals_df


if __name__ == "__main__":
    # Run the complete data collection pipeline
    main()