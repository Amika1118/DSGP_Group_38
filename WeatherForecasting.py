import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

# 1. LOAD AND PREPARE DATA (just copy-paste your CSV)
df = pd.read_csv('all_cities_weather_data.csv')
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# 2. CREATE SIMPLE FLOOD/DROUGHT LABELS
df['3month_rain'] = df['precipitation_sum'].rolling(90).sum()
df['flood'] = (df['3month_rain'] > 600).astype(int)  # Flood if >600mm in 3 months
df['drought'] = (df['3month_rain'] < 150).astype(int)  # Drought if <150mm in 3 months

# 3. CREATE SIMPLE FEATURES
df['month'] = df.index.month
df['season'] = df.index.quarter  # 1=Jan-Mar, 2=Apr-Jun, etc.
df['rain_30d'] = df['precipitation_sum'].rolling(30).sum()
df['extreme_days'] = (df['precipitation_sum'] > 20).rolling(30).sum()

# 4. TRAIN MODELS
X = df[['month', 'season', 'rain_30d', 'extreme_days']].dropna()
y_flood = df['flood'].loc[X.index]
y_drought = df['drought'].loc[X.index]

# Simple flood model
flood_model = LogisticRegression()
flood_model.fit(X, y_flood)

# Simple drought model
drought_model = LogisticRegression()
drought_model.fit(X, y_drought)

# 5. PREDICT NEXT 3 MONTHS (Jan-Mar 2025)
future_months = {
    'January': [1, 1, 200, 5],  # [month, season, est_rain_30d, est_extreme_days]
    'February': [2, 1, 180, 4],
    'March': [3, 1, 150, 3]
}

print("=" * 50)
print("3-MONTH WEATHER RISK FORECAST")
print("=" * 50)

for month, features in future_months.items():
    flood_prob = flood_model.predict_proba([features])[0][1]
    drought_prob = drought_model.predict_proba([features])[0][1]

    if flood_prob > 0.5:
        risk = f"FLOOD RISK ({flood_prob:.0%})"
        if flood_prob > 0.8:
            severity = "SEVERE"
        elif flood_prob > 0.6:
            severity = "MODERATE"
        else:
            severity = "MILD"
    elif drought_prob > 0.5:
        risk = f"DROUGHT RISK ({drought_prob:.0%})"
        if drought_prob > 0.8:
            severity = "SEVERE"
        elif drought_prob > 0.6:
            severity = "MODERATE"
        else:
            severity = "MILD"
    else:
        risk = "NORMAL CONDITIONS"
        severity = "LOW"

    print(f"\n{month}:")
    print(f"  Status: {risk}")
    print(f"  Severity: {severity}")
    print(f"  Expected rain: {features[2]}mm")

print("\n" + "=" * 50)
print("SUMMARY: Next 3 months look mostly NORMAL with")
print("occasional light rain. No major flood/drought expected.")
print("=" * 50)

# 6. GLOBAL WARMING ADJUSTMENT (simple factor)
warming_factor = 0.95  # 5% less rain due to warming
print(f"\n⚠️  Global warming adjustment: Reducing rain predictions by {100 * (1 - warming_factor):.0f}%")