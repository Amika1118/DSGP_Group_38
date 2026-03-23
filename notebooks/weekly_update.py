import os
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


VEGETABLES = ["Bitter Gourd", "Brinjals", "Cabbage", "Carrot", "Pumpkin", "Tomatoes"]

VEG_CODE = {
    "Bitter Gourd": 0, "Brinjals": 1, "Cabbage": 2,
    "Carrot": 3,       "Pumpkin": 4,  "Tomatoes": 5,
}

BACKEND_VEG_MAP = {
    1: "Bitter Gourd", 2: "Brinjals", 3: "Cabbage",
    4: "Carrot",       5: "Pumpkin",  6: "Tomatoes",
}

CEYPETCO_URL = "https://ceypetco.gov.lk/historical-prices/"

WEEK_RANGES = [
    (1,1,1,7),    (1,8,1,14),   (1,15,1,21),  (1,22,1,28),  (1,29,2,4),
    (2,5,2,11),   (2,12,2,18),  (2,19,2,25),  (2,26,3,4),   (3,5,3,11),
    (3,12,3,18),  (3,19,3,25),  (3,26,4,1),   (4,2,4,8),    (4,9,4,15),
    (4,16,4,22),  (4,23,4,29),  (4,30,5,6),   (5,7,5,13),   (5,14,5,20),
    (5,21,5,27),  (5,28,6,3),   (6,4,6,10),   (6,11,6,17),  (6,18,6,24),
    (6,25,7,1),   (7,2,7,8),    (7,9,7,15),   (7,16,7,22),  (7,23,7,29),
    (7,30,8,5),   (8,6,8,12),   (8,13,8,19),  (8,20,8,26),  (8,27,9,2),
    (9,3,9,9),    (9,10,9,16),  (9,17,9,23),  (9,24,9,30),  (10,1,10,7),
    (10,8,10,14), (10,15,10,21),(10,22,10,28),(10,29,11,4),
    (11,5,11,11), (11,12,11,18),(11,19,11,25),(11,26,12,2),
    (12,3,12,9),  (12,10,12,16),(12,17,12,23),(12,24,12,31),
]


def get_custom_week(d: date) -> int:
    for wn, (sm, sd, em, ed) in enumerate(WEEK_RANGES, 1):
        if date(d.year, sm, sd) <= d <= date(d.year, em, ed):
            return wn
    return 52


def week_to_csv_monday(year: int, week_num: int) -> date:
    sm, sd     = WEEK_RANGES[week_num - 1][:2]
    thursday   = date(year, sm, sd)
    return thursday - timedelta(days=thursday.weekday())


def scrape_lad_price(effective_date: date) -> float:
    print("  [fuel] Scraping CEYPETCO for LAD price ...")
    try:
        resp = requests.get(CEYPETCO_URL, timeout=20,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Cannot reach CEYPETCO: {exc}")

    soup  = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if not table:
        raise RuntimeError("Price table not found on CEYPETCO page.")

    price_rows = []
    for tr in table.find_all("tr")[1:]:
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cols) < 4:
            continue
        try:
            price_rows.append((datetime.strptime(cols[0], "%d.%m.%Y").date(), float(cols[3])))
        except (ValueError, IndexError):
            continue

    if not price_rows:
        raise RuntimeError("No rows parsed from CEYPETCO table.")

    price_rows.sort(key=lambda x: x[0], reverse=True)
    for rev_date, lad in price_rows:
        if rev_date <= effective_date:
            print(f"  [fuel] LAD revision date {rev_date}: LKR {lad:.0f}")
            return lad
    oldest = price_rows[-1][1]
    print(f"  [fuel] WARNING using LKR {oldest:.0f}")
    return oldest


def get_backend_data(backend_df, year, custom_week):
    mask = (backend_df["year"] == year) & (backend_df["week_num"] == custom_week)
    rows = backend_df[mask].copy()
    if rows.empty:
        return {}
    first = rows.iloc[0]
    result = {
        "USD_LKR":             float(first["USD_LKR_avg"]),
        "ExchangeRate_Change": float(first["RateChange_avg"]),
        "avg_flood_prob":      float(first["avg_prob_flood_risk"]),
        "avg_drought_prob":    float(first["avg_prob_drought"]),
        "avg_precipitation":   float(first["avg_prob_normal"]),
    }
    rows["Vegetable"] = rows["vegetable"].map(BACKEND_VEG_MAP)
    result["wholesale"] = dict(zip(rows["Vegetable"], rows["price"].astype(float)))
    print(f"  [backend] W{custom_week}  USD={result['USD_LKR']:.2f}  "
          f"flood={result['avg_flood_prob']:.4f}  drought={result['avg_drought_prob']:.4f}")
    print(f"  [backend] wholesale: { {k: round(v,2) for k,v in result['wholesale'].items()} }")
    return result


def _safe(a, b, mode="diff"):
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return (a - b) if mode == "diff" else ((a - b) / b * 100 if b != 0 else np.nan)


def build_row(veg, cur_price, dt, fuel_price, backend, ph, fh, wh, month_avg):
    r = {}
    m, n = dt.month, len(ph)
    r["Date"]      = dt.strftime("%Y-%m-%d")
    r["Vegetable"] = veg
    r["Price"]     = cur_price
    r["Month"]     = m
    r["Quarter"]   = dt.quarter

    for lag, key in [(1,"Price_Lag_1"),(2,"Price_Lag_2"),(3,"Price_Lag_3"),
                     (4,"Price_Lag_4"),(8,"Price_Lag_8"),(12,"Price_Lag_12"),(52,"Price_Lag_52")]:
        r[key] = ph[n-lag] if n >= lag else np.nan

    l4, l8, l12 = ph[max(0,n-4):n], ph[max(0,n-8):n], ph[max(0,n-12):n]
    r["Rolling_Mean_4"]  = np.mean(l4)  if l4  else np.nan
    r["Rolling_Mean_8"]  = np.mean(l8)  if l8  else np.nan
    r["Rolling_Mean_12"] = np.mean(l12) if l12 else np.nan
    r["Rolling_Std_4"]   = (np.std(l4,ddof=0) if len(l4)>1 else (0. if len(l4)==1 else np.nan))
    r["Rolling_Min_4"]   = np.min(l4) if l4 else np.nan
    r["Rolling_Max_4"]   = np.max(l4) if l4 else np.nan

    lag1  = ph[n-1]  if n>=1  else np.nan
    lag4  = ph[n-4]  if n>=4  else np.nan
    lag12 = ph[n-12] if n>=12 else np.nan
    lag52 = ph[n-52] if n>=52 else np.nan
    pc1   = (ph[n-1]-ph[n-2]) if n>=2 else np.nan

    r["Price_Change_1wk"]      = _safe(cur_price, lag1)
    r["Price_Change_Pct_1wk"]  = _safe(cur_price, lag1,  "pct")
    r["Price_Change_4wk"]      = _safe(cur_price, lag4)
    r["Price_Change_12wk"]     = _safe(cur_price, lag12)
    r["Price_Change_Pct_4wk"]  = _safe(cur_price, lag4,  "pct")
    r["Price_Change_Pct_12wk"] = _safe(cur_price, lag12, "pct")
    cc = _safe(cur_price, lag1)
    r["Price_Acceleration"]    = _safe(cc, pc1)

    ma = month_avg.get(m, np.nan)
    r["Month_Avg_Price"]    = ma
    r["Price_vs_Month_Avg"] = _safe(cur_price, ma)

    for v in VEGETABLES:
        r[f"Is_{v.replace(' ','_')}"] = 1 if veg==v else 0

    fp, fn = fuel_price, len(fh)
    r["Fuel_Price"] = fp
    fl = [fh[fn-i] if fn>=i else np.nan for i in range(1,5)]
    r["Fuel_Lag_1"], r["Fuel_Lag_2"], r["Fuel_Lag_3"], r["Fuel_Lag_4"] = fl
    valid_fl = [x for x in fl if not np.isnan(x)]
    r["Fuel_Rolling_Mean_4"] = np.mean(valid_fl) if valid_fl else np.nan

    r["Month_Sin"] = np.sin(2*np.pi*m/12)
    r["Month_Cos"] = np.cos(2*np.pi*m/12)
    woy = dt.isocalendar().week
    r["Week_Sin"]  = np.sin(2*np.pi*woy/52)
    r["Week_Cos"]  = np.cos(2*np.pi*woy/52)
    r["Season"]    = {12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}[m]

    vol4 = r["Rolling_Std_4"]
    r["Fuel_x_Lag1"]              = fp*lag1  if not np.isnan(lag1)  else np.nan
    r["Fuel_x_Lag52"]             = fp*lag52 if not np.isnan(lag52) else np.nan
    r["Price_Level_x_Volatility"] = cur_price*vol4             if not np.isnan(vol4) else np.nan
    r["Momentum_x_Volatility"]    = r["Price_Change_1wk"]*vol4 if not np.isnan(vol4) else np.nan
    r["Month_x_Price"]            = m*cur_price
    r["Fuel_x_Month"]             = fp*m

    all_p = ph + [cur_price]
    rstds = [np.std(all_p[max(0,i-4):i],ddof=0) for i in range(4,len(all_p)+1)]
    pct   = np.mean(np.array(rstds) <= vol4)*100 if rstds else np.nan
    r["Volatility_Percentile"] = pct
    r["High_Volatility"]       = 1 if (not np.isnan(pct) and pct>75) else 0
    pvl = ph[max(0,n-5):n-1] if n>=2 else []
    pv  = np.std(pvl,ddof=0) if len(pvl)>1 else 0.
    r["Volatility_Change"]     = _safe(vol4, pv)

    r["Vegetable_Code"] = VEG_CODE[veg]

    ws, wn = backend.get("wholesale", {}).get(veg, np.nan), len(wh)
    r["Wholesale_Price"]         = ws
    r["Wholesale_Lag1"]          = wh[wn-1] if wn>=1 else np.nan
    r["Wholesale_Lag2"]          = wh[wn-2] if wn>=2 else np.nan
    r["Wholesale_Rolling_Mean4"] = np.mean(wh[max(0,wn-4):wn]) if wh else np.nan

    r["USD_LKR"]             = backend.get("USD_LKR",            np.nan)
    r["ExchangeRate_Change"] = backend.get("ExchangeRate_Change", np.nan)
    r["avg_flood_prob"]      = backend.get("avg_flood_prob",      np.nan)
    r["avg_drought_prob"]    = backend.get("avg_drought_prob",    np.nan)
    r["avg_precipitation"]   = backend.get("avg_precipitation",   np.nan)

    return r


def update_features(features_path, backend_path, new_prices, output_path, run_date=None):
    if run_date is None:
        run_date = date.today()

    custom_week = get_custom_week(run_date)
    year        = run_date.year
    csv_monday  = week_to_csv_monday(year, custom_week)
    csv_date    = pd.Timestamp(csv_monday)

    print(f"\n{'='*62}")
    print(f"  Weekly Feature Update")
    print(f"  Run date    : {run_date}")
    print(f"  Custom week : W{custom_week} of {year}")
    print(f"  CSV date    : {csv_monday}")
    print(f"{'='*62}\n")

    df = pd.read_csv(features_path)
    df["Date"] = pd.to_datetime(df["Date"])
    orig_cols  = df.columns.tolist()

    if (df["Date"] == csv_date).any():
        print(f"  W{custom_week} ({csv_monday}) already in CSV. Nothing to do.")
        return df

    backend_df = pd.read_csv(backend_path)
    backend    = get_backend_data(backend_df, year, custom_week)

    if not backend:
        print(f"\n  ERROR: Backend_Data.csv has no data for year={year} week={custom_week}.")
        print("  Make sure Arosha and Amika have updated Backend_Data.csv first.\n")
        input("Press Enter to close...")
        sys.exit(1)

    fuel_price = scrape_lad_price(run_date)

    print()
    new_rows = []
    for veg in VEGETABLES:
        if veg not in new_prices:
            print(f"  WARNING: No price for {veg} - skipping.")
            continue
        hist = df[df["Vegetable"] == veg].sort_values("Date")
        row  = build_row(
            veg=veg, cur_price=float(new_prices[veg]),
            dt=csv_date, fuel_price=fuel_price, backend=backend,
            ph=list(hist["Price"]), fh=list(hist["Fuel_Price"]),
            wh=list(hist["Wholesale_Price"]),
            month_avg=hist.groupby("Month")["Price"].mean().to_dict(),
        )
        new_rows.append(row)
        ws_val = row["Wholesale_Price"]
        ws_str = f"{ws_val:.2f}" if not np.isnan(ws_val) else "N/A"
        print(f"  OK {veg:15s}  price={new_prices[veg]:>8.2f}  "
              f"wholesale={ws_str:>8}  fuel={fuel_price:.0f}  USD={row['USD_LKR']:.2f}")

    new_df   = pd.DataFrame(new_rows)[orig_cols]
    combined = pd.concat(
        [df.assign(Date=df["Date"].dt.strftime("%Y-%m-%d")), new_df],
        ignore_index=True
    )
    combined.to_csv(output_path, index=False)

    print(f"\n  Saved: {output_path}")
    print(f"  Rows: {len(df)} -> {len(combined)}  (+{len(new_rows)})\n")
    return combined


# ══════════════════════════════════════════════════════════════════════════════
#  DOUBLE-CLICK TO RUN — keep these 4 files in the same folder as this script:
#    weekly_features_engineered_v3.csv
#    Backend_Data.csv
#    future_price_predictions-market.csv
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))

    # Automatically read prices from last week's notebook predictions
    pred_df    = pd.read_csv(os.path.join(here, "content", "future_price_predictions-market.csv"))
    new_prices = dict(zip(pred_df["Vegetable"], pred_df["Predicted Price"]))

    print("  [prices] Loaded from future_price_predictions-market.csv:")
    for veg, price in new_prices.items():
        print(f"    {veg:15s}: {price:.2f}")

    update_features(
        features_path = os.path.join(here, "content", "weekly_features_engineered_v3.csv"),
        backend_path  = os.path.join(here, "..", "WholeSale-Price-Model", "Backend", "Data", "Backend_Data.csv"),
        new_prices    = new_prices,
        output_path   = os.path.join(here, "content", "weekly_features_engineered_v3.csv"),
    )

    input("\nDone! Press Enter to close...")
