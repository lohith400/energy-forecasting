"""
GridSense India — Flask Backend
Provides /predict, /get_history, and /get_temperature endpoints.

Changes from previous version:
  1. Removed ALL sinusoidal/diurnal synthetic temperature curves.
     Fallback now uses flat monthly average instead of fake diurnal shape.
  2. /predict now returns all_regional_forecasts: full 24-hour curves for
     all 5 regions (not just the selected one).
  3. Historical /get_temperature now correctly reads per-region temperature
     columns (North, South, East, West, NorthEast) from the CSV.
  4. predict_24_hours ingest path is cleaner — national_hourly_temps[i]
     is fed directly with no modification.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import math
import os
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_PATH   = os.path.join(BASE_DIR, "smartgrid_lstm_model.keras")
LOAD_FILE    = os.path.join(BASE_DIR, "hourlyLoadDataIndia.xlsx")
TEMP_FILE    = os.path.join(BASE_DIR, "historical_hourly_temp.csv")

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="/static")
CORS(app)

# ── Serve frontend ─────────────────────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/index.html")
def serve_index_html():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/app.js")
def serve_app_js():
    return send_from_directory(FRONTEND_DIR, "app.js")

@app.route("/styles.css")
def serve_styles():
    return send_from_directory(FRONTEND_DIR, "styles.css")

# ── Constants ──────────────────────────────────────────────────────────────────
REGION_CITIES = {
    "North":     {"city": "New Delhi",  "lat": 28.6139, "lon": 77.2090},
    "South":     {"city": "Bengaluru",  "lat": 12.9716, "lon": 77.5946},
    "East":      {"city": "Kolkata",    "lat": 22.5726, "lon": 88.3639},
    "West":      {"city": "Mumbai",     "lat": 19.0760, "lon": 72.8777},
    "NorthEast": {"city": "Guwahati",   "lat": 26.1445, "lon": 91.7362},
}

# Actual demand columns in hourlyLoadDataIndia.xlsx
REGION_DEMAND_COL = {
    "North":     "Northen Region Hourly Demand",
    "West":      "Western Region Hourly Demand",
    "East":      "Eastern Region Hourly Demand",
    "South":     "Southern Region Hourly Demand",
    "NorthEast": "North-Eastern Region Hourly Demand",
}

# Temperature columns in historical_hourly_temp.csv
# CSV header: datetime,hourly_temperature,North,South,East,West,NorthEast
REGION_TEMP_COL = {
    "North":     "North",
    "South":     "South",
    "East":      "East",
    "West":      "West",
    "NorthEast": "NorthEast",
}

# Load-weighted grid shares (North 30%, West 26.6%, South 23.2%, East 15.4%, NE 4.8%)
REGION_FRACTIONS = {
    "North":     0.300,
    "West":      0.266,
    "South":     0.232,
    "East":      0.154,
    "NorthEast": 0.048,
}

# Exact feature set the LSTM was trained on — must never change
LSTM_COLS    = [
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "dow_sin", "dow_cos",
    "is_weekend",
    "temperature_max",          # national weighted temperature
    "National Hourly Demand",   # target (index 8)
]
TARGET_IDX   = 8
SEQUENCE_LEN = 168             # 1 week lookback

# ── Global state ───────────────────────────────────────────────────────────────
MODEL         = None
SCALER        = None
DF            = None           # Full preprocessed DataFrame
MONTHLY_TEMPS = {}             # {month(1-12): avg_national_weighted_temp}
DATASET_END   = None           # Last pd.Timestamp in load dataset


# ── Regional monthly baseline temps — used ONLY as last-resort fallback ────────
# (replaces sinusoidal curve: a flat monthly average is more honest than a fake
#  diurnal shape when we have no real data)
REGIONAL_MONTHLY_TEMPS = {
    "North":     {1:15,2:19,3:25,4:32,5:35,6:34,7:31,8:30,9:29,10:26,11:20,12:15},
    "South":     {1:22,2:24,3:27,4:29,5:28,6:25,7:24,8:24,9:24,10:24,11:23,12:21},
    "East":      {1:20,2:23,3:28,4:31,5:32,6:30,7:29,8:29,9:29,10:28,11:24,12:20},
    "West":      {1:24,2:25,3:27,4:29,5:30,6:29,7:27,8:27,9:27,10:28,11:27,12:25},
    "NorthEast": {1:18,2:20,3:24,4:27,5:28,6:29,7:29,8:29,9:28,10:26,11:22,12:19},
}


def load_data_and_model():
    global MODEL, SCALER, DF, MONTHLY_TEMPS, DATASET_END

    # ── Temperature data ──────────────────────────────────────────────────────
    print("Loading temperature data…")
    df_temp = pd.read_csv(TEMP_FILE)
    df_temp["datetime"] = pd.to_datetime(df_temp["datetime"])
    df_temp["month"]    = df_temp["datetime"].dt.month
    # Monthly average of the national weighted temperature (hourly_temperature column)
    MONTHLY_TEMPS = df_temp.groupby("month")["hourly_temperature"].mean().to_dict()
    print(f"  Monthly avg national temps: { {k: round(v,1) for k,v in MONTHLY_TEMPS.items()} }")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading hourly load data…")
    df_raw = pd.read_excel(LOAD_FILE)
    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
    df_raw = df_raw.sort_values("datetime").reset_index(drop=True)
    DATASET_END = df_raw["datetime"].max()
    print(f"  Dataset: {df_raw['datetime'].min().date()} → {DATASET_END.date()} ({len(df_raw):,} rows)")

    # ── Merge on datetime ─────────────────────────────────────────────────────
    df = df_raw.merge(df_temp, on="datetime", how="left")

    # National weighted temperature (used as LSTM feature)
    df["temperature_max"] = df["hourly_temperature"].ffill().bfill()

    # Regional temperature columns for /get_temperature historical mode
    for col in REGION_TEMP_COL.values():
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # ── Feature engineering (must match training exactly) ─────────────────────
    df["year"]      = df["datetime"].dt.year
    df["month"]     = df["datetime"].dt.month
    df["hour"]      = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]      / 24.0)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]      / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]     / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]     / 12.0)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] /  7.0)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] /  7.0)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    DF = df.reset_index(drop=True)

    # ── Fit scaler on same columns the model was trained with ─────────────────
    lstm_data = DF[LSTM_COLS].values.astype("float32")
    SCALER = MinMaxScaler()
    SCALER.fit(lstm_data)
    print("  Scaler fitted on 9-column LSTM feature set.")

    # ── Load LSTM model ───────────────────────────────────────────────────────
    if os.path.exists(MODEL_PATH):
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        print(f"  Model loaded: {MODEL_PATH}")
    else:
        print(f"  ⚠ Model file NOT found at {MODEL_PATH}")

    print("Server ready.\n")


try:
    load_data_and_model()
except Exception as e:
    print(f"Startup error: {e}")
    traceback.print_exc()


# ── Context window helper ──────────────────────────────────────────────────────
def _get_context_sequence(target_dt: datetime) -> np.ndarray:
    """
    Return the 168-row raw feature matrix immediately before target_dt.
    Uses actual historical rows when available; falls back to dataset tail.
    """
    target_ts = pd.Timestamp(target_dt)
    if DF is not None and target_ts > DF["datetime"].iloc[0]:
        mask = DF["datetime"] < target_ts
        if mask.sum() >= SEQUENCE_LEN:
            end_pos   = mask.sum()
            start_pos = end_pos - SEQUENCE_LEN
            return DF.iloc[start_pos:end_pos][LSTM_COLS].values.astype("float32")
    return DF.tail(SEQUENCE_LEN)[LSTM_COLS].values.astype("float32")


# ── Core prediction engine ─────────────────────────────────────────────────────
def predict_24_hours(date_str: str, national_hourly_temps: list, is_holiday: bool) -> list:
    """
    Recursively predict 24 hours of national demand (in MW) starting at midnight.

    national_hourly_temps: list of exactly 24 floats — the load-weighted national
    temperature for each hour.  Each value is fed directly into the LSTM feature
    row for that hour. No synthetic/diurnal curves are generated here.

    Returns: list of 24 GW values (index 0 = midnight, 23 = 11 PM).
    """
    if MODEL is None or SCALER is None or DF is None:
        raise ValueError("Model or data not loaded.")

    if len(national_hourly_temps) != 24:
        raise ValueError(f"national_hourly_temps must have exactly 24 values, got {len(national_hourly_temps)}")

    base_time    = datetime.strptime(date_str, "%Y-%m-%d")   # midnight
    future_times = [base_time + timedelta(hours=i) for i in range(24)]

    # Seed the rolling context window with 168 actual rows
    context_raw    = _get_context_sequence(base_time)
    context_scaled = SCALER.transform(context_raw)

    predictions_mw = []

    for i, dt in enumerate(future_times):
        # ── Single-step forecast ──────────────────────────────────────────────
        pred_scaled = MODEL.predict(
            context_scaled[np.newaxis, :, :], verbose=0
        ).flatten()[0]

        # Inverse-transform to MW
        dummy          = np.zeros(len(LSTM_COLS))
        dummy[TARGET_IDX] = pred_scaled
        pred_mw = SCALER.inverse_transform([dummy])[0, TARGET_IDX]

        # Optional holiday adjustment
        if is_holiday:
            pred_mw *= 0.88

        predictions_mw.append(pred_mw)

        # ── Build next input row ──────────────────────────────────────────────
        hour  = dt.hour
        month = dt.month
        dow   = dt.weekday()

        # Feed real (or API-provided) temperature directly — no synthetic curve
        hour_temp = float(national_hourly_temps[i])

        new_row = [
            math.sin(2 * math.pi * hour  / 24.0),   # hour_sin
            math.cos(2 * math.pi * hour  / 24.0),   # hour_cos
            math.sin(2 * math.pi * month / 12.0),   # month_sin
            math.cos(2 * math.pi * month / 12.0),   # month_cos
            math.sin(2 * math.pi * dow   /  7.0),   # dow_sin
            math.cos(2 * math.pi * dow   /  7.0),   # dow_cos
            1 if dow >= 5 else 0,                    # is_weekend
            hour_temp,                               # temperature_max (national weighted)
            pred_mw,                                 # National Hourly Demand (recursive)
        ]

        # Slide context window forward by one step
        context_scaled = np.vstack([
            context_scaled[1:],
            SCALER.transform([new_row])[0]
        ])

    # Convert MW → GW
    return [p / 1000.0 for p in predictions_mw]


# ── Regional fraction lookup from actual data ──────────────────────────────────
def _get_regional_fractions_for_date(date_str: str) -> dict:
    """
    Derive per-region demand fractions from actual data for the given date.
    Falls back to fixed long-run averages for future/missing dates.
    """
    if DF is None:
        return REGION_FRACTIONS.copy()

    target_date = pd.Timestamp(date_str).date()
    day_mask    = DF["datetime"].dt.date == target_date
    day_df      = DF[day_mask]

    if len(day_df) == 0:
        return REGION_FRACTIONS.copy()

    nat_total = day_df["National Hourly Demand"].sum()
    if nat_total == 0:
        return REGION_FRACTIONS.copy()

    fracs = {}
    for region, col in REGION_DEMAND_COL.items():
        fracs[region] = float(day_df[col].sum() / nat_total) if col in day_df.columns else REGION_FRACTIONS[region]
    return fracs


# ── /predict ───────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data       = request.json
    region     = data.get("region", "North")
    date_str   = data.get("date") or datetime.now().strftime("%Y-%m-%d")
    hour       = int(data.get("hour", 12))
    is_holiday = bool(data.get("is_holiday", False))

    national_hourly_temps = data.get("national_hourly_temperatures")

    try:
        target_date   = pd.Timestamp(date_str).date()
        is_historical = (DF is not None and target_date <= DATASET_END.date())

        # ── Temperature source selection ──────────────────────────────────────
        if is_historical:
            # Always use actual recorded temperatures for historical dates
            day_mask = DF["datetime"].dt.date == target_date
            day_df   = DF[day_mask].sort_values("datetime")
            if len(day_df) >= 24:
                national_hourly_temps = day_df["temperature_max"].values[:24].tolist()
            else:
                # Incomplete day: flat monthly average (no fake diurnal curve)
                avg = MONTHLY_TEMPS.get(target_date.month, 30.0)
                national_hourly_temps = [round(avg, 2)] * 24

        elif not national_hourly_temps:
            # Future date with no temps supplied: flat monthly average fallback
            # (no sinusoidal/diurnal simulation)
            avg = MONTHLY_TEMPS.get(target_date.month, 30.0)
            national_hourly_temps = [round(avg, 2)] * 24

        # Strict length check
        if len(national_hourly_temps) != 24:
            return jsonify({
                "error": f"national_hourly_temperatures must have 24 elements, got {len(national_hourly_temps)}"
            }), 400

        # ── Predict 24-hour national curve ────────────────────────────────────
        if MODEL is None:
            baseline_gw   = 160.0 * REGION_FRACTIONS.get(region, 0.2)
            national_gw   = [baseline_gw / REGION_FRACTIONS.get(region, 0.2)] * 24
        else:
            national_gw = predict_24_hours(date_str, national_hourly_temps, is_holiday)

        # ── Regional fractions ────────────────────────────────────────────────
        fracs = _get_regional_fractions_for_date(date_str)

        # Selected region 24-hour curve
        fraction        = fracs.get(region, REGION_FRACTIONS.get(region, 0.2))
        hourly_forecast = [round(v * fraction, 2) for v in national_gw]

        # All 5 regions' full 24-hour curves
        all_regional_forecasts = {
            r: [round(v * fracs.get(r, REGION_FRACTIONS[r]), 2) for v in national_gw]
            for r in REGION_FRACTIONS
        }

        # Selected-hour values for display card and comparison bar
        target_hour_pred = hourly_forecast[hour]
        regional_comparison = {
            r: round(national_gw[hour] * fracs.get(r, REGION_FRACTIONS[r]), 2)
            for r in REGION_FRACTIONS
        }

        # ── Status label ──────────────────────────────────────────────────────
        base_demand = 160.0 * REGION_FRACTIONS.get(region, 0.2)
        if target_hour_pred > base_demand * 1.15:
            status = "Alert: High Demand"
        elif target_hour_pred > base_demand * 1.05:
            status = "Warning: Elevated Load"
        else:
            status = "Normal Load"

        low  = round(max(0.0, target_hour_pred * 0.93), 2)
        high = round(target_hour_pred * 1.07, 2)

        return jsonify({
            "predicted_demand_gw":   round(target_hour_pred, 2),
            "confidence_low":        low,
            "confidence_high":       high,
            "status":                status,
            "hourly_forecast":       hourly_forecast,          # selected region 24h
            "all_regional_forecasts": all_regional_forecasts,  # all 5 regions 24h
            "regional_comparison":   regional_comparison,      # single-hour bar chart
            "national_hourly_gw":    [round(v, 2) for v in national_gw],  # national 24h
            "is_historical":         is_historical,
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── /get_history ───────────────────────────────────────────────────────────────
@app.route("/get_history", methods=["GET"])
def get_history():
    """
    Returns ground-truth demand for a historical date and region.
    Query params: date (YYYY-MM-DD), region
    """
    date_str = request.args.get("date", "")
    region   = request.args.get("region", "North")

    if not date_str:
        return jsonify({"error": "date parameter required"}), 400

    if DF is None:
        return jsonify({"available": False, "hourly_actual": []}), 200

    try:
        target_date = pd.Timestamp(date_str).date()
        day_mask    = DF["datetime"].dt.date == target_date
        day_df      = DF[day_mask].sort_values("datetime")

        if len(day_df) == 0:
            return jsonify({"available": False, "hourly_actual": [], "date": date_str, "region": region})

        col = REGION_DEMAND_COL.get(region)
        if col and col in day_df.columns:
            actuals_mw = day_df[col].values[:24]
        else:
            nat        = day_df["National Hourly Demand"].values[:24]
            actuals_mw = nat * REGION_FRACTIONS.get(region, 0.2)

        actuals_gw = [round(float(v) / 1000.0, 2) for v in actuals_mw]
        while len(actuals_gw) < 24:
            actuals_gw.append(actuals_gw[-1] if actuals_gw else 0.0)

        return jsonify({
            "available":     True,
            "hourly_actual": actuals_gw,
            "date":          date_str,
            "region":        region,
        })

    except Exception as e:
        print(f"History error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── /get_temperature ────────────────────────────────────────────────────────────
@app.route("/get_temperature", methods=["GET"])
def get_temperature_route():
    """
    For historical dates: reads actual per-region and national temperatures from DF.
    For future dates:     calls Open-Meteo via weather_api.py.
    Fallback (API error): flat monthly average — no sinusoidal simulation.
    """
    region   = request.args.get("region", "North")
    date_str = request.args.get("date", "")
    hour_str = request.args.get("hour", "12")

    try:
        hour = int(hour_str)
    except ValueError:
        return jsonify({"error": "hour must be an integer"}), 400

    target_date   = pd.Timestamp(date_str).date()
    is_historical = (DF is not None and target_date <= DATASET_END.date())

    if is_historical:
        day_mask = DF["datetime"].dt.date == target_date
        day_df   = DF[day_mask].sort_values("datetime")

        if len(day_df) >= 24:
            # National weighted temperature (used by LSTM)
            national_hourly = day_df["temperature_max"].values[:24].tolist()

            # Per-region temperature from CSV columns (North, South, East, West, NorthEast)
            reg_col = REGION_TEMP_COL.get(region)
            if reg_col and reg_col in day_df.columns:
                regional_hourly = day_df[reg_col].values[:24].tolist()
            else:
                # Column missing — fall back to national curve
                regional_hourly = national_hourly
        else:
            # Incomplete historical day: flat monthly average (no diurnal curve)
            avg             = MONTHLY_TEMPS.get(target_date.month, 30.0)
            national_hourly = [round(avg, 2)] * 24
            regional_hourly = national_hourly

        city = REGION_CITIES.get(region, {}).get("city", region)
        return jsonify({
            "temperature_celsius":          round(regional_hourly[hour], 1),
            "regional_hourly_temperatures": [round(t, 2) for t in regional_hourly],
            "national_hourly_temperatures": [round(t, 2) for t in national_hourly],
            "city":   city,
            "region": region,
            "date":   date_str,
            "hour":   hour,
            "source": "Historical dataset",
        })

    else:
        # Future date — call Open-Meteo
        from weather_api import get_temperature
        result = get_temperature(region, date_str, hour)

        if "error" in result:
            # Open-Meteo unavailable: flat monthly average fallback (no diurnal curve)
            month = target_date.month

            nat_avg         = MONTHLY_TEMPS.get(month, 30.0)
            national_hourly = [round(nat_avg, 2)] * 24

            reg_avg         = REGIONAL_MONTHLY_TEMPS.get(region, {}).get(month, 28.0)
            regional_hourly = [round(reg_avg, 2)] * 24

            city = REGION_CITIES.get(region, {}).get("city", region)
            return jsonify({
                "temperature_celsius":          round(regional_hourly[hour], 1),
                "regional_hourly_temperatures": regional_hourly,
                "national_hourly_temperatures": national_hourly,
                "city":   city,
                "region": region,
                "date":   date_str,
                "hour":   hour,
                "source": "Fallback: monthly average (Open-Meteo unavailable)",
            })

        return jsonify(result)


if __name__ == "__main__":
    print("Starting GridSense Flask Server on http://localhost:5000 …")
    app.run(port=5000, debug=False)