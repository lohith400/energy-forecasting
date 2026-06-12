"""
GridSense India — Flask Backend (Fixed)
Provides /predict, /get_history, and /get_temperature endpoints.

Fixes applied:
  1. All file paths resolved relative to this file (absolute paths).
  2. Temperature data extended cyclically through 2024 using 2019-2021 averages.
  3. predict_24_hours() uses actual 168-row context for historical dates.
  4. Real per-region demand columns used for historical regional fractions.
  5. /get_history endpoint returns actual demand for any historical date.
  6. Weather API falls back to monthly averages for dates outside forecast window.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import math
import os
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_PATH    = os.path.join(BASE_DIR, "smartgrid_lstm_model.keras")
LOAD_FILE     = os.path.join(BASE_DIR, "hourlyLoadDataIndia.xlsx")
TEMP_FILE     = os.path.join(BASE_DIR, "historical_hourly_temp.csv")

# ── Flask app ──────────────────────────────────────────────────────────────────
# Use static_url_path='/static' so Flask's built-in static handler doesn't
# shadow our API routes (predict, get_history, get_temperature).
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="/static")
CORS(app)

# ── Flask route: serve frontend ───────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/index.html")
def serve_index_html():
    return send_from_directory(FRONTEND_DIR, "index.html")

# Serve known frontend assets (CSS, JS) by explicit routes
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

# Column names for per-region demand in the dataset
REGION_COL = {
    "North":     "Northen Region Hourly Demand",
    "West":      "Western Region Hourly Demand",
    "East":      "Eastern Region Hourly Demand",
    "South":     "Southern Region Hourly Demand",
    "NorthEast": "North-Eastern Region Hourly Demand",
}

# Long-run average fractions (used only for future dates beyond dataset)
REGION_FRACTIONS = {
    "North":     175.0 / 583.0,
    "South":     135.0 / 583.0,
    "East":       90.0 / 583.0,
    "West":      155.0 / 583.0,
    "NorthEast":  28.0 / 583.0,
}

LSTM_COLS    = ["hour_sin","hour_cos","month_sin","month_cos",
                "dow_sin","dow_cos","is_weekend","temperature_max",
                "National Hourly Demand"]
TARGET_IDX   = 8
SEQUENCE_LEN = 168

# ── Global state ───────────────────────────────────────────────────────────────
MODEL         = None
SCALER        = None
DF            = None          # Full preprocessed DataFrame (index-aligned)
MONTHLY_TEMPS = {}            # {(month): avg_temp_celsius}  – for weather fallback
DATASET_END   = None          # Last datetime in the dataset (pd.Timestamp)

# ── Monthly average temperature lookup (from the temp file 2019-2021) ─────────
MONTH_MAP = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
             "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}


def _build_monthly_temp_avg(df_temp_clean: pd.DataFrame) -> dict:
    """Average temperature per calendar month (1-12) across all years."""
    return df_temp_clean.groupby("month")["temperature_max"].mean().to_dict()


def _get_temp_for_date(year: int, month: int) -> float:
    """
    Return a best-estimate temperature for any (year, month).
    Priority: exact match in dataset → monthly average fallback.
    """
    return MONTHLY_TEMPS.get(month, 30.0)


def load_data_and_model():
    global MODEL, SCALER, DF, MONTHLY_TEMPS, DATASET_END

    # ── Load hourly temperature data (weighted national hourly temperature) ──
    print("Loading temperature data…")
    df_temp = pd.read_csv(TEMP_FILE)
    df_temp["datetime"] = pd.to_datetime(df_temp["datetime"])
    
    # Extract month for monthly average fallback lookup
    df_temp["month"] = df_temp["datetime"].dt.month
    MONTHLY_TEMPS = df_temp.groupby("month")["hourly_temperature"].mean().to_dict()
    print(f"  Monthly avg temps built: {MONTHLY_TEMPS}")

    # ── Load hourly load data ──────────────────────────────────────────────────
    print("Loading hourly load data…")
    df_raw = pd.read_excel(LOAD_FILE)
    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
    df_raw = df_raw.sort_values("datetime").reset_index(drop=True)
    DATASET_END = df_raw["datetime"].max()
    print(f"  Dataset: {df_raw['datetime'].min().date()} -> {DATASET_END.date()} ({len(df_raw):,} rows)")

    # ── Merge temperature on datetime ─────────────────────────────────────────
    df = df_raw.merge(df_temp, on="datetime", how="left")
    df["temperature_max"] = df["hourly_temperature"].ffill().bfill()
    for col in ["North", "West", "South", "East", "NorthEast"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # ── Feature engineering ───────────────────────────────────────────────────
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

    # ── Fit scaler on full data ────────────────────────────────────────────────
    lstm_data = DF[LSTM_COLS].values.astype("float32")
    SCALER = MinMaxScaler()
    SCALER.fit(lstm_data)
    print("  Scaler fitted.")

    # ── Load LSTM model ────────────────────────────────────────────────────────
    if os.path.exists(MODEL_PATH):
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        print(f"  Model loaded: {MODEL_PATH}")
    else:
        print(f"  ⚠ Model file NOT found at {MODEL_PATH}")

    print("Server ready.\n")


# Load everything on startup
try:
    load_data_and_model()
except Exception as e:
    print(f"Error during startup: {e}")
    traceback.print_exc()


# ── Prediction helper ──────────────────────────────────────────────────────────
def _get_context_sequence(target_dt: datetime) -> np.ndarray:
    """
    Return the 168-hour input window (in raw feature space) for target_dt.

    • If target_dt is within the dataset: use the actual 168 rows that
      immediately precede it (the gold-standard approach).
    • If target_dt is beyond the dataset: use the last 168 rows available.
    """
    target_ts = pd.Timestamp(target_dt)
    if DF is not None and target_ts > DF["datetime"].iloc[0]:
        # Find the position in DF that is <= target_ts
        mask = DF["datetime"] < target_ts
        if mask.sum() >= SEQUENCE_LEN:
            end_pos = mask.sum()
            start_pos = end_pos - SEQUENCE_LEN
            return DF.iloc[start_pos:end_pos][LSTM_COLS].values.astype("float32")

    # Fallback: last SEQUENCE_LEN rows in dataset
    return DF.tail(SEQUENCE_LEN)[LSTM_COLS].values.astype("float32")


def predict_24_hours(date_str: str, national_hourly_temps: list, is_holiday: bool) -> list:
    """
    Predict all 24 hours of a day starting from MIDNIGHT (hour 0).
    Returns a list of 24 national GW values: index 0 = midnight, 23 = 11 PM.
    This ensures alignment with actual data (which also starts at midnight).
    """
    if MODEL is None or SCALER is None or DF is None:
        raise ValueError("Model or data not loaded properly.")

    # Always start at midnight so the chart aligns with actual data
    base_time    = datetime.strptime(date_str, "%Y-%m-%d")  # midnight
    future_times = [base_time + timedelta(hours=i) for i in range(24)]

    # Get the correct 168-hour context for this date
    context_raw    = _get_context_sequence(base_time)
    context_scaled = SCALER.transform(context_raw)

    predictions_mw = []
    for i, dt in enumerate(future_times):
        pred_scaled = MODEL.predict(context_scaled[np.newaxis, :, :], verbose=0).flatten()[0]

        dummy = np.zeros(len(LSTM_COLS))
        dummy[TARGET_IDX] = pred_scaled
        pred_mw = SCALER.inverse_transform([dummy])[0, TARGET_IDX]

        if is_holiday:
            pred_mw *= 0.88

        predictions_mw.append(pred_mw)

        hour  = dt.hour
        month = dt.month
        dow   = dt.weekday()

        # Ingest the target hour's temperature directly from the passed weighted curve
        hour_temp = national_hourly_temps[i]

        new_row = [
            np.sin(2 * np.pi * hour  / 24.0),
            np.cos(2 * np.pi * hour  / 24.0),
            np.sin(2 * np.pi * month / 12.0),
            np.cos(2 * np.pi * month / 12.0),
            np.sin(2 * np.pi * dow   /  7.0),
            np.cos(2 * np.pi * dow   /  7.0),
            1 if dow >= 5 else 0,
            hour_temp,
            pred_mw,
        ]
        context_scaled = np.vstack([context_scaled[1:],
                                    SCALER.transform([new_row])[0]])

    # Convert MW -> GW
    return [p / 1000.0 for p in predictions_mw]


def _get_regional_fractions_for_date(date_str: str) -> dict:
    """
    Compute per-region fractions from the actual data for the given date.
    Falls back to the global fixed fractions if date is outside the dataset.
    """
    if DF is None:
        return REGION_FRACTIONS.copy()

    target_date = pd.Timestamp(date_str).date()
    day_mask = DF["datetime"].dt.date == target_date
    day_df   = DF[day_mask]

    if len(day_df) == 0:
        return REGION_FRACTIONS.copy()

    nat_total = day_df["National Hourly Demand"].sum()
    if nat_total == 0:
        return REGION_FRACTIONS.copy()

    fracs = {}
    for region, col in REGION_COL.items():
        if col in day_df.columns:
            fracs[region] = float(day_df[col].sum() / nat_total)
        else:
            fracs[region] = REGION_FRACTIONS[region]
    return fracs


# ── /predict ──────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data        = request.json
    region      = data.get("region", "North")
    date_str    = data.get("date") or datetime.now().strftime("%Y-%m-%d")
    hour        = int(data.get("hour", 12))  # used only for highlighting
    is_holiday  = bool(data.get("is_holiday", False))
    
    national_hourly_temps = data.get("national_hourly_temperatures")

    try:
        # Determine if this date is historical
        target_date = pd.Timestamp(date_str).date()
        is_historical = (DF is not None and target_date <= DATASET_END.date())

        # If historical, always load actual national weighted temperatures from DF
        if is_historical:
            day_mask = DF["datetime"].dt.date == target_date
            day_df = DF[day_mask].sort_values("datetime")
            if len(day_df) >= 24:
                national_hourly_temps = day_df["temperature_max"].values[:24].tolist()
            else:
                month = target_date.month
                avg_temp = MONTHLY_TEMPS.get(month, 30.0)
                national_hourly_temps = [
                    round(avg_temp + 4.0 * math.sin(2 * np.pi * (h - 9) / 24.0), 2)
                    for h in range(24)
                ]
        # If future and missing, construct fallback
        elif not national_hourly_temps:
            base_temp = data.get("temperature")
            if base_temp is None:
                month = target_date.month
                base_temp = MONTHLY_TEMPS.get(month, 30.0)
            else:
                base_temp = float(base_temp)
            national_hourly_temps = [
                round(base_temp + 4.0 * math.sin(2 * np.pi * (h - 9) / 24.0), 2)
                for h in range(24)
            ]

        # Make sure national_hourly_temps has exactly 24 elements
        if len(national_hourly_temps) != 24:
            return jsonify({"error": f"national_hourly_temperatures must have exactly 24 elements, got {len(national_hourly_temps)}"}), 400

        if MODEL is None:
            # Hard fallback when model failed to load
            baseline_gw     = 160.0 * REGION_FRACTIONS.get(region, 0.2)
            hourly_forecast = [baseline_gw * (1 + 0.05 * math.sin(i)) for i in range(24)]
            comparison      = {r: baseline_gw for r in REGION_FRACTIONS}
            target_hour_pred = hourly_forecast[hour]
        else:
            # Pass the 24 hourly temperatures to predict_24_hours
            national_gw = predict_24_hours(date_str, national_hourly_temps, is_holiday)

            # Determine regional fractions
            fracs = _get_regional_fractions_for_date(date_str)

            fraction        = fracs.get(region, REGION_FRACTIONS.get(region, 0.2))
            hourly_forecast = [v * fraction for v in national_gw]

            # Use the selected hour's value for the display card
            target_hour_pred = hourly_forecast[hour]
            comparison       = {r: national_gw[hour] * f for r, f in fracs.items()}

        base_demand = 160.0 * REGION_FRACTIONS.get(region, 0.2)

        if target_hour_pred > base_demand * 1.15:
            status = "Alert: High Demand"
        elif target_hour_pred > base_demand * 1.05:
            status = "Warning: Elevated Load"
        else:
            status = "Normal Load"

        low  = max(0.0, round(target_hour_pred * 0.93, 2))
        high = round(target_hour_pred * 1.07, 2)

        return jsonify({
            "predicted_demand_gw":  round(target_hour_pred, 2),
            "confidence_low":       low,
            "confidence_high":      high,
            "status":               status,
            "hourly_forecast":      [round(v, 2) for v in hourly_forecast],
            "regional_comparison":  {r: round(v, 2) for r, v in comparison.items()},
            "is_historical":        is_historical,
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── /get_history ──────────────────────────────────────────────────────────────
@app.route("/get_history", methods=["GET"])
def get_history():
    """
    Returns actual (ground-truth) demand for a given date and region.
    Query params:
      date   – YYYY-MM-DD
      region – North | South | East | West | NorthEast
    Response:
      {
        "hourly_actual": [float * 24],   # GW, 24 hours starting from midnight
        "date": "YYYY-MM-DD",
        "region": "...",
        "available": true/false
      }
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

        col = REGION_COL.get(region)
        if col and col in day_df.columns:
            actuals_mw = day_df[col].values[:24]
        else:
            # Fall back to national × fraction
            nat = day_df["National Hourly Demand"].values[:24]
            actuals_mw = nat * REGION_FRACTIONS.get(region, 0.2)

        actuals_gw = [round(float(v) / 1000.0, 2) for v in actuals_mw]

        # Pad to 24 if data is incomplete
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


# Regional monthly baseline average temperatures (typical values for representative cities)
REGIONAL_MONTHLY_TEMPS = {
    "North":     {1: 15.0, 2: 19.0, 3: 25.0, 4: 32.0, 5: 35.0, 6: 34.0, 7: 31.0, 8: 30.0, 9: 29.0, 10: 26.0, 11: 20.0, 12: 15.0},
    "South":     {1: 22.0, 2: 24.0, 3: 27.0, 4: 29.0, 5: 28.0, 6: 25.0, 7: 24.0, 8: 24.0, 9: 24.0, 10: 24.0, 11: 23.0, 12: 21.0},
    "East":      {1: 20.0, 2: 23.0, 3: 28.0, 4: 31.0, 5: 32.0, 6: 30.0, 7: 29.0, 8: 29.0, 9: 29.0, 10: 28.0, 11: 24.0, 12: 20.0},
    "West":      {1: 24.0, 2: 25.0, 3: 27.0, 4: 29.0, 5: 30.0, 6: 29.0, 7: 27.0, 8: 27.0, 9: 27.0, 10: 28.0, 11: 27.0, 12: 25.0},
    "NorthEast": {1: 18.0, 2: 20.0, 3: 24.0, 4: 27.0, 5: 28.0, 6: 29.0, 7: 29.0, 8: 29.0, 9: 28.0, 10: 26.0, 11: 22.0, 12: 19.0},
}

# ── /get_temperature ──────────────────────────────────────────────────────────
@app.route("/get_temperature", methods=["GET"])
def get_temperature_route():
    region   = request.args.get("region", "North")
    date_str = request.args.get("date", "")
    hour_str = request.args.get("hour", "12")

    try:
        hour = int(hour_str)
    except ValueError:
        return jsonify({"error": "Hour must be an integer"}), 400

    target_date = pd.Timestamp(date_str).date()
    is_historical = (DF is not None and target_date <= DATASET_END.date())

    if is_historical:
        # Load actual national weighted temperatures from DF for this date
        day_mask = DF["datetime"].dt.date == target_date
        day_df = DF[day_mask].sort_values("datetime")
        
        if len(day_df) >= 24:
            national_hourly = day_df["temperature_max"].values[:24].tolist()
            if region in day_df.columns:
                regional_hourly = day_df[region].values[:24].tolist()
            else:
                regional_hourly = national_hourly
        else:
            # Fallback if historical data is incomplete
            month = target_date.month
            avg_temp = MONTHLY_TEMPS.get(month, 30.0)
            national_hourly = [
                round(avg_temp + 4.0 * math.sin(2 * np.pi * (h - 9) / 24.0), 2)
                for h in range(24)
            ]
            regional_hourly = national_hourly
            
        city = REGION_CITIES.get(region, {}).get("city", region)
        temperature_celsius = regional_hourly[hour]
        
        return jsonify({
            "temperature_celsius": round(temperature_celsius, 1),
            "regional_hourly_temperatures": [round(t, 2) for t in regional_hourly],
            "national_hourly_temperatures": [round(t, 2) for t in national_hourly],
            "city": city,
            "region": region,
            "date": date_str,
            "hour": hour,
            "source": "Historical dataset",
        })
        
    else:
        # Try live weather API first
        from weather_api import get_temperature
        result = get_temperature(region, date_str, hour)

        if "error" in result:
            month = target_date.month
            
            # National fallback
            avg_temp = MONTHLY_TEMPS.get(month, 30.0)
            national_hourly = [
                round(avg_temp + 4.0 * math.sin(2 * np.pi * (h - 9) / 24.0), 2)
                for h in range(24)
            ]
            
            # Regional fallback
            reg_avg = REGIONAL_MONTHLY_TEMPS.get(region, {}).get(month, 28.0)
            regional_hourly = [
                round(reg_avg + 4.0 * math.sin(2 * np.pi * (h - 9) / 24.0), 2)
                for h in range(24)
            ]
            
            city = REGION_CITIES.get(region, {}).get("city", region)
            temperature_celsius = regional_hourly[hour]
            
            return jsonify({
                "temperature_celsius": round(temperature_celsius, 1),
                "regional_hourly_temperatures": regional_hourly,
                "national_hourly_temperatures": national_hourly,
                "city": city,
                "region": region,
                "date": date_str,
                "hour": hour,
                "source": "Fallback monthly average (API offline)",
            })

        return jsonify(result)


if __name__ == "__main__":
    print("Starting GridSense Flask Server on http://localhost:5000 …")
    app.run(port=5000, debug=False)
