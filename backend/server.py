"""
GridSense India — Flask Backend (Improved Model Version)
server.py

Matches the improved run_forecaster.py which uses 14 features + target:
    hour_sin, hour_cos, month_sin, month_cos,
    dow_sin, dow_cos, is_weekend, is_holiday,
    temperature_max,
    lag_1h, lag_24h, lag_168h,
    roll_mean_24h, roll_std_24h,
    National Hourly Demand  ← target (index 14)

INFERENCE STRATEGY FOR LAG/ROLLING FEATURES:
    - lag_1h:        the prediction from the previous hour (recursive)
    - lag_24h:       from DF if historical; else from predictions buffer
    - lag_168h:      from DF if historical; else from predictions buffer
    - roll_mean_24h: rolling mean of last 24 demand values
    - roll_std_24h:  rolling std of last 24 demand values
    All of these are maintained in a rolling buffer during the 24-step loop.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import math
import os
import traceback
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

BACKEND_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_PATH   = os.path.join(BACKEND_DIR, "smartgrid_lstm_model.keras")
META_PATH    = os.path.join(BACKEND_DIR, "model_metadata.json")
LOAD_FILE    = os.path.join(BACKEND_DIR, "hourlyLoadDataIndia.xlsx")
TEMP_FILE    = os.path.join(BACKEND_DIR, "historical_hourly_temp.csv")

# ── Flask ──────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="/static")
CORS(app)

@app.route("/")
def serve_index():       return send_from_directory(FRONTEND_DIR, "index.html")
@app.route("/index.html")
def serve_index_html():  return send_from_directory(FRONTEND_DIR, "index.html")
@app.route("/app.js")
def serve_app_js():      return send_from_directory(FRONTEND_DIR, "app.js")
@app.route("/styles.css")
def serve_styles():      return send_from_directory(FRONTEND_DIR, "styles.css")

# ── Feature contract (must match run_forecaster.py exactly) ───────────────────
LSTM_COLS = [
    "hour_sin",        # 0
    "hour_cos",        # 1
    "month_sin",       # 2
    "month_cos",       # 3
    "dow_sin",         # 4
    "dow_cos",         # 5
    "is_weekend",      # 6
    "is_holiday",      # 7
    "temperature_max", # 8
    "lag_1h",          # 9
    "lag_24h",         # 10
    "lag_168h",        # 11
    "roll_mean_24h",   # 12
    "roll_std_24h",    # 13
    "National Hourly Demand",  # 14 — TARGET
]
TARGET_IDX   = 14
SEQUENCE_LEN = 168

# ── Region config ──────────────────────────────────────────────────────────────
REGION_CITIES = {
    "North":     {"city": "New Delhi",  "lat": 28.6139, "lon": 77.2090},
    "South":     {"city": "Bengaluru",  "lat": 12.9716, "lon": 77.5946},
    "East":      {"city": "Kolkata",    "lat": 22.5726, "lon": 88.3639},
    "West":      {"city": "Mumbai",     "lat": 19.0760, "lon": 72.8777},
    "NorthEast": {"city": "Guwahati",   "lat": 26.1445, "lon": 91.7362},
}
REGION_DEMAND_COL = {
    "North":     "Northen Region Hourly Demand",
    "West":      "Western Region Hourly Demand",
    "East":      "Eastern Region Hourly Demand",
    "South":     "Southern Region Hourly Demand",
    "NorthEast": "North-Eastern Region Hourly Demand",
}
REGION_TEMP_COL = {
    "North": "North", "South": "South",
    "East": "East",   "West": "West", "NorthEast": "NorthEast",
}
REGION_FRACTIONS = {
    "North": 0.300, "West": 0.266, "South": 0.232,
    "East": 0.154,  "NorthEast": 0.048,
}
REGIONAL_MONTHLY_TEMPS = {
    "North":     {1:15,2:19,3:25,4:32,5:35,6:34,7:31,8:30,9:29,10:26,11:20,12:15},
    "South":     {1:22,2:24,3:27,4:29,5:28,6:25,7:24,8:24,9:24,10:24,11:23,12:21},
    "East":      {1:20,2:23,3:28,4:31,5:32,6:30,7:29,8:29,9:29,10:28,11:24,12:20},
    "West":      {1:24,2:25,3:27,4:29,5:30,6:29,7:27,8:27,9:27,10:28,11:27,12:25},
    "NorthEast": {1:18,2:20,3:24,4:27,5:28,6:29,7:29,8:29,9:28,10:26,11:22,12:19},
}

# ── Global state ───────────────────────────────────────────────────────────────
MODEL         = None
SCALER        = None
DF            = None
MONTHLY_TEMPS = {}
DATASET_END   = None


def load_data_and_model():
    global MODEL, SCALER, DF, MONTHLY_TEMPS, DATASET_END

    # ── Temperature ───────────────────────────────────────────────────────────
    print("Loading temperature data...")
    df_temp = pd.read_csv(TEMP_FILE)
    df_temp["datetime"] = pd.to_datetime(df_temp["datetime"])
    df_temp["month"]    = df_temp["datetime"].dt.month
    MONTHLY_TEMPS = df_temp.groupby("month")["hourly_temperature"].mean().to_dict()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading hourly load data...")
    df_raw = pd.read_excel(LOAD_FILE)
    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
    df_raw = df_raw.sort_values("datetime").reset_index(drop=True)
    DATASET_END = df_raw["datetime"].max()
    print(f"  Dataset: {df_raw['datetime'].min().date()} -> {DATASET_END.date()} ({len(df_raw):,} rows)")

    # ── Merge ─────────────────────────────────────────────────────────────────
    df = df_raw.merge(df_temp, on="datetime", how="left")
    df["temperature_max"] = df["hourly_temperature"].ffill().bfill()
    for col in REGION_TEMP_COL.values():
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # ── Feature engineering (identical to run_forecaster.py) ─────────────────
    df["hour"]       = df["datetime"].dt.hour
    df["month"]      = df["datetime"].dt.month
    df["dayofweek"]  = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_holiday"] = 0    # default; overridden at inference by user input

    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]      / 24.0)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]      / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]     / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]     / 12.0)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] /  7.0)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] /  7.0)

    # Lag features
    df["lag_1h"]   = df["National Hourly Demand"].shift(1)
    df["lag_24h"]  = df["National Hourly Demand"].shift(24)
    df["lag_168h"] = df["National Hourly Demand"].shift(168)

    # Rolling features
    df["roll_mean_24h"] = df["National Hourly Demand"].shift(1).rolling(24).mean()
    df["roll_std_24h"]  = df["National Hourly Demand"].shift(1).rolling(24).std()

    # Fill NaN from lag/rolling with ffill for the context window
    df["lag_1h"]        = df["lag_1h"].ffill().bfill()
    df["lag_24h"]       = df["lag_24h"].ffill().bfill()
    df["lag_168h"]      = df["lag_168h"].ffill().bfill()
    df["roll_mean_24h"] = df["roll_mean_24h"].ffill().bfill()
    df["roll_std_24h"]  = df["roll_std_24h"].ffill().bfill()

    DF = df.reset_index(drop=True)

    # -- Scaler ----------------------------------------------------------------
    # Try to load saved scaler parameters from model_metadata.json first.
    # This guarantees server uses the exact same scaler that was fit during training.
    if os.path.exists(META_PATH):
        print("  Loading scaler from model_metadata.json...")
        with open(META_PATH) as f:
            meta = json.load(f)
        sc_params = meta["scaler"]
        SCALER = MinMaxScaler()
        # Reconstruct scaler from saved parameters
        SCALER.data_min_  = np.array(sc_params["data_min"],  dtype="float32")
        SCALER.data_max_  = np.array(sc_params["data_max"],  dtype="float32")
        SCALER.scale_     = np.array(sc_params["scale"],     dtype="float32")
        SCALER.min_       = np.array(sc_params["min"],       dtype="float32")
        SCALER.data_range_= SCALER.data_max_ - SCALER.data_min_
        SCALER.n_features_in_ = len(LSTM_COLS)
        SCALER.n_samples_seen_ = 1
        print("  Scaler reconstructed from metadata.")
    else:
        print("  model_metadata.json not found — fitting scaler from DF.")
        lstm_data = DF[LSTM_COLS].values.astype("float32")
        SCALER = MinMaxScaler()
        SCALER.fit(lstm_data)
        print("  Scaler fitted from live data.")

    # ── Model ─────────────────────────────────────────────────────────────────
    if os.path.exists(MODEL_PATH):
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        print(f"  Model loaded: {MODEL_PATH}")
        print(f"  Input shape: {MODEL.input_shape}")
    else:
        print(f"  [Warning] Model NOT found at {MODEL_PATH}")

    print("Server ready.\n")


try:
    load_data_and_model()
except Exception as e:
    print(f"Startup error: {e}")
    traceback.print_exc()


# ── Context window builder ─────────────────────────────────────────────────────
def _get_context_sequence(target_dt: datetime) -> np.ndarray:
    """
    Returns the 168-row feature matrix immediately before target_dt.
    All 15 columns including lag and rolling features are included.
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
def predict_24_hours(
    date_str: str,
    national_hourly_temps: list,
    is_holiday: bool,
) -> list:
    """
    Recursively predict 24 hours of national demand (MW).

    Lag and rolling features are maintained in a deque buffer and updated
    after every predicted step — matching exactly how they were computed
    during training (shift(1) rolling on past demand values).

    Returns list of 24 GW values.
    """
    if MODEL is None or SCALER is None or DF is None:
        raise ValueError("Model or data not loaded.")

    base_time    = datetime.strptime(date_str, "%Y-%m-%d")
    future_times = [base_time + timedelta(hours=i) for i in range(24)]

    # ── Seed context from historical data ─────────────────────────────────────
    context_raw    = _get_context_sequence(base_time)
    context_scaled = SCALER.transform(context_raw)

    # ── Demand buffer for recursive lag/rolling computation ───────────────────
    # Initialise with the last 168 actual demand values from the context window
    # (column index TARGET_IDX = 14 in raw space)
    demand_buffer = deque(
        context_raw[:, TARGET_IDX].tolist(),
        maxlen=168
    )

    predictions_mw = []

    for i, dt in enumerate(future_times):

        # ── Step 1: predict scaled demand ─────────────────────────────────────
        pred_scaled = MODEL.predict(
            context_scaled[np.newaxis, :, :], verbose=0
        ).flatten()[0]

        # Inverse-transform to MW
        dummy             = np.zeros(len(LSTM_COLS))
        dummy[TARGET_IDX] = pred_scaled
        pred_mw = SCALER.inverse_transform([dummy])[0, TARGET_IDX]

        predictions_mw.append(pred_mw)

        # ── Step 2: update demand buffer ──────────────────────────────────────
        demand_buffer.append(pred_mw)
        buf = list(demand_buffer)   # list of up to 168 values

        # ── Step 3: compute lag/rolling from buffer ────────────────────────────
        lag_1h   = buf[-1]                          # just-predicted value
        lag_24h  = buf[-24]  if len(buf) >= 24  else buf[0]
        lag_168h = buf[-168] if len(buf) >= 168 else buf[0]

        # roll_mean/std use the last 24 values before this step
        window_24 = buf[-24:] if len(buf) >= 24 else buf
        roll_mean = float(np.mean(window_24))
        roll_std  = float(np.std(window_24))

        # ── Step 4: build next feature row ────────────────────────────────────
        hour  = dt.hour
        month = dt.month
        dow   = dt.weekday()

        new_row = [
            math.sin(2 * math.pi * hour  / 24.0),   # hour_sin
            math.cos(2 * math.pi * hour  / 24.0),   # hour_cos
            math.sin(2 * math.pi * month / 12.0),   # month_sin
            math.cos(2 * math.pi * month / 12.0),   # month_cos
            math.sin(2 * math.pi * dow   /  7.0),   # dow_sin
            math.cos(2 * math.pi * dow   /  7.0),   # dow_cos
            1 if dow >= 5 else 0,                    # is_weekend
            1 if is_holiday else 0,                  # is_holiday
            float(national_hourly_temps[i]),         # temperature_max
            lag_1h,                                  # lag_1h
            lag_24h,                                 # lag_24h
            lag_168h,                                # lag_168h
            roll_mean,                               # roll_mean_24h
            roll_std,                                # roll_std_24h
            pred_mw,                                 # National Hourly Demand
        ]

        # ── Step 5: slide context window ──────────────────────────────────────
        context_scaled = np.vstack([
            context_scaled[1:],
            SCALER.transform([new_row])[0]
        ])

    # Convert MW → GW
    return [p / 1000.0 for p in predictions_mw]


# ── Regional fraction helper ───────────────────────────────────────────────────
def _get_regional_fractions_for_date(date_str: str) -> dict:
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
    return {
        r: float(day_df[col].sum() / nat_total) if col in day_df.columns else REGION_FRACTIONS[r]
        for r, col in REGION_DEMAND_COL.items()
    }


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

        # ── Temperature source ────────────────────────────────────────────────
        if is_historical:
            day_mask = DF["datetime"].dt.date == target_date
            day_df   = DF[day_mask].sort_values("datetime")
            if len(day_df) >= 24:
                national_hourly_temps = day_df["temperature_max"].values[:24].tolist()
            else:
                avg = MONTHLY_TEMPS.get(target_date.month, 30.0)
                national_hourly_temps = [round(avg, 2)] * 24
        elif not national_hourly_temps:
            avg = MONTHLY_TEMPS.get(target_date.month, 30.0)
            national_hourly_temps = [round(avg, 2)] * 24

        if len(national_hourly_temps) != 24:
            return jsonify({"error": f"national_hourly_temperatures must have 24 values, got {len(national_hourly_temps)}"}), 400

        # ── Predict ───────────────────────────────────────────────────────────
        if MODEL is None:
            baseline = 160.0 * REGION_FRACTIONS.get(region, 0.2)
            national_gw = [baseline / REGION_FRACTIONS.get(region, 0.2)] * 24
        else:
            national_gw = predict_24_hours(date_str, national_hourly_temps, is_holiday)

        fracs = _get_regional_fractions_for_date(date_str)

        fraction        = fracs.get(region, REGION_FRACTIONS.get(region, 0.2))
        hourly_forecast = [round(v * fraction, 2) for v in national_gw]

        all_regional_forecasts = {
            r: [round(v * fracs.get(r, REGION_FRACTIONS[r]), 2) for v in national_gw]
            for r in REGION_FRACTIONS
        }

        target_hour_pred   = hourly_forecast[hour]
        regional_comparison = {
            r: round(national_gw[hour] * fracs.get(r, REGION_FRACTIONS[r]), 2)
            for r in REGION_FRACTIONS
        }

        base_demand = 160.0 * REGION_FRACTIONS.get(region, 0.2)
        if target_hour_pred > base_demand * 1.15:
            status = "Alert: High Demand"
        elif target_hour_pred > base_demand * 1.05:
            status = "Warning: Elevated Load"
        else:
            status = "Normal Load"

        return jsonify({
            "predicted_demand_gw":    round(target_hour_pred, 2),
            "confidence_low":         round(max(0.0, target_hour_pred * 0.93), 2),
            "confidence_high":        round(target_hour_pred * 1.07, 2),
            "status":                 status,
            "hourly_forecast":        hourly_forecast,
            "all_regional_forecasts": all_regional_forecasts,
            "regional_comparison":    regional_comparison,
            "national_hourly_gw":     [round(v, 2) for v in national_gw],
            "is_historical":          is_historical,
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── /get_history ───────────────────────────────────────────────────────────────
@app.route("/get_history", methods=["GET"])
def get_history():
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
        actuals_mw = day_df[col].values[:24] if col and col in day_df.columns \
            else day_df["National Hourly Demand"].values[:24] * REGION_FRACTIONS.get(region, 0.2)
        actuals_gw = [round(float(v) / 1000.0, 2) for v in actuals_mw]
        while len(actuals_gw) < 24:
            actuals_gw.append(actuals_gw[-1] if actuals_gw else 0.0)
        return jsonify({"available": True, "hourly_actual": actuals_gw, "date": date_str, "region": region})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── /get_temperature ────────────────────────────────────────────────────────────
@app.route("/get_temperature", methods=["GET"])
def get_temperature_route():
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
            national_hourly = day_df["temperature_max"].values[:24].tolist()
            reg_col         = REGION_TEMP_COL.get(region)
            regional_hourly = day_df[reg_col].values[:24].tolist() \
                if reg_col and reg_col in day_df.columns else national_hourly
        else:
            avg             = MONTHLY_TEMPS.get(target_date.month, 30.0)
            national_hourly = [round(avg, 2)] * 24
            regional_hourly = national_hourly
        city = REGION_CITIES.get(region, {}).get("city", region)
        return jsonify({
            "temperature_celsius":          round(regional_hourly[hour], 1),
            "regional_hourly_temperatures": [round(t, 2) for t in regional_hourly],
            "national_hourly_temperatures": [round(t, 2) for t in national_hourly],
            "city": city, "region": region, "date": date_str, "hour": hour,
            "source": "Historical dataset",
        })
    else:
        from weather_api import get_temperature
        result = get_temperature(region, date_str, hour)
        if "error" in result:
            month           = target_date.month
            nat_avg         = MONTHLY_TEMPS.get(month, 30.0)
            national_hourly = [round(nat_avg, 2)] * 24
            reg_avg         = REGIONAL_MONTHLY_TEMPS.get(region, {}).get(month, 28.0)
            regional_hourly = [round(reg_avg, 2)] * 24
            city = REGION_CITIES.get(region, {}).get("city", region)
            return jsonify({
                "temperature_celsius":          round(regional_hourly[hour], 1),
                "regional_hourly_temperatures": regional_hourly,
                "national_hourly_temperatures": national_hourly,
                "city": city, "region": region, "date": date_str, "hour": hour,
                "source": "Fallback: monthly average (Open-Meteo unavailable)",
            })
        return jsonify(result)


if __name__ == "__main__":
    print("Starting GridSense Flask Server on http://localhost:5000 …")
    app.run(port=5000, debug=False)

