"""
GridSense India — Flask Backend
Provides /predict and /get_temperature endpoints for the forecasting dashboard.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import math
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Representative city coordinates for each region
REGION_CITIES = {
    "North": {"city": "New Delhi", "lat": 28.6139, "lon": 77.2090},
    "South": {"city": "Bengaluru", "lat": 12.9716, "lon": 77.5946},
    "East": {"city": "Kolkata", "lat": 22.5726, "lon": 88.3639},
    "West": {"city": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    "NorthEast": {"city": "Guwahati", "lat": 26.1445, "lon": 91.7362},
}

# Regional shares of national demand (to scale down National forecast)
REGION_FRACTIONS = {
    "North": 175.0 / 583.0,
    "South": 135.0 / 583.0,
    "East": 90.0 / 583.0,
    "West": 155.0 / 583.0,
    "NorthEast": 28.0 / 583.0,
}

MODEL = None
SCALER = None
LSTM_COLS = ["hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos", "is_weekend", "temperature_max", "National Hourly Demand"]
TARGET_IDX = 8
SEQUENCE_LEN = 168
LAST_KNOWN_DATA = None  # DataFrame of the last 168 hours of the dataset

def load_data_and_model():
    global MODEL, SCALER, LAST_KNOWN_DATA
    
    # Load Model
    model_path = "smartgrid_lstm_model.keras"
    if os.path.exists(model_path):
        MODEL = tf.keras.models.load_model(model_path)
        print("Loaded trained LSTM model.")
    else:
        print("Model file not found. Ensure run_forecaster.py has been run.")
        return
        
    # Load and preprocess Data to fit the Scaler exactly as in run_forecaster.py
    print("Loading data to fit scaler and get baseline sequence...")
    df_raw = pd.read_excel("hourlyLoadDataIndia.xlsx")
    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
    df_raw = df_raw.sort_values("datetime").reset_index(drop=True)
    
    df_temp = pd.read_excel("monthly_temp.xlsx")
    # Quick clean of temp data (as done in run_forecaster.py)
    temp_cols_lower = [c.lower() for c in df_temp.columns]
    year_col  = df_temp.columns[[i for i,c in enumerate(temp_cols_lower) if "year"  in c][0]] if any("year"  in c for c in temp_cols_lower) else df_temp.columns[0]
    month_col = df_temp.columns[[i for i,c in enumerate(temp_cols_lower) if "month" in c][0]] if any("month" in c for c in temp_cols_lower) else df_temp.columns[1]
    remaining = [c for c in df_temp.columns if c not in [year_col, month_col]]
    temp_val_col = next((c for c in remaining if "temp" in c.lower()), remaining[-1])
    
    df_temp_clean = df_temp[[year_col, month_col, temp_val_col]].copy()
    df_temp_clean.columns = ["year", "month", "temperature_max"]
    MONTH_MAP = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                 "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
    if pd.api.types.is_string_dtype(df_temp_clean["month"]) or isinstance(df_temp_clean["month"].iloc[0], str):
        df_temp_clean["month"] = df_temp_clean["month"].astype(str).str[:3].str.lower().map(MONTH_MAP)
    df_temp_clean["year"] = df_temp_clean["year"].astype(int)
    df_temp_clean["month"] = df_temp_clean["month"].astype(int)
    
    df_raw["year"]  = df_raw["datetime"].dt.year
    df_raw["month"] = df_raw["datetime"].dt.month
    
    df = df_raw.merge(df_temp_clean, on=["year","month"], how="left")
    df["temperature_max"] = df["temperature_max"].ffill().bfill()
    
    # Feature Engineering (time encodings)
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    df["is_weekend"] = df["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)
    
    lstm_data = df[LSTM_COLS].values.astype("float32")
    
    SCALER = MinMaxScaler()
    SCALER.fit(lstm_data)
    
    # Save the last 168 hours as baseline
    LAST_KNOWN_DATA = df.tail(SEQUENCE_LEN).copy()
    print("Scaler fit and baseline sequence loaded.")

# Load everything on startup
import traceback
try:
    load_data_and_model()
except Exception as e:
    print(f"Error loading model/data: {e}")
    traceback.print_exc()

def predict_24_hours(start_date_str, start_hour, input_temp, is_holiday):
    """
    Iteratively predict 24 hours of demand using the LSTM model.
    Since we don't know the exact history of a *future* selected date,
    we use the last known 168 hours from the dataset, but we override 
    the "future" time features (hour, month, dow) and temperature 
    for the sequence we are predicting.
    """
    if MODEL is None or SCALER is None or LAST_KNOWN_DATA is None:
        raise ValueError("Model or data not loaded properly.")
        
    # We will predict 24 hours into the "future"
    # To do this, we need a sliding window of size 168.
    
    # 1. Grab the last 168 hours of known data and scale it
    current_seq_df = LAST_KNOWN_DATA.copy()
    
    # Build list of future timestamps (24 hours)
    base_time = datetime.strptime(start_date_str, "%Y-%m-%d")
    base_time = base_time.replace(hour=int(start_hour))
    future_times = [base_time + timedelta(hours=i) for i in range(24)]
    
    predictions_mw = []
    
    # We will maintain our 168-step sequence in scaled form
    current_seq_scaled = SCALER.transform(current_seq_df[LSTM_COLS].values.astype("float32"))
    
    for dt in future_times:
        # Predict the next hour
        # shape: (1, 168, 9)
        pred_scaled = MODEL.predict(current_seq_scaled[np.newaxis, :, :], verbose=0).flatten()[0]
        
        # Inverse transform to get MW
        dummy = np.zeros(len(LSTM_COLS))
        dummy[TARGET_IDX] = pred_scaled
        pred_mw = SCALER.inverse_transform([dummy])[0, TARGET_IDX]
        
        # Apply holiday heuristic (-12%) since the model doesn't explicitly have an is_holiday feature
        if is_holiday:
            pred_mw *= 0.88
            
        predictions_mw.append(pred_mw)
        
        # Now we need to update the sequence for the NEXT step.
        # Construct the features for this newly predicted timestep
        hour = dt.hour
        month = dt.month
        dow = dt.weekday()
        
        new_row = [
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * month / 12.0),
            np.cos(2 * np.pi * month / 12.0),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
            1 if dow >= 5 else 0,
            input_temp,  # using the requested temp
            pred_mw
        ]
        
        # Scale the new row
        new_row_scaled = SCALER.transform([new_row])[0]
        
        # Append and pop oldest
        current_seq_scaled = np.vstack([current_seq_scaled[1:], new_row_scaled])

    # Convert from MW to GW for the frontend
    return [p / 1000.0 for p in predictions_mw]


@app.route('/predict', methods=['POST'])
def predict():
    """
    Returns a 24-hour demand forecast using the LSTM model.
    """
    data = request.json
    region = data.get('region', 'North')
    date_str = data.get('date') # e.g. "2024-05-15"
    hour = int(data.get('hour', 12))
    temperature = float(data.get('temperature', 35.0))
    is_holiday = data.get('is_holiday', False)

    if not date_str:
        # fallback to today
        date_str = datetime.now().strftime("%Y-%m-%d")

    try:
        if MODEL is None:
            # Fallback to a hardcoded baseline if model fails to load
            baseline_gw = 150.0 * REGION_FRACTIONS.get(region, 0.2)
            hourly_forecast = [baseline_gw * (1 + 0.1 * math.sin(i)) for i in range(24)]
            comparison = {r: baseline_gw for r in REGION_FRACTIONS}
        else:
            # Get national prediction in GW
            national_forecast_gw = predict_24_hours(date_str, hour, temperature, is_holiday)
            
            fraction = REGION_FRACTIONS.get(region, 0.2)
            hourly_forecast = [val * fraction for val in national_forecast_gw]
            
            target_hour_pred_national = national_forecast_gw[0]
            comparison = {r: target_hour_pred_national * frac for r, frac in REGION_FRACTIONS.items()}
        
        # Determine status
        target_hour_pred = hourly_forecast[0]
        base_demand = 583.0 * REGION_FRACTIONS.get(region, 0.2)
        
        if target_hour_pred > base_demand * 1.15:
            status = "Alert: High Demand"
        elif target_hour_pred > base_demand * 1.05:
            status = "Warning: Elevated Load"
        else:
            status = "Normal Load"

        response_payload = {
            "prediction_gw": round(target_hour_pred, 2),
            "status": status,
            "hourly_forecast": [round(val, 2) for val in hourly_forecast],
            "regional_comparison": {r: round(val, 2) for r, val in comparison.items()}
        }
        return jsonify(response_payload)
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/get_temperature', methods=['GET'])
def get_temperature_route():
    region = request.args.get('region', 'North')
    date_str = request.args.get('date', '')
    hour = request.args.get('hour', '12')
    
    try:
        hour = int(hour)
    except ValueError:
        return jsonify({"error": "Hour must be an integer"}), 400

    from weather_api import get_temperature
    result = get_temperature(region, date_str, hour)
    
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


if __name__ == '__main__':
    print("Starting GridSense Flask Server...")
    app.run(port=5000, debug=False)
