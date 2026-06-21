# GridSense India — Electricity Demand Forecasting Dashboard

An advanced deep learning (LSTM) and gradient boosting (LightGBM/XGBoost) powered web application for forecasting electricity demand across Indian power grid regions in real-time, integrated with a multi-region load-weighted hourly weather API.

---

## 📋 Project Overview

**GridSense India** predicts hourly electricity demand for the Indian national grid and its regional subdivisions (North, South, East, West, North-East). 

Since the forecasting model predicts the **National Hourly Demand** (and scales it to regional grids using fractions), the temperature feature fed into the model represents the **weighted national average hourly temperature**. The system features:
* **Deep Learning & Gradient Boosting:** Trained on historical hourly grid load records merged with true hourly weighted temperatures.
* **Multi-Region Weather Integration:** Real-time 24-hour temperature forecasts for all 5 representative grid cities queried in parallel from the Open-Meteo API.
* **Interactive Dual Y-Axis UI:** Visualizes both regional and national weighted temperature curves overlaying demand curves in Chart.js.
* **Dual Operational Modes:** Auto-detects historical dates (showing actual vs. predicted demand alongside true weather records) vs. future dates (live forecast).

---

## ✨ Key Features

* ✅ **Regional Forecasts** — Predict demand for 5 Indian power grid regions separately (North, West, South, East, North-East).
* ✅ **Multi-Region Weather API** — Fetches 24-hour forecasts for New Delhi, Bengaluru, Kolkata, Mumbai, and Guwahati in parallel and calculates the national load-weighted hourly temperature.
* ✅ **Historical Analysis Mode** — Query dates from Jan 2019 to Apr 2024 to overlay actual grid demand against predictions and compare them directly with actual historical temperature records.
* ✅ **Dual Y-Axis Charting** — View both the Selected Region's temperature curve (solid pink) and the National Weighted average temperature curve (dashed teal) plotted on a secondary Y-axis overlaying the hourly demand curve.
* ✅ **Holiday Override** — Adjust forecasts for national public holidays and weekends (applies a pre-set demand reduction factor).
* ✅ **Confidence Intervals** — Understand prediction uncertainty with dynamically generated low/high bounds (±7%).

---

## 📁 Project Structure

```
energy-forecasting/
├── frontend/                      # Web UI (HTML/CSS/JavaScript)
│   ├── index.html               # Dashboard layout, dual-axis legends & structure
│   ├── app.js                   # Application state, API requests, & Chart.js logic
│   └── styles.css               # Responsive design system & themes
│
├── backend/                      # Python ML models & API
│   ├── server.py                # Flask API server & endpoints (/predict, /get_temperature)
│   ├── run_forecaster.py        # ML training pipeline (LSTM/LightGBM/XGBoost)
│   ├── weather_api.py           # Parallel Open-Meteo temperature fetcher
│   └── outputs/                 # Zipped model summaries, evaluation plots & logs
│
├── historical_hourly_temp.csv    # 5-year load-weighted hourly temperature dataset
├── hourlyLoadDataIndia.xlsx      # POSOCO / Grid India hourly demand dataset (Jan 2019 – Apr 2024)
├── smartgrid_lstm_model.keras    # Retrained deep learning model
├── requirements_no_directml.txt  # Python packages checklist (CPU execution)
├── README.md                     # This file
└── venv/                         # Python virtual environment
```

---

## 🚀 Getting Started

### Prerequisites
* **Python 3.11+**
* **Internet Connection** (for live Open-Meteo API requests)

### 1. Setup Environment
```bash
# Navigate to the project directory
cd energy-forecasting

# Activate the virtual environment
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Windows (CMD)
.\venv\Scripts\activate.bat
# macOS/Linux
source venv/Scripts/activate
```

### 2. Run Flask Backend Server
```bash
# Set console encoding & launch server
$env:PYTHONIOENCODING="utf-8"
python backend/server.py
```
The backend initializes the dataset, fits the scaler, loads the LSTM model, and starts listening on `http://127.0.0.1:5000`.

### 3. Open Frontend UI
You can open `frontend/index.html` directly in your browser, or serve it using a local HTTP server:
```bash
cd frontend
python -m http.server 8000
```
Then visit `http://localhost:8000` in your web browser.

---

## 🔧 Backend API Reference

### `GET /get_temperature`
Fetches 24-hour temperature profiles for the selected region and computes the load-weighted national average.

* **Parameters:**
  * `region`: `North` | `West` | `South` | `East` | `NorthEast`
  * `date`: `YYYY-MM-DD`
  * `hour`: `0-23` (target hour for display value)

* **Response Example:**
  ```json
  {
    "temperature_celsius": 29.0,
    "regional_hourly_temperatures": [21.3, 21.0, 20.8, ..., 21.7],
    "national_hourly_temperatures": [26.21, 26.02, 25.7, ..., 27.68],
    "city": "Bengaluru",
    "region": "South",
    "date": "2026-06-14",
    "hour": 12,
    "source": "Open-Meteo Multi-Region Forecast"
  }
  ```

### `POST /predict`
Predicts 24-hour electricity demand for the selected region.

* **Request Body Example:**
  ```json
  {
    "region": "South",
    "date": "2026-06-18",
    "hour": 12,
    "is_holiday": false,
    "national_hourly_temperatures": [26.21, 26.02, 25.7, ..., 27.68]
  }
  ```

* **Response Example:**
  ```json
  {
    "predicted_demand_gw": 31.5,
    "confidence_low": 29.3,
    "confidence_high": 33.7,
    "status": "Normal Load",
    "hourly_forecast": [28.2, 27.9, 27.4, ..., 29.5],
    "regional_comparison": {
      "North": 42.1,
      "West": 35.8,
      "South": 31.5,
      "East": 20.2,
      "NorthEast": 6.1
    },
    "is_historical": false
  }
  ```

---

## 🤖 Machine Learning Model Details

* **Core Predictor:** Multi-layer LSTM neural network (trained on 46,728 rows of hourly grid load data).
* **Feature Vector:** Incorporates cyclical time encodings (hour, month, day-of-week), weekend indicators, lag features, rolling statistics, and the national load-weighted hourly temperature curve.
* **Top-down Regional Splitting:** For any date, the LSTM predicts the 24-hour national demand profile. The server then computes the actual regional fractions for that date (if historical) or uses long-run baseline fractions to split the national demand into regional forecasts.
* **Weights:**
  - North: 30.0%
  - West: 26.6%
  - South: 23.2%
  - East: 15.4%
  - NorthEast: 4.8%

---

**Enjoy forecasting India's electricity demand with GridSense! ⚡**
