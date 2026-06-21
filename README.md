# GridSense India — Electricity Demand Forecasting Dashboard

An advanced deep learning (LSTM) and gradient boosting (LightGBM/XGBoost) powered web application for forecasting electricity demand across Indian power grid regions in real-time, integrated with a multi-region load-weighted hourly weather API.

---

## 📋 Project Overview

**GridSense India** (branded as **E-City**) predicts hourly electricity demand for the Indian national grid and its regional subdivisions (North, South, East, West, North-East). 

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
* ✅ **Holiday Override** — Adjust forecasts for national public holidays and weekends (applies a pre-set demand reduction factor of 8% to 12%).
* ✅ **Confidence Intervals** — Understand prediction uncertainty with dynamically generated low/high bounds (±7%).

---

## 📁 Project Structure

```
energy-forecasting/
├── backend/                      # Python ML models & API
│   ├── outputs/                 # Model summaries, evaluation plots & logs
│   ├── SmartGrid_Forecaster_LSTM_LightGBM.ipynb  # Model development notebook
│   ├── historical_hourly_temp.csv # 5-year load-weighted hourly temperature dataset
│   ├── hourlyLoadDataIndia.xlsx  # POSOCO / Grid India hourly demand dataset (Jan 2019 – Apr 2024)
│   ├── model_metadata.json      # Scaler parameters & fitted metrics
│   ├── monthly_temp.xlsx        # Dataset
│   ├── run_forecaster.py        # ML training pipeline (LSTM/LightGBM/XGBoost)
│   ├── server.py                # Flask API server & endpoints (/predict, /get_temperature)
│   ├── smartgrid_lstm_model.keras # Retrained deep learning model
│   └── weather_api.py           # Parallel Open-Meteo temperature fetcher
│
├── frontend/                     # Web UI (HTML/CSS/JavaScript)
│   ├── app.js                   # Application state, API requests, & Chart.js logic
│   ├── index.html               # Dashboard layout, dual-axis legends & structure
│   └── styles.css               # Responsive design system & themes
│
├── scratch/                      # Temporary/development scripts
│   ├── fetch_all_regions_historical_weather.py
│   └── fetch_historical_hourly_weather.py
│
├── README.md                     # This unified file
├── requirements.txt              # Unified Python dependency checklist
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

# Create the virtual environment (if not already present)
python -m venv venv

# Activate the virtual environment
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Windows (CMD)
.\venv\Scripts\activate.bat
# macOS/Linux
source venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
```

### 2. Run Flask Backend Server
```bash
# Set console encoding & launch server
# Windows (PowerShell)
$env:PYTHONIOENCODING="utf-8"
python backend/server.py

# Windows (CMD)
set PYTHONIOENCODING=utf-8
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

* **Response Example (JSON):**
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

* **Request Body (JSON):**
  ```json
  {
    "region": "South",
    "date": "2026-06-18",
    "hour": 12,
    "is_holiday": false,
    "national_hourly_temperatures": [26.21, 26.02, 25.7, ..., 27.68]
  }
  ```

* **Response Example (JSON):**
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

### `GET /regions/summary`
Returns the current predicted load for all 5 power grid regions.
* **Response Example (JSON):**
  ```json
  {
    "North": { "current_load": 42.1, "load_status": "Normal Load", "region": "North" },
    "South": { "current_load": 31.5, "load_status": "Normal Load", "region": "South" },
    "East": { "current_load": 20.2, "load_status": "Normal Load", "region": "East" },
    "West": { "current_load": 35.8, "load_status": "High Load", "region": "West" },
    "NorthEast": { "current_load": 6.1, "load_status": "Normal Load", "region": "NorthEast" }
  }
  ```

### `GET /model/metrics`
Returns model performance metrics from `model_metadata.json`.
* **Response Example (JSON):**
  ```json
  {
    "mae": 320.5,
    "rmse": 480.2,
    "mape": 2.46,
    "r2_score": 0.9754,
    "training_period": "2019–2024",
    "total_predictions": 43824
  }
  ```

### `GET /weather`
An alias that fetches the current temperature (Celsius) for the given region.
* **Parameters:**
  * `region`: `North` | `West` | `South` | `East` | `NorthEast`

---

## 🎨 Frontend Details & Customization

The interface is built with **HTML5**, **CSS3**, and **vanilla JavaScript** using **Chart.js** for visualization.

### Configuration
* **API Base URL**: The backend API URL is configured in `frontend/app.js`:
  ```javascript
  const API_BASE = 'http://localhost:5000';
  ```
  Update this variable if serving the Flask API on a different address or port.
* **Timezone**: The dashboard operates on **Indian Standard Time (IST = UTC+5:30)**.

### Customization
* **Change Theme Colors**: You can customize the dashboard palette by editing the CSS Variables in `frontend/styles.css`:
  ```css
  :root {
    --navy: #1A3C6E;
    --blue: #2E75B6;
    --amber: #B7770D;
    /* ... */
  }
  ```
* **Modify Chart Rendering**: Chart settings, colors, and datasets can be altered inside `frontend/app.js` functions `renderHourlyChart()` and `renderRegionalChart()`.

---

## 🤖 Machine Learning Model Details

* **Core Predictor:** Multi-layer LSTM neural network (trained on 46,728 rows of hourly grid load data).
* **Feature Vector:** Incorporates cyclical time encodings (hour, month, day-of-week), weekend indicators, lag features, rolling statistics, and the national load-weighted hourly temperature curve.
* **Top-down Regional Splitting:** For any date, the LSTM predicts the 24-hour national demand profile. The server then computes the actual regional fractions for that date (if historical) or uses long-run baseline fractions to split the national demand into regional forecasts.
* **Baseline Region Load Shares:**
  - North: 30.0%
  - West: 26.6%
  - South: 23.2%
  - East: 15.4%
  - NorthEast: 4.8%

### Training Pipeline (`run_forecaster.py`)
To train or retrain models:
```bash
python backend/run_forecaster.py
```
This loads data from Excel files, splits into training/testing sets chronologically, trains LSTM models, generates evaluations, and saves output plots and models to `backend/outputs/`.

---

## 🔧 Troubleshooting

1. **Dashboard displays but charts won't load**:
   * Open the browser console (F12) to check for JavaScript errors.
   * Verify the Flask backend is running on `http://localhost:5000`.
2. **CORS issues**:
   * Flask CORS is configured out of the box in `server.py` to allow cross-origin requests. Ensure it isn't blocked by network policies.
3. **Temperature auto-fetch not working**:
   * Weather API might be rate-limited. The system will fall back to seasonal monthly averages automatically, or you can enter values manually.
