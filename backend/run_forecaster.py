# ============================================================
# GridSense India — Improved LSTM Forecaster
# run_forecaster.py
#
# KEY IMPROVEMENT over previous version:
#   Old LSTM_COLS (8 features + target):
#     hour_sin, hour_cos, month_sin, month_cos,
#     dow_sin, dow_cos, is_weekend, temperature_max,
#     National Hourly Demand
#
#   New LSTM_COLS (13 features + target):
#     hour_sin, hour_cos, month_sin, month_cos,
#     dow_sin, dow_cos, is_weekend, is_holiday,
#     temperature_max,
#     lag_1h, lag_24h, lag_168h,
#     roll_mean_24h, roll_std_24h,
#     National Hourly Demand  ← target (index 14)
#
# INFERENCE CONSISTENCY RULE:
#   Every feature added here MUST be generated the same way
#   in server.py during prediction. This file and server.py
#   are kept in sync via the LSTM_COLS list and TARGET_IDX.
# ============================================================

# ── CELL 1: Installs ──────────────────────────────────────────────────────────
print("Packages ready!")

# ── CELL 2: File paths ────────────────────────────────────────────────────────
import os

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
LOAD_FILE = os.path.join(BACKEND_DIR, "hourlyLoadDataIndia.xlsx")
TEMP_FILE = os.path.join(BACKEND_DIR, "historical_hourly_temp.csv")

for f in [LOAD_FILE, TEMP_FILE]:
    if os.path.exists(f):
        print(f"  Found: {f} ({os.path.getsize(f)/1e6:.2f} MB)")
    else:
        print(f"  MISSING: {f}")

# ── CELL 3: Imports and config ────────────────────────────────────────────────
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

print(f"TensorFlow : {tf.__version__}")
print(f"NumPy      : {np.__version__}")
print(f"Pandas     : {pd.__version__}")
print(f"GPU        : {len(tf.config.list_physical_devices('GPU')) > 0}")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dark plot theme
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a", "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#444466", "axes.labelcolor": "#c8c8ff",
    "axes.titlecolor": "#e0e0ff", "xtick.color": "#aaaacc",
    "ytick.color": "#aaaacc", "text.color": "#e0e0ff",
    "grid.color": "#2a2a4a", "grid.linestyle": "--",
    "grid.linewidth": 0.5, "font.size": 11,
    "legend.facecolor": "#1a1a2e", "legend.edgecolor": "#444466",
    "lines.linewidth": 1.5,
})
ACCENT_CYAN   = "#00d4aa"
ACCENT_PINK   = "#ff6b9d"
ACCENT_GOLD   = "#ffd166"
ACCENT_PURPLE = "#7c83fd"

def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=plt.rcParams["figure.facecolor"])
    plt.close()
    print(f"  [saved] outputs/{name}")

def mape(y_true, y_pred):
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

TARGET = "National Hourly Demand"
print("Config done.")

# ── CELL 4: Load and merge data ───────────────────────────────────────────────
print("\nLoading hourly load data...")
df_raw = pd.read_excel(LOAD_FILE)
df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
df_raw = df_raw.sort_values("datetime").reset_index(drop=True)
print(f"  Rows: {len(df_raw):,}  |  {df_raw['datetime'].min().date()} -> {df_raw['datetime'].max().date()}")

print("Loading temperature data...")
df_temp = pd.read_csv(TEMP_FILE)
df_temp["datetime"] = pd.to_datetime(df_temp["datetime"])
print(f"  Temp columns: {df_temp.columns.tolist()}")

# Merge on datetime
df = df_raw.merge(df_temp, on="datetime", how="left")
df["temperature_max"] = df["hourly_temperature"].ffill().bfill()
df["year"]  = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month

print(f"  After merge: {df.shape}")
print(f"  Temp NaNs: {df['temperature_max'].isnull().sum()}")

# Use last 24000 rows for faster training (covers ~2.7 years of hourly data)
df = df.tail(24000).reset_index(drop=True)
print(f"  Using last 24,000 rows for training.")

# ── CELL 5: Feature engineering ───────────────────────────────────────────────
print("\nEngineering features...")

df["hour"]       = df["datetime"].dt.hour
df["dayofweek"]  = df["datetime"].dt.dayofweek   # 0=Mon, 6=Sun
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

# is_holiday: 0 for all rows by default during training.
# At inference time, the user passes is_holiday=True/False which adjusts
# the prediction by 12% — we include it as a feature so the model learns
# that "holiday=1" correlates with lower demand.
# For training we set all rows to 0 (we don't have a holiday calendar in
# the dataset). If you have a holiday list, populate this column properly.
df["is_holiday"] = 0

# Cyclical time encodings (prevents discontinuity at midnight/month boundaries)
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]     / 24.0)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]     / 24.0)
df["month_sin"] = np.sin(2 * np.pi * df["month"]    / 12.0)
df["month_cos"] = np.cos(2 * np.pi * df["month"]    / 12.0)
df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
print("  [OK] Cyclical time features")

# ── LAG FEATURES ─────────────────────────────────────────────────────────────
# These are the most powerful predictors: demand right now is strongly
# correlated with demand 1h ago, 24h ago (same hour yesterday), and
# 168h ago (same hour last week).
df["lag_1h"]   = df[TARGET].shift(1)     # demand 1 hour ago
df["lag_24h"]  = df[TARGET].shift(24)    # demand same hour yesterday
df["lag_168h"] = df[TARGET].shift(168)   # demand same hour last week
print("  [OK] Lag features: lag_1h, lag_24h, lag_168h")

# ── ROLLING FEATURES ──────────────────────────────────────────────────────────
# Rolling mean captures the recent trend level.
# Rolling std captures volatility (e.g. ramp events, storms).
# shift(1) ensures no data leakage (we only use past values).
df["roll_mean_24h"] = df[TARGET].shift(1).rolling(24).mean()
df["roll_std_24h"]  = df[TARGET].shift(1).rolling(24).std()
print("  [OK] Rolling features: roll_mean_24h, roll_std_24h")

# Drop NaN rows created by lag/rolling
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"  [OK] Rows after dropping NaN: {len(df):,}")

# ── CELL 6: Define LSTM feature set ───────────────────────────────────────────
#
# CRITICAL: This list defines the model's input contract.
# server.py MUST generate features in EXACTLY this order at inference time.
# Any change here requires a matching change in server.py LSTM_COLS.
#
LSTM_COLS = [
    "hour_sin",       # 0  — cyclical hour
    "hour_cos",       # 1
    "month_sin",      # 2  — cyclical month
    "month_cos",      # 3
    "dow_sin",        # 4  — cyclical day of week
    "dow_cos",        # 5
    "is_weekend",     # 6  — binary weekend flag
    "is_holiday",     # 7  — binary holiday flag (NEW)
    "temperature_max",# 8  — national weighted temperature (NEW position)
    "lag_1h",         # 9  — demand 1h ago (NEW)
    "lag_24h",        # 10 — demand 24h ago (NEW)
    "lag_168h",       # 11 — demand 168h ago (NEW)
    "roll_mean_24h",  # 12 — 24h rolling mean demand (NEW)
    "roll_std_24h",   # 13 — 24h rolling std demand (NEW)
    TARGET,           # 14 — National Hourly Demand (TARGET)
]

TARGET_IDX   = 14    # index of target in LSTM_COLS
SEQUENCE_LEN = 168   # 1 week lookback (unchanged)

print(f"\nLSTM feature count : {len(LSTM_COLS) - 1} features + 1 target")
print(f"TARGET_IDX         : {TARGET_IDX}")
print(f"SEQUENCE_LEN       : {SEQUENCE_LEN}")
print(f"Features           : {LSTM_COLS}")

# ── CELL 7: Scale and build sequences ─────────────────────────────────────────
print("\nScaling data...")
lstm_data   = df[LSTM_COLS].values.astype("float32")
scaler_lstm = MinMaxScaler()
lstm_scaled = scaler_lstm.fit_transform(lstm_data)
print(f"  Data shape: {lstm_scaled.shape}")

# Save scaler parameters so server.py can reconstruct it
# (In production, save the scaler object with joblib)
scaler_data_min  = scaler_lstm.data_min_.tolist()
scaler_data_max  = scaler_lstm.data_max_.tolist()
scaler_scale     = scaler_lstm.scale_.tolist()
scaler_min       = scaler_lstm.min_.tolist()
print("  Scaler fitted and parameters recorded.")

# Chronological 80/20 split
split_idx  = int(len(df) * 0.80)

def make_sequences(data, seq_len, split):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i, :])
        y.append(data[i, TARGET_IDX])
    X, y = np.array(X), np.array(y)
    return X[:split], X[split:], y[:split], y[split:]

lstm_split = split_idx - SEQUENCE_LEN
X_train, X_test, y_train, y_test = make_sequences(lstm_scaled, SEQUENCE_LEN, lstm_split)

print(f"\n  Train : {X_train.shape}  |  Test: {X_test.shape}")
print(f"  Train dates: {df['datetime'].iloc[0].date()} -> {df['datetime'].iloc[split_idx-1].date()}")
print(f"  Test  dates: {df['datetime'].iloc[split_idx].date()} -> {df['datetime'].iloc[-1].date()}")

# ── CELL 8: Build improved LSTM model ─────────────────────────────────────────
#
# Architecture improvements:
#   - Bidirectional LSTM: learns patterns from both directions in the sequence
#   - Additional Dense layer for better non-linear mapping
#   - BatchNormalization removed (MinMaxScaler handles scaling)
#   - Dropout tuned conservatively (0.15) to avoid underfitting
#
n_features = X_train.shape[2]   # = len(LSTM_COLS) = 15
print(f"\nBuilding model with input shape: ({SEQUENCE_LEN}, {n_features})")

model = Sequential([
    Input(shape=(SEQUENCE_LEN, n_features)),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.15),
    LSTM(64, return_sequences=False),
    Dropout(0.15),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1),
], name="SmartGrid_LSTM_v2")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"],
)
model.summary()

# ── CELL 9: Train ─────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=5, min_lr=1e-6, verbose=1),
]

print("\nTraining LSTM...")
t0 = time.time()
history = model.fit(
    X_train, y_train,
    epochs           = 50,        # EarlyStopping will stop earlier if needed
    batch_size       = 256,
    validation_split = 0.1,
    callbacks        = callbacks,
    verbose          = 1,
)
train_time = time.time() - t0
print(f"\nTraining complete in {train_time:.1f}s ({train_time/60:.1f} min)")

# ── CELL 10: Evaluate ─────────────────────────────────────────────────────────
pred_scaled = model.predict(X_test, verbose=0).flatten()

dummy = np.zeros((len(pred_scaled), len(LSTM_COLS)))
dummy[:, TARGET_IDX] = pred_scaled
pred_mw = scaler_lstm.inverse_transform(dummy)[:, TARGET_IDX]

y_test_mw = df[TARGET].values[split_idx:][:len(pred_mw)]

mae_val  = mean_absolute_error(y_test_mw, pred_mw)
rmse_val = float(np.sqrt(mean_squared_error(y_test_mw, pred_mw)))
r2_val   = r2_score(y_test_mw, pred_mw)
mape_val = mape(y_test_mw, pred_mw)

print(f"\n{'='*50}")
print(f"  IMPROVED LSTM Results")
print(f"{'='*50}")
print(f"  MAE   = {mae_val:,.1f} MW")
print(f"  RMSE  = {rmse_val:,.1f} MW")
print(f"  R2    = {r2_val:.4f}")
print(f"  MAPE  = {mape_val:.3f} %")
print(f"  Time  = {train_time:.1f}s")
print(f"{'='*50}")

# ── CELL 11: Plots ────────────────────────────────────────────────────────────
# Training loss
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history.history["loss"],     color=ACCENT_CYAN, label="Train Loss")
ax.plot(history.history["val_loss"], color=ACCENT_PINK, linestyle="--", label="Val Loss")
ax.set_title("Improved LSTM — Training History")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.legend(); ax.grid(True); plt.tight_layout()
save_fig("improved_01_training_loss.png")

# Actual vs Predicted (last 30 days)
test_dates = df["datetime"].values[split_idx:][:len(pred_mw)]
last_n     = min(30 * 24, len(pred_mw))
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(test_dates[-last_n:], y_test_mw[-last_n:], color=ACCENT_CYAN,
        linewidth=1.2, label="Actual", alpha=0.9)
ax.plot(test_dates[-last_n:], pred_mw[-last_n:],   color=ACCENT_GOLD,
        linewidth=1.2, linestyle="--", label="Predicted", alpha=0.85)
ax.fill_between(test_dates[-last_n:], y_test_mw[-last_n:], pred_mw[-last_n:],
                alpha=0.15, color=ACCENT_PINK, label="Error band")
ax.set_title(f"Improved LSTM — Last 30 Days (MAPE={mape_val:.2f}%, R2={r2_val:.4f})")
ax.set_xlabel("Date"); ax.set_ylabel("Load (MW)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(); ax.grid(True); plt.tight_layout()
save_fig("improved_02_actual_vs_predicted.png")

# ── CELL 12: Save model and metadata ─────────────────────────────────────────
MODEL_SAVE_PATH = os.path.join(BACKEND_DIR, "smartgrid_lstm_model.keras")
model.save(MODEL_SAVE_PATH)
print(f"\nModel saved: {MODEL_SAVE_PATH}")

# Save feature metadata — server.py reads this to know LSTM_COLS and TARGET_IDX
import json
metadata = {
    "lstm_cols":    LSTM_COLS,
    "target_idx":   TARGET_IDX,
    "sequence_len": SEQUENCE_LEN,
    "n_features":   n_features,
    "scaler": {
        "data_min":  scaler_data_min,
        "data_max":  scaler_data_max,
        "scale":     scaler_scale,
        "min":       scaler_min,
    },
    "metrics": {
        "mae_mw":  round(mae_val, 1),
        "rmse_mw": round(rmse_val, 1),
        "r2":      round(r2_val, 4),
        "mape_pct": round(mape_val, 3),
    }
}
with open(os.path.join(BACKEND_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print("Metadata saved: model_metadata.json")
print("\nDone. Saved both files directly in the backend directory.")