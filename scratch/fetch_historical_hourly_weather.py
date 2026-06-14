import requests
import json
import pandas as pd

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 28.6139,
    "longitude": 77.2090,
    "start_date": "2019-01-01",
    "end_date": "2024-04-30",
    "hourly": "temperature_2m",
    "timezone": "Asia/Kolkata"
}

print("Fetching historical weather from Open-Meteo...")
try:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    
    print(f"Successfully fetched {len(temps)} hourly temperature records.")
    print(f"Start: {times[0]} -> {temps[0]}°C")
    print(f"End  : {times[-1]} -> {temps[-1]}°C")
    
    # Save to CSV
    df = pd.DataFrame({"datetime": times, "hourly_temperature": temps})
    df.to_csv("historical_hourly_temp.csv", index=False)
    print("Saved to historical_hourly_temp.csv")
    
except Exception as e:
    print(f"Error: {e}")
