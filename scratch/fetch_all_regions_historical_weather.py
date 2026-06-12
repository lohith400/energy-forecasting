import requests
import json
import pandas as pd
import time

# Region information and weight based on load share
REGION_CITIES = {
    "North":     {"lat": 28.6139, "lon": 77.2090, "weight": 175.0 / 583.0},
    "South":     {"lat": 12.9716, "lon": 77.5946, "weight": 135.0 / 583.0},
    "East":      {"lat": 22.5726, "lon": 88.3639, "weight": 90.0 / 583.0},
    "West":      {"lat": 19.0760, "lon": 72.8777, "weight": 155.0 / 583.0},
    "NorthEast": {"lat": 26.1445, "lon": 91.7362, "weight": 28.0 / 583.0},
}

url = "https://archive-api.open-meteo.com/v1/archive"
start_date = "2019-01-01"
end_date = "2024-04-30"

region_data = {}

for region, info in REGION_CITIES.items():
    print(f"Fetching historical weather for {region} region...")
    params = {
        "latitude": info["lat"],
        "longitude": info["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "timezone": "Asia/Kolkata"
    }
    
    success = False
    for attempt in range(5):
        try:
            response = requests.get(url, params=params, timeout=45)
            response.raise_for_status()
            data = response.json()
            
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [])
            
            print(f"  Received {len(temps)} records.")
            region_data[region] = temps
            # Store datetime on the first successful fetch
            if "datetime" not in region_data:
                region_data["datetime"] = times
            
            success = True
            break
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed for {region}: {e}")
            time.sleep(2 * (attempt + 1))
            
    if not success:
        print(f"Error: Failed to fetch weather for {region} after 5 attempts.")
        exit(1)
        
    time.sleep(1) # Be nice to Open-Meteo

# Verify all datasets have the same length
lengths = [len(v) for k, v in region_data.items() if k != "datetime"]
if len(set(lengths)) > 1:
    print(f"Error: Mismatched lengths of region data: {lengths}")
    exit(1)

# Calculate weighted national average temperature
df = pd.DataFrame(region_data)
df["weighted_temp"] = 0.0

for region, info in REGION_CITIES.items():
    df["weighted_temp"] += df[region] * info["weight"]

print("\nWeighted Temperature Summary:")
print(df[["datetime", "weighted_temp"] + list(REGION_CITIES.keys())].head())

# Save to CSV
df_out = pd.DataFrame({
    "datetime": df["datetime"],
    "hourly_temperature": df["weighted_temp"]
})
df_out.to_csv("historical_hourly_temp.csv", index=False)
print("\nSaved weighted average to historical_hourly_temp.csv")
