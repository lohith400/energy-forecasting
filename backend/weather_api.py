"""
weather_api.py — Multi-Region Open-Meteo weather data fetcher for GridSense India.
Fetches hourly temperatures for all 5 representative grid region cities,
computes the load-weighted national hourly average, and returns the curves.
"""

import requests
import concurrent.futures

# Representative cities and their grid load weights
REGION_CITIES = {
    "North":     {"city": "New Delhi",  "lat": 28.6139, "lon": 77.2090, "weight": 0.300},
    "South":     {"city": "Bengaluru",  "lat": 12.9716, "lon": 77.5946, "weight": 0.232},
    "East":      {"city": "Kolkata",    "lat": 22.5726, "lon": 88.3639, "weight": 0.154},
    "West":      {"city": "Mumbai",     "lat": 19.0760, "lon": 72.8777, "weight": 0.266},
    "NorthEast": {"city": "Guwahati",   "lat": 26.1445, "lon": 91.7362, "weight": 0.048},
}


def fetch_city_hourly_temp(city_info: dict, date: str) -> list:
    """
    Fetch the 24-hour temperature forecast (12:00 AM to 11:00 PM) for a single city on a target date.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":      city_info["lat"],
        "longitude":     city_info["lon"],
        "hourly":        "temperature_2m",
        "timezone":      "Asia/Kolkata",
        "forecast_days": 7,
    }
    
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    hourly_times = data.get("hourly", {}).get("time", [])
    hourly_temps = data.get("hourly", {}).get("temperature_2m", [])
    
    # Filter for the target date prefix (e.g. "2026-06-13T")
    target_prefix = f"{date}T"
    day_temps = [
        temp for time_str, temp in zip(hourly_times, hourly_temps)
        if time_str.startswith(target_prefix)
    ]
    
    if len(day_temps) != 24:
        raise ValueError(
            f"Expected exactly 24 temperature values for {date}, but got {len(day_temps)}. "
            f"Date may be outside the 7-day forecast window."
        )
        
    return day_temps


def get_temperature(region: str, date: str, hour: int) -> dict:
    """
    Query the Open-Meteo API for 24-hour temperature forecasts for all 5 representative grid cities,
    calculate the load-weighted national average temperature curve, and return both the regional
    and national hourly temperature curves.
    """
    if region not in REGION_CITIES:
        return {"error": f"Invalid region '{region}'. Choose from: {list(REGION_CITIES.keys())}"}

    results = {}
    errors = []
    
    # Query all 5 cities in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_reg = {
            executor.submit(fetch_city_hourly_temp, info, date): reg
            for reg, info in REGION_CITIES.items()
        }
        for future in concurrent.futures.as_completed(future_to_reg):
            reg = future_to_reg[future]
            try:
                results[reg] = future.result()
            except Exception as e:
                errors.append(f"{reg}: {str(e)}")
                
    if errors:
        return {
            "error": (
                f"Failed to fetch forecast for one or more regions: {'; '.join(errors)}. "
                "The date may be outside the 7-day forecast window or the API is unavailable."
            )
        }
        
    # Calculate load-weighted national temperature curve for all 24 hours
    national_hourly_temperatures = [0.0] * 24
    for h_idx in range(24):
        weighted_sum = 0.0
        for reg, info in REGION_CITIES.items():
            weighted_sum += results[reg][h_idx] * info["weight"]
        national_hourly_temperatures[h_idx] = round(weighted_sum, 2)
        
    # Selected region temperatures
    selected_reg_temps = results[region]
    target_hour_temp = selected_reg_temps[hour]
    
    return {
        "temperature_celsius": round(target_hour_temp, 1),
        "regional_hourly_temperatures": [round(t, 2) for t in selected_reg_temps],
        "national_hourly_temperatures": national_hourly_temperatures,
        "city": REGION_CITIES[region]["city"],
        "region": region,
        "date": date,
        "hour": hour,
        "source": "Open-Meteo Multi-Region Forecast",
    }


if __name__ == "__main__":
    from datetime import datetime, timedelta

    today = datetime.now()
    d1 = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    
    print("\n--- Testing Multi-Region Weather API ---\n")
    print(f"Fetching weather forecast for: {d1}")
    result = get_temperature("South", d1, 12)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Selected Region : {result['region']} ({result['city']})")
        print(f"Temp at Hour 12 : {result['temperature_celsius']} C")
        print(f"Regional Curve  : {result['regional_hourly_temperatures']}")
        print(f"National Curve  : {result['national_hourly_temperatures']}")
        print(f"Source          : {result['source']}")