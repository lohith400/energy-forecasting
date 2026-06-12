"""
GridSense India — Multi-Region Weather API (weather_api.py)

Fetches 24-hour hourly temperature forecasts for all 5 representative
grid region cities concurrently via Open-Meteo, computes the load-weighted
national temperature curve, and returns structured data for the backend.

Grid load weights:
    North      (New Delhi)  30.0%
    West       (Mumbai)     26.6%
    South      (Bengaluru)  23.2%
    East       (Kolkata)    15.4%
    NorthEast  (Guwahati)    4.8%

This file is unchanged from the audited version — it was already correct.
"""

import requests
import concurrent.futures

REGION_CITIES = {
    "North":     {"city": "New Delhi",  "lat": 28.6139, "lon": 77.2090, "weight": 0.300},
    "South":     {"city": "Bengaluru",  "lat": 12.9716, "lon": 77.5946, "weight": 0.232},
    "East":      {"city": "Kolkata",    "lat": 22.5726, "lon": 88.3639, "weight": 0.154},
    "West":      {"city": "Mumbai",     "lat": 19.0760, "lon": 72.8777, "weight": 0.266},
    "NorthEast": {"city": "Guwahati",   "lat": 26.1445, "lon": 91.7362, "weight": 0.048},
}


def fetch_city_hourly_temp(city_info: dict, date: str) -> list:
    """
    Fetch the 24-hour temperature forecast for a single city on the target date.
    Returns a list of exactly 24 floats (one per hour, 00:00 to 23:00 IST).
    Raises ValueError if the date is outside the 7-day forecast window.
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

    target_prefix = f"{date}T"
    day_temps = [
        temp
        for time_str, temp in zip(hourly_times, hourly_temps)
        if time_str.startswith(target_prefix)
    ]

    if len(day_temps) != 24:
        raise ValueError(
            f"Expected 24 temperature values for {date}, got {len(day_temps)}. "
            "Date may be outside the 7-day forecast window."
        )

    return day_temps


def get_temperature(region: str, date: str, hour: int) -> dict:
    """
    Query Open-Meteo for all 5 grid cities concurrently, compute the
    load-weighted national hourly temperature curve, and return the
    response contract expected by server.py /get_temperature.

    Returns:
        {
            "temperature_celsius":          float,   # selected region at target hour
            "regional_hourly_temperatures": list[float],  # selected region 24h curve
            "national_hourly_temperatures": list[float],  # weighted national 24h curve
            "city":   str,
            "region": str,
            "date":   str,
            "hour":   int,
            "source": str,
        }
    On error:
        { "error": str }
    """
    if region not in REGION_CITIES:
        return {"error": f"Invalid region '{region}'. Choose from: {list(REGION_CITIES.keys())}"}

    results = {}
    errors  = []

    # Fetch all 5 cities in parallel — reduces wall-clock time by ~4×
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
                errors.append(f"{reg}: {e}")

    if errors:
        return {
            "error": (
                f"Failed to fetch forecast for one or more regions: {'; '.join(errors)}. "
                "The date may be outside the 7-day Open-Meteo window or the API is unavailable."
            )
        }

    # Compute load-weighted national average for each of the 24 hours
    national_hourly_temperatures = []
    for h in range(24):
        weighted_sum = sum(
            results[reg][h] * info["weight"]
            for reg, info in REGION_CITIES.items()
        )
        national_hourly_temperatures.append(round(weighted_sum, 2))

    selected_temps = results[region]

    return {
        "temperature_celsius":          round(selected_temps[hour], 1),
        "regional_hourly_temperatures": [round(t, 2) for t in selected_temps],
        "national_hourly_temperatures": national_hourly_temperatures,
        "city":   REGION_CITIES[region]["city"],
        "region": region,
        "date":   date,
        "hour":   hour,
        "source": "Open-Meteo Multi-Region Forecast",
    }


if __name__ == "__main__":
    from datetime import datetime, timedelta
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"\n--- Testing Multi-Region Weather API for {tomorrow} ---\n")
    result = get_temperature("South", tomorrow, 12)
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Region  : {result['region']} ({result['city']})")
        print(f"Hour 12 : {result['temperature_celsius']} °C")
        print(f"Regional: {result['regional_hourly_temperatures']}")
        print(f"National: {result['national_hourly_temperatures']}")
        print(f"Source  : {result['source']}")