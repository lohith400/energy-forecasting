"""
weather_api.py — Open-Meteo weather data fetcher for GridSense India.
Returns temperature for a given region / date / hour.
For dates outside the 7-day forecast window, returns an error so the
server can substitute the historical monthly average.
"""

import requests

REGION_CITIES = {
    "North":     {"city": "New Delhi",  "lat": 28.6139, "lon": 77.2090},
    "South":     {"city": "Bengaluru",  "lat": 12.9716, "lon": 77.5946},
    "East":      {"city": "Kolkata",    "lat": 22.5726, "lon": 88.3639},
    "West":      {"city": "Mumbai",     "lat": 19.0760, "lon": 72.8777},
    "NorthEast": {"city": "Guwahati",   "lat": 26.1445, "lon": 91.7362},
}


def get_temperature(region: str, date: str, hour: int) -> dict:
    """
    Fetch temperature from Open-Meteo for the given region / date / hour.

    Returns a dict with 'temperature_celsius' on success, or 'error' on failure.
    Callers should handle the 'error' key and apply their own fallback.
    """
    if region not in REGION_CITIES:
        return {"error": f"Invalid region '{region}'. Choose from: {list(REGION_CITIES.keys())}"}

    info = REGION_CITIES[region]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":      info["lat"],
        "longitude":     info["lon"],
        "hourly":        "temperature_2m",
        "timezone":      "Asia/Kolkata",
        "forecast_days": 7,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        hourly_times = data["hourly"]["time"]
        hourly_temps = data["hourly"]["temperature_2m"]

        target = f"{date}T{str(hour).zfill(2)}:00"
        if target not in hourly_times:
            return {
                "error": (
                    f"Date {date} is outside the 7-day forecast window. "
                    "A historical monthly average temperature will be used instead."
                )
            }

        index       = hourly_times.index(target)
        temperature = hourly_temps[index]

        return {
            "temperature_celsius": round(temperature, 1),
            "city":   info["city"],
            "region": region,
            "date":   date,
            "hour":   hour,
            "source": "Open-Meteo (open-meteo.com)",
        }

    except requests.exceptions.ConnectionError:
        return {"error": "No internet connection. A monthly average will be used."}
    except requests.exceptions.Timeout:
        return {"error": "Weather API timed out. A monthly average will be used."}
    except requests.exceptions.HTTPError as e:
        return {"error": f"Weather API HTTP error: {str(e)}"}
    except (KeyError, IndexError, ValueError):
        return {"error": "Could not parse weather response. A monthly average will be used."}


if __name__ == "__main__":
    from datetime import datetime, timedelta

    today = datetime.now()
    d1 = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    d2 = (today + timedelta(days=2)).strftime("%Y-%m-%d")
    d_hist = "2024-04-20"   # historical — will return error for fallback

    test_cases = [
        ("North",     d1,     14),
        ("South",     d2,     20),
        ("West",      d2,     18),
        ("NorthEast", d1,      8),
        ("North",     d_hist,  9),   # historical date — expect graceful error
    ]

    print("\n── Weather API Test Results ──\n")
    for region, date, hour in test_cases:
        result = get_temperature(region, date, hour)
        if "error" in result:
            print(f"  {region:10s} | {date} {hour:02d}:00 | FALLBACK NEEDED: {result['error']}")
        else:
            print(f"  {result['region']:10s} | {result['city']:12s} | "
                  f"{date} {hour:02d}:00 IST | "
                  f"{result['temperature_celsius']}°C | "
                  f"Source: {result['source']}")