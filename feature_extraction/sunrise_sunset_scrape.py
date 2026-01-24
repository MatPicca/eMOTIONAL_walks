import requests
import pandas as pd
from datetime import date, timedelta
from zoneinfo import ZoneInfo 

'''
Scripti to scrape sunrise and sunset times for Copenhagen from the Sunrise-Sunset API.
https://sunrise-sunset.org/api
'''
lat, lon = 55.6761, 12.5683  # Copenhagen
start = date(2023, 11, 1)
end = date(2024, 7, 31)
tz_cph = ZoneInfo("Europe/Copenhagen")

rows = []
print("Collecting sunrise/sunset data")

for d in (start + timedelta(n) for n in range((end - start).days + 1)):
    r = requests.get(
        "https://api.sunrise-sunset.org/json",
        params={"lat": lat, "lng": lon, "date": d.isoformat(), "formatted": 0},
    )
    data = r.json()["results"]
    print(f"Date: {d}, Sunrise: {data['sunrise']}, Sunset: {data['sunset']}")

    # Convert to datetime and localize
    sunrise_utc = pd.to_datetime(data["sunrise"], utc=True)
    sunset_utc = pd.to_datetime(data["sunset"], utc=True)
    sunrise_local = sunrise_utc.tz_convert(tz_cph)
    sunset_local = sunset_utc.tz_convert(tz_cph)

    rows.append(
        {
            "date": d,
            "sunrise_local": sunrise_local.isoformat(),
            "sunset_local": sunset_local.isoformat(),
        }
    )

# === SAVE ===
df = pd.DataFrame(rows)
df.to_csv("/home/s232713/data/sunset_sunrise/copenhagen_sunrise_sunset_2023_2024.csv", index=False)
print(f"Saved {len(df)} rows to copenhagen_sunrise_sunset_2023_2024.csv")

