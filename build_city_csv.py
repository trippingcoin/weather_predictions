from datetime import datetime, timedelta
from meteostat import Point, Hourly
import pandas as pd, os, ssl, time


ssl._create_default_https_context = ssl._create_unverified_context

CITY_COORDS = {
    "Astana": (51.13, 71.43),
    "Almaty": (43.24, 76.95),
    "Shymkent": (42.31, 69.59),
    "Kokshetau": (53.29, 69.38),
    "Atyrau": (47.11, 51.91),
    "Aktobe": (50.28, 57.23),
    "Semey": (50.40, 80.25),
    "Zhezqazghan": (47.80, 67.71),
    "Taldykorgan": (45.02, 78.38),
    "Oskemen": (49.95, 82.63),
    "Karaganda": (49.82, 73.10),
    "Kostanay": (53.21, 63.63),
    "Kyzylorda": (44.84, 65.50),
    "Aktau": (43.64, 51.17),
    "Petropavl": (54.87, 69.13),
    "Pavlodar": (52.29, 76.95),
    "Turkistan": (43.30, 68.27),
    "Oral": (51.20, 51.37),
    # --- Европа ---
    "Vienna (Austria)": (48.21, 16.37),
    "Brussels (Belgium)": (50.85, 4.35),
    "Sofia (Bulgaria)": (42.70, 23.32),
    "Zagreb (Croatia)": (45.84, 15.96),
    "Prague (Czech Republic)": (50.09, 14.42),
    "Copenhagen (Denmark)": (55.69, 12.57),
    "Tallinn (Estonia)": (59.44, 24.75),
    "Helsinki (Finland)": (60.17, 24.94),
    "Paris (France)": (48.86, 2.32),
    "Berlin (Germany)": (52.52, 13.39),
    "Athens (Greece)": (37.98, 23.73),
    "Budapest (Hungary)": (47.50, 19.04),
    "Reykjavik (Iceland)": (64.15, -21.94),
    "Dublin (Ireland)": (53.35, -6.26),
    "Rome (Italy)": (41.89, 12.48),
    "Riga (Latvia)": (56.95, 24.11),
    "Vilnius (Lithuania)": (54.69, 25.28),
    "Luxembourg (Luxembourg)": (49.61, 6.13),
    "Valletta (Malta)": (35.90, 14.51),
    "Chisinau (Moldova)": (47.02, 28.83),
    "Monaco (Monaco)": (43.73, 7.42),
    "Podgorica (Montenegro)": (42.44, 19.26),
    "Amsterdam (Netherlands)": (52.37, 4.89),
    "Skopje (North Macedonia)": (42.00, 21.43),
    "Oslo (Norway)": (59.91, 10.74),
    "Warsaw (Poland)": (52.23, 21.01),
    "Lisbon (Portugal)": (38.71, -9.14),
    "Bucharest (Romania)": (44.44, 26.10),
    "Belgrade (Serbia)": (44.82, 20.46),
    "Bratislava (Slovakia)": (48.14, 17.11),
    "Ljubljana (Slovenia)": (46.05, 14.51),
    "Madrid (Spain)": (40.42, -3.70),
    "Stockholm (Sweden)": (59.33, 18.07),
    "Bern (Switzerland)": (46.95, 7.45),
    "Kyiv (Ukraine)": (50.45, 30.52),
    "London (United Kingdom)": (51.51, -0.13),
}

end = datetime.utcnow()
start = end - timedelta(days=30)
os.makedirs("data", exist_ok=True)

all_records = []

print(f"Fetching hourly data from {start.date()} to {end.date()}...\n")

for city, (lat, lon) in CITY_COORDS.items():
    print(f"{city} ({lat}, {lon}) ...")
    try:
        location = Point(lat, lon)
        data = Hourly(location, start, end)
        df = data.fetch()
        if df.empty or len(df) < 24:
            print(f"Skipping {city}: insufficient data ({len(df)} records)")
            continue

        if df.empty:
            print(f"No data for {city}")
            continue

        df = df.reset_index()
        df["city"] = city
        all_records.append(df)
        print(f"{len(df)} hourly records collected")
    except Exception as e:
        print(f"Failed for {city}: {e}")
    time.sleep(0.5)

if all_records:
    final_df = pd.concat(all_records, ignore_index=True)
    final_df.to_csv("data/weather_data.csv", index=False)
    print(f"\nSaved {len(final_df)} rows → weather_data.csv")
else:
    print("No data collected.")