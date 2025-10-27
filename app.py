from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)

# --- Загрузка данных (пример с weather.csv) ---
import time, requests, pandas as pd

API_KEY = "cd59f6e78b8ec3e0d4b69f9e55d80e1c"

# Казахстан 🇰🇿
CITY_QUERIES = {
    "Astana":        ["Astana,KZ", "Nur-Sultan,KZ", "Akmola,KZ"],
    "Almaty":        ["Almaty,KZ", "Alma-Ata,KZ"],
    "Shymkent":      ["Shymkent,KZ", "Chimkent,KZ"],

    "Akmola (Kokshetau)":   ["Kokshetau,KZ", "Kokchetav,KZ"],
    "Atyrau":               ["Atyrau,KZ"],
    "Aktobe":               ["Aktobe,KZ", "Aqtobe,KZ"],
    "Abai (Semey)":         ["Semey,KZ", "Semei,KZ", "Semipalatinsk,KZ"],
    "Ulytau (Zhezqazghan)": ["Zhezqazghan,KZ", "Zhezkazgan,KZ", "Dzhezkazgan,KZ", "Jezkazgan,KZ"],
    "Jetisu (Taldykorgan)": ["Taldykorgan,KZ", "Taldyqorghan,KZ"],
    "Oskemen":              ["Oskemen,KZ", "Ust-Kamenogorsk,KZ"],
    "Karaganda":            ["Karaganda,KZ"],
    "Kostanay":             ["Kostanay,KZ", "Kustanay,KZ"],
    "Kyzylorda":            ["Kyzylorda,KZ"],
    "Mangystau (Aktau)":    ["Aktau,KZ"],
    "Petropavl":            ["Petropavl,KZ", "Petropavlovsk,KZ"],
    "Pavlodar":             ["Pavlodar,KZ"],
    "Turkistan":            ["Turkistan,KZ", "Turkestan,KZ"],
    "Oral":                 ["Oral,KZ", "Uralsk,KZ"],
}

# Европы 🇪🇺
EUROPE_CAPITALS = {
    "Vienna (Austria)": ["Vienna,AT"],
    "Brussels (Belgium)": ["Brussels,BE"],
    "Sofia (Bulgaria)": ["Sofia,BG"],
    "Zagreb (Croatia)": ["Zagreb,HR"],
    "Prague (Czech Republic)": ["Prague,CZ"],
    "Copenhagen (Denmark)": ["Copenhagen,DK"],
    "Tallinn (Estonia)": ["Tallinn,EE"],
    "Helsinki (Finland)": ["Helsinki,FI"],
    "Paris (France)": ["Paris,FR"],
    "Berlin (Germany)": ["Berlin,DE"],
    "Athens (Greece)": ["Athens,GR"],
    "Budapest (Hungary)": ["Budapest,HU"],
    "Reykjavik (Iceland)": ["Reykjavik,IS"],
    "Dublin (Ireland)": ["Dublin,IE"],
    "Rome (Italy)": ["Rome,IT"],
    "Riga (Latvia)": ["Riga,LV"],
    "Vilnius (Lithuania)": ["Vilnius,LT"],
    "Luxembourg (Luxembourg)": ["Luxembourg,LU"],
    "Valletta (Malta)": ["Valletta,MT"],
    "Chisinau (Moldova)": ["Chisinau,MD"],
    "Monaco (Monaco)": ["Monaco,MC"],
    "Podgorica (Montenegro)": ["Podgorica,ME"],
    "Amsterdam (Netherlands)": ["Amsterdam,NL"],
    "Skopje (North Macedonia)": ["Skopje,MK"],
    "Oslo (Norway)": ["Oslo,NO"],
    "Warsaw (Poland)": ["Warsaw,PL"],
    "Lisbon (Portugal)": ["Lisbon,PT"],
    "Bucharest (Romania)": ["Bucharest,RO"],
    "Belgrade (Serbia)": ["Belgrade,RS"],
    "Bratislava (Slovakia)": ["Bratislava,SK"],
    "Ljubljana (Slovenia)": ["Ljubljana,SI"],
    "Madrid (Spain)": ["Madrid,ES"],
    "Stockholm (Sweden)": ["Stockholm,SE"],
    "Bern (Switzerland)": ["Bern,CH"],
    "Kyiv (Ukraine)": ["Kyiv,UA"],
    "London (United Kingdom)": ["London,GB"],
}

# Объединяем словари
CITY_QUERIES.update(EUROPE_CAPITALS)

def geocode_best(q_list):
    for q in q_list:
        r = requests.get("https://api.openweathermap.org/geo/1.0/direct",
                         params={"q": q, "limit": 1, "appid": API_KEY}, timeout=20)
        if r.status_code != 200:
            time.sleep(1)
            continue
        data = r.json()
        if data:
            item = data[0]
            return item["lat"], item["lon"], item["name"], item.get("country")
        time.sleep(0.5)
    return None

def fetch_forecast(lat, lon):
    r = requests.get("https://api.openweathermap.org/data/2.5/forecast",
                     params={"lat": lat, "lon": lon, "appid": API_KEY,
                             "units": "metric", "lang": "en"}, timeout=30)
    if r.status_code != 200:
        return None
    return r.json()

rows, per_city_counts = [], {}

for label, qlist in CITY_QUERIES.items():
    geo = geocode_best(qlist)
    if not geo:
        print(f"  {label}: could not find coordinates")
        continue
    lat, lon, resolved_name, country = geo
    print(f" {label} → {resolved_name}, {country} ({lat:.4f}, {lon:.4f})")

    data = fetch_forecast(lat, lon)
    if not data or "list" not in data:
        print(f"  {label}: no forecast data available")
        continue

    added = 0
    for it in data["list"]:
        rows.append({
            "city": label,
            "datetime": it.get("dt_txt"),
            "temp": it["main"]["temp"],
            "feels_like": it["main"]["feels_like"],
            "pressure": it["main"]["pressure"],
            "humidity": it["main"]["humidity"],
            "clouds": it["clouds"]["all"],
            "wind_speed": it["wind"]["speed"],
            "wind_deg": it["wind"].get("deg"),
            "pop": it.get("pop", 0),
            "rain_3h": it.get("rain", {}).get("3h", 0),
            "snow_3h": it.get("snow", {}).get("3h", 0),
            "weather_main": it["weather"][0]["main"],
        })
        added += 1

    per_city_counts[label] = added
    print(f" {label}: added {added} rows")
    time.sleep(1)

df = pd.DataFrame(rows)
df["RainOrNot"] = ((df["rain_3h"] > 0) | (df["snow_3h"] > 0)).astype(int)
df.to_csv("weather_data.csv", index=False, encoding="utf-8")

df = pd.read_csv("weather_data.csv")

df["hours_ahead"] = (pd.to_datetime(df["datetime"]) - pd.Timestamp.now()).dt.total_seconds() / 3600
df["hours_ahead"] = df["hours_ahead"].clip(lower=0)

X = df[["temp", "humidity", "pressure", "wind_speed", "hours_ahead"]]
y = df["RainOrNot"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train_predict", methods=["POST"])
def train_predict():
    try:
        data = request.get_json()
        model_type = data.get("model")

        # Проверка входных данных
        for key in ["temperature", "humidity", "pressure", "wind_speed", "hours_ahead"]:
            if key not in data:
                return jsonify({"error": f"Missing parameter: {key}"}), 400

        user_input = np.array([
            data["temperature"],
            data["humidity"],
            data["pressure"],
            data["wind_speed"],
            data["hours_ahead"]
        ]).reshape(1, -1)

        # Выбор модели
        if model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=150, random_state=42)
        elif model_type == "Gradient Boosting":
            model = GradientBoostingClassifier(random_state=42)
        elif model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            return jsonify({"error": "Invalid model type"}), 400

        # Разделение и обучение
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)

        # Оценка точности
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Предсказание вероятности дождя
        user_input_scaled = scaler.transform(user_input)
        rain_prob = float(model.predict_proba(user_input_scaled)[0][1])

        return jsonify({
            "accuracy": round(acc * 100, 2),
            "rain_probability": round(rain_prob * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)