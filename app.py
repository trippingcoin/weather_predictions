from flask import Flask, render_template, request, jsonify
import os, time, requests
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

app = Flask(__name__)

try:
    from xgboost import XGBClassifier

    _xgb_available = True
except Exception:
    _xgb_available = False


# --- Загрузка данных (пример с weather.csv) ---
API_KEY = "cd59f6e78b8ec3e0d4b69f9e55d80e1c"
DATA_FILE = "weather_data.csv"

# Казахстан 🇰🇿
CITY_QUERIES = {
    "Astana": ["Astana,KZ", "Nur-Sultan,KZ", "Akmola,KZ"],
    "Almaty": ["Almaty,KZ", "Alma-Ata,KZ"],
    "Shymkent": ["Shymkent,KZ", "Chimkent,KZ"],
    "Akmola (Kokshetau)": ["Kokshetau,KZ", "Kokchetav,KZ"],
    "Atyrau": ["Atyrau,KZ"],
    "Aktobe": ["Aktobe,KZ", "Aqtobe,KZ"],
    "Abai (Semey)": ["Semey,KZ", "Semei,KZ", "Semipalatinsk,KZ"],
    "Ulytau (Zhezqazghan)": [
        "Zhezqazghan,KZ",
        "Zhezkazgan,KZ",
        "Dzhezkazgan,KZ",
        "Jezkazgan,KZ",
    ],
    "Jetisu (Taldykorgan)": ["Taldykorgan,KZ", "Taldyqorghan,KZ"],
    "Oskemen": ["Oskemen,KZ", "Ust-Kamenogorsk,KZ"],
    "Karaganda": ["Karaganda,KZ"],
    "Kostanay": ["Kostanay,KZ", "Kustanay,KZ"],
    "Kyzylorda": ["Kyzylorda,KZ"],
    "Mangystau (Aktau)": ["Aktau,KZ"],
    "Petropavl": ["Petropavl,KZ", "Petropavlovsk,KZ"],
    "Pavlodar": ["Pavlodar,KZ"],
    "Turkistan": ["Turkistan,KZ", "Turkestan,KZ"],
    "Oral": ["Oral,KZ", "Uralsk,KZ"],
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

MODEL_FEATURES = [
    "temp",
    "feels_like",
    "pressure",
    "humidity",
    "clouds",
    "wind_speed",
    "hour_sin",
    "hour_cos",
    "mon_sin",
    "mon_cos",
    "is_weekend",
    "season",
    "city",
]

# Объединяем словари
CITY_QUERIES.update(EUROPE_CAPITALS)

models_cache = {}
pipelines_cache = {}


# Utilities: fetch / build dataset
# ---------------------------
def geocode_best(q_list):
    for q in q_list:
        try:
            r = requests.get(
                "https://api.openweathermap.org/geo/1.0/direct",
                params={"q": q, "limit": 1, "appid": API_KEY},
                timeout=10,
            )
            if r.status_code != 200:
                time.sleep(0.2)
                continue
            data = r.json()
            if data:
                item = data[0]
                return item["lat"], item["lon"], item.get("name", q)
        except Exception:
            time.sleep(0.2)
            continue
    return None


def fetch_forecast(lat, lon):
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"},
            timeout=20,
        )
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def build_weather_dataset():
    rows = []
    for label, qlist in CITY_QUERIES.items():
        geo = geocode_best(qlist)
        if not geo:
            print(f"geocode fail: {label}")
            continue
        lat, lon, resolved = geo
        data = fetch_forecast(lat, lon)
        if not data or "list" not in data:
            print(f"no forecast for {label}")
            continue
        for it in data["list"]:
            rows.append(
                {
                    "city": label,
                    "datetime": it.get("dt_txt"),
                    "temp": it["main"]["temp"],
                    "feels_like": it["main"]["feels_like"],
                    "pressure": it["main"]["pressure"],
                    "humidity": it["main"]["humidity"],
                    "clouds": it["clouds"]["all"],
                    "wind_speed": it["wind"]["speed"],
                    "wind_deg": it["wind"].get("deg", np.nan),
                    "pop": it.get("pop", 0.0),
                    "rain_3h": it.get("rain", {}).get("3h", 0.0),
                    "snow_3h": it.get("snow", {}).get("3h", 0.0),
                    "weather_main": (
                        it["weather"][0]["main"] if it.get("weather") else None
                    ),
                }
            )
        time.sleep(0.5)
    if not rows:
        raise RuntimeError("No data rows fetched from API. Check API key or network.")
    df = pd.DataFrame(rows)
    df["RainOrNot"] = (
        (df["rain_3h"].fillna(0) > 0) | (df["snow_3h"].fillna(0) > 0)
    ).astype(int)
    df.to_csv(DATA_FILE, index=False, encoding="utf-8")
    print(f"Built dataset saved to {DATA_FILE}, rows={len(df)}")
    return df


def prepare_df_for_model(df_raw):
    df = df_raw.copy()
    # ensure datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        df["datetime"] = pd.NaT

    # Basic imputation for numeric fields commonly used
    num_cols = [
        "temp",
        "feels_like",
        "pressure",
        "humidity",
        "clouds",
        "wind_speed",
        "pop",
        "wind_deg",
        "rain_3h",
        "snow_3h",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())

    # Fill categorical
    for c in ["city", "weather_main"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str)

    # Time features
    df["hour"] = df["datetime"].dt.hour.fillna(0).astype(int)
    df["month"] = df["datetime"].dt.month.fillna(1).astype(int)
    df["dow"] = df["datetime"].dt.dayofweek.fillna(0).astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["mon_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["mon_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Season (coarse)
    def season_from_month(m):
        if m in [12, 1, 2]:
            return "Winter"
        if m in [3, 4, 5]:
            return "Spring"
        if m in [6, 7, 8]:
            return "Summer"
        return "Autumn"

    df["season"] = df["month"].apply(season_from_month)

    # Simple composites
    df["wind_power"] = df["wind_speed"] ** 2
    df["humid_cloud"] = df["humidity"] * df["clouds"] / 100.0

    # Select features for modeling (we keep season and city as categorical)
    # Note: we intentionally do NOT use lag/rolling features here to allow single-row predict
    features = []
    for f in MODEL_FEATURES:
        if f in ["season", "city"]:
            features.append(f)
        elif f in df.columns:
            features.append(f)
        else:
            # if some expected numeric column is missing, create it with median/0
            df[f] = 0.0
            features.append(f)

    X = df[features].copy()
    y = df["RainOrNot"].copy() if "RainOrNot" in df.columns else None

    return X, y, df


def build_preprocessor(X_train):
    # Find numeric vs categorical in X_train
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    num_pipe = Pipeline(
        [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    )
    try:
        cat_pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
    except TypeError:
        # fallback для старых sklearn
        cat_pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ]
        )
    preproc = ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)]
    )
    return preproc, num_cols, cat_cols


# ---------------------------
# Model creation helper
# ---------------------------
def get_model_by_name(name):
    name = name.strip().lower()

    if name == "decision tree":
        return DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=42
        )

    if name in ("lasso", "sgd", "lasso (sgdclassifier)"):
        return SGDClassifier(
            loss="log_loss",
            penalty="l1",
            alpha=0.001,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )

    if name in ("svm", "svm (rbf)"):
        return SVC(
            kernel="rbf",
            C=1.5,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )

    if name in ("random forest", "random_forest"):
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=20,
            min_samples_split=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    if name in ("xgboost", "xgb", "xgboost classifier") and _xgb_available:
        return XGBClassifier(
            n_estimators=400,
            learning_rate=0.5,
            max_depth=3,
            subsample=0.4,
            colsample_bytree=0.5,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )

    if name in ("gradient boosting", "gradient_boosting"):
        return GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42
        )

    # fallback
    return LogisticRegression(max_iter=1000)


# ---------------------------
# Training wrapper (caches models and preprocessors)
# ---------------------------
def train_and_cache(model_name, X_all, y_all):
    key = model_name.lower()
    if key in models_cache and key in pipelines_cache:
        return models_cache[key], pipelines_cache[key]

    # build preprocessor from X_all
    preproc, num_cols, cat_cols = build_preprocessor(X_all)
    # build pipeline
    model = get_model_by_name(model_name)
    pipe = Pipeline([("pre", preproc), ("clf", model)])

    # train/test split (grouping avoided here; consistent random split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    pipe.fit(X_train, y_train)

    # evaluate on test
    y_pred = pipe.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    models_cache[key] = {"pipeline": pipe, "metrics": metrics}
    pipelines_cache[key] = {
        "preproc": preproc,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }
    return models_cache[key], pipelines_cache[key]


# ---------------------------
# Load or build dataset (on startup)
# ---------------------------
if os.path.exists(DATA_FILE):
    try:
        df_raw = pd.read_csv(DATA_FILE)
        print(f"Loaded {DATA_FILE}, rows={len(df_raw)}")
    except Exception as e:
        print("Failed to load existing data file, rebuilding:", e)
        df_raw = build_weather_dataset()
else:
    print(f"{DATA_FILE} not found — building from API...")
    df_raw = build_weather_dataset()

# Prepare training X,y and full prepared df
X_all, y_all, df_prepared = prepare_df_for_model(df_raw)

# Keep the list of available models for the frontend
AVAILABLE_MODELS = [
    "Decision Tree",
    "Lasso (SGDClassifier)",
    "SVM (RBF)",
    "Random Forest",
    "Gradient Boosting",
]
if _xgb_available:
    AVAILABLE_MODELS.insert(4, "XGBoost")


# ---------------------------
# Flask endpoints
# ---------------------------
@app.route("/")
def index():
    # Serve a simple form if you have index.html in templates, else return available models
    try:
        return render_template("index.html", models=AVAILABLE_MODELS)
    except Exception:
        return jsonify({"available_models": AVAILABLE_MODELS})


@app.route("/predict", methods=["POST"])
def predict_api():
    """
    Endpoint: accepts {"city": "...", "hours_ahead": 3, "model": "best" or "Random Forest"}
    -> fetches forecast, runs model, returns rain_probability + accuracy.
    If model='best', selects model with highest cached accuracy.
    """
    req = request.get_json(force=True)
    city = req.get("city")
    hours = float(req.get("hours_ahead", 3))

    model_name = str(req.get("model", "Random Forest")).strip()
    if model_name.lower() in ("best", "auto", "auto (best model)"):
        # выбираем лучшую модель
        if models_cache:
            best_model_key = max(
                models_cache.items(), key=lambda kv: kv[1]["metrics"].get("accuracy", 0)
            )[0]
            used_model_name = (
                best_model_key  # имя модели из keys (подкорректируй формат)
            )
        else:
            # если кэша нет — обучаем всех кандидатов и выбираем (см. ниже)
            used_model_name = "Random Forest"
        model_name = "Best Model"
    else:
        used_model_name = model_name

    # --- геокодирование ---
    qlist = CITY_QUERIES.get(city, [f"{city},KZ"])
    geo = geocode_best(qlist)
    if not geo:
        return jsonify({"error": f"Cannot geocode city '{city}'"}), 400
    lat, lon, _ = geo

    data = fetch_forecast(lat, lon)
    if not data or "list" not in data:
        return jsonify({"error": f"No forecast available for {city}"}), 400

    # --- выбираем ближайший прогноз ---
    target = pd.Timestamp.now() + pd.to_timedelta(hours, unit="h")
    fl = pd.DataFrame(data["list"])
    fl["datetime"] = pd.to_datetime(fl["dt_txt"])
    idx = (fl["datetime"] - target).abs().argsort()[0]
    nearest = fl.iloc[idx]

    rec = {
        "city": city,
        "datetime": nearest["dt_txt"],
        "temp": nearest["main"]["temp"],
        "feels_like": nearest["main"]["feels_like"],
        "pressure": nearest["main"]["pressure"],
        "humidity": nearest["main"]["humidity"],
        "clouds": nearest["clouds"]["all"],
        "wind_speed": nearest["wind"]["speed"],
        "weather_main": (
            nearest["weather"][0]["main"] if nearest.get("weather") else None
        ),
    }

    df_user = pd.DataFrame([rec])
    X_user, _, _ = prepare_df_for_model(df_user)

    try:
        model_info, _ = train_and_cache(used_model_name, X_all, y_all)
    except Exception as e:
        return jsonify({"error": f"Training failed: {e}"}), 500

    pipe = model_info["pipeline"]

    try:
        if hasattr(pipe, "predict_proba"):
            prob = float(pipe.predict_proba(X_user)[0][1])
        else:
            pred = pipe.predict(X_user)[0]
            prob = float(pred)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    metrics = model_info.get("metrics", {})
    accuracy = metrics.get("accuracy", None)

    return jsonify(
        {
            "model": model_name,
            "used_model": used_model_name,
            "is_best_model": model_name.lower().startswith("best"),
            "city": city,
            "datetime": rec["datetime"],
            "rain_probability": round(prob * 100, 2),
            "accuracy": round(accuracy * 100, 2) if accuracy is not None else None,
            "metrics": metrics,
        }
    )


# Optional endpoint to retrain/clear cache
@app.route("/retrain", methods=["POST"])
def retrain():
    """
    Force retrain of a given model or all models.
    JSON: { "model": "Random Forest" } or { "model": "all" }
    """
    req = request.get_json(force=True)
    model = req.get("model", "all")
    if model == "all":
        models_cache.clear()
        pipelines_cache.clear()
        return jsonify({"status": "cleared"}), 200
    else:
        k = model.lower()
        models_cache.pop(k, None)
        pipelines_cache.pop(k, None)
        return jsonify({"status": f"cleared {model}"}), 200


# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    print("Available models:", AVAILABLE_MODELS)
    app.run(debug=True, host="127.0.0.1", port=5000)
