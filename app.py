from flask import Flask, render_template, request, jsonify
import os, time, requests
import pandas as pd
import numpy as np
import warnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import randint, uniform
import io
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay






warnings.filterwarnings("ignore")

app = Flask(__name__)

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Missing API_KEY. Please set it in your .env file.")

# API_KEY = "cd59f6e78b8ec3e0d4b69f9e55d80e1c"
DATA_FILE = "data/weather_data.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

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
    "wind_speed",        
    "wind_deg",        
    "hour_sin",       
    "hour_cos",
    "mon_sin",        
    "mon_cos",
    "is_weekend",       
    "season",       
    "city",              
    "wind_power",        
    "humid_cloud",       
]

CITY_QUERIES.update(EUROPE_CAPITALS)

models_cache = {}
pipelines_cache = {}



def plot_feature_importance(model, X, y=None, top_n=15, save_path="feature_importance.png"):
    """
    Строит бар-чарт важности признаков для RandomForest или GradientBoosting.
    """
    if hasattr(model, "feature_importances_"):
        # Основной вариант для деревьев
        importances = model.feature_importances_
        features = getattr(model, "feature_names_in_", None)
        if features is None and hasattr(X, "columns"):
            features = X.columns
        elif features is None:
            features = [f"f{i}" for i in range(len(importances))]

        if len(importances) != len(features):
            print(f"[Warning] len(importances)={len(importances)} != len(features)={len(features)}")
            indices = np.argsort(importances)[::-1][:top_n]
            features = [f"f{i}" for i in indices]
            importances = importances[indices]

        df = pd.DataFrame({"feature": features, "importance": importances})
        df = df.sort_values("importance", ascending=False).head(top_n)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="importance", y="feature", data=df, palette="viridis")
        plt.title("Figure 2: Feature Importances")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Saved feature importance chart] {save_path}")

    elif y is not None:
        # Permutation importance как fallback
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        importances = result.importances_mean
        features = X.columns
        df = pd.DataFrame({"feature": features, "importance": importances})
        df = df.sort_values("importance", ascending=False).head(top_n)

        plt.figure(figsize=(10,6))
        sns.barplot(x="importance", y="feature", data=df, palette="viridis")
        plt.title("Figure 2: Permutation Feature Importances")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Saved permutation importance chart] {save_path}")

    else:
        print("Cannot compute feature importance for this model.")

def plot_model_metrics(models_cache, save_path="model_metrics_overview.png"):
    """
    Строит сравнительные графики (bar charts) для всех моделей:
    Accuracy (Train/Test/CV), F1, Precision, Recall.
    Сохраняет всё в один PNG файл.
    """
    if not models_cache:
        print("No trained models found in cache.")
        return

    # Собираем метрики в таблицу
    data = []
    for name, info in models_cache.items():
        metrics = info.get("metrics", {})
        if not metrics:
            continue
        data.append({
            "Model": name,
            "Train Accuracy": metrics.get("train_accuracy", 0),
            "Test Accuracy": metrics.get("test_accuracy", 0),
            "CV Accuracy": metrics.get("cv_accuracy", 0),
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "F1": metrics.get("f1", 0),
        })

    df = pd.DataFrame(data)
    df = df.sort_values("Test Accuracy", ascending=False)

    # --- Визуализация ---
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("📊 Model Performance Comparison", fontsize=16, fontweight="bold")

    # Accuracy (Train/Test/CV)
    df_melt_acc = df.melt(id_vars="Model", value_vars=["Train Accuracy", "Test Accuracy", "CV Accuracy"], var_name="Type", value_name="Accuracy")
    sns.barplot(ax=axes[0,0], data=df_melt_acc, x="Model", y="Accuracy", hue="Type", palette="viridis")
    axes[0,0].set_title("Accuracy (Train/Test/CV)")
    axes[0,0].tick_params(axis="x", rotation=45)

    # F1
    sns.barplot(ax=axes[0,1], data=df, x="Model", y="F1", palette="coolwarm")
    axes[0,1].set_title("F1 Score")
    axes[0,1].tick_params(axis="x", rotation=45)

    # Precision
    sns.barplot(ax=axes[1,0], data=df, x="Model", y="Precision", palette="mako")
    axes[1,0].set_title("Precision")
    axes[1,0].tick_params(axis="x", rotation=45)

    # Recall
    sns.barplot(ax=axes[1,1], data=df, x="Model", y="Recall", palette="rocket")
    axes[1,1].set_title("Recall")
    axes[1,1].tick_params(axis="x", rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved metrics overview chart] {save_path}")

def generate_full_report(models_cache, X_all, y_all, preproc, save_dir="reports"):
    """
    Создает 4 визуализации:
    1. Correlation heatmap (взаимосвязь признаков)
    2. Feature importance (Random Forest)
    3. Model comparison (Accuracy/F1)
    4. Confusion matrix (лучшая модель)
    Сохраняет все в PNG.
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Figure 1: Correlation heatmap ---
    corr_path = os.path.join(save_dir, "figure1_correlation_heatmap.png")
    numeric_X = X_all.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_X.corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Figure 1: Correlation heatmap (feature relationships)")
    plt.tight_layout()
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"[Saved] {corr_path}")

    # --- Figure 2: Feature importance (Random Forest) ---
    rf_model_info = models_cache.get("random forest") or models_cache.get("random_forest")
    if rf_model_info:
        rf_pipeline = rf_model_info["pipeline"]
        imp_path = os.path.join(save_dir, "figure2_rf_importance.png")
        plot_feature_importance(rf_pipeline.named_steps["clf"], X_all, y_all, top_n=15, save_path=imp_path)
    else:
        print("Random Forest model not found in cache — skipping feature importance plot.")

    # --- Figure 3: Model comparison ---
    comp_path = os.path.join(save_dir, "figure3_model_comparison.png")
    data = []
    for name, info in models_cache.items():
        m = info.get("metrics", {})
        if not m:
            continue
        data.append({
            "Model": name.title(),
            "Accuracy": m.get("test_accuracy", 0),
            "F1": m.get("f1", 0),
        })
    if data:  
        df = pd.DataFrame(data).sort_values("Accuracy", ascending=False)
        plt.figure(figsize=(10, 6))
        df_melt = df.melt(id_vars="Model", value_vars=["Accuracy", "F1"], var_name="Metric", value_name="Score")
        sns.barplot(data=df_melt, x="Model", y="Score", hue="Metric", palette="viridis")
        plt.title("Figure 3: Model comparison (Accuracy & F1)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(comp_path, dpi=300)
        plt.close()
        print(f"[Saved] {comp_path}")
    else:
        print("No model metrics found — skipping model comparison.")
        return  

    # --- Figure 4: Confusion matrix for best model ---
    best_name, best_info = max(models_cache.items(), key=lambda kv: kv[1]["metrics"].get("test_accuracy", 0))
    best_pipe = best_info["pipeline"]
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
    y_pred = best_pipe.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(save_dir, "figure4_confusion_matrix.png")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"Figure 4: Confusion matrix for best model ({best_name.title()})")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Saved] {cm_path}")

    print("\nAll 4 figures generated in:", save_dir)

def save_model(name, model):
    path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_').lower()}.pkl")
    joblib.dump(model, path)
    print(f"[Saved] {name} → {path}")

def load_model_if_exists(name):
    path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_').lower()}.pkl")
    if os.path.exists(path):
        print(f"[Loaded saved model] {path}")
        return joblib.load(path)
    return None

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

    rename_map = {
        "time": "datetime",
        "rhum": "humidity",
        "wspd": "wind_speed",
        "wdir": "wind_deg",
        "prcp": "rain_3h",
        "snow": "snow_3h",
        "pres": "pressure",
        "temp": "temp",
        "dwpt": "feels_like",
        "city": "city",
    }
    df = df.rename(columns=rename_map)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        df["datetime"] = pd.NaT

    num_cols = [
        "temp", "feels_like", "pressure", "humidity",
        "clouds", "wind_speed", "pop", "wind_deg",
        "rain_3h", "snow_3h"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())
        else:
            df[c] = 0.0

    for c in ["city", "weather_main"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str)
        else:
            df[c] = pd.Series(["Unknown"] * len(df), index=df.index, dtype=str)

    df["hour"] = df["datetime"].dt.hour.fillna(0).astype(int)
    df["month"] = df["datetime"].dt.month.fillna(1).astype(int)
    df["dow"] = df["datetime"].dt.dayofweek.fillna(0).astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["mon_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["mon_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    def season_from_month(m):
        if m in [12, 1, 2]:
            return "Winter"
        if m in [3, 4, 5]:
            return "Spring"
        if m in [6, 7, 8]:
            return "Summer"
        return "Autumn"

    df["season"] = df["month"].apply(season_from_month)
    df["wind_power"] = df["wind_speed"] ** 2
    df["humid_cloud"] = df["humidity"] * df.get("clouds", 0) / 100.0

    df["temp_prev"] = df["temp"].shift(1).fillna(df["temp"].iloc[0])
    df["humidity_prev"] = df["humidity"].shift(1).fillna(df["humidity"].iloc[0])
    df["pressure_prev"] = df["pressure"].shift(1).fillna(df["pressure"].iloc[0])

    df["pressure_change"] = df["pressure"] - df["pressure_prev"]
    df["temp_change_3h"] = df["temp"] - df["temp"].shift(3).fillna(df["temp"].iloc[0])
    df["humidity_change_3h"] = df["humidity"] - df["humidity"].shift(3).fillna(df["humidity"].iloc[0])
    df["pressure_trend_12h"] = df["pressure"].rolling(window=12, min_periods=1).mean()
    df["wind_power_change"] = df["wind_power"] - df["wind_power"].shift(1).fillna(0)
    df["dew_point"] = df["temp"] - ((100 - df["humidity"]) / 5)
    df["dew_point_diff"] = df["dew_point"] - df["feels_like"]

    EXTRA_FEATURES = ["temp_change_3h", "humidity_change_3h", "pressure_trend_12h", "wind_power_change", "dew_point_diff"]
    for f in EXTRA_FEATURES:
        if f not in df.columns:
            df[f] = 0.0
        df[f] = df[f].fillna(0)

    if "RainOrNot" not in df.columns:
        df["RainOrNot"] = ((df["rain_3h"] > 0) | (df["snow_3h"] > 0)).astype(int)

    all_features = MODEL_FEATURES + EXTRA_FEATURES
    for f in all_features:
        if f not in df.columns:
            df[f] = 0.0

    X = df[all_features].copy()
    y = df["RainOrNot"].copy() if "RainOrNot" in df.columns else None

    return X, y, df


def build_preprocessor(X_train):
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("pca", PCA(n_components=0.95))
    ])
    try:
        cat_pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
    except TypeError:
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


def build_dl_model(input_dim):
    model = Sequential([
        Dense(512, activation="relu", input_dim=input_dim),
        Dropout(0.5),
        Dense(256, activation="relu"),
        Dropout(0.4),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def get_model_by_name(name, X=None, y=None):
    """
    Returns a tuned model by name.
    Uses RandomizedSearchCV for hyperparameter optimization and saves best model to disk.
    """
    name = name.strip().lower()

    if name == "decision tree":
        cached = load_model_if_exists(name)
        if cached:
            return cached

        base = DecisionTreeClassifier(class_weight="balanced", random_state=42)
        if X is not None and y is not None:
            param_dist = {
                "max_depth": randint(6, 20),
                "min_samples_split": randint(2, 10),
                "min_samples_leaf": randint(1, 5),
                "criterion": ["gini", "entropy"],
            }
            rs = RandomizedSearchCV(base, param_dist, n_iter=15, cv=3,
                                    scoring="accuracy", n_jobs=-1, random_state=42)
            rs.fit(X, y)
            best = rs.best_estimator_
            print(f"[Decision Tree] Best params: {rs.best_params_}")
            save_model(name, best)
            return best
        return base

    if name in ("svm", "svm (rbf)"):
        cached = load_model_if_exists(name)
        if cached:
            return cached

        base = SVC(kernel="rbf", class_weight="balanced", probability=True)
        if X is not None and y is not None:
            param_dist = {
                "C": uniform(0.1, 10),
                "gamma": ["scale", "auto", 0.1, 0.01, 0.001],
            }
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            rs = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=30,
                                    cv=cv, scoring="accuracy", n_jobs=-1, random_state=42)
            rs.fit(X, y)
            best = rs.best_estimator_
            print(f"[SVM] Best params: {rs.best_params_}")
            save_model(name, best)
            return best
        return base

    if name in ("logistic regression", "logreg"):
        cached = load_model_if_exists(name)
        if cached:
            return cached

        base = LogisticRegression(
            solver="saga",
            class_weight="balanced",
            max_iter=3000,  
            n_jobs=-1,
            random_state=42
        )

        if X is not None and y is not None:
            param_dist = [
                {"penalty": ["l1", "l2"], "C": uniform(0.1, 5)},  
            ]

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            rs = RandomizedSearchCV(
                base,
                param_distributions=param_dist,
                n_iter=10,  
                cv=cv,
                scoring="f1",
                n_jobs=-1,
                random_state=42,
                verbose=2
            )

            rs.fit(X, y)
            best = rs.best_estimator_
            print(f"[LogReg] Best params: {rs.best_params_}")
            save_model(name, best)
            return best

        return base

    if name in ("random forest", "random_forest"):
        cached = load_model_if_exists(name)
        if cached:
            print("[Random Forest] Loaded cached model")
            return cached

        print("[Random Forest] Starting hyperparameter tuning...")
        base = RandomForestClassifier(
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        )

        if X is not None and y is not None:
            param_dist = {
                "n_estimators": randint(1000, 1500),
                "max_depth": randint(8, 30),
                "min_samples_leaf": randint(1, 5),
                "max_features": ["sqrt", "log2"],
            }
            rs = RandomizedSearchCV(base, param_dist, n_iter=50, cv=3,
                                    scoring="accuracy", n_jobs=-1, random_state=42)
            rs.fit(X, y)
            best = rs.best_estimator_
            print(f"[Random Forest] Best params: {rs.best_params_}")
            save_model(name, best)
            print("[Random Forest] Model saved")
            return best

        return base

    if name in ("gradient boosting", "gradient_boosting"):
        cached = load_model_if_exists(name)
        if cached:
            print("[Gradient Boosting] Loaded cached model")
            return cached

        print("[Gradient Boosting] Starting hyperparameter tuning...")
        base = GradientBoostingClassifier(random_state=42)

        if X is not None and y is not None:
            param_dist = {
                "n_estimators": randint(500, 1000),
                "learning_rate": uniform(0.05, 0.15),
                "max_depth": randint(3, 7),
                "subsample": uniform(0.7, 0.3),
                "max_features": ["sqrt", "log2", None]
            }
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            rs = RandomizedSearchCV(
                base,
                param_distributions=param_dist,
                n_iter=25,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
                random_state=42
            )
            rs.fit(X, y)
            best = rs.best_estimator_
            print(f"[Gradient Boosting] Best params: {rs.best_params_}")
            save_model(name, best)
            print("[Gradient Boosting] Model saved")
            return best

        return base

    if name in ("knn", "k-nearest neighbors"):
        cached = load_model_if_exists(name)
        if cached:
            return cached

        base = KNeighborsClassifier(n_jobs=-1)

        # If X and y are provided, perform hyperparameter search
        if X is not None and y is not None:
            param_dist = {
                "n_neighbors": randint(3, 150), 
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski", "chebyshev", "seuclidean"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "leaf_size": randint(10, 60),
                "p": [1, 2],  
            }
            cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
            rs = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=120,
                                    cv=cv, scoring="accuracy", n_jobs=-1, random_state=42, verbose=2)
            rs.fit(X, y)
            best = rs.best_estimator_
            print(f"[KNN] Best params: {rs.best_params_}")
            save_model(name, best)
            return best

        return base

    if name in ("deep learning", "neural network", "mlp"):
        model = KerasClassifier(
            build_fn=lambda: build_dl_model(X_all.shape[1]),
            epochs=300,
            batch_size=32,
            verbose=0,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
        )
        return model

    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

def train_and_cache(model_name, X_all, y_all):
    key = model_name.lower()
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_file = os.path.join(models_dir, f"{key}_model.pkl")
    metrics_file = os.path.join(models_dir, f"{key}_metrics.pkl")

    if os.path.exists(model_file) and os.path.exists(metrics_file):
        try:
            loaded_pipe = joblib.load(model_file)
            loaded_metrics = joblib.load(metrics_file)
            models_cache[key] = {"pipeline": loaded_pipe, "metrics": loaded_metrics}
            if "preproc" in pipelines_cache:
                preproc = pipelines_cache["preproc"]["preproc"]
                num_cols = pipelines_cache["preproc"]["num_cols"]
                cat_cols = pipelines_cache["preproc"]["cat_cols"]
            else:
                preproc, num_cols, cat_cols = build_preprocessor(X_all)
                pipelines_cache["preproc"] = {
                    "preproc": preproc,
                    "num_cols": num_cols,
                    "cat_cols": cat_cols,
                }
            pipelines_cache[key] = {
                "preproc": preproc,
                "num_cols": num_cols,
                "cat_cols": cat_cols,
            }
            return models_cache[key], pipelines_cache[key]
        except Exception as e:
            print(f"Failed to load cached model for {key}: {e}")

    if "preproc" in pipelines_cache:
        preproc = pipelines_cache["preproc"]["preproc"]
        num_cols = pipelines_cache["preproc"]["num_cols"]
        cat_cols = pipelines_cache["preproc"]["cat_cols"]
    else:
        preproc, num_cols, cat_cols = build_preprocessor(X_all)
        pipelines_cache["preproc"] = {
            "preproc": preproc,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
        }
    if key in models_cache and key in pipelines_cache:
        return models_cache[key], pipelines_cache[key]

    X_proc = preproc.fit_transform(X_all)
    model = get_model_by_name(model_name, X_proc, y_all)


    pipe = ImbPipeline([
        ("pre", preproc),
        ("smote", SMOTE(random_state=42, sampling_strategy=1.0)),
        ("clf", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_all, y_all, cv=cv, scoring="accuracy")
    cv_acc = cv_scores.mean()

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = {
        "train_accuracy": float(pipe.score(X_train, y_train)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "cv_accuracy": float(cv_acc),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    print(
        f"[{model_name}] Train: {metrics['train_accuracy']:.3f}, "
        f"Test: {metrics['test_accuracy']:.3f}, CV: {metrics['cv_accuracy']:.3f}"
    )

    models_cache[key] = {"pipeline": pipe, "metrics": metrics}
    pipelines_cache[key] = {
        "preproc": preproc,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }

    try:
        joblib.dump(pipe, model_file)
        joblib.dump(metrics, metrics_file)
    except Exception as e:
        print(f"Failed to save model or metrics for {key}: {e}")

    return models_cache[key], pipelines_cache[key]

def train_blended_model(X_all, y_all):
    """
    Ensemble blending using several strong base models + Logistic Regression as meta-model.
    """
    preproc, num_cols, cat_cols = build_preprocessor(X_all)
    X_proc = preproc.fit_transform(X_all)
    rf_model = get_model_by_name("Random Forest", X_proc, y_all)
    gb_model = get_model_by_name("Gradient Boosting", X_proc, y_all)
    base_models = [
        ("rf", rf_model),
        ("gb", gb_model),
    ]

    blend_train, blend_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    meta_train = np.zeros((blend_train.shape[0], len(base_models)))
    meta_test = np.zeros((blend_test.shape[0], len(base_models)))

    for i, (name, model) in enumerate(base_models):
        pipe = ImbPipeline([
            ("pre", preproc),
            ("smote", SMOTE(random_state=42)),
            ("clf", model),
        ])
        pipe.fit(blend_train, y_train)

        meta_train[:, i] = pipe.predict_proba(blend_train)[:, 1]
        meta_test[:, i] = pipe.predict_proba(blend_test)[:, 1]

    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    meta_model.fit(meta_train, y_train)
    y_pred = meta_model.predict(meta_test)

    metrics = {
        "train_accuracy": float(meta_model.score(meta_train, y_train)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    print(f"[Blending Ensemble] Test Accuracy: {metrics['test_accuracy']:.3f}")

    models_cache["ensemble_blend"] = {
        "pipeline": (base_models, meta_model, preproc),
        "metrics": metrics,
    }
    return models_cache["ensemble_blend"]

if os.path.exists(DATA_FILE):
    try:
        df_raw = pd.read_csv(DATA_FILE)
        print(f"Loaded {DATA_FILE}, rows={len(df_raw)}")
    except Exception as e:
        print("Failed to load existing data file")
else:
    print(f"{DATA_FILE} not found")

X_all, y_all, df_prepared = prepare_df_for_model(df_raw)

time_cols = ["temp", "feels_like", "pressure", "humidity", "wind_speed", "wind_deg", "clouds"]
for col in time_cols:
    df_prepared[col] = df_prepared[col].interpolate(method="linear").fillna(df_prepared[col].median())

for col in ["rain_3h", "snow_3h", "pop"]:
    df_prepared[col] = df_prepared[col].fillna(0.0)

df_prepared = df_prepared.fillna(0)

EXTRA_FEATURES = ["temp_change_3h", "humidity_change_3h", "pressure_trend_12h", "wind_power_change", "dew_point_diff"]
X_all = df_prepared[MODEL_FEATURES + EXTRA_FEATURES].copy()
y_all = df_prepared["RainOrNot"].copy()

AVAILABLE_MODELS = [
    "Decision Tree",
    "SVM (RBF)",
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
    "KNN",
    "Ensemble Blending",
]

@app.route("/")
def index():
    try:
        return render_template("index.html", models=AVAILABLE_MODELS)
    except Exception:
        return jsonify({"available_models": AVAILABLE_MODELS})


@app.route("/predict", methods=["POST"])
def predict_api():
    """
    Endpoint: accepts {"city": "...", "hours_ahead": 3, "model": "best" or "Random Forest"}
    Fetches forecast, runs the selected model, and returns rain probability + model accuracy.
    """
    req = request.get_json(force=True)
    city = req.get("city")
    hours = float(req.get("hours_ahead", 3))
    requested_model = str(req.get("model", "Random Forest")).strip()

    if requested_model.lower() in ("best", "auto", "auto (best model)"):
        if "best" in models_cache:
            selected_model_name = models_cache["best"]["used_model"]
            model_label = "Best Model (Cached)"
        elif models_cache:
            best_item = max(
                models_cache.items(),
                key=lambda kv: kv[1]["metrics"].get("cv_accuracy", 0)
            )
            selected_model_name = best_item[0]
            model_label = "Best Model"
            models_cache["best"] = {**best_item[1], "used_model": selected_model_name}
        else:
            selected_model_name = "Random Forest"
            model_label = "Best Model (Default: Random Forest)"
    else:
        selected_model_name = requested_model
        model_label = requested_model

    qlist = CITY_QUERIES.get(city, [f"{city},KZ"])
    geo = geocode_best(qlist)
    if not geo:
        return jsonify({"error": f"Cannot geocode city '{city}'"}), 400
    lat, lon, _ = geo

    data = fetch_forecast(lat, lon)
    if not data or "list" not in data:
        return jsonify({"error": f"No forecast data available for '{city}'"}), 400
    if not isinstance(data.get("list"), list) or not data["list"]:
        return jsonify({"error": "Forecast data is empty or invalid."}), 400

    target_time = pd.Timestamp.now() + pd.to_timedelta(hours, unit="h")
    forecast_df = pd.DataFrame(data["list"])
    forecast_df["datetime"] = pd.to_datetime(forecast_df["dt_txt"])
    nearest = forecast_df.iloc[(forecast_df["datetime"] - target_time).abs().argsort()[0]]

    rec = {
        "city": city,
        "datetime": nearest["dt_txt"],
        "temp": nearest["main"]["temp"],
        "feels_like": nearest["main"]["feels_like"],
        "pressure": nearest["main"]["pressure"],
        "humidity": nearest["main"]["humidity"],
        "clouds": nearest["clouds"]["all"],
        "wind_speed": nearest["wind"]["speed"],
        "weather_main": nearest["weather"][0]["main"] if nearest.get("weather") else "Unknown",
    }
    df_user = pd.DataFrame([rec])
    X_user, _, _ = prepare_df_for_model(df_user)

    if selected_model_name.lower() == "ensemble blending":
        try:
            model_info = train_blended_model(X_all, y_all)
            base_models, meta_model, preproc = model_info["pipeline"]

            X_user_proc = preproc.transform(X_user)
            meta_features = [m.predict_proba(X_user_proc)[:, 1][0] for _, m in base_models]
            meta_input = np.array(meta_features).reshape(1, -1)
            prob = float(meta_model.predict_proba(meta_input)[0][1])
            accuracy = model_info["metrics"].get("test_accuracy", None)

            return jsonify({
                "model": "Ensemble Blending",
                "used_model": "Blending Ensemble",
                "city": city,
                "datetime": rec["datetime"],
                "rain_probability": round(prob * 100, 2),
                "accuracy": round(accuracy * 100, 2) if accuracy else None,
                "metrics": model_info["metrics"],
            })
        except Exception as e:
            return jsonify({"error": f"Ensemble prediction failed: {e}"}), 500

    try:
        model_info, _ = train_and_cache(selected_model_name, X_all, y_all)
        pipe = model_info["pipeline"]

        if hasattr(pipe, "predict_proba"):
            prob = float(pipe.predict_proba(X_user)[0][1])
        else:
            prob = float(pipe.predict(X_user)[0])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    metrics = model_info.get("metrics", {})
    accuracy = metrics.get("test_accuracy") or metrics.get("cv_accuracy")

    plot_model_metrics(models_cache)

    preproc = pipelines_cache.get("preproc", {}).get("preproc")
    generate_full_report(models_cache, X_all, y_all, preproc)

    return jsonify({
        "model": model_label,
        "used_model": selected_model_name,
        "is_best_model": requested_model.lower().startswith("best"),
        "city": city,
        "datetime": rec["datetime"],
        "rain_probability": round(prob * 100, 2),
        "accuracy": round(accuracy * 100, 2) if accuracy is not None else None,
        "metrics": metrics,
    })


if __name__ == "__main__":
    print("Available models:", AVAILABLE_MODELS)
    app.run(debug=False, host="127.0.0.1", port=5000, threaded=False)
# 127.0.0.1 - - [10/Nov/2025 22:09:09] "GET / HTTP/1.1" 200 -
# 127.0.0.1 - - [10/Nov/2025 22:09:09] "GET /favicon.ico HTTP/1.1" 404 -
# [Decision Tree] Train: 0.840, Test: 0.806, CV: 0.791
# 127.0.0.1 - - [10/Nov/2025 22:09:13] "POST /predict HTTP/1.1" 200 -
# [Random Forest] Train: 0.965, Test: 0.916, CV: 0.912
# 127.0.0.1 - - [10/Nov/2025 22:09:42] "POST /predict HTTP/1.1" 200 -
# [Gradient Boosting] Train: 0.882, Test: 0.874, CV: 0.866
# 127.0.0.1 - - [10/Nov/2025 22:12:01] "POST /predict HTTP/1.1" 200 -
# [Logistic Regression] Train: 0.824, Test: 0.823, CV: 0.822
# 127.0.0.1 - - [10/Nov/2025 22:13:22] "POST /predict HTTP/1.1" 200 -
# [SVM (RBF)] Train: 0.830, Test: 0.827, CV: 0.823
# 127.0.0.1 - - [10/Nov/2025 22:14:42] "POST /predict HTTP/1.1" 200 -
# [KNN] Train: 1.000, Test: 0.925, CV: 0.920
# 127.0.0.1 - - [10/Nov/2025 22:14:50] "POST /predict HTTP/1.1" 200 -
# [Deep Learning (MLP)] Train: 0.824, Test: 0.823, CV: 0.822
# 127.0.0.1 - - [10/Nov/2025 22:15:05] "POST /predict HTTP/1.1" 200 -
# [Blending Ensemble] Test Accuracy: 0.931
# 127.0.0.1 - - [10/Nov/2025 22:17:29] "POST /predict HTTP/1.1" 200 -
# 127.0.0.1 - - [10/Nov/2025 22:17:37] "POST /predict HTTP/1.1" 200 -
