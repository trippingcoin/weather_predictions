# 🌦️ Weather Predictions — Machine Learning Forecast App

A Flask-based web application that predicts the **probability of rain** for any city using **machine learning models** and live data from the **OpenWeatherMap API**.

---

## 🚀 Features

- 🌍 **Real-time weather forecast** using OpenWeatherMap API  
- 🤖 Multiple **machine learning models**:
  - Random Forest  
  - Gradient Boosting  
  - Decision Tree  
  - SVM (RBF Kernel)  
  - Lasso (SGD Classifier)  
  - XGBoost *(if available)*  
- 🧠 Auto Model Selection — automatically picks the most accurate model  
- ⚖️ Handles imbalanced data using **SMOTE**  
- 📊 Displays model **accuracy**, precision, recall, and F1-score  
- 🧩 Modular architecture — easy to extend with deep learning models  
- 💾 Caches trained models to avoid re-training  

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-------------|
| **Backend** | Flask (Python) |
| **ML / AI** | scikit-learn, XGBoost, imbalanced-learn |
| **Data Source** | OpenWeatherMap API |
| **Visualization** | HTML, CSS, JavaScript (frontend form) |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/weather_predictions.git
cd weather_predictions
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

*(If using macOS with Apple Silicon, also install TensorFlow support if needed:)*  
```bash
pip install tensorflow-macos tensorflow-metal
```

---

## 🔑 API Setup

Create a `.env` file in the project root:

```bash
API_KEY=your_api_key_here
```

You can get a free API key from [OpenWeatherMap](https://openweathermap.org/api).

---

## ▶️ Run the App

```bash
python3 app.py
```

The Flask server will start at:

```
http://127.0.0.1:5000
```

---

## 🌤️ Example Output

```
City: Astana
Forecast Time: 2025-10-29 18:00:00
Rain Probability: 0.62%
Model Used: Random Forest
Accuracy: 93.4%
Prediction: ☀️ Mostly clear.
```

---

## 🧩 Extending the Project

You can add more models easily inside `get_model_by_name()`:

```python
if name == "neural network":
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    model = Sequential([
        Dense(64, activation="relu", input_dim=input_dim),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
```

---

## 📈 Future Improvements

- Add **deep learning (TensorFlow / Keras)** model integration  
- Display **historical performance charts**  
- Build **frontend dashboard** for predictions  
- Store results in a **database** (PostgreSQL / SQLite)  

---

## 👨‍💻 Author

**Torekhan Pugashbek**  
📍 Kazakhstan  
💬 Passionate about AI, Go, and backend systems.

---

## 🪪 License

MIT License © 2025 Torekhan Pugashbek