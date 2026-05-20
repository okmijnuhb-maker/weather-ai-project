#  WeatherAI — Climate Intelligence System

**AI-Based Weather & Climate Intelligence System for Human Task Optimization**  
Student: J. Charan Reddy | Reg No: 23BCB7120

## 🔗 Live Demo

 **https://weather-ai-project-j-charan-reddy.streamlit.app/**

---

##  Purpose

WeatherAI predicts rainfall probability and temperature using machine learning trained on 142,193 Australian weather records. It combines live weather data from the OpenWeather API with ML predictions to give industry-specific task recommendations and risk alerts — all in a real-time interactive dashboard.

---

##  ML Techniques Used

| Task | Model | Result |
|---|---|---|
| Rain Prediction | XGBoost Classifier | Accuracy: 82.30% · AUC: 0.9305 |
| Temperature Forecast | XGBoost Regressor | R²: 0.8913 · MAE: 0.2513 |
| Feature Scaling | StandardScaler | 16 numerical columns |
| Imbalance Handling | class_weight = balanced | 3.46:1 ratio |
| Threshold Optimization | F1-score tuning | Best threshold: 0.61 (not default 0.50) |

Models compared: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM — XGBoost selected for best balance of accuracy and deployment size.

---

##  Key Findings

- **Top rain predictor:** Humidity3pm (importance: 0.28)
- **Top temperature predictor:** Season (importance: 0.44)
- **Dataset:** 145,460 rows · 23 columns · 0 duplicates
- **Class imbalance:** No Rain 77.58% vs Rain 22.42%
- **High null columns:** Sunshine (48%), Evaporation (43%), Cloud3pm (41%) — filled with median
- **Data leakage fixed:** Temp9am and Temp3pm dropped from regression features
- **AUC 0.9305** — model strongly discriminates rain vs no-rain days

---

##  App Pages

1. **Home** — Live weather via OpenWeather API (temperature, humidity, wind, AQI, 5-day forecast)
2. **Predict** — ML rain probability + temperature forecast with feature importance
3. **Recommend** — Industry-wise task advice (Agriculture, Construction, Transport, Tourism, Energy, Logistics)
4. **Alerts** — Risk level (Low / Medium / High) with active weather alerts
5. **Report** — Downloadable weather summary report
6. **Insights** — Full model metrics, evaluation charts, preprocessing summary

---

##  Run Locally

```bash
git clone https://github.com/okmijnuhb-maker/weather-ai-project.git
cd weather-ai-project
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:
```toml
OPENWEATHER_API_KEY = "your_api_key_here"
```

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

Python · Streamlit · XGBoost · Scikit-learn · Pandas · NumPy · Plotly · OpenWeather API · GitHub · Streamlit Cloud
