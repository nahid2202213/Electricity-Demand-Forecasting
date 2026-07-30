# ⚡ Electricity Demand Forecasting

A machine learning project that forecasts hourly electricity demand using an **XGBoost** model trained on historical demand, weather, and time-based features. The trained model is deployed as an interactive **Streamlit web app** so users can enter their own inputs and get an instant demand prediction.

## 🚀 Live Demo

Try the app here: **[electricity-demand-forecasting-nahid-hasan.streamlit.app](https://electricity-demand-forecasting-nahid-hasan.streamlit.app/)**

## 📌 Project Overview

Accurate electricity demand forecasting is critical for utility companies, grid operators, and businesses — it helps manage peak demand, reduce load shedding, and plan better around renewable energy integration.

This project covers:
- Analysis of historical hourly electricity demand data (2020–2024)
- Engineering of time-based and weather-based features (lags, rolling statistics, etc.)
- Training and comparing **XGBoost, Random Forest, and Linear Regression** models
- Hyperparameter tuning, 5-fold time-series cross-validation, and feature importance / residual analysis
- Deploying the best model in an interactive **Streamlit app** with live weather fetch

## 🗂️ Repository Structure

```
Electricity-Demand-Forecasting/
├── app/
│   ├── Electricity Demand Dataset.csv              # Historical hourly demand + weather data
│   ├── ML Project Electricity Demand Forecasting.ipynb   # Full EDA, feature engineering & model training
│   ├── app.py                                      # Streamlit app for live prediction
│   ├── electicity_xgb_prediction_model.pkl         # Trained XGBoost model
│   └── requirements.txt                            # Python dependencies (app-level)
├── requirements.txt                                # Python dependencies (root-level, used by Streamlit Cloud)
├── .devcontainer/                                  # Dev container config
└── README.md
```

## 📊 Dataset

- **~43,800 hourly records** (Jan 2020 – Dec 2024)
- Columns: `Timestamp`, `hour`, `dayofweek`, `month`, `year`, `dayofyear`, `Temperature`, `Humidity`, `Demand`

## 🛠️ Feature Engineering

To improve model accuracy, several additional features were created:
- `Demand_lag_24hr` – demand from the same hour, 24 hours ago
- `demand_lag_168hr` – demand from the same hour, 1 week ago
- `demand_rolling_mean_24hr` – rolling average demand over the last 24 hours
- `demand_rolling_std_24hr` – rolling standard deviation of demand over the last 24 hours
- Calendar features: `weekofyear`, `quarter`, `is_weekend`

## 🤖 Model

- **Algorithm:** XGBoost Regressor (hyperparameter-tuned)
- **Validation:** 5-fold Time Series Cross-Validation, plus a held-out chronological test set (2024 onward)
- **Performance (test set):**

| Model | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|
| **XGBoost (tuned)** | **148.19** | **99.74** | **0.989** | **2.10%** |
| XGBoost (baseline) | 174.83 | 123.32 | 0.985 | 2.58% |
| Random Forest | 193.94 | 138.44 | 0.981 | 2.88% |
| Linear Regression | 243.61 | 180.72 | 0.970 | 3.75% |

XGBoost was compared against Random Forest and Linear Regression baselines, then tuned (learning rate, tree depth, subsampling) — reducing RMSE by ~15% over the initial configuration. Feature importance and residual analysis are also included in the notebook.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/nahid2202213/Electricity-Demand-Forecasting.git
cd Electricity-Demand-Forecasting/app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```
The app opens with the date and hour pre-filled to the current time (Bangladesh timezone). Enter a city and click **Fetch Live Weather** to auto-fill temperature and humidity (fetched from the free Open-Meteo API — no API key needed), or edit those values manually. Click **Predict Demand** to get an instant forecast. All other model features (day of week, quarter, recent demand trends, etc.) are calculated automatically behind the scenes.

### 💻 Option: Run Instantly with GitHub Codespaces

No local setup needed — this repo includes a `.devcontainer` config, so you can run the app directly in the cloud:

1. Click **Code → Codespaces → Create codespace on main**
2. Wait for the environment to build (Python, dependencies, and Streamlit are installed automatically)
3. The app will launch automatically and open in a preview tab

## 📦 Requirements

- streamlit
- pandas
- numpy
- matplotlib
- xgboost
- scikit-learn
- joblib
- requests
- tzdata

## 🔮 Future Improvements

- Try additional models (LightGBM, LSTM, Prophet)
- Automated hyperparameter search (GridSearchCV / Optuna) instead of manual configs
- Use actual weather *forecasts* (not just current conditions) for future-dated predictions
- Display prediction confidence intervals
- Fix and re-include the holiday feature (originally computed but dropped) with the correct country's holiday calendar

## 👤 Author

**Nahid**
— [GitHub Profile](https://github.com/nahid2202213)