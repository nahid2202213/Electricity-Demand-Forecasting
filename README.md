# ⚡ Electricity Demand Forecasting

A machine learning project that forecasts hourly electricity demand using an **XGBoost** model trained on historical demand, weather, and time-based features. The trained model is deployed as an interactive **Streamlit web app** so users can enter their own inputs and get an instant demand prediction.

## 📌 Project Overview

Accurate electricity demand forecasting is critical for utility companies, grid operators, and businesses — it helps manage peak demand, reduce load shedding, and plan better around renewable energy integration.

This project covers:
- Analysis of historical hourly electricity demand data (2020–2024)
- Engineering of time-based and weather-based features (lags, rolling statistics, etc.)
- Training an **XGBoost Regressor** to predict demand
- Deploying the trained model in an interactive **Streamlit app**

## 🗂️ Repository Structure

```
Electricity-Demand-Forecasting/
├── Electricity Demand Forecasting/
│   ├── Electricity Demand Dataset.csv              # Historical hourly demand + weather data
│   ├── ML Project Electricity Demand Forecasting.ipynb   # Full EDA, feature engineering & model training
│   ├── app.py                                      # Streamlit app for live prediction
│   ├── electicity_xgb_prediction_model.pkl         # Trained XGBoost model
│   └── requirements.txt                            # Python dependencies
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

- **Algorithm:** XGBoost Regressor
- **Validation:** Time Series Split (appropriate cross-validation for time-series data)
- **Performance (test set):**
  - RMSE: **174.82**
  - MAE: **123.47**

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/nahid2202213/Electricity-Demand-Forecasting.git
cd "Electricity-Demand-Forecasting/Electricity Demand Forecasting"
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```
The app will open in your browser, where you can enter values such as hour, day, month, temperature, and humidity to get an instant electricity demand prediction.

### 💻 Option: Run Instantly with GitHub Codespaces

No local setup needed — this repo includes a `.devcontainer` config, so you can run the app directly in the cloud:

1. Click **Code → Codespaces → Create codespace on main**
2. Wait for the environment to build (Python, dependencies, and Streamlit are installed automatically)
3. The app will launch automatically and open in a preview tab

## 📦 Requirements

- streamlit
- pandas
- numpy
- xgboost
- scikit-learn
- joblib

## 🔮 Future Improvements

- Compare additional models (Random Forest, LSTM, Prophet, etc.)
- Deeper hyperparameter tuning
- Real-time weather API integration
- Display prediction confidence intervals

## 👤 Author

**Nahid**
— [GitHub Profile](https://github.com/nahid2202213)
