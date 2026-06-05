import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("electicity_xgb_prediction_model.pkl")

st.title("Electricity Demand Forecasting")

hour = st.number_input("Hour", 0, 23, 12)
dayofweek = st.number_input("Day of Week", 0, 6, 0)
month = st.number_input("Month", 1, 12, 1)
year = st.number_input("Year", 2020, 2035, 2026)
dayofyear = st.number_input("Day of Year", 1, 366, 1)
weekofyear = st.number_input("Week of Year", 1, 53, 1)
quarter = st.number_input("Quarter", 1, 4, 1)
is_weekend = st.selectbox("Weekend?", [0, 1])

temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")

demand_lag_24hr = st.number_input("Demand Lag 24hr")
demand_lag_168hr = st.number_input("Demand Lag 168hr")

demand_rolling_mean_24hr = st.number_input("Rolling Mean 24hr")
demand_rolling_std_24hr = st.number_input("Rolling Std 24hr")

if st.button("Predict"):

    input_data = pd.DataFrame({
        'hour':[hour],
        'dayofweek':[dayofweek],
        'month':[month],
        'year':[year],
        'dayofyear':[dayofyear],
        'weekofyear':[weekofyear],
        'quarter':[quarter],
        'is_weekend':[is_weekend],
        'Temperature':[temperature],
        'Humidity':[humidity],
        'Demand_lag_24hr':[demand_lag_24hr],
        'demand_lag_168hr':[demand_lag_168hr],
        'demand_rolling_mean_24hr':[demand_rolling_mean_24hr],
        'demand_rolling_std_24hr':[demand_rolling_std_24hr]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Demand: {prediction[0]:.2f}")
