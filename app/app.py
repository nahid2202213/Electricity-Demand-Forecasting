import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
import joblib

BANGLADESH_TZ = ZoneInfo("Asia/Dhaka")

# ---------------------------------------------------------------------------
# Setup — load model and historical data (paths relative to this file so
# it works regardless of the working directory Streamlit runs from)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "electicity_xgb_prediction_model.pkl"))


@st.cache_data
def load_history():
    df = pd.read_csv(os.path.join(BASE_DIR, "Electricity Demand Dataset.csv"))
    return df


history = load_history()


def fetch_weather(city_name: str):
    """Look up a city's coordinates, then fetch current temperature and
    humidity from the free Open-Meteo API (no API key required)."""
    geo_resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city_name, "count": 1},
        timeout=10,
    )
    geo_resp.raise_for_status()
    results = geo_resp.json().get("results")
    if not results:
        return None

    lat = results[0]["latitude"]
    lon = results[0]["longitude"]
    resolved_name = results[0].get("name", city_name)
    country = results[0].get("country", "")

    weather_resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m",
        },
        timeout=10,
    )
    weather_resp.raise_for_status()
    current = weather_resp.json().get("current", {})

    return {
        "location": f"{resolved_name}, {country}".strip(", "),
        "temperature": current.get("temperature_2m"),
        "humidity": current.get("relative_humidity_2m"),
    }


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "temperature" not in st.session_state:
    st.session_state.temperature = 25.0
if "humidity" not in st.session_state:
    st.session_state.humidity = 60.0

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("⚡ Electricity Demand Forecasting")
st.write(
    "Enter a date and hour, then either fetch live weather for a city or "
    "enter temperature/humidity manually, to get a predicted electricity demand."
)

col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input("Date", datetime.now(BANGLADESH_TZ).date())
with col2:
    hour = st.slider("Hour of Day", 0, 23, datetime.now(BANGLADESH_TZ).hour)
st.caption("Defaults to the current date and hour (Bangladesh time) — change them for a different forecast.")

st.subheader("🌤️ Weather")
city = st.text_input("City", placeholder="e.g. Dhaka")

if st.button("Fetch Live Weather"):
    if not city.strip():
        st.warning("Please enter a city name first.")
    else:
        with st.spinner(f"Fetching weather for {city}..."):
            try:
                weather = fetch_weather(city.strip())
            except requests.RequestException:
                weather = None

        if weather is None or weather["temperature"] is None:
            st.error("Couldn't find weather for that city. Try a different name.")
        else:
            st.session_state.temperature = float(weather["temperature"])
            st.session_state.humidity = float(weather["humidity"])
            st.success(
                f"Weather fetched for {weather['location']}: "
                f"{weather['temperature']}°C, {weather['humidity']}% humidity"
            )

col3, col4 = st.columns(2)
with col3:
    temperature = st.number_input("Temperature (°C)", key="temperature")
with col4:
    humidity = st.number_input(
        "Humidity (%)", min_value=0.0, max_value=100.0, key="humidity"
    )
st.caption("You can fetch live weather above, or edit these values manually.")

if st.button("Predict Demand"):
    # --- Derive calendar features automatically from the selected date ---
    dayofweek = selected_date.weekday()          # 0 = Monday ... 6 = Sunday
    month = selected_date.month
    year = selected_date.year
    dayofyear = selected_date.timetuple().tm_yday
    weekofyear = selected_date.isocalendar()[1]
    quarter = (month - 1) // 3 + 1
    is_weekend = 1 if dayofweek >= 5 else 0

    # --- Estimate recent-demand features from historical patterns ---
    # (We don't have live/actual recent demand at prediction time, so we
    # approximate using the historical average demand for the same hour
    # and day of week.)
    same_slot = history[
        (history["hour"] == hour) & (history["dayofweek"] == dayofweek)
    ]
    if same_slot.empty:
        same_slot = history[history["hour"] == hour]

    demand_lag_24hr = same_slot["Demand"].mean()
    demand_lag_168hr = same_slot["Demand"].mean()

    same_hour = history[history["hour"] == hour]
    demand_rolling_mean_24hr = same_hour["Demand"].mean()
    demand_rolling_std_24hr = same_hour["Demand"].std()

    input_data = pd.DataFrame({
        "hour": [hour],
        "dayofweek": [dayofweek],
        "month": [month],
        "year": [year],
        "dayofyear": [dayofyear],
        "weekofyear": [weekofyear],
        "quarter": [quarter],
        "is_weekend": [is_weekend],
        "Temperature": [st.session_state.temperature],
        "Humidity": [st.session_state.humidity],
        "Demand_lag_24hr": [demand_lag_24hr],
        "demand_lag_168hr": [demand_lag_168hr],
        "demand_rolling_mean_24hr": [demand_rolling_mean_24hr],
        "demand_rolling_std_24hr": [demand_rolling_std_24hr],
    })

    prediction = model.predict(input_data)
    st.success(f"Predicted Demand: {prediction[0]:.2f}")

    with st.expander("See details used for this prediction"):
        st.dataframe(input_data)
