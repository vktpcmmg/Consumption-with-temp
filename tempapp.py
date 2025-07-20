import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import base64
import datetime

# Load Tata Power logo as base64
def get_image_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_image_base64("tata_logo.png")
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{logo_base64}" width="100">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; color: #0072C6;'>🔌 Electricity Consumption Predictor </h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>🔷 Designed by <span style='color: #0072C6;'>Tata Power - MMG</span></h4>", unsafe_allow_html=True)
st.markdown("*_Note: Based on ~100K smart meter data for FY 24–25 + live weather forecast._*")

# Cache data loading
@st.cache_data
def load_data():
    df = pd.read_csv("consumptionai.csv")
    df.columns = df.columns.str.strip()
    df.rename(columns={'Connected  Load': 'Connected Load'}, inplace=True)
    return df

@st.cache_data
def load_historical_weather():
    return pd.read_csv("weather.csv")

# Train models and label encoders
@st.cache_resource
def train_models(df):
    label_encoders = {}
    for col in ['Zone', 'Category']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    input_features = ['Connected Load', 'Zone', 'Category', 'Avg_Temp_C', 'Avg_Humidity']
    months = ['May', 'Jun', 'Jul', 'August', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    models = {}

    for month in months:
        X = df[input_features]
        y = df[month]
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        models[month] = model

    return models, label_encoders

# Open-Meteo API call for forecast weather
def fetch_forecast_weather(lat, lon, year, month):
    # We will fetch daily data for the full month and average it
    start_date = f"{year}-{month:02d}-01"
    # Get last day of month
    if month == 12:
        end_date = f"{year+1}-01-01"
    else:
        end_date = f"{year}-{month+1:02d}-01"
    end_date_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d") - datetime.timedelta(days=1)
    end_date = end_date_dt.strftime("%Y-%m-%d")

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "relativehumidity_2m_max", "relativehumidity_2m_min"],
        "timezone": "auto"
    }
    url = "https://api.open-meteo.com/v1/forecast"
    res = requests.get(url, params=params)
    data = res.json()
    if "daily" not in data:
        return None

    temps_max = data["daily"]["temperature_2m_max"]
    temps_min = data["daily"]["temperature_2m_min"]
    hum_max = data["daily"]["relativehumidity_2m_max"]
    hum_min = data["daily"]["relativehumidity_2m_min"]

    avg_temp = (np.mean(temps_max) + np.mean(temps_min)) / 2
    avg_humidity = (np.mean(hum_max) + np.mean(hum_min)) / 2

    return avg_temp, avg_humidity

# Mapping of zones to lat/lon (adjust coordinates as needed)
zone_coords = {
    "City South": (19.07, 72.88),
    "Urban": (19.13, 72.90),
    "West Suburb": (19.20, 72.82),
    "East Suburb": (19.22, 72.90),
    "North Suburb": (19.30, 72.85),
    "Metro": (19.08, 72.89)
}

# Load data
df = load_data()
weather_hist = load_historical_weather()

# Prepare df for model training by merging historical weather with consumption data
df_weather_merged = df.merge(weather_hist, how='left', left_on=['Zone'], right_on=['Zone'])

# Label encode after merge for consistency
models, label_encoders = train_models(df_weather_merged)

# UI Inputs
connected_load = st.number_input("Connected Load (kW/KVA)", min_value=0.1, value=10.0)

zone = st.selectbox("Select Zone", list(zone_coords.keys()))

category = st.selectbox("Select Category", label_encoders['Category'].classes_)

month_name = st.selectbox("Select Month", ['May', 'Jun', 'Jul', 'August', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'])

# Convert month name to number for API call
month_map = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'August':8, 'Sept':9, 'Oct':10, 'Nov':11, 'Dec':12}
month_num = month_map[month_name]

# Encode inputs
zone_enc = label_encoders['Zone'].transform([zone])[0]
category_enc = label_encoders['Category'].transform([category])[0]

# Predict button & logic
if connected_load <= 0:
    st.error("⚠️ Please enter a valid load. Zero or negative load does not exist.")
else:
    if st.button("🔍 Predict Consumption"):

        # Fetch forecast weather from Open-Meteo
        forecast_weather = fetch_forecast_weather(zone_coords[zone][0], zone_coords[zone][1], 2025, month_num)
        if forecast_weather is None:
            st.warning("⚠️ Could not fetch forecast weather for the selected zone/month.")
        else:
            avg_temp_forecast, avg_humidity_forecast = forecast_weather

            # Fetch historical weather for comparison
            hist_row = weather_hist[(weather_hist['Zone'] == zone) & (weather_hist['Month'] == month_name)]
            if hist_row.empty:
                st.warning("⚠️ Historical weather data not found for this zone/month.")
                avg_temp_hist = np.nan
                avg_humidity_hist = np.nan
            else:
                avg_temp_hist = hist_row.iloc[0]['Avg_Temp_C']
                avg_humidity_hist = hist_row.iloc[0]['Avg_Humidity']

            st.markdown(f"### 🌡️ Weather Data for {month_name} 2025 in {zone}")
            st.write(f"Forecasted Avg Temp: **{avg_temp_forecast:.2f} °C**")
            st.write(f"Forecasted Avg Humidity: **{avg_humidity_forecast:.2f} %**")
            st.write(f"Historical Avg Temp: **{avg_temp_hist:.2f} °C**")
            st.write(f"Historical Avg Humidity: **{avg_humidity_hist:.2f} %**")

            # Prepare model input with forecast weather
            input_data = np.array([[connected_load, zone_enc, category_enc, avg_temp_forecast, avg_humidity_forecast]])

            prediction = models[month_name].predict(input_data)[0]
            st.success(f"📊 Predicted electricity consumption for **{month_name} 2025**: **{prediction:.2f} kWh**")
