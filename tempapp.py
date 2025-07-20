import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import base64
import datetime
import matplotlib.pyplot as plt

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

@st.cache_data
def load_data():
    df = pd.read_csv("consumptionai.csv")
    df.columns = df.columns.str.strip()
    df.rename(columns={'Connected  Load': 'Connected Load'}, inplace=True)
    return df

@st.cache_data
def load_historical_weather():
    return pd.read_csv("weather.csv")

def fetch_forecast_weather(lat, lon, year, month):
    start_date = f"{year}-{month:02d}-01"
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

zone_coords = {
    "City South": (19.07, 72.88),
    "Urban": (19.13, 72.90),
    "West Suburb": (19.20, 72.82),
    "East Suburb": (19.22, 72.90),
    "North Suburb": (19.30, 72.85),
    "Metro": (19.08, 72.89)
}

df = load_data()
weather_hist = load_historical_weather()

# Merge weather for model B
df_weather_merged = df.merge(weather_hist, how='left', left_on=['Zone','Month'], right_on=['Zone','Month'])

# Encode labels
label_encoders = {}
for col in ['Zone', 'Category']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Model A: Without weather
input_features_A = ['Connected Load', 'Zone', 'Category']
models_A = {}
months = ['May', 'Jun', 'Jul', 'August', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']

for month in months:
    X = df[input_features_A]
    y = df[month]
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    models_A[month] = model

# Model B: With weather
# Encode weather merged df
for col in ['Zone', 'Category']:
    df_weather_merged[col] = label_encoders[col].transform(df_weather_merged[col])

input_features_B = ['Connected Load', 'Zone', 'Category', 'Avg_Temp_C', 'Avg_Humidity']
models_B = {}

for month in months:
    X = df_weather_merged[input_features_B]
    y = df_weather_merged[month]
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    models_B[month] = model

# Inputs
connected_load = st.number_input("Connected Load (kW/KVA)", min_value=0.1, value=10.0)
zone = st.selectbox("Select Zone", list(zone_coords.keys()))
category = st.selectbox("Select Category", label_encoders['Category'].classes_)
month_name = st.selectbox("Select Month", months)

month_map = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'August':8, 'Sept':9, 'Oct':10, 'Nov':11, 'Dec':12}
month_num = month_map[month_name]

zone_enc = label_encoders['Zone'].transform([zone])[0]
category_enc = label_encoders['Category'].transform([category])[0]

if connected_load <= 0:
    st.error("⚠️ Please enter a valid load (positive number).")
else:
    if st.button("🔍 Predict Consumption"):
        forecast_weather = fetch_forecast_weather(zone_coords[zone][0], zone_coords[zone][1], 2025, month_num)
        if forecast_weather is None:
            st.warning("⚠️ Could not fetch forecast weather for the selected zone/month.")
        else:
            avg_temp_forecast, avg_humidity_forecast = forecast_weather

            # Historical weather for comparison
            hist_row = weather_hist[(weather_hist['Zone'] == zone) & (weather_hist['Month'] == month_name)]
            if hist_row.empty:
                avg_temp_hist = np.nan
                avg_humidity_hist = np.nan
            else:
                avg_temp_hist = hist_row.iloc[0]['Avg_Temp_C']
                avg_humidity_hist = hist_row.iloc[0]['Avg_Humidity']

            # Prediction WITHOUT weather
            input_A = np.array([[connected_load, zone_enc, category_enc]])
            pred_A = models_A[month_name].predict(input_A)[0]

            # Prediction WITH weather
            input_B = np.array([[connected_load, zone_enc, category_enc, avg_temp_forecast, avg_humidity_forecast]])
            pred_B = models_B[month_name].predict(input_B)[0]

            st.success(f"Predicted consumption without weather: {pred_A:.2f} kWh")
            st.success(f"Predicted consumption with weather: {pred_B:.2f} kWh")

            # Bar chart
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            bars = ax.bar(['Without Weather', 'With Weather'], [pred_A, pred_B], color=['skyblue', 'orange'])
            ax.set_ylabel('Predicted Consumption (kWh)')
            ax.set_title(f'Prediction Comparison for {month_name} in {zone}')
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}', ha='center', va='bottom')
            st.pyplot(fig)
