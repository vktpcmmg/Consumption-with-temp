import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import base64
from openai import OpenAI
import requests

# ------------ LOGO DISPLAY ------------
def get_image_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = get_image_base64("tata_logo.png")

st.markdown(f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{logo_base64}" width="100">
    </div>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #0072C6;'>🔌 Electricity Consumption Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>🔷 Designed by <span style='color: #0072C6;'>Tata Power - MMG</span></h4>", unsafe_allow_html=True)
st.markdown("*_Note: Based on Smart Meter data (FY 24–25) & Weather Comparison_*")

# ------------ FILE UPLOAD AND LOADING ------------
uploaded_file = st.file_uploader("Upload CSV file with 2 Sheets (Sheet1: ConsumptionAI, Sheet2: Historical Weather)", type=["csv", "xlsx"])
if uploaded_file is not None:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, sheet_name="Sheet1")
        weather_hist = pd.read_excel(uploaded_file, sheet_name="Sheet2")
    else:
        st.error("⚠️ Please upload an Excel file with 2 sheets.")

    df.columns = df.columns.str.strip()
    df.rename(columns={'Connected  Load': 'Connected Load'}, inplace=True)

    # ------------ LABEL ENCODING ------------
    label_encoders = {}
    for col in ['Zone', 'Category']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    input_features = ['Connected Load', 'Zone', 'Category', 'Temp Anomaly', 'Humidity Anomaly']
    months = ['May', 'Jun', 'Jul', 'August', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    models = {}

    # ------------ MERGE HISTORICAL WEATHER ANOMALY PLACEHOLDER ------------
    for month in months:
        df['Temp Anomaly'] = 0.0
        df['Humidity Anomaly'] = 0.0
        X = df[input_features]
        y = df[month]
        model = RandomForestRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)
        models[month] = model

    # ------------ USER INPUTS ------------
    st.write("### Enter Details to Predict Consumption")
    connected_load = st.number_input("Connected Load (kW/KVA)", min_value=0.0, value=10.0)
    zone = st.selectbox("Select Zone", label_encoders['Zone'].classes_)
    category = st.selectbox("Select Category", label_encoders['Category'].classes_)
    month = st.selectbox("Select Month", months)
    lat = st.number_input("Latitude of Zone", value=19.0, format="%.4f")
    lon = st.number_input("Longitude of Zone", value=72.0, format="%.4f")

    # ------------ WEATHER FORECAST FETCH ------------
    def fetch_forecast(lat, lon, month_name):
        import datetime
        month_num = datetime.datetime.strptime(month_name[:3], "%b").month
        year = 2025
        start = f"{year}-{month_num:02d}-01"
        end = f"{year}-{month_num:02d}-28"

        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
            f"&daily=temperature_2m_mean,relativehumidity_2m_mean&timezone=auto"
        )

        res = requests.get(url)
        data = res.json()
        daily_temps = data['daily']['temperature_2m_mean']
        daily_humidity = data['daily']['relativehumidity_2m_mean']
        return np.mean(daily_temps), np.mean(daily_humidity)

    if connected_load <= 0:
        st.error("⚠️ Please enter a valid load.")
    else:
        if st.button("🔍 Predict Consumption"):
            zone_enc = label_encoders['Zone'].transform([zone])[0]
            category_enc = label_encoders['Category'].transform([category])[0]

            forecast_temp, forecast_humid = fetch_forecast(lat, lon, month)

            # Historical monthly weather
            month_weather = weather_hist[weather_hist['Month'] == month]
            zone_weather = month_weather[month_weather['Zone'] == zone]
            if zone_weather.empty:
                st.warning("⚠️ No historical weather data for this zone/month. Proceeding without weather anomaly.")
                temp_anomaly = 0.0
                humid_anomaly = 0.0
            else:
                hist_temp = zone_weather['temperature_2m_mean'].values[0]
                hist_humid = zone_weather['relativehumidity_2m_mean'].values[0]
                temp_anomaly = forecast_temp - hist_temp
                humid_anomaly = forecast_humid - hist_humid

            input_data = np.array([[connected_load, zone_enc, category_enc, temp_anomaly, humid_anomaly]])
            prediction = models[month].predict(input_data)[0]
            st.success(f"📊 Predicted electricity consumption for **{month}**: **{prediction:.2f} kWh**")
