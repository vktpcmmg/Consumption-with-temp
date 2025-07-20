import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ⚠️ Replace with your actual GitHub raw Excel file URL
EXCEL_URL = "https://raw.githubusercontent.com/vktpcmmg/Consumption-with-temp/main/consumptiontemp.xlsx"

st.set_page_config(page_title="Electricity Consumption Predictor", layout="centered")
st.title("⚡ Electricity Consumption Predictor with Weather Comparison")

@st.cache_data
def load_data():
    df = pd.read_excel(EXCEL_URL, sheet_name='Sheet1')
    df.columns = df.columns.str.strip()
    df.rename(columns={'Connected  Load': 'Connected Load'}, inplace=True)
    return df

@st.cache_data
def load_weather():
    df = pd.read_excel(EXCEL_URL, sheet_name='Sheet2')
    df.columns = df.columns.str.strip()
    return df

@st.cache_resource
def train_models(consumption_df):
    label_encoders = {}
    for col in ["Zone", "Category"]:
        le = LabelEncoder()
        consumption_df[col] = le.fit_transform(consumption_df[col])
        label_encoders[col] = le

    input_features = ["Connected Load", "Zone", "Category"]
    months = ['May', 'Jun', 'Jul', 'August', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    models = {}

    for month in months:
        X = consumption_df[input_features]
        y = consumption_df[month]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        models[month] = model

    return models, label_encoders

# Load data
consumption_df = load_data()
weather_df = load_weather()
models, label_encoders = train_models(consumption_df)

# UI Inputs
st.subheader("Enter Details to Predict Consumption")
connected_load = st.number_input("Connected Load (kW/KVA)", min_value=0.1, value=10.0)
zone = st.selectbox("Zone", label_encoders["Zone"].classes_)
category = st.selectbox("Category", label_encoders["Category"].classes_)
month = st.selectbox("Month", ['May', 'Jun', 'Jul', 'August', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'])

zone_enc = label_encoders["Zone"].transform([zone])[0]
category_enc = label_encoders["Category"].transform([category])[0]

if st.button("📊 Predict Consumption"):
    input_data = np.array([[connected_load, zone_enc, category_enc]])
    prediction = models[month].predict(input_data)[0]
    st.success(f"✅ Predicted Consumption for **{month}**: **{prediction:.2f} kWh**")

    # Weather Comparison
    weather_row = weather_df[(weather_df["Zone"] == zone) & (weather_df["Month"] == month)]
    if not weather_row.empty:
        hist_temp = weather_row["Historical Temp"].values[0]
        forecast_temp = weather_row["Forecasted Temp"].values[0]
        delta = forecast_temp - hist_temp
        st.info(f"""
        🌡️ **{month} - {zone} Zone Weather**
        - Historical Temp: **{hist_temp}°C**
        - Forecasted Temp: **{forecast_temp}°C**
        - Δ Change: **{delta:+.2f}°C**
        """)
    else:
        st.warning("⚠️ Weather data not found for selected Zone & Month.")
