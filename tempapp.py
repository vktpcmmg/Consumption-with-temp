import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import base64

# --- Logo Setup ---
def get_image_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = get_image_base64("tata_logo.png")

st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{logo_base64}" width="100">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='text-align: center; color: #0072C6;'>🔌 Electricity Consumption Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: gray;'>🔷 Designed by <span style='color: #0072C6;'>Tata Power - MMG</span></h4>",
    unsafe_allow_html=True
)

# --- GitHub Raw CSV URLs ---
CONSUMPTION_CSV_URL = "https://raw.githubusercontent.com/your-username/your-repo/main/consumptionai.csv"
WEATHER_CSV_URL = "https://raw.githubusercontent.com/your-username/your-repo/main/weather_data.csv"

# --- Load and preprocess data ---
@st.cache_data
def load_data():
    df = pd.read_csv(CONSUMPTION_CSV_URL)
    weather = pd.read_csv(WEATHER_CSV_URL)

    df.columns = df.columns.str.strip()
    df.rename(columns={'Connected  Load': 'Connected Load'}, inplace=True)
    weather.columns = weather.columns.str.strip()

    return df, weather

# --- Train models ---
@st.cache_resource
def train_models(df):
    label_encoders = {}
    for col in ['Zone', 'Category']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    input_features = ['Connected Load', 'Zone', 'Category', 'Temp Delta']
    months = ['May', 'Jun', 'Jul', 'August', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    models = {}

    for month in months:
        temp_col = f'{month}_Delta'
        if temp_col not in df.columns:
            df[temp_col] = 0
        X = df[['Connected Load', 'Zone', 'Category']]
        X['Temp Delta'] = df[temp_col]
        y = df[month]
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        models[month] = model

    return models, label_encoders

# --- UI ---
st.markdown("*_Note: Based on smart meter and weather data (FY 24–25)._*\n")
st.write("Enter details to predict monthly electricity usage (kWh/KVAh).")

# --- Load data ---
df, weather = load_data()
models, label_encoders = train_models(df)

# --- User input ---
connected_load = st.number_input("Connected Load (kW/KVA)", min_value=0.0, value=10.0)
zone = st.selectbox("Select Zone", label_encoders['Zone'].classes_)
category = st.selectbox("Select Category", label_encoders['Category'].classes_)
month = st.selectbox("Select Month", ['May', 'Jun', 'Jul', 'August', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'])

# --- Weather Delta ---
zone_weather = weather[weather['Zone'] == zone]
hist_temp = zone_weather[zone_weather['Type'] == 'Historical'][month].values[0]
forecast_temp = zone_weather[zone_weather['Type'] == 'Forecast'][month].values[0]
temp_delta = forecast_temp - hist_temp

# --- Encode inputs ---
zone_enc = label_encoders['Zone'].transform([zone])[0]
category_enc = label_encoders['Category'].transform([category])[0]

# --- Predict button ---
if connected_load <= 0:
    st.error("⚠️ Please enter a valid connected load.")
else:
    if st.button("🔍 Predict Consumption"):
        input_data = np.array([[connected_load, zone_enc, category_enc, temp_delta]])
        prediction = models[month].predict(input_data)[0]
        st.success(f"📊 Predicted consumption for **{month}**: **{prediction:.2f} kWh**")
        st.info(f"🌡️ Compared to average, temperature difference this month is **{temp_delta:+.1f}°C**.")
