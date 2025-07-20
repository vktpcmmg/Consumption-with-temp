import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import base64

# Function to load and encode logo image for header
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
    "<h1 style='text-align: center; color: #0072C6;'>🔌 Electricity Consumption Predictor </h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: gray;'>🔷 Designed by <span style='color: #0072C6;'>Tata Power - MMG</span></h4>",
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    # Load consumption data - Sheet1
    df_cons = pd.read_csv("consumptionai.csv")
    df_cons.columns = df_cons.columns.str.strip()
    df_cons.rename(columns={'Connected  Load': 'Connected Load'}, inplace=True)
    # Load weather data - Sheet2
    df_weather = pd.read_csv("weather.csv")
    df_weather.columns = df_weather.columns.str.strip()
    # Standardize month and zone columns
    df_cons['Month'] = df_cons['Month'].str.strip()
    df_cons['Zone'] = df_cons['Zone'].str.strip()
    df_weather['Month'] = df_weather['Month'].str.strip()
    df_weather['Zone'] = df_weather['Zone'].str.strip()
    # Merge on Zone and Month
    df = pd.merge(df_cons, df_weather, on=['Zone', 'Month'], how='left')
    return df

@st.cache_resource
def train_models(df):
    # Label encode categorical columns Zone and Category
    label_encoders = {}
    for col in ['Zone', 'Category']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    input_features = ['Connected Load', 'Zone', 'Category', 'Avg_Temp_C', 'Avg_Humidity']
    months = ['May', 'Jun', 'Jul', 'August', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']

    models = {}
    for month in months:
        if month not in df.columns:
            continue
        df_month = df[df['Month'] == month]
        if df_month.empty:
            continue
        X = df_month[input_features]
        y = df_month[month]
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        models[month] = model

    return models, label_encoders

# Load data
df = load_data()

# Train models
models, label_encoders = train_models(df)

st.markdown("*_Note: Model uses historical weather data to enhance prediction accuracy._*")
st.write("Enter details to predict monthly electricity usage (kWh/KVAh).")

# User inputs
connected_load = st.number_input("Connected Load (kW/KVA)", min_value=0.0, value=10.0)
zone = st.selectbox("Select Zone", label_encoders['Zone'].classes_)
category = st.selectbox("Select Category", label_encoders['Category'].classes_)
month = st.selectbox("Select Month", ['May', 'Jun', 'Jul', 'August', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'])

# Find weather data for selected zone & month
weather_row = df[(df['Zone'] == label_encoders['Zone'].transform([zone])[0]) & (df['Month'] == month)]
if not weather_row.empty:
    avg_temp = weather_row.iloc[0]['Avg_Temp_C']
    avg_humidity = weather_row.iloc[0]['Avg_Humidity']
else:
    avg_temp = None
    avg_humidity = None

st.markdown(f"**Average Temperature for {zone} in {month}:** {avg_temp if avg_temp else 'N/A'} °C")
st.markdown(f"**Average Humidity for {zone} in {month}:** {avg_humidity if avg_humidity else 'N/A'} %")

# Encode inputs
zone_enc = label_encoders['Zone'].transform([zone])[0]
category_enc = label_encoders['Category'].transform([category])[0]

# Predict button with validation
if connected_load <= 0:
    st.error("⚠️ Please enter a valid load. Zero or negative load does not exist.")
else:
    if st.button("🔍 Predict Consumption"):
        if month not in models:
            st.error(f"Model for month {month} not available.")
        elif avg_temp is None or avg_humidity is None:
            st.error("Weather data not available for selected zone/month.")
        else:
            input_data = np.array([[connected_load, zone_enc, category_enc, avg_temp, avg_humidity]])
            prediction = models[month].predict(input_data)[0]
            st.success(f"📊 Predicted electricity consumption for **{month}**: **{prediction:.2f} kWh**")
