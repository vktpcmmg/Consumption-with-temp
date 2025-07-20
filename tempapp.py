import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import requests
import base64
import datetime
import matplotlib.pyplot as plt

# --- Helper functions ---
def get_image_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def fetch_forecast_weather(lat, lon, year, month):
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

# Coordinates for each zone (modify if needed)
zone_coords = {
    "City South": (19.07, 72.88),
    "Urban": (19.13, 72.90),
    "West Suburb": (19.20, 72.82),
    "East Suburb": (19.22, 72.90),
    "North Suburb": (19.30, 72.85),
    "Metro": (19.08, 72.89)
}

# --- Load & preprocess data ---
@st.cache_data
def load_and_prepare_data():
    # Load consumption wide data
    df = pd.read_csv("consumptionai.csv")
    df.columns = df.columns.str.strip()
    df.rename(columns={'Connected  Load': 'Connected Load'}, inplace=True)

    # Melt months into 'Month' and 'Consumption'
    month_cols = ['May', 'Jun', 'Jul', 'August', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    df_long = df.melt(
        id_vars=['meter_number', 'Category', 'Connected Load', 'Zone', 'District'],
        value_vars=month_cols,
        var_name='Month',
        value_name='Consumption'
    )

    # Load weather data
    weather = pd.read_csv("weather.csv")
    weather.columns = weather.columns.str.strip()

    # Strip spaces in key columns
    df_long['Zone'] = df_long['Zone'].str.strip()
    df_long['Month'] = df_long['Month'].str.strip()
    weather['Zone'] = weather['Zone'].str.strip()
    weather['Month'] = weather['Month'].str.strip()

    # Merge consumption + weather
    df_merged = pd.merge(df_long, weather, how='left', on=['Zone', 'Month'])

    return df_merged

df = load_and_prepare_data()

# Encode categorical columns for modeling
@st.cache_resource
def encode_and_train_models(df):
    label_encoders = {}
    for col in ['Zone', 'Category', 'Month']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Model A input: Connected Load, Zone, Category, Month (encoded)
    input_features_A = ['Connected Load', 'Zone', 'Category', 'Month']

    # Model B input: Connected Load, Zone, Category, Month + weather features
    input_features_B = ['Connected Load', 'Zone', 'Category', 'Month', 'Avg_Temp_C', 'Avg_Humidity']

    models = {}

    # Train Model A
    X_A = df[input_features_A]
    y = df['Consumption']
    model_A = RandomForestRegressor(n_estimators=20, random_state=42)
    model_A.fit(X_A, y)

    # Train Model B
    # Fill missing weather values if any (simple fill with mean)
    df['Avg_Temp_C'] = df['Avg_Temp_C'].fillna(df['Avg_Temp_C'].mean())
    df['Avg_Humidity'] = df['Avg_Humidity'].fillna(df['Avg_Humidity'].mean())
    X_B = df[input_features_B]
    model_B = RandomForestRegressor(n_estimators=20, random_state=42)
    model_B.fit(X_B, y)

    models['A'] = model_A
    models['B'] = model_B

    return models, label_encoders

models, label_encoders = encode_and_train_models(df)

# --- Streamlit UI ---
logo_base64 = get_image_base64("tata_logo.png")
st.markdown(
    f"""<div style="text-align: center;">
        <img src="data:image/png;base64,{logo_base64}" width="100"></div>""",
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align: center; color: #0072C6;'>🔌 Electricity Consumption Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>🔷 Designed by <span style='color: #0072C6;'>Tata Power - MMG</span></h4>", unsafe_allow_html=True)
st.markdown("*_Note: Based on ~100K smart meter data for FY 24–25 + live weather forecast._*")

connected_load = st.number_input("Connected Load (kW/KVA)", min_value=0.1, value=10.0)

# Show zone names decoded from label encoder
zone_names = label_encoders['Zone'].inverse_transform(np.arange(len(label_encoders['Zone'].classes_)))
zone = st.selectbox("Select Zone", zone_names)

category_names = label_encoders['Category'].inverse_transform(np.arange(len(label_encoders['Category'].classes_)))
category = st.selectbox("Select Category", category_names)

month_names = label_encoders['Month'].inverse_transform(np.arange(len(label_encoders['Month'].classes_)))
month = st.selectbox("Select Month", month_names)

# Encode inputs for model
zone_enc = label_encoders['Zone'].transform([zone])[0]
category_enc = label_encoders['Category'].transform([category])[0]
month_enc = label_encoders['Month'].transform([month])[0]

if connected_load <= 0:
    st.error("⚠️ Please enter a valid positive load.")
else:
    if st.button("🔍 Predict Consumption"):

        # Predict WITHOUT weather (Model A)
        input_A = np.array([[connected_load, zone_enc, category_enc, month_enc]])
        pred_A = models['A'].predict(input_A)[0]

        # Fetch live weather forecast for selected zone and month
        if zone in zone_coords:
            lat, lon = zone_coords[zone]
            month_num = datetime.datetime.strptime(month, "%b").month if len(month) == 3 else datetime.datetime.strptime(month, "%B").month
            forecast = fetch_forecast_weather(lat, lon, 2025, month_num)
            if forecast is None:
                st.warning("⚠️ Could not fetch live weather forecast. Showing prediction without weather.")
                pred_B = pred_A
            else:
                avg_temp_forecast, avg_humidity_forecast = forecast
                # Predict WITH weather (Model B)
                input_B = np.array([[connected_load, zone_enc, category_enc, month_enc, avg_temp_forecast, avg_humidity_forecast]])
                pred_B = models['B'].predict(input_B)[0]
        else:
            st.warning("Zone coordinates not found for weather forecast. Showing prediction without weather.")
            pred_B = pred_A

        st.success(f"Predicted consumption WITHOUT weather: {pred_A:.2f} kWh")
        st.success(f"Predicted consumption WITH weather: {pred_B:.2f} kWh")

        # Plot bar chart comparison
        fig, ax = plt.subplots()
        bars = ax.bar(['Without Weather', 'With Weather'], [pred_A, pred_B], color=['skyblue', 'orange'])
        ax.set_ylabel('Predicted Consumption (kWh)')
        ax.set_title(f'Prediction Comparison for {month} in {zone}')
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.2f}', ha='center')
        st.pyplot(fig)
