import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import base64

# Load logo
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

# Cache data loading
@st.cache_data
def load_data():
    df = pd.read_excel("consumptiontemp.xlsx", sheet_name=None)
    df1 = df["Sheet1"]
    df2 = df["Sheet2"]
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    df1.rename(columns={'Connected  Load': 'Connected Load'}, inplace=True)
    return df1, df2

@st.cache_resource
def train_models(df1, df2):
    label_encoders = {}
    for col in ['Zone', 'Category']:
        le = LabelEncoder()
        df1[col] = le.fit_transform(df1[col])
        df2[col] = le.transform(df2[col])
        label_encoders[col] = le

    input_features = ['Connected Load', 'Zone', 'Category', 'Max Temp', 'Min Temp']
    months = ['May', 'Jun', 'Jul', 'August', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    models = {}

    df1 = pd.merge(df1, df2, on=['Zone', 'Category', 'Month'], how='left')

    for month in months:
        df_month = df1[df1['Month'] == month]
        X = df_month[input_features]
        y = df_month[month]
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        models[month] = model

    return models, label_encoders

st.markdown("*_Note: This is based on smart meter and historical weather data from FY 24–25._*")
st.write("Enter details to predict monthly electricity usage (kWh/KVAh) with weather insights.")

# Load data and train models
df1, df2 = load_data()
models, label_encoders = train_models(df1, df2)

# User inputs
connected_load = st.number_input("Connected Load (kW/KVA)", min_value=0.0, value=10.0)
zone = st.selectbox("Select Zone", label_encoders['Zone'].classes_)
category = st.selectbox("Select Category", label_encoders['Category'].classes_)
month = st.selectbox("Select Month", ['May', 'Jun', 'Jul', 'August', 'Sept', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'])

# Fetch weather for selected month from historical data
selected_weather = df2[(df2['Zone'] == zone) & (df2['Category'] == category) & (df2['Month'] == month)]
if not selected_weather.empty:
    max_temp = selected_weather['Max Temp'].values[0]
    min_temp = selected_weather['Min Temp'].values[0]
else:
    max_temp = min_temp = 30.0  # default values

# Encode inputs
zone_enc = label_encoders['Zone'].transform([zone])[0]
category_enc = label_encoders['Category'].transform([category])[0]

if connected_load <= 0:
    st.error("⚠️ Please enter a valid load. Zero or negative load does not exist.")
else:
    if st.button("🔍 Predict Consumption"):
        input_data = np.array([[connected_load, zone_enc, category_enc, max_temp, min_temp]])
        prediction = models[month].predict(input_data)[0]
        st.success(f"📊 Predicted electricity consumption for **{month}**: **{prediction:.2f} kWh**")
