import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Electricity Consumption Predictor", page_icon="🔌", layout="centered")

# Load model
model = joblib.load("consumption_model.pkl")  # replace with your actual model path if different

# Load data
df = pd.read_csv("consumptionai.csv")
weather_df = pd.read_csv("weather.csv")

# Mapping of month names
month_mapping = {
    'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12,
    'January': 1, 'February': 2, 'March': 3
}

# Title and Header
st.markdown("<h1 style='text-align: center; color: #0072C6;'>🔌 Electricity Consumption Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>🔷 Designed by <b><span style='color:#0072C6;'>Tata Power - MMG</span></b></h5>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic;'>Note: This is based on around 100K smart meter data from FY 24–25.</p>", unsafe_allow_html=True)

st.markdown("### Enter details to predict monthly electricity usage (kWh/KVAh)")

# Centered input section
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    connected_load = st.number_input("Connected Load (kW/KVA)", min_value=0.0, step=0.1, format="%.2f")
    zone = st.selectbox("Select Zone", df['Zone'].unique())
    category = st.selectbox("Select Category", df['Category'].unique())
    month = st.selectbox("Select Month", list(month_mapping.keys()))

    if st.button("🔍 Predict Consumption"):
        if model:
            input_df = pd.DataFrame({
                'Connected Load': [connected_load],
                'Category': [category],
                'Zone': [zone],
                'Month': [month]
            })
            prediction = model.predict(input_df)[0]
            st.success(f"✅ Predicted Monthly Consumption for {month}: **{prediction:.2f} kWh/KVAh**")
        else:
            st.error("⚠️ Model not loaded. Please check the backend.")
