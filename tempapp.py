import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="centered")
st.image("tata_logo.png", width=350)
st.title("Electricity Consumption Predictor with Weather AI")

# Load consumption and weather data from local files
@st.cache_data
def load_data():
    df_cons = pd.read_csv("consumptionai.csv")
    df_weather = pd.read_csv("weather.csv")
    
    df_cons.columns = df_cons.columns.str.strip()
    df_weather.columns = df_weather.columns.str.strip()

    df_cons.rename(columns={'Connected  Load': 'Connected Load'}, inplace=True)
    df_weather.rename(columns={"Avg_Temp_C": "Temp", "Avg_Humidity": "Humidity"}, inplace=True)

    return df_cons, df_weather

df_cons, df_weather = load_data()

# User inputs
st.sidebar.header("Enter Details")
category = st.sidebar.selectbox("Category", sorted(df_cons['Category'].unique()))
zone = st.sidebar.selectbox("Zone", sorted(df_cons['Zone'].unique()))
connected_load = st.sidebar.number_input("Connected Load (kW)", min_value=1.0, step=0.5)
month = st.sidebar.selectbox("Month", ["May", "Jun", "Jul", "August", "Sept", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"])

# Filter for training based on similar weather history
def get_similar_weather_data(df_weather, target_zone, target_month):
    target_weather = df_weather[(df_weather['Zone'] == target_zone) & (df_weather['Month'] == month)]
    if target_weather.empty:
        return df_weather  # fallback

    temp, humid = target_weather.iloc[0]["Temp"], target_weather.iloc[0]["Humidity"]
    
    # Compute similarity (euclidean distance)
    df_weather["similarity"] = ((df_weather["Temp"] - temp) ** 2 + (df_weather["Humidity"] - humid) ** 2) ** 0.5
    similar_zones = df_weather.sort_values("similarity").head(6)["Zone"].unique()
    return similar_zones

# Training
def train_model(month):
    df_filtered = df_cons[df_cons['Category'] == category]
    df_filtered = df_filtered[df_filtered['Zone'].isin(get_similar_weather_data(df_weather, zone, month))]

    X = df_filtered[["Connected Load"]]
    y = df_filtered[month]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

if st.sidebar.button("Predict Consumption"):
    model = train_model(month)
    prediction = model.predict(np.array([[connected_load]]))[0]
    st.success(f"Predicted Consumption for {month} FY24-25: **{prediction:.2f} kWh**")

    # Plot
    fig, ax = plt.subplots()
    ax.bar([month], [prediction], color='skyblue')
    ax.set_ylabel("Consumption (kWh)")
    ax.set_title("Predicted Monthly Consumption")
    st.pyplot(fig)

    st.caption("Model trained using similar weather patterns across zones.")
