import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# App title and logo
st.set_page_config(page_title="Weather-Aware Consumption AI", layout="centered")
st.image("tata_logo.png", width=350)
st.markdown("<h3 style='text-align: center;'>Electricity Consumption Prediction with Weather Intelligence</h3>", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("consumptionai.csv")
    df.columns = df.columns.str.strip()
    df.rename(columns={'Connected  Load': 'Connected Load'}, inplace=True)
    
    weather_df = pd.read_csv("weather.csv")
    weather_df.columns = weather_df.columns.str.strip()
    return df, weather_df

df_cons, df_weather = load_data()

# Melt consumption data to long format
df_long = df_cons.melt(
    id_vars=["meter_number", "Category", "Connected Load", "Zone", "District"],
    var_name="Month",
    value_name="Consumption"
)

# Merge with weather data
df_merged = pd.merge(
    df_long,
    df_weather,
    on=["Zone", "Month"],
    how="left"
)

# Encode Category
df_merged['Category'] = df_merged['Category'].astype('category')
df_merged['Cat_Code'] = df_merged['Category'].cat.codes

# Train model
features = df_merged[["Connected Load", "Zone", "Month", "Cat_Code", "Avg_Temp_C", "Avg_Humidity"]]
features = pd.get_dummies(features, columns=["Zone", "Month"], drop_first=True)
target = df_merged["Consumption"]

model = RandomForestRegressor(random_state=42)
model.fit(features, target)

# Sidebar input block centered using columns
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.subheader("Enter Details to Predict Consumption")
    load = st.number_input("Connected Load (kW)", min_value=1.0, step=0.5)
    category = st.selectbox("Category", df_merged['Category'].unique())
    zone = st.selectbox("Zone", sorted(df_merged['Zone'].unique()))
    month = st.selectbox("Month", sorted(df_merged['Month'].unique()))

# Get weather values for selected Zone & Month
weather_row = df_weather[(df_weather["Zone"] == zone) & (df_weather["Month"] == month)]
if not weather_row.empty:
    avg_temp = weather_row["Avg_Temp_C"].values[0]
    avg_humidity = weather_row["Avg_Humidity"].values[0]
else:
    avg_temp = avg_humidity = 0  # fallback

# Prepare input for prediction
input_dict = {
    "Connected Load": load,
    "Cat_Code": df_merged[df_merged['Category'] == category]["Cat_Code"].iloc[0],
    "Avg_Temp_C": avg_temp,
    "Avg_Humidity": avg_humidity,
}
# Add dummy variables
for z in sorted(df_merged['Zone'].unique()):
    input_dict[f"Zone_{z}"] = 1 if z == zone else 0
for m in sorted(df_merged['Month'].unique())[1:]:  # drop_first=True
    input_dict[f"Month_{m}"] = 1 if m == month else 0

input_df = pd.DataFrame([input_dict])

# Predict
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Predict Consumption (kWh)"):
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Consumption: **{prediction:.2f} kWh**")

