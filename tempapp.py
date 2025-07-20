import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

st.set_page_config(page_title="Electricity Consumption Predictor with Weather", layout="wide")
st.title("🔌 Electricity Consumption Forecast with Weather Impact")
st.markdown("Built by **Tata Power**")

# --- Load Excel with 2 Sheets ---
@st.cache_data
def load_data():
    df_all = pd.read_excel("consumptiontemp.xlsx", sheet_name=None)
    df1 = df_all['Sheet1']
    df2 = df_all['Sheet2']
    
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    df1.rename(columns={'Connected  Load': 'Connected Load'}, inplace=True)
    df2.rename(columns={'Connected  Load': 'Connected Load'}, inplace=True)

    return df1, df2

df1, df2 = load_data()

# --- Train Models ---
@st.cache_data
def train_models(df1, df2):
    label_encoders = {}
    categorical_cols = ['Zone', 'Category', 'Month']

    for col in categorical_cols:
        if col in df2.columns:
            le = LabelEncoder()
            df2[col] = le.fit_transform(df2[col])
            label_encoders[col] = le
        else:
            st.warning(f"Column '{col}' missing in df2")

    models = {}
    for zone in df2['Zone'].unique():
        zone_df = df1[df1['Zone'] == zone]
        if not zone_df.empty:
            X = zone_df[['Connected Load', 'Month']]
            y = zone_df['Consumption']
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            models[zone] = model

    return models, label_encoders

models, label_encoders = train_models(df1.copy(), df2.copy())

# --- User Inputs ---
st.sidebar.header("🔧 Prediction Inputs")
zone = st.sidebar.selectbox("Select Zone", df1['Zone'].unique())
category = st.sidebar.selectbox("Select Category", df1['Category'].unique())
month = st.sidebar.selectbox("Select Month", df1['Month'].unique())
connected_load = st.sidebar.number_input("Connected Load (kW)", min_value=1.0, step=1.0)

# --- Predict ---
def make_prediction(zone, category, month, connected_load):
    if zone in models:
        zone_model = models[zone]
        le_month = label_encoders['Month']
        month_encoded = le_month.transform([month])[0]

        X_new = np.array([[connected_load, month_encoded]])
        prediction = zone_model.predict(X_new)[0]
        return prediction
    else:
        st.error(f"No model trained for zone: {zone}")
        return None

predicted_consumption = make_prediction(zone, category, month, connected_load)

# --- Show Prediction ---
if predicted_consumption:
    st.success(f"📈 **Predicted Consumption:** {predicted_consumption:.2f} kWh")

# --- Weather Impact Chart ---
st.subheader("🌦️ Weather Forecast vs Historical")

# Filter weather data
df2_filtered = df2[(df2['Zone'] == zone) & (df2['Category'] == category) & (df2['Month'] == label_encoders['Month'].transform([month])[0])]

if not df2_filtered.empty:
    fig, ax = plt.subplots()
    ax.plot(df2_filtered['Temperature'], label='Forecast Temp (°C)', marker='o')
    ax.plot(df2_filtered['Historical Temperature'], label='Historical Temp (°C)', linestyle='--', marker='x')
    ax.set_title(f"Temperature Comparison - {zone} - {month}")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("No weather data found for the selected filters.")
