import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Tata Power AI Model", layout="centered")

st.image("https://upload.wikimedia.org/wikipedia/en/thumb/6/64/Tata_Power_Logo.svg/2560px-Tata_Power_Logo.svg.png", width=200)

st.title("Electricity Consumption Prediction")
st.markdown("Upload includes: `consumptionai.csv` (Sheet1) & `weather.csv` (Sheet2). Prediction uses weather + load + location.")

@st.cache_data
def load_data():
    df_cons = pd.read_csv("consumptionai.csv")
    df_weather = pd.read_csv("weather.csv")
    
    # Clean columns
    df_cons.columns = df_cons.columns.str.strip()
    df_cons.rename(columns={"Connected Load": "Connected Load"}, inplace=True)
    df_weather.columns = df_weather.columns.str.strip()

    # Strip text
    df_cons['Zone'] = df_cons['Zone'].astype(str).str.strip()
    df_weather['Zone'] = df_weather['Zone'].astype(str).str.strip()
    df_weather['Month'] = df_weather['Month'].astype(str).str.strip()

    return df_cons, df_weather

df_cons, df_weather = load_data()

# --- Sidebar Inputs ---
st.sidebar.header("Input Parameters")

zone = st.sidebar.selectbox("Select Zone", sorted(df_cons["Zone"].dropna().unique()))
district = st.sidebar.selectbox("Select District", sorted(df_cons[df_cons["Zone"] == zone]["District"].dropna().unique()))
category = st.sidebar.selectbox("Select Category", sorted(df_cons["Category"].dropna().unique()))
month = st.sidebar.selectbox("Select Month", ['May','Jun','Jul','August','Sept','Oct','Nov','Dec','Jan','Feb','Mar','Apr'])
connected_load = st.sidebar.number_input("Connected Load (kW)", min_value=1.0, max_value=1000.0, value=10.0)

# --- Weather for selected zone+month ---
weather_row = df_weather[(df_weather["Zone"] == zone) & (df_weather["Month"] == month)]
if not weather_row.empty:
    temperature = weather_row.iloc[0]["Temperature"]
    humidity = weather_row.iloc[0]["Humidity"]
else:
    temperature = 30  # default fallback
    humidity = 50

# --- Prepare Data for Model ---
def train_model(month):
    df_train = df_cons[["Category", "Connected Load", "Zone", "District", month]].copy()
    df_train = df_train.dropna()

    df_train["Temperature"] = df_train["Zone"].map(
        lambda z: df_weather[(df_weather["Zone"] == z) & (df_weather["Month"] == month)]["Temperature"].values[0]
        if not df_weather[(df_weather["Zone"] == z) & (df_weather["Month"] == month)].empty else 30
    )

    df_train["Humidity"] = df_train["Zone"].map(
        lambda z: df_weather[(df_weather["Zone"] == z) & (df_weather["Month"] == month)]["Humidity"].values[0]
        if not df_weather[(df_weather["Zone"] == z) & (df_weather["Month"] == month)].empty else 50
    )

    X = df_train[["Connected Load", "Temperature", "Humidity"]]
    y = df_train[month]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(month)

# --- Prediction ---
input_data = pd.DataFrame({
    "Connected Load": [connected_load],
    "Temperature": [temperature],
    "Humidity": [humidity]
})

predicted_consumption = model.predict(input_data)[0]

# --- Display ---
st.subheader("Prediction Result")
st.markdown(f"""
**Zone**: {zone}  
**District**: {district}  
**Category**: {category}  
**Month**: {month}  
**Connected Load**: {connected_load} kW  
**Temperature**: {temperature} °C  
**Humidity**: {humidity} %

### 🔮 Predicted Consumption: `{predicted_consumption:.2f}` units
""")
