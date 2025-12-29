import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Flight On Time Prediction", layout="centered")


@st.cache_resource
def load_encoders():
    return joblib.load("models/label_encoders_v7.pkl")


@st.cache_resource
def load_model():
    return joblib.load("models/randomforest_v7_final.pkl")


encoders = load_encoders()
model = load_model()

st.title("Flight On Time Prediction")

st.write(
    "Preencha os dados abaixo para prever se o voo chegará no horário ou atrasado."
)

airline = st.selectbox(
    'Airline (Codigo da companhia aérea, ex: "AA" para American Airlines)',
    options=("AA", "DL", "UA", "WN", "AS", "B6", "F9", "NK", "HA", "G4"),
)


airport_options = (
    "ATL",
    "DEN",
    "DFW",
    "ORD",
    "CLT",
    "PHX",
    "LAX",
    "LAS",
    "MCO",
    "SEA",
    "LGA",
    "IAH",
    "DCA",
    "EWR",
    "BOS",
    "SFO",
    "MIA",
    "DTW",
    "JFK",
    "MSP",
)

origin = st.selectbox(
    "Origin (Código do aeroporto de origem)",
    options=airport_options,
)

destination = st.selectbox(
    "Destination (Código do aeroporto de destino)",
    options=airport_options,
)

month = st.number_input("Mês (1-12)", min_value=1, max_value=12, step=1)

day_of_week = st.number_input(
    "Day of Week (1=Monday, 7=Sunday)", min_value=1, max_value=7, step=1
)

crsdep_time = st.number_input(
    "CRS Departure Time (Formato HHMM, ex: 1330)", min_value=0, max_value=2359, step=1
)
distance = st.number_input("Distance (Distância do voo em milhas)", min_value=1, step=1)

if st.button("Predict"):
    X = pd.DataFrame(
        {
            "Airline": [airline],
            "Origin": [origin],
            "Dest": [destination],
            "Month": [month],
            "DayOfWeek": [day_of_week],
            "CRSDepTime": [crsdep_time],
            "Distance": [distance],
        }
    )

    X["dephour"] = (X["CRSDepTime"] // 100).clip(0, 23).astype("int8")
    X["is_weekend"] = X["DayOfWeek"].isin([6, 7]).astype("int8")
    X["quarter"] = ((X["Month"] - 1) // 3 + 1).astype("int8")
    X["time_of_day"] = X["dephour"].apply(
        lambda h: "Morning"
        if 6 <= h < 12
        else "Afternoon"
        if 12 <= h < 18
        else "Evening"
        if 18 <= h < 22
        else "Night"
    )

    X["origin_delay_rate"] = 0.23
    X["carrier_delay_rate"] = 0.23
    X["origin_traffic"] = 0

    for col in ["Airline", "Origin", "Dest", "time_of_day"]:
        X[col] = encoders[col].transform(X[col].astype(str))

    pred = model.predict(
        X[
            [
                "Month",
                "DayOfWeek",
                "dephour",
                "is_weekend",
                "quarter",
                "Distance",
                "origin_delay_rate",
                "carrier_delay_rate",
                "origin_traffic",
                "Airline",
                "Origin",
                "Dest",
                "time_of_day",
            ]
        ]
    )
    if pred[0] == 0:
        st.balloons()
        st.success("ON TIME ✅")
    else:
        st.warning("DELAYED ⚠️")
