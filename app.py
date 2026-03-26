import streamlit as st
import pickle
import pandas as pd
import train_model.py
model = pickle.load(open("model.pkl", "rb"))

st.title("🚕 Ride Demand Predictor (India)")

hour = st.slider("Select Hour", 0, 23)
day = st.selectbox("Select Day", [1,2,3,4,5,6,7])

input_df = pd.DataFrame([[hour, day]], columns=["hour", "day"])

prediction = model.predict(input_df)[0]

if prediction == 0:
    st.success("🟢 Low Demand")
elif prediction == 1:
    st.warning("🟡 Medium Demand")
else:
    st.error("🔴 High Demand (Surge likely)")
