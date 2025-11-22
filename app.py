import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('model.pkl')

st.title("🌍 EcoScan - Carbon Footprint Estimator")
st.write("Enter your lifestyle details to estimate your carbon footprint.")

def user_inputs():
    body_type = st.selectbox("Body Type", ["Underweight", "Normal", "Overweight", "Obese"])
    sex = st.selectbox("Sex", ["Male", "Female"])
    diet = st.selectbox("Diet", ["Vegan", "Vegetarian", "Pescatarian", "Omnivore"])
    shower_freq = st.number_input("How often do you shower per week?", 0, 50, 7)
    heating = st.selectbox("Heating Energy Source", ["Coal", "Wood", "Natural Gas", "Electricity"])
    transport = st.selectbox("Transport Mode", ["Public", "Private", "Walk/Bicycle"])
    vehicle_type = st.selectbox("Vehicle Type", ["None", "Diesel", "Petrol", "Electric", "Hybrid"])
    distance = st.number_input("Vehicle Monthly Distance (km)", 0, 10000, 100)
    grocery = st.number_input("Monthly Grocery Bill (₹)", 0, 50000, 2000)
    air_travel = st.number_input("Flights per year", 0, 50, 0)
    waste_size = st.selectbox("Waste Bag Size", ["Small", "Medium", "Large"])
    waste_count = st.number_input("Weekly Waste Bags", 0, 50, 2)
    screen_time = st.number_input("Daily Screen Time (hours)", 0, 24, 4)
    clothes = st.number_input("Clothes Bought Per Month", 0, 50, 2)
    internet = st.number_input("Daily Internet Usage (hours)", 0, 24, 5)
    energy_eff = st.selectbox("Energy Efficient Home?", ["Yes", "No", "Sometimes"])

    paper = st.selectbox("Recycle Paper?", ["Yes", "No"])
    plastic = st.selectbox("Recycle Plastic?", ["Yes", "No"])
    metal = st.selectbox("Recycle Metal?", ["Yes", "No"])
    glass = st.selectbox("Recycle Glass?", ["Yes", "No"])

    cooking = st.selectbox("Cooking Method", ["Gas", "Electric", "Induction", "Wood"])

    data = pd.DataFrame({
        'BodyType':[body_type],
        'Sex':[sex],
        'Diet':[diet],
        'HowOftenShower':[shower_freq],
        'HeatingEnergySource':[heating],
        'Transport':[transport],
        'VehicleType':[vehicle_type],
        'VehicleMonthlyDistance':[distance],
        'MonthlyGroceryBill':[grocery],
        'AirTravel':[air_travel],
        'WasteBagSize':[waste_size],
        'WasteBagWeeklyCount':[waste_count],
        'DailyTVPCuse':[screen_time],
        'MonthlyClothesBought':[clothes],
        'DailyInternetUsage':[internet],
        'EnergyEfficiency':[energy_eff],
        'RecyclePaper':[paper],
        'RecyclePlastic':[plastic],
        'RecycleMetal':[metal],
        'RecycleGlass':[glass],
        'Cooking':[cooking]
    })
    return data

input_df = user_inputs()

if st.button("Estimate Carbon Footprint"):
    try:
        result = model.predict(input_df)[0]
        st.success(f"Your estimated carbon footprint: **{result:.2f} kg CO₂/month**")
    except Exception as e:
        st.error("Error predicting. Check input format.")
        st.write(e)


