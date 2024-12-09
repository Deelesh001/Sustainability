import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained Linear Regression model
@st.cache_resource
def load_model():
    with open("sustainability_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# App title
st.title("Dummy Vehicle Sustainability Predictor")

# Sidebar for user input
st.sidebar.header("Input Vehicle Attributes")

# Collect user input
def user_input():
    vehicle_mass = st.sidebar.number_input("Vehicle Mass (kg)", min_value=500.0, max_value=5000.0, step=100.0, value=1500.0)
    emissions_wltp = st.sidebar.number_input("Emissions WLTP (g/km)", min_value=0.0, max_value=400.0, step=10.0, value=100.0)
    engine_capacity = st.sidebar.number_input("Engine Capacity (cm3)", min_value=0.0, max_value=6000.0, step=100.0, value=2000.0)
    engine_power = st.sidebar.number_input("Engine Power (KW)", min_value=0.0, max_value=500.0, step=10.0, value=100.0)
    fuel_consumption = st.sidebar.number_input("Fuel Consumption (L/100km)", min_value=0.0, max_value=50.0, step=1.0, value=6.0)
    electric_range = st.sidebar.number_input("Electric Range (km)", min_value=0.0, max_value=500.0, step=10.0, value=0.0)

    data = {
        "Vehicle Mass (kg)": vehicle_mass,
        "Emissions WLTP (g/km)": emissions_wltp,
        "Engine Capacity (cm3)": engine_capacity,
        "Engine Power (KW)": engine_power,
        "Fuel Consumption (L/100km)": fuel_consumption,
        "Electric Range (km)": electric_range,
    }
    return pd.DataFrame([data])

# Get user input data
input_data = user_input()

# Display input data
st.subheader("Input Data")
st.write(input_data)

# Standardize input data based on the original model's training
@st.cache_resource
def preprocess_data(data):
    # Load the scaler used during training
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    standardized_data = scaler.transform(data)
    return standardized_data

processed_data = preprocess_data(input_data)

# Predict sustainability score
if st.button("Predict Sustainability Score"):
    prediction = model.predict(processed_data)
    st.subheader("Predicted Sustainability Score")
    st.write(f"{prediction[0]:.2f}")

# Footer
st.write("\n\n")
st.info("Dummy model trained using Linear Regression on a small synthetic dataset.")
