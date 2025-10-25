# for å kunne bruke dette lokalt, kjør: streamlit run website.py
# for å installere alt riktig aktiver py inviroment: .venv\Scripts\Activate.ps1
# så installer streamlit: pip install streamlit
# og pandas og numpy: pip install pandas numpy

import streamlit as st
import pandas as pd
import numpy as np

# --- Page Configuration ---
# Set the title and a favicon for your app's browser tab
st.set_page_config(page_title="Do You Have Alzheimer's", page_icon="🧠")

# --- Title Section ---
st.title("Do You Have Alzheimer's 🧠")
st.write("This app helps you assess the likelihood of having Alzheimer's disease based on various health and lifestyle factors.")
st.write("Please fill in the following information:")

# --- Input Section ---
# --- Demographic Information ---
st.header("Demographic Information")
age = st.number_input("Age", min_value=0, max_value=120, value=30)

gender = st.selectbox("Gender", options=["Male", "Female"])

Ethnicity = st.selectbox("Ethnicity", options=["Caucasian", "African American", "Asian", "Other"])

education_level = st.selectbox("Education Level", options=["None", "High School", "Bachelor's", "Higher",])

# --- Lifestyle Information ---
st.header("Lifestyle Information")
BMI = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=22.0, format="%.1f")

smoking = st.checkbox("Smoking")

Alcohol_consumption = st.slider("Alcohol Consumption (units per week)", min_value=0, max_value=20, value=0)

Physical_activity = st.slider("Physical Activity (Hours per week", min_value=0, max_value=10, value=2)

Diet_quality = st.slider("Diet Quality", min_value=0, max_value=10, value=0)

Sleep_quality = st.slider("Sleep Quality", min_value=4, max_value=10, value=4) # Fordi at dataset har minimum 4

# --- Medical History ---
st.header("Medical History")

Family_history = st.checkbox("Family History of Alzheimer's Disease")

Cardio_vascular_disease = st.checkbox("History of Cardiovascular Disease")

Diabetes = st.checkbox("History of Diabetes")

Depression = st.checkbox("History of Depression")

History_of_head_injury = st.checkbox("History of Head Injury")

Hypertension = st.checkbox("Hypertension")

# --- Health Information ---
st.header("Health Information")

SystolicBP = st.number_input("Systolic Blood Pressure (mm Hg)", min_value=90, max_value=180, value=120)
DiastolicBP = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=60, max_value=120, value=80)

# --- Symptom Assessment ---
st.header("Symptom Assessment")
Confusion = st.checkbox("Confusion", options=["No", "Yes"])
Memory_complaints = st.selectbox("Memory Complaints", options=["No", "Yes"])
Difficulty_executing_tasks = st.selectbox("Difficulty Executing Tasks", options=["No"," Yes"])
Forgetfulness = st.checkbox("Forget", options=["No", "Yes"])

st.write("---")

# Submit Button
if st.button("Assess Alzheimer's Risk"):
    # Here you would typically process the input data and run your prediction model
    # For demonstration purposes, we'll just display the collected inputs
    st.header("Input Summary")
    st.subheader("demographic Information")
    st.write(f"Age: {age}")
    st.write(f"Gender: {gender}")
    st.write(f"Ethnicity: {Ethnicity}")
    st.write(f"Education Level: {education_level}")

    st.subheader("Lifestyle Information")
    st.write(f"BMI: {BMI}")
    st.write(f"Smoking: {Smoking}")
    st.write(f"Alcohol Consumption: {Alcohol_consumption} units/week")
    st.write(f"Physical Activity: {Physical_activity} hours/week")
    st.write(f"Diet Quality: {Diet_quality}")
    st.write(f"Sleep Quality: {Sleep_quality}")

    st.subheader("Medical History")
    st.write(f"Family History of Alzheimer's Disease: {Family_history}")
    st.write(f"History of Cardiovascular Disease: {Cardio_vascular_disease}")
    st.write(f"History of Diabetes: {Diabetes}")
    st.write(f"History of Depression: {Depression}")
    st.write(f"History of Head Injury: {History_of_head_injury}")
    st.write(f"Hypertension: {Hypertension}")

    st.subheader("Health Information")
    st.write(f"Systolic Blood Pressure: {SystolicBP} mm Hg")
    st.write(f"Diastolic Blood Pressure: {DiastolicBP} mm Hg")

    st.subheader("Symptom Assessment")
    st.write(f"Confusion: {Confusion}")
    st.write(f"Memory Complaints: {Memory_complaints}")
    st.write(f"Difficulty Executing Tasks: {Difficulty_executing_tasks}")
    st.write(f"Forgetfulness: {Forgetfulness}")

    st.header("Prediction Result")