# for å kunne bruke dette lokalt, kjør: streamlit run website.py
# for å installere alt riktig aktiver py inviroment: .venv\Scripts\Activate.ps1
# så installer streamlit: pip install streamlit
# og pandas og numpy: pip install pandas numpy
import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Skrevet av Gemini

@st.cache_resource
def load_model():
    loaded_model = joblib.load("logistic_regression_alzheimers_model.pkl")
    loaded_scaler = joblib.load("standard_scaler_alzheimers.pkl")
    loaded_label_encoder = joblib.load("label_encoder_alzheimers.pkl")
    return loaded_model, loaded_scaler, loaded_label_encoder

model, scaler, label_encoder = load_model()

# --- Page Configuration ---
st.set_page_config(page_title="Do You Have Alzheimer's", page_icon="🧠")

# --- Title Section ---
st.title("Do You Have Alzheimer's 🧠")
st.write("This app helps you assess the likelihood of having Alzheimer's disease based on various health and lifestyle factors.")
st.write("Please fill in the following information:")

# --- Input Section ---
# --- Demographic Information ---
st.header("Demographic Information")
Age = st.number_input("Age", min_value=0, max_value=120, value=30)

Gender = st.selectbox("Gender", options=["Male", "Female"])

Ethnicity = st.selectbox("Ethnicity", options=["Caucasian", "African American", "Asian", "Other"])

Education_level = st.selectbox("Education Level", options=["None", "High School", "Bachelor's", "Higher", ])

# --- Lifestyle Information ---
st.header("Lifestyle Information")
BMI = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=22.0, format="%.1f")

Smoking = st.checkbox("Smoking")

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

# --- Cognitive and Functional Assessments ---
st.header("Cognitive and Functional Assessments")
MemoryComplaints = st.checkbox("Memory Complaints")
BehavioralProblems = st.checkbox("Behavioural Problems")

# --- Symptom Assessment ---
st.header("Symptom Assessment")
Confusion = st.checkbox("Confusion")
Disorientation = st.checkbox("Disorientation")
PersonalityChanges = st.checkbox("Personality Changes")
DifficultyCompletingTasks = st.checkbox("Difficulty Executing Tasks")
Forgetfulness = st.checkbox("Forgetfulness")

st.write("---")

# Hjulpet av Gemini
# Submit Button
if st.button("Assess Alzheimer's Risk"):

    # Mapping
    Gender_map = {"Male" : 0, "Female" : 1}
    Ethnicity_map = {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3}
    Education_level_map = {"None": 0, "High School": 1, "Bachelor's": 2, "Higher": 3}

    # 1. Create a dictionary from the user's input
    input_data = {
        'Age': Age,
        'Gender': Gender_map[Gender],
        'Ethnicity': Ethnicity_map[Ethnicity],
        'EducationLevel': Education_level_map[Education_level],
        'BMI': BMI,
        'Smoking': 1 if Smoking else 0,
        'AlcoholConsumption': Alcohol_consumption,
        'PhysicalActivity': Physical_activity,
        'DietQuality': Diet_quality,
        'SleepQuality': Sleep_quality,
        'FamilyHistoryAlzheimers': 1 if Family_history else 0,
        'CardiovascularDisease': 1 if Cardio_vascular_disease else 0,
        'Diabetes': 1 if Diabetes else 0,
        'Depression': 1 if Depression else 0,
        'HeadInjury': 1 if History_of_head_injury else 0,
        'Hypertension': 1 if Hypertension else 0,
        'SystolicBP': SystolicBP,
        'DiastolicBP': DiastolicBP,
        'MemoryComplaints': 1 if MemoryComplaints else 0,
        'BehavioralProblems': 1 if BehavioralProblems else 0,
        'Confusion': 1 if Confusion else 0,
        'Disorientation': 1 if Disorientation else 0,
        'PersonalityChanges': 1 if PersonalityChanges else 0,
        'DifficultyCompletingTasks': 1 if DifficultyCompletingTasks else 0,
        'Forgetfulness': 1 if Forgetfulness else 0
    }

    # 2. Convert the dictionary to a pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # 3. Scale the numerical features
    # Define the columns that need to be scaled
    columns_to_scale = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
                        'SystolicBP', 'DiastolicBP']
    input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])

    # 4. Make a prediction and get probabilities
    prediction_proba = model.predict_proba(input_df)
    prediction = model.predict(input_df)

    # 5. Display the results
    st.header("Prediction Result")

    # Get the predicted class label using the label encoder

    # Output Mapping (based on alphabetical order from LabelEncoder)
    diagnosis_map = {
        0: "No Alzheimer's Disease",
        1: "Alzheimer's Disease",
    }

    # Get the predicted class label using the map
    predicted_class_label = diagnosis_map[prediction[0]]

    st.write(f"**Predicted Diagnosis:** `{predicted_class_label}`")

    # Display the probabilities for each class
    st.write("**Prediction Probabilities:**")
    for class_index, class_name in diagnosis_map.items():
        st.write(f"- {class_name}: `{prediction_proba[0][class_index]:.2%}`")

st.info("Disclaimer: This is a machine learning prediction and not a medical diagnosis. Please consult a healthcare professional for any health concerns.")