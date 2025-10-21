import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('sleep_disorder_model.pkl')

st.title('Sleep Disorder Prediction')

st.write("Enter your details below to predict if you have a sleep disorder.")

# Create input fields for user data
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=1, max_value=100, value=30)
occupation = st.selectbox('Occupation',
                          ['Software Engineer', 'Doctor', 'Sales Representative', 'Teacher', 'Nurse', 'Engineer',
                           'Accountant', 'Lawyer', 'Salesperson', 'Manager', 'Scientist'])
sleep_duration = st.number_input('Sleep Duration (hours)', min_value=1.0, max_value=12.0, value=7.0, step=0.1)
quality_of_sleep = st.slider('Quality of Sleep (1-10)', 1, 10, 7)
physical_activity = st.number_input('Physical Activity Level (minutes/day)', min_value=0, value=30)
stress_level = st.slider('Stress Level (1-10)', 1, 10, 5)
bmi_category = st.selectbox('BMI Category', ['Normal', 'Overweight', 'Obese', 'Normal Weight'])
systolic_bp = st.number_input('Systolic Blood Pressure', min_value=90, max_value=180, value=120)
diastolic_bp = st.number_input('Diastolic Blood Pressure', min_value=60, max_value=120, value=80)
heart_rate = st.number_input('Heart Rate', min_value=50, max_value=100, value=70)
daily_steps = st.number_input('Daily Steps', min_value=1000, max_value=20000, value=5000)

if st.button('Predict'):
    # Create a dataframe from user input
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Occupation': [occupation],
        'Sleep Duration': [sleep_duration],
        'Quality of Sleep': [quality_of_sleep],
        'Physical Activity Level': [physical_activity],
        'Stress Level': [stress_level],
        'BMI Category': [bmi_category],
        'Heart Rate': [heart_rate],
        'Daily Steps': [daily_steps],
        'Systolic_BP': [systolic_bp],
        'Diastolic_BP': [diastolic_bp]
    })

    # Encode categorical variables
    for column in input_data.columns:
        if input_data[column].dtype == 'object':
            le = LabelEncoder()
            input_data[column] = le.fit_transform(input_data[column])

    # Make prediction
    prediction = model.predict(input_data)

    # Map prediction to disorder name
    disorder_mapping = {0: 'None', 1: 'Sleep Apnea', 2: 'Insomnia'}
    result = disorder_mapping.get(prediction[0], "Unknown")

    st.success(f'The predicted sleep disorder is: {result}')
