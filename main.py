#run the file using "streamlit run .\main.py" in terminal:
# F:\Data Science\Pandas\ML codebasics course\Project 1\
# project_1_model_retraining_resources\app> streamlit run .\main.py

import streamlit as st
import prediction as prd


st.title("Insurance Premium Prediction App")
st.markdown("<p style='font-size:16px;'>Fill in the details below to get your insurance premium prediction. Provide accurate information for the best estimate. Click on the <b>Predict</b> button at the bottom to get the value!!</p>", unsafe_allow_html=True)

# Create three columns for better layout
col1, col2, col3 = st.columns(3)

# Column 1 Inputs
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    gender = st.selectbox("Gender", ['Female', 'Male'])
    region = st.selectbox("Region", ['Southeast', 'Northeast', 'Southwest', 'Northwest'])
    marital_status = st.selectbox("Marital Status", ['Unmarried', 'Married'])
    bmi_category = st.selectbox("BMI Category", ['Normal', 'Overweight', 'Obesity', 'Underweight'])
    
# Column 2 Inputs
with col2:
    number_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
    physical_activity = st.selectbox("Physical Activity", ['Medium', 'Low', 'High'])
    stress_level = st.selectbox("Stress Level", ['Medium', 'High', 'Low'])
    income_lakhs = st.number_input("Income (Lakhs)", min_value=0.1, step=0.1)

# Column 3 Inputs
with col3:
    smoking_status = st.selectbox("Smoking Status", ['No Smoking', 'Occasional', 'Regular'])
    employment_status = st.selectbox("Employment Status", ['Self-Employed', 'Freelancer', 'Salaried'])
    medical_history = st.selectbox("Medical History", ['High blood pressure', 'No Disease', 'Thyroid','High blood pressure & Heart disease', 'Diabetes & Thyroid','Diabetes', 'Heart disease', 'Diabetes & High blood pressure','Diabetes & Heart disease'])
    insurance_plan = st.selectbox("Insurance Plan", ['Gold', 'Silver', 'Bronze'])

# Store input values in a dictionary
input_data = {
    "age": age,
    "gender": gender,
    "region": region,
    "marital_status": marital_status,
    "number_of_dependents": number_of_dependents,
    "physical_activity": physical_activity,
    "stress_level": stress_level,
    "income_lakhs": income_lakhs,
    "bmi_category": bmi_category,
    "smoking_status": smoking_status,
    "employment_status": employment_status,
    "medical_history": medical_history,
    "insurance_plan": insurance_plan
}

# Predict button
if st.button("Predict"):
    try:
        Prediction = prd.predict(input_data)
        st.success(f"Annual Insurance Premium: Rs.{round(Prediction[0])}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")