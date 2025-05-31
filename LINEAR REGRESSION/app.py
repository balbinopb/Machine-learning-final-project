import streamlit as st
from linear_regression import SalaryPredictor

@st.cache_resource
def load_model():
    return SalaryPredictor()

predictor = load_model()

st.title("Salary Prediction App")

# Numeric input
experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)

# Dropdowns for categorical inputs
unique_values = predictor.get_unique_values()

education = st.selectbox("Education Level", unique_values['Education'])
location = st.selectbox("Location", unique_values['Location'])
job_title = st.selectbox("Job Title", unique_values['Job_Title'])
gender = st.selectbox("Gender", unique_values['Gender'])

if st.button("Predict Salary"):
    input_data = {
        'Experience': experience,
        'Education': education,
        'Location': location,
        'Job_Title': job_title,
        'Gender': gender
    }
    salary_pred = predictor.predict_salary(input_data)
    st.success(f"Predicted Salary: ${salary_pred:,.2f}")
