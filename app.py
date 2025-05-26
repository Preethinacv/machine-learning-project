import streamlit as st
import pandas as pd
import joblib

# Load the saved model and scaler
logi = joblib.load('logis.pkl')
scaler = joblib.load('scaler.pkl')

# Prediction function
def predict_loan_eligibility(input_data):
    # Convert input data to a DataFrame
    input_data_df = pd.DataFrame([input_data])  # Single-row DataFrame
    
    # Scale input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data_df)
    
    # Predict using the loaded model
    prediction = logi.predict(input_data_scaled)
    
    # Return result
    return "Eligible" if prediction[0] == 1 else "Not Eligible"

# Streamlit App
def main():
    # Streamlit page title and description
    st.title("Loan Eligibility Prediction App")
    st.write("Enter the required details below to check loan eligibility.")

    # Input fields for user data
    gender = st.selectbox("Gender", options=["Male", "Female"])
    married = st.selectbox("Married", options=["Yes", "No"])
    dependents = st.selectbox("Dependents", options=["0", "1", "2", "3+"])
    education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self-Employed", options=["Yes", "No"])
    loan_amount = st.number_input("Loan Amount", min_value=1, max_value=1000, step=1)
    loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0, max_value=480, step=1)
    credit_history = st.selectbox("Credit History", options=["0", "1"])
    property_area = st.selectbox("Property Area", options=["Rural", "Semiurban", "Urban"])
    total_income = st.number_input("Total Income (Applicant + Co-applicant)", min_value=0, step=1)

    # Map inputs to numeric values
    input_data = {
        "Gender": 1 if gender == "Male" else 0,
        "Married": 1 if married == "Yes" else 0,
        "Dependents": 4 if dependents == "3+" else int(dependents),
        "Education": 1 if education == "Graduate" else 0,
        "Self_Employed": 1 if self_employed == "Yes" else 0,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": int(credit_history),
        "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area],
        "total_income": total_income
    }

    # Prediction button
    if st.button("Check Eligibility"):
        # Predict loan eligibility
        result = predict_loan_eligibility(input_data)
        st.success(f"Loan Eligibility Prediction: {result}")

# Run the app
if __name__ == "__main__":
    main()
