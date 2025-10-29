import streamlit as st
import pandas as pd
import joblib

# Load the saved model and scaler once at app start
model = joblib.load('best_credit_risk_model.pkl')
scaler = joblib.load('scaler.pkl')

# List of features expected by model
feature_names = ['age', 'sex', 'job', 'saving_accounts', 'checking_account', 'credit_amount', 'duration',
                 'marital_status', 'education_level', 'number_of_dependents', 'income', 'employment_status',
                 'existing_loans_count', 'credit_history_length', 'previous_defaults', 'credit_score',
                 'installment_rate', 'loan_type', 'interest_rate', 'collateral']

# Streamlit UI
st.title("Credit Risk Prediction")

# Create input widgets for each feature
input_data = {}

input_data['age'] = st.number_input('Age', min_value=18, max_value=100, value=30)
input_data['sex'] = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Female' if x==0 else 'Male')
input_data['job'] = st.number_input('Job (encoded)', min_value=0, max_value=5, value=2)
input_data['saving_accounts'] = st.number_input('Saving Accounts (encoded)', min_value=0, max_value=5, value=0)
input_data['checking_account'] = st.number_input('Checking Account (encoded)', min_value=0, max_value=5, value=1)
input_data['credit_amount'] = st.number_input('Credit Amount', min_value=100, max_value=1000000, value=5000)
input_data['duration'] = st.number_input('Duration (months)', min_value=1, max_value=120, value=36)
input_data['marital_status'] = st.number_input('Marital Status (encoded)', min_value=0, max_value=5, value=1)
input_data['education_level'] = st.number_input('Education Level (encoded)', min_value=0, max_value=5, value=2)
input_data['number_of_dependents'] = st.number_input('Number of Dependents', min_value=0, max_value=10, value=1)
input_data['income'] = st.number_input('Income', min_value=0, max_value=1000000, value=35000)
input_data['employment_status'] = st.number_input('Employment Status (encoded)', min_value=0, max_value=5, value=1)
input_data['existing_loans_count'] = st.number_input('Existing Loans Count', min_value=0, max_value=10, value=0)
input_data['credit_history_length'] = st.number_input('Credit History Length', min_value=0, max_value=50, value=4)
input_data['previous_defaults'] = st.number_input('Previous Defaults', min_value=0, max_value=10, value=0)
input_data['credit_score'] = st.number_input('Credit Score', min_value=0, max_value=1000, value=650)
input_data['installment_rate'] = st.number_input('Installment Rate', min_value=0, max_value=10, value=3)
input_data['loan_type'] = st.number_input('Loan Type (encoded)', min_value=0, max_value=5, value=1)
input_data['interest_rate'] = st.number_input('Interest Rate', min_value=0.0, max_value=100.0, value=5.0)
input_data['collateral'] = st.number_input('Collateral (encoded)', min_value=0, max_value=10, value=1)

if st.button("Predict Credit Risk"):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Scale numerical features using loaded scaler
    numerical_cols = ['age', 'credit_amount', 'duration', 'number_of_dependents', 'income',
                      'existing_loans_count', 'credit_history_length', 'previous_defaults',
                      'credit_score', 'installment_rate', 'interest_rate']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    # Predict
    prediction = model.predict(input_df)[0]
    result = "Good Credit Risk" if prediction == 0 else "Bad Credit Risk"
    
    st.success(f"Prediction: {result}")
