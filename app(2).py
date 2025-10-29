import streamlit as st
import pandas as pd
import joblib

# Load saved model and scaler
model = joblib.load('best_credit_risk_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define encoding dictionaries to convert user-friendly inputs to model expected encoded values
sex_map = {'Female': 0, 'Male': 1}
saving_accounts_map = {'little': 0, 'moderate': 1, 'quite rich': 2, 'rich': 3, 'no info': 4}
checking_account_map = {'no': 0, 'little': 1, 'moderate': 2, 'rich': 3, 'no info': 4}
marital_status_map = {'single': 0, 'married': 1, 'divorced': 2}
education_level_map = {'none': 0, 'high school': 1, 'graduate': 2, 'postgraduate': 3, 'phd': 4}
employment_status_map = {'unemployed': 0, 'employed': 1, 'self-employed': 2, 'retired': 3, 'student': 4, 'other': 5}
loan_type_map = {'personal': 0, 'business': 1, 'car': 2, 'mortgage': 3, 'education': 4, 'domestic appliances':5}
collateral_map = {'none': 0, 'property': 1, 'vehicle': 2, 'guarantor': 3}

st.title("Credit Risk Prediction")

# User inputs with friendly options
age = st.number_input('Age', 18, 100, 30)
sex = st.selectbox('Sex', list(sex_map.keys()))
job = st.number_input('Job (0-5 scale)', 0, 5, 2)
saving_accounts = st.selectbox('Saving Accounts', list(saving_accounts_map.keys()))
checking_account = st.selectbox('Checking Account', list(checking_account_map.keys()))
credit_amount = st.number_input('Credit Amount', 100, 1000000, 5000)
duration = st.number_input('Duration (months)', 1, 120, 36)
marital_status = st.selectbox('Marital Status', list(marital_status_map.keys()))
education_level = st.selectbox('Education Level', list(education_level_map.keys()))
number_of_dependents = st.number_input('Number of Dependents', 0, 10, 1)
income = st.number_input('Income', 0, 1000000, 35000)
employment_status = st.selectbox('Employment Status', list(employment_status_map.keys()))
existing_loans_count = st.number_input('Existing Loans Count', 0, 10, 0)
credit_history_length = st.number_input('Credit History Length (years)', 0, 50, 4)
previous_defaults = st.number_input('Previous Defaults', 0, 10, 0)
credit_score = st.number_input('Credit Score', 0, 1000, 650)
installment_rate = st.number_input('Installment Rate', 0, 10, 3)
loan_type = st.selectbox('Loan Type', list(loan_type_map.keys()))
interest_rate = st.number_input('Interest Rate (%)', 0.0, 100.0, 5.0)
collateral = st.selectbox('Collateral', list(collateral_map.keys()))

if st.button('Predict Credit Risk'):
    # Map user-friendly inputs to encoded values
    input_dict = {
        'age': age,
        'sex': sex_map[sex],
        'job': job,
        'saving_accounts': saving_accounts_map[saving_accounts],
        'checking_account': checking_account_map[checking_account],
        'credit_amount': credit_amount,
        'duration': duration,
        'marital_status': marital_status_map[marital_status],
        'education_level': education_level_map[education_level],
        'number_of_dependents': number_of_dependents,
        'income': income,
        'employment_status': employment_status_map[employment_status],
        'existing_loans_count': existing_loans_count,
        'credit_history_length': credit_history_length,
        'previous_defaults': previous_defaults,
        'credit_score': credit_score,
        'installment_rate': installment_rate,
        'loan_type': loan_type_map[loan_type],
        'interest_rate': interest_rate,
        'collateral': collateral_map[collateral]
    }

    input_df = pd.DataFrame([input_dict])

    # Scale numerical features
    numerical_cols = ['age', 'credit_amount', 'duration', 'number_of_dependents',
                      'income', 'existing_loans_count', 'credit_history_length',
                      'previous_defaults', 'credit_score', 'installment_rate', 'interest_rate']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Predict using your model
    prediction = model.predict(input_df)[0]

    result_text = "Good Credit Risk" if prediction == 0 else "Bad Credit Risk"
    st.success(f'Prediction: {result_text}')
