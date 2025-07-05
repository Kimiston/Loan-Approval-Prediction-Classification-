import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

# Set working directory where the model file is stored
os.chdir('E:\MLCLASSIFICATIONKSK')


model = pickle.load(open('E:\MLCLASSIFICATIONKSK\model.pkl', 'rb'))
#scaler=pickle.load(open('D:\\project steps\\scaler.pkl','rb'))
gen_encoder=pickle.load(open('E:\MLCLASSIFICATIONKSK\gender_encoder.pkl', 'rb'))
edu_encoder=pickle.load(open('E:\MLCLASSIFICATIONKSK\education_encoder.pkl', 'rb'))
own_encoder=pickle.load(open('E:\MLCLASSIFICATIONKSK\ownership_encoder.pkl', 'rb'))
intend_encoder=pickle.load(open('E:\MLCLASSIFICATIONKSK\intent_encoder.pkl', 'rb'))
pre_encoder=pickle.load(open('E:\MLCLASSIFICATIONKSK\previous_encoder.pkl', 'rb'))


# App title
st.title("Loan Approval Prediction")
st.write("This is to predict the chances of loan approval depends on all the given required inputs")


# Inputs from the user
Age = st.slider(
    "Age (years)",
    min_value=20, max_value=144, value=22
)

Gender = st.selectbox(
    "Gender",
    options=["female", "male"]
)

Education = st.selectbox(
    "Highest education completed",
    options=["Associate", "Bachelor", "Doctorate", "High School", "Master"]
)

Income = st.slider(
    "Annual income (₹)",
    min_value=8_000, max_value=7_201_000, value=50_000, step=1_000
)
Emp_Exp = st.slider(
    "Years of employment experience",
    min_value=0, max_value=125, value=5
) 
Home_Ownership = st.selectbox(
    "Home‑ownership status",
    options=["MORTGAGE", "OTHER", "OWN", "RENT"]
)

# ------------- requested loan -------------
Loan_Amount = st.slider(
    "Loan amount requested (₹)",
    min_value=500, max_value=35_000, value=5_000, step=500
)

Loan_Intent = st.selectbox(
    "Purpose of the loan",
    options=[
        "DEBTCONSOLIDATION",
        "EDUCATION",
        "HOMEIMPROVEMENT",
        "MEDICAL",
        "PERSONAL",
        "VENTURE",
    ]
)

Interest_Rate = st.slider(
    "Expected interest rate (%)",
    min_value=5.42, max_value=20.00, value=10.00, step=0.01
)

Percent_Income = st.slider(
    "Loan‑payment as a fraction of income",
    min_value=0.00, max_value=0.66, value=0.10, step=0.01
)

# ------------- credit history -------------
Cred_Hist_Length = st.slider(
    "Credit‑history length (years)",
    min_value=2, max_value=30, value=5
)

Credit_Score = st.slider(
    "Credit score",
    min_value=390, max_value=850, value=650
)

Prev_Default = st.selectbox(
    "Previous loan default on file?",
    options=["No", "Yes"]
)



# Prediction
#if st.button("Predict Score"):
    #new_extra=encoder.transform([extra])
    #new_data = np.array([Study, pre_score, new_extra, s_hour, s_question])
    #new_data_encode=scalerlabel.transform(new_data)
    #prediction = model.predict(new_data)
    #result = prediction
    
    #st.success(f"Prediction: {result}")
    #st.write("Model Raw Output:", prediction)



    # Prediction
if st.button("Predict Loan Status"):
    #new_extra=encoder.transform([extra])
    column_name= ['person_age', 'person_gender', 'person_education', 'person_income',
       'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
       'credit_score', 'previous_loan_defaults_on_file']
    new_data = [[Age, Gender, Education, Income,Emp_Exp, Home_Ownership, Loan_Amount,Loan_Intent, Interest_Rate, Percent_Income, Cred_Hist_Length, Credit_Score, Prev_Default]]
    new_df=pd.DataFrame(new_data,columns=column_name)
    st.write(new_df)
    new_df['person_gender']=gen_encoder.transform(new_df['person_gender'])
    new_df['person_education']=edu_encoder.transform(new_df['person_education'])
    new_df['person_home_ownership']=own_encoder.transform(new_df['person_home_ownership'])
    new_df['loan_intent']=intend_encoder.transform(new_df['loan_intent'])
    new_df['previous_loan_defaults_on_file']=pre_encoder.transform(new_df['previous_loan_defaults_on_file']) 
    prediction = model.predict(new_df)
    result = "Approved" if prediction[0] == 1 else "Declined"
    
    st.success(f"Prediction: {result}")
    #st.write("Model Raw Output:",prediction)