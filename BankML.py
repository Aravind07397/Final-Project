import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained Decision Tree model
with open('decision_tree_model.pkl', 'rb') as file:
    dtree_model = pickle.load(file)

# Define the app title
st.title("Bank Dataset Prediction App")

# Create input fields for user to input data
st.header("Input Features")

NAME_CONTRACT_TYPE_x = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
CODE_GENDER = st.selectbox("Gender", ["M", "F"])
AMT_INCOME_TOTAL = st.number_input("Income Total", min_value=0.0, value=50000.0, step=1000.0)
AMT_CREDIT_x = st.number_input("Credit Amount", min_value=0.0, value=100000.0, step=1000.0)
NAME_INCOME_TYPE = st.selectbox("Income Type", ["Working", "State servant", "Commercial associate", "Pensioner", "Unemployed", "Student"])
NAME_EDUCATION_TYPE = st.selectbox("Education Type", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
DAYS_BIRTH = st.number_input("Days of Birth (Negative Value)", min_value=-25000, max_value=-5000, value=-10000, step=1)
DAYS_EMPLOYED = st.number_input("Days Employed (Negative Value)", min_value=-18000, max_value=0, value=-5000, step=1)
OCCUPATION_TYPE = st.selectbox("Occupation Type", ["Laborers", "Core staff", "Accountants", "Managers", "Drivers", "Sales staff", "Cleaning staff", "Cooking staff", "Private service staff", "Medicine staff", "Security staff", "High skill tech staff", "Waiters/barmen staff", "Low-skill Laborers", "Realty agents", "Secretaries", "IT staff", "HR staff"])
EXT_SOURCE_2 = st.number_input("External Source 2", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Preprocess user input
input_data = pd.DataFrame({
    'NAME_CONTRACT_TYPE_x': [NAME_CONTRACT_TYPE_x],
    'CODE_GENDER': [CODE_GENDER],
    'AMT_INCOME_TOTAL': [AMT_INCOME_TOTAL],
    'AMT_CREDIT_x': [AMT_CREDIT_x],
    'NAME_INCOME_TYPE': [NAME_INCOME_TYPE],
    'NAME_EDUCATION_TYPE': [NAME_EDUCATION_TYPE],
    'DAYS_BIRTH': [DAYS_BIRTH],
    'DAYS_EMPLOYED': [DAYS_EMPLOYED],
    'OCCUPATION_TYPE': [OCCUPATION_TYPE],
    'EXT_SOURCE_2': [EXT_SOURCE_2]
})

# Encode categorical variables using the same encoding as during training
def encode_input(input_df):
    encoding_dict = {
        'NAME_CONTRACT_TYPE_x': {'Cash loans': 0, 'Revolving loans': 1},
        'CODE_GENDER': {'F': 0, 'M': 1},
        'NAME_INCOME_TYPE': {'Commercial associate': 0, 'Pensioner': 1, 'State servant': 2, 'Student': 3, 'Unemployed': 4, 'Working': 5},
        'NAME_EDUCATION_TYPE': {'Academic degree': 0, 'Higher education': 1, 'Incomplete higher': 2, 'Lower secondary': 3, 'Secondary / secondary special': 4},
        'OCCUPATION_TYPE': {'Accountants': 0, 'Cleaning staff': 1, 'Cooking staff': 2, 'Core staff': 3, 'Drivers': 4, 'HR staff': 5,
                            'High skill tech staff': 6, 'IT staff': 7, 'Laborers': 8, 'Low-skill Laborers': 9, 'Managers': 10,
                            'Medicine staff': 11, 'Private service staff': 12, 'Realty agents': 13, 'Sales staff': 14,
                            'Secretaries': 15, 'Security staff': 16, 'Waiters/barmen staff': 17}
    }
    for col in encoding_dict:
        input_df[col] = input_df[col].map(encoding_dict[col])
    return input_df

input_data_encoded = encode_input(input_data)

# Define numerical features
numerical_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT_x', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'EXT_SOURCE_2']

# Scale numerical features using the same scaler as during training
scaler = StandardScaler()
input_data_encoded[numerical_features] = scaler.fit_transform(input_data_encoded[numerical_features])

# Make prediction
if st.button("Predict"):
    prediction = dtree_model.predict(input_data_encoded)
    if prediction[0] == 1:
        st.error("The model predicts: DEFAULT")
    else:
        st.success("The model predicts: NOT DEFAULT")
