import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Set the background color to black
st.markdown(
    """
    <style>
    .stApp {
        background-color: black;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained Decision Tree model
with open('decision_tree_model.pkl', 'rb') as file:
    dtree_model = pickle.load(file)

# Load the dataset
file_path = "C:\\Users\\aravi\\OneDrive\\Desktop\\Python Scripts\\Final Project\\Bank_dataset.csv"
df = pd.read_csv(file_path)

# Load the scaler used during training (assuming it is saved similarly)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the app title
st.title("Bank Dataset Prediction App")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select an option", ["Prediction", "Data", "EDA - Visual"])

if options == "Prediction":
    st.header("**_Input Features_**")
    
    # Create input fields for user to input data
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
    input_data_encoded[numerical_features] = scaler.transform(input_data_encoded[numerical_features])

    # Make prediction
    if st.button("Predict"):
        prediction = dtree_model.predict(input_data_encoded)
        if prediction[0] == 1:
            st.error("The model predicts: DEFAULT")
        else:
            st.success("The model predicts: NOT DEFAULT")

elif options == "Data":
    st.header("**_Dataset and Classification Report_**")
    st.subheader("First 15 Rows of the Dataset")
    st.dataframe(df.head(15))

    st.subheader("Classification Report")
    st.text("""
        Decision Tree Accuracy: 0.9915
        Classification Report:
               precision    recall  f1-score   support

           0       1.00      0.99      1.00    258362
           1       0.95      0.96      0.95     24379

    accuracy                           0.99    282741
   macro avg       0.97      0.98      0.97    282741
weighted avg       0.99      0.99      0.99    282741
""")

elif options == "EDA - Visual":
    st.header("**_Exploratory Data Analysis_**")

    # Top Row: Distribution, Comparison, F1 vs F2
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Distribution")
        plt.figure(figsize=(8, 6))  # Medium size
        sns.countplot(x='TARGET', data=df)
        st.pyplot(plt)

    with col2:
        st.subheader("Comparison")
        plt.figure(figsize=(8, 6))  # Medium size
        sns.barplot(x='CODE_GENDER', y='AMT_CREDIT_x', data=df)
        st.pyplot(plt)

    with col3:
        st.subheader("F1 vs F2")
        plt.figure(figsize=(8, 6))  # Medium size
        sns.barplot(x='NAME_CONTRACT_TYPE_x', y='AMT_INCOME_TOTAL', data=df)
        st.pyplot(plt)

    # Middle Row: Box plots, Pair plot, Correlation Plot
    col4, col5, col6 = st.columns(3)

    with col4:
        st.subheader("Box plots")
        plt.figure(figsize=(8, 6))  # Medium size
        sns.boxplot(x='NAME_EDUCATION_TYPE', y='AMT_INCOME_TOTAL', data=df)
        st.pyplot(plt)

    with col5:
        st.subheader("Pair plot")
        sns.pairplot(df[['AMT_INCOME_TOTAL', 'AMT_CREDIT_x', 'EXT_SOURCE_2', 'DAYS_BIRTH']])
        st.pyplot(plt)

    with col6:
        st.subheader("Correlation Plot")
        plt.figure(figsize=(8, 6))  # Medium size
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt)

    # Bottom Row: Normal Dist, Normal - 2 F, Additional Plot 1
    col7, col8, col9 = st.columns(3)

    with col7:
        st.subheader("Normal Dist.")
        plt.figure(figsize=(8, 6))  # Medium size
        sns.histplot(df['AMT_CREDIT_x'], kde=True)
        st.pyplot(plt)

    with col8:
        st.subheader("Normal - 2 F")
        plt.figure(figsize=(8, 6))  # Medium size
        sns.scatterplot(x='DAYS_BIRTH', y='EXT_SOURCE_2', data=df)
        st.pyplot(plt)

    with col9:
        st.subheader("Additional Plot 1")
        plt.figure(figsize=(8, 6))  # Medium size
        sns.barplot(x='NAME_INCOME_TYPE', y='AMT_INCOME_TOTAL', data=df)
        st.pyplot(plt)

    # Additional Plot 2
    st.subheader("Additional Plot 2")
    plt.figure(figsize=(8, 6))  # Medium size
    sns.boxplot(x='OCCUPATION_TYPE', y='AMT_CREDIT_x', data=df)
    st.pyplot(plt)
