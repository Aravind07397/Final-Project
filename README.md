# Bank Risk Controller System 
# Project Overview
 The Bank Risk Controller System is a machine learning-based application designed to predict the likelihood of a customer defaulting on a loan. This system helps banks and financial institutions assess the risk associated with lending money to clients, enabling informed decision-making and improved risk management.

The application uses a Decision Tree Classifier trained on a comprehensive dataset, evaluating customers based on various input features such as income, credit amount, employment history, and more. The prediction outcomes assist banks in categorizing customers as either high or low risk, thereby optimizing loan approval processes.

# Features
User-Friendly Interface: A web-based interface built using Streamlit for easy data input and visualization.
Accurate Risk Prediction: The system uses a Decision Tree model to predict whether a customer will default.
Exploratory Data Analysis (EDA): Visualize the dataset through various graphs and plots.
Model Evaluation: Provides a detailed classification report, including precision, recall, F1-score, and accuracy.
Scalable Deployment: Ready for deployment on platforms like Render or Heroku.
Tech Stack
Python: Core programming language used for model building and backend.
Streamlit: Framework for developing the web-based interface.
Scikit-learn: Machine learning library for model training and evaluation.
Pandas & NumPy: Data manipulation and analysis.
Matplotlib & Seaborn: Libraries for data visualization.
Pickle: For saving and loading the machine learning model and scaler.
Setup Instructions
Clone the Repository:

# bash
Copy code
git clone https://github.com/yourusername/bank-risk-controller-system.git
cd bank-risk-controller-system
Create a Virtual Environment:

# bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:

# bash
Copy code
pip install -r requirements.txt
Download the Dataset:

Place the dataset (Bank_dataset.csv) in the project's root directory.
Train the Model and Save Scaler (if not using the pre-trained model):

# bash
Copy code
python train_model.py
Run the Application:

# bash
Copy code
streamlit run app.py
Usage
Prediction: Input customer details through the interface and get instant risk predictions.
Data Analysis: Explore the dataset visually through the EDA section.
Model Evaluation: View the performance of the model through the classification report.
Model Details
Algorithm: Decision Tree Classifier

# Input Features:

NAME_CONTRACT_TYPE_x
CODE_GENDER
AMT_INCOME_TOTAL
AMT_CREDIT_x
NAME_INCOME_TYPE
NAME_EDUCATION_TYPE
DAYS_BIRTH
DAYS_EMPLOYED
OCCUPATION_TYPE
EXT_SOURCE_2
# Performance Metrics:

Accuracy: 99.15%
Precision: 1.00 (Class 0), 0.95 (Class 1)
Recall: 0.99 (Class 0), 0.96 (Class 1)
F1-Score: 1.00 (Class 0), 0.95 (Class 1)
Dataset
Source: Internal bank data (or publicly available datasets, if applicable).
Size: ~280,000 records
Features: Income, credit amount, gender, employment history, etc.
Target: Loan Default (0 = No Default, 1 = Default)
Scalability and Deployment
The application is designed to be easily deployable on cloud platforms like Render, Heroku, or AWS. The web-based interface ensures that the system can be accessed from any device, providing scalability for organizations of different sizes.

# Contributing
We welcome contributions to enhance the Bank Risk Controller System. To contribute:

# Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -m 'Add some feature').
Push to the branch (git push origin feature-branch).
Create a pull request.

