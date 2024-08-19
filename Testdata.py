import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings


# Load the dataset
file_path = "C:\\Users\\aravi\\OneDrive\\Desktop\\Python Scripts\\Final Project\\Bank_dataset.csv"
df = pd.read_csv(file_path)

# Display the first few rows
df.head()
# Select the 10 input features
features = [
    'NAME_CONTRACT_TYPE_x', 'CODE_GENDER', 'AMT_INCOME_TOTAL',
    'AMT_CREDIT_x', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
    'DAYS_BIRTH', 'DAYS_EMPLOYED', 'OCCUPATION_TYPE', 'EXT_SOURCE_2'
]

# Define the target variable
target = 'TARGET'

# Create feature matrix X and target vector y
X = df[features]
y = df[target]

# Check for missing values
print(X.isnull().sum())

# For numerical features, fill missing values with mean
numerical_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT_x', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'EXT_SOURCE_2']
for col in numerical_features:
    X[col].fillna(X[col].mean(), inplace=True)

# For categorical features, fill missing values with mode
categorical_features = ['NAME_CONTRACT_TYPE_x', 'CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE']
for col in categorical_features:
    X[col].fillna(X[col].mode()[0], inplace=True)

warnings.filterwarnings('ignore')

le = LabelEncoder()

for col in categorical_features:
    X[col] = le.fit_transform(X[col])

scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Split the data (80% training and 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree model
dtree = DecisionTreeClassifier(random_state=42)

# Train the model
dtree.fit(X_train, y_train)

# Predict on the test set
y_pred_dtree = dtree.predict(X_test)


accuracy_dtree = accuracy_score(y_test, y_pred_dtree)
print(f"Decision Tree Accuracy: {accuracy_dtree:.4f}")

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred_dtree))

# Confusion Matrix
conf_matrix_dtree = confusion_matrix(y_test, y_pred_dtree)
sns.heatmap(conf_matrix_dtree, annot=True, fmt='d', cmap='Greens')
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Save the model to a file
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(dtree, file)