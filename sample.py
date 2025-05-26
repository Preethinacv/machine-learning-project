# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib

# Load the dataset
df = pd.read_csv("loanprediction.csv")

# Dataset overview
print(df.info())
print(df.nunique())

# Feature Engineering: Combine ApplicantIncome and CoapplicantIncome
df['total_income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.drop(columns=['ApplicantIncome', 'CoapplicantIncome'], inplace=True)

# Handle missing values using mode
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']:
    df[col].fillna(value=df[col].mode()[0], inplace=True)

df = df.replace(to_replace="3+", value=4)

# Label encoding for categorical variables
df.replace({
    "Loan_Status": {"N": 0, "Y": 1},
    "Married": {"No": 0, "Yes": 1},
    "Gender": {"Female": 0, "Male": 1},
    "Self_Employed": {"No": 0, "Yes": 1},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
    "Education": {"Not Graduate": 0, "Graduate": 1}
}, inplace=True)

# Drop unnecessary columns
x = df.drop(columns=['Loan_ID', 'Loan_Status'])
y = df['Loan_Status']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=2)

# Feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)  # Fit and scale the training data
x_test_scaled = scaler.transform(x_test)        # Transform the test data using the same scaler

# Define Logistic Regression model
logi = LogisticRegression(random_state=2)
logi.fit(x_train_scaled, y_train)  # Fit the model with scaled data

# Predictions
y_pred = logi.predict(x_test_scaled)

# Model evaluation
def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=['Not Eligible', 'Eligible'], yticklabels=['Not Eligible', 'Eligible'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Display metrics
metrics_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")

# Save the preprocessing and model
joblib.dump(logi, 'logis.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler
joblib.dump(x_train_scaled, 'x_trains.pkl')  # Save the scaled training data (if needed)
print("Model, scaler, and training data saved successfully!")
