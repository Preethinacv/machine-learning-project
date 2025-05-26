## Machine Learning Web App that predicts whether a loan applicant is eligible for a loan based on personal, financial, and credit details. It uses Logistic Regression for prediction and Streamlit for the frontend.

#### Model Training (main.py)
Loads and preprocesses the dataset (loanprediction.csv).

Handles missing values, encodes categorical variables, and creates a total_income feature.

Scales features using StandardScaler.

Trains a Logistic Regression model to predict loan eligibility.

Saves the trained model and scaler using joblib.

#### Web App (app.py)
Built with Streamlit to provide a simple user interface.

Takes user input for loan-related details.

Loads the saved model and scaler.

Predicts loan eligibility based on input and displays Eligible or Not Eligible.
