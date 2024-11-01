# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib

# Load your pre-trained model (assuming it's saved as 'model.pkl')
model = joblib.load('model.pkl')

# Load dataset (assuming you have a CSV file with your data)
@st.cache  # Cache the data to make the app faster
def load_data():
    data = pd.read_csv('Telco Customer Churn.csv')
    return data

data = load_data()

# Sidebar for user input parameters
st.sidebar.header("Customer Input Parameters")

# Function to get user input
def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    SeniorCitizen = st.sidebar.selectbox("Senior Citizen", (0, 1))
    Partner = st.sidebar.selectbox("Partner", ("Yes", "No"))
    Dependents = st.sidebar.selectbox("Dependents", ("Yes", "No"))
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    MonthlyCharges = st.sidebar.slider("Monthly Charges", 10.0, 150.0, 70.0)
    TotalCharges = st.sidebar.slider("Total Charges", 0.0, 8000.0, 2000.0)

    # Convert categorical features to match model’s training requirements
    data = {
        'gender': 1 if gender == "Male" else 0,  # Male=1, Female=0
        'SeniorCitizen': SeniorCitizen,
        'Partner': 1 if Partner == "Yes" else 0,  # Yes=1, No=0
        'Dependents': 1 if Dependents == "Yes" else 0,  # Yes=1, No=0
        'tenure': tenure,
        'PhoneService': 1,  # Assuming default value (add other features as necessary)
        'MultipleLines': 0,  # Assuming default value, set accordingly
        'InternetService': 1,  # Assuming default value, set accordingly
        'OnlineSecurity': 0,
        'OnlineBackup': 0,
        'DeviceProtection': 0,
        'TechSupport': 0,
        'StreamingTV': 0,
        'StreamingMovies': 0,
        'Contract': 1,  # Default for Contract Type (set accordingly)
        'PaperlessBilling': 1,
        'PaymentMethod': 1,  # Default payment method encoding
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    
    # Convert to DataFrame with correct order of columns
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the main title and introduction
st.title("Customer Churn Prediction App")
st.write("""
This app predicts whether a customer will churn based on demographic and account information. Use the sidebar to input customer details.
""")

# Show the dataset
if st.checkbox("Show Raw Data"):
    st.write(data.head())

# Visualizations for Gender and Churn distributions
st.subheader("Gender Distribution")
gender_counts = data['gender'].value_counts()
fig_gender = go.Figure(data=[go.Pie(labels=gender_counts.index, values=gender_counts.values, hole=0.4)])
fig_gender.update_layout(title_text="Gender Distribution")
st.plotly_chart(fig_gender)

st.subheader("Churn Distribution")
churn_counts = data['Churn'].value_counts()
fig_churn = go.Figure(data=[go.Pie(labels=churn_counts.index, values=churn_counts.values, hole=0.4)])
fig_churn.update_layout(title_text="Churn Distribution")
st.plotly_chart(fig_churn)

# Model Prediction
st.subheader("Predict Churn")

if st.button("Predict Churn"):
    # Make sure input_df has the same columns as the training data
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]
    st.write(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
    st.write(f"Churn Probability: {prediction_proba:.2f}")
