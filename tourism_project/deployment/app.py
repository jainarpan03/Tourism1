import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="jarpan03/Tourism-Package-Prediction", filename="best_tourism_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Package Prediction App")
st.write("The Tourism Package Prediction App is an internal tool for 'Visit with Us,' predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.")
st.write("Kindly enter the customer details to check whether they are likely to take the package.")

# Collect user input
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
CityTier = st.selectbox("City Tier", ["Metro", "Tier-2","Tier-3"])
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=10)
Occupation = st.selectbox("Occupation", ["Free Lancer","Large Business", "Salaried", "Small Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=2)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar = st.number_input("Preferred Property Star (1–5)", min_value=1, max_value=5, value=3)
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced","Unmarried"])
NumberOfTrips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=2)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score (1–5)", min_value=1, max_value=5, value=3)
OwnCar = st.selectbox("Has Own Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (below 5 years)", min_value=0, max_value=10, value=0)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income (in ₹)", min_value=0, max_value=5000000, value=50000)


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age' :Age,
    'TypeofContact':TypeofContact,
    'CityTier' : 1 if CityTier == "Metro" else 2 if CityTier == "Tier-2" else 3,
    'DurationOfPitch':DurationOfPitch,
    'Occupation':Occupation,
    'Gender':Gender,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0    
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase" if prediction == 1 else "not purchase"
    st.write(f"Based on the information provided, the customer is likely to {result} the newly introduced Wellness Tourism Package.")
