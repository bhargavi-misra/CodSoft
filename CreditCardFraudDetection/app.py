import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load your model, scaler, and label encoders
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

merchant_le = joblib.load('merchant_le.pkl')
category_le = joblib.load('category_le.pkl')
gender_le = joblib.load('gender_le.pkl')
city_le = joblib.load('city_le.pkl')
state_le = joblib.load('state_le.pkl')
zip_le = joblib.load('zip_le.pkl')

# Function to safely transform categorical inputs
def safe_transform(le, val):
    if val not in le.classes_:
        return -1  # Unknown category handled as -1
    else:
        return le.transform([val])[0]

st.title("Credit Card Fraud Detection")

# Collect user inputs
cc_num = st.number_input("Credit Card Number (integer)", min_value=0, step=1)
merchant = st.text_input("Merchant")
category = st.text_input("Category")
amt = st.number_input("Amount", min_value=0.0, format="%.2f")
gender = st.selectbox("Gender", options=['M', 'F', 'Other'])
city = st.text_input("City")
state = st.text_input("State")
zip_code = st.text_input("Zip Code")

if st.button("Predict Fraud"):

    # Encode categorical variables safely
    merchant_enc = safe_transform(merchant_le, merchant)
    category_enc = safe_transform(category_le, category)
    gender_enc = safe_transform(gender_le, gender)
    city_enc = safe_transform(city_le, city)
    state_enc = safe_transform(state_le, state)
    zip_enc = safe_transform(zip_le, zip_code)

    # Prepare feature array in the correct order
    features = np.array([cc_num, merchant_enc, category_enc, amt, gender_enc, city_enc, state_enc, zip_enc]).reshape(1, -1)

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict probability
    pred_prob = model.predict_proba(features_scaled)[0][1]

    # Threshold can be your chosen one (e.g. 0.4444)
    threshold = 0.4444
    pred_class = int(pred_prob >= threshold)

    st.write(f"**Fraud Probability:** {pred_prob:.4f}")
    st.write(f"**Prediction:** {'Fraud' if pred_class == 1 else 'Not Fraud'}")
