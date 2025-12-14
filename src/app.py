import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="SA House Price Predictor", layout="centered")

st.title("🏡 SA House Price Predictor")
st.markdown("Enter property details below to get an estimated market price.")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        bedrooms = st.number_input("Bedrooms", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
        bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
        
    with col2:
        erf_size = st.number_input("Erf Size (m²)", min_value=0.0, value=500.0, step=10.0)
        property_type = st.selectbox("Type of Property", ["House", "Townhouse", "Apartment / Flat"])
        
    submitted = st.form_submit_button("Predict Price")

if submitted:
    # Prepare payload
    payload = {
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Erf_Size": erf_size,
        "Type_of_Property": property_type
    }
    
    # Call API
    try:
        # Assuming API is running locally on port 8000
        response = requests.post("http://localhost:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            price = result["predicted_price"]
            st.success(f"💰 Estimated Price: R {price:,.2f}")
        else:
            st.error(f"Error: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Is the backend running?")
