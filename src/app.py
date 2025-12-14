import streamlit as st
import requests
import pandas as pd

# Page config
st.set_page_config(
    page_title="SA House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border-color: #ff3333;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/25/25694.png", width=100)
    st.title("About")
    st.info(
        """
        This app predicts house prices in South Africa (Johannesburg) using a Random Forest model.
        
        **Features used:**
        - Bedrooms
        - Bathrooms
        - Erf Size (m²)
        - Property Type
        """
    )
    st.markdown("---")
    st.caption("Built with FastAPI & Streamlit")

# Main Content
st.title("🏡 SA House Price Predictor")
st.markdown("### Estimate the market value of your property")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Property Details")
    with st.form("prediction_form"):
        bedrooms = st.slider("Bedrooms", 0.0, 10.0, 3.0, 0.5)
        bathrooms = st.slider("Bathrooms", 0.0, 10.0, 2.0, 0.5)
        erf_size = st.number_input("Erf Size (m²)", min_value=0.0, value=500.0, step=10.0)
        property_type = st.selectbox("Type of Property", ["House", "Townhouse", "Apartment / Flat"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮 Predict Price")

with col2:
    st.markdown("#### Prediction Result")
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
            with st.spinner("Calculating..."):
                response = requests.post("http://localhost:8000/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                price = result["predicted_price"]
                
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Estimated Market Value</div>
                        <div class="metric-value">R {price:,.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Add some context/visuals (mockup)
                st.success("Prediction successful!")
                st.balloons()
                
            else:
                st.error(f"Error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Is the backend running?")
    else:
        st.info("👈 Enter details and click Predict to see the result here.")
        st.image("https://img.freepik.com/free-vector/house-searching-concept-illustration_114360-466.jpg?w=740&t=st=1696434000~exp=1696434600~hmac=...", caption="Find your dream home value")

