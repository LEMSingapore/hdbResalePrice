import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Set page config
st.set_page_config(
    page_title="HDB Resale Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("HDB Resale Price Predictor")
st.write("Predict HDB resale prices based on various features")

# Load the model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'XBR_trained_hdb_resale_modelV5.pkl')
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

if model is None:
    st.error("Could not load the model. Please check if the model file exists in the correct location.")
    st.stop()

# Create columns for input
col1, col2 = st.columns(2)

with col1:
    # Numerical inputs
    floor_area = st.number_input("Floor Area (sqm)", min_value=1.0, max_value=500.0, value=148.0)
    lease_commence_date = st.number_input("Lease Commencement Date", min_value=1960, max_value=2024, value=1992)
    postal_code = st.number_input("Postal Code", min_value=100000, max_value=999999, value=520329)
    current_year = 2024  # You can make this dynamic if needed

with col2:
    # Categorical inputs
    towns = [
        'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 
        'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 
        'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
        'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 
        'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
        'TOA PAYOH', 'WOODLANDS', 'YISHUN'
    ]
    selected_town = st.selectbox("Town", towns)

    flat_types = ['3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']
    selected_flat_type = st.selectbox("Flat Type", flat_types)

# Create prediction button
if st.button("Predict Price"):
    try:
        # Create the feature array (same order as training data)
        features = [
            floor_area,
            lease_commence_date,
            postal_code,
            current_year
        ]
        
        # Add town one-hot encoding
        for town in towns:
            features.append(1 if town == selected_town else 0)
        
        # Add flat_type one-hot encoding
        for flat_type in flat_types:
            features.append(1 if flat_type == selected_flat_type else 0)
        
        # Make prediction
        prediction = model.predict([features])
        
        # Round to nearest 1,000
        rounded_price = round(prediction[0] / 1000) * 1000
        
        # Display prediction
        st.success(f"Predicted Resale Price: ${rounded_price:,.2f}")
        
        # Additional details
        st.write("### Property Details")
        details = {
            "Floor Area": f"{floor_area} sqm",
            "Town": selected_town,
            "Flat Type": selected_flat_type,
            "Lease Commencement Date": lease_commence_date,
            "Postal Code": postal_code
        }
        
        # Display details in a nice format
        for key, value in details.items():
            st.write(f"**{key}:** {value}")
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Add some information about the model
with st.expander("Model Information"):
    st.write("""
    This model uses XGBoost Regression to predict HDB resale prices based on historical data.
    The model achieves:
    - Mean Absolute Error: ~$15,749
    - R-squared Score: 0.982
    
    Note: Predictions are estimates and actual prices may vary.
    """)
