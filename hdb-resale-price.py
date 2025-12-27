import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# -----------------------------
# Page config & global styling
# -----------------------------
st.set_page_config(
    page_title="HDB Resale Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# Custom CSS – same visual style as sentiment app / portfolio
CUSTOM_CSS = """
<style>
:root {
  --bg-page: #ffffff;
  --bg-card: #f9fafb;
  --text-main: #111827;
  --text-muted: #6b7280;
  --accent: #2563eb;
}

/* Overall page */
html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--bg-page) !important;
  color: var(--text-main) !important;
}

/* Typography */
h1, h2, h3, h4, h5, h6,
p, label, span, div,
.stMarkdown, .stMarkdown p {
  color: var(--text-main);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen,
               Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
}

/* Main content padding */
.block-container {
  padding-top: 2.5rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background-color: #111827;
}

[data-testid="stSidebar"] * {
  color: #e5e7eb !important;
}

/* Links / accents */
a, .stMarkdown a {
  color: var(--accent);
}

/* Cards */
.app-card {
  background-color: var(--bg-card);
  padding: 1.25rem 1.5rem 1.5rem;
  border-radius: 1rem;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
  border: 1px solid #e5e7eb;
}

/* Section titles inside cards */
.app-section-title {
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
}

/* Muted text */
.app-muted {
  color: var(--text-muted) !important;
  font-size: 0.9rem;
}

/* Result labels */
.app-badge {
  display: inline-block;
  padding: 0.35rem 0.7rem;
  border-radius: 999px;
  background-color: #e5f3ff;
  color: var(--accent);
  font-size: 0.8rem;
  font-weight: 600;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("### Chang Chee Young | AI & ML Developer")
st.title("HDB Resale Price Predictor")
st.markdown(
    "<p class='app-muted'>Predict Singapore HDB resale prices using an "
    "XGBoost regression model trained on historical transaction data.</p>",
    unsafe_allow_html=True,
)

# -----------------------------
# Load the model
# -----------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "models",
        "XBR_trained_hdb_resale_modelV4a.pkl",
    )
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


model = load_model()

if model is None:
    st.error(
        "Could not load the model. Please check if the model file exists "
        "in `models/XBR_trained_hdb_resale_modelV4a.pkl`."
    )
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("#### App Info")
    st.markdown(
        "- Built with **XGBoost Regression**\n"
        "- Deployed via **Streamlit**\n"
        "- Uses **one-hot encoded town & flat type**\n"
        "- Part of my portfolio on `changcheeyoung.github.io`"
    )
    st.markdown("---")
    st.markdown("#### How to use")
    st.markdown(
        "1. Enter property details.\n"
        "2. Click **Predict Price**.\n"
        "3. Review the predicted price and details."
    )

# -----------------------------
# Input Card (single main column, 2 sub-columns)
# -----------------------------
st.markdown("<div class='app-card'>", unsafe_allow_html=True)
st.markdown(
    "<div class='app-section-title'>🧾 Property Details</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='app-muted'>Fill in the fields below for the flat you are "
    "interested in. The model will estimate the resale price based on "
    "historical transactions.</p>",
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    # Numerical inputs
    floor_area = st.number_input(
        "Floor Area (sqm)",
        min_value=1.0,
        max_value=500.0,
        value=148.0,
    )
    lease_commence_date = st.number_input(
        "Lease Commencement Date",
        min_value=1960,
        max_value=2024,
        value=1992,
    )
    postal_code = st.number_input(
        "Postal Code",
        min_value=10000,
        max_value=999999,
        value=520329,
    )
    current_year = 2024  # You can make this dynamic if needed

with col2:
    # Categorical inputs
    towns = [
        "ANG MO KIO",
        "BEDOK",
        "BISHAN",
        "BUKIT BATOK",
        "BUKIT MERAH",
        "BUKIT PANJANG",
        "BUKIT TIMAH",
        "CENTRAL AREA",
        "CHOA CHU KANG",
        "CLEMENTI",
        "GEYLANG",
        "HOUGANG",
        "JURONG EAST",
        "JURONG WEST",
        "KALLANG/WHAMPOA",
        "MARINE PARADE",
        "PASIR RIS",
        "PUNGGOL",
        "QUEENSTOWN",
        "SEMBAWANG",
        "SENGKANG",
        "SERANGOON",
        "TAMPINES",
        "TOA PAYOH",
        "WOODLANDS",
        "YISHUN",
    ]
    selected_town = st.selectbox("Town", towns)

    flat_types = [
        "3 ROOM",
        "4 ROOM",
        "5 ROOM",
        "EXECUTIVE",
        "MULTI-GENERATION",
    ]
    selected_flat_type = st.selectbox("Flat Type", flat_types)

st.markdown(
    "<p class='app-muted' style='margin-top:0.25rem;'>"
    "Click <strong>Predict Price</strong> below to estimate the resale price."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Results Card
# -----------------------------
st.markdown("<div class='app-card'>", unsafe_allow_html=True)
st.markdown(
    "<div class='app-section-title'>📊 Predicted Price</div>",
    unsafe_allow_html=True,
)

predict_clicked = st.button("Predict Price")

if predict_clicked:
    try:
        # Create the feature array (same order as training data)
        features = [
            floor_area,
            lease_commence_date,
            postal_code,
            current_year,
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
        st.markdown(
            "<span class='app-badge'>Prediction</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"### ${rounded_price:,.0f}",
        )
        st.markdown(
            "<p class='app-muted'>Estimated resale price (rounded to the "
            "nearest $1,000). Actual transacted prices may vary.</p>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            "<span class='app-badge'>Property Summary</span>",
            unsafe_allow_html=True,
        )

        details = {
            "Floor Area": f"{floor_area} sqm",
            "Town": selected_town,
            "Flat Type": selected_flat_type,
            "Lease Commencement Date": lease_commence_date,
            "Postal Code": postal_code,
            "Assumed Current Year": current_year,
        }

        for key, value in details.items():
            st.write(f"**{key}:** {value}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
else:
    st.markdown(
        "<p class='app-muted'>Fill in the property details above and click "
        "<strong>Predict Price</strong> to see the estimated resale value.</p>",
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Model Information
# -----------------------------
with st.expander("Model Information"):
    st.write(
        """
This model uses **XGBoost Regression** to predict HDB resale prices based on
historical transaction data and engineered features.

The model achieves:
- Mean Absolute Error: ~**$15,749**
- R-squared Score: **0.982**

> Note: Predictions are estimates based on historical patterns.  
> Actual transacted prices may vary due to market conditions and flat-specific factors.
"""
    )
