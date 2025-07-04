import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
from geopy.distance import geodesic

# --- Load model and encoders
model = joblib.load("fraud_detector.jb")
encoder = joblib.load("label_enc.jb")


def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km


# --- App Title & Description
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>💳 Fraud Detection System</h1>
    <p style='text-align: center;'>Enter transaction details below to check for potential fraud.</p>
    <hr style='border: 1px solid #4CAF50;'>
    """,
    unsafe_allow_html=True,
)

# --- Main Input Form
with st.form("fraud_form"):
    st.subheader("📋 Transaction Details")
    col1, col2 = st.columns(2)
    with col1:
        merchant = st.text_input("Merchant Name")
        category = st.text_input("Category")
        amt = st.number_input("Transaction Amount ($)", min_value=0.0, format="%.2f")
        cc_num = st.text_input("Credit Card Number")
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        lat = st.number_input("Latitude", format="%.6f")
        long = st.number_input("Longitude", format="%.6f")
        merch_lat = st.number_input("Merchant Latitude", format="%.6f")
        merch_long = st.number_input("Merchant Longitude", format="%.6f")
        hour = st.slider("Transaction Hour", 0, 23, 12)
        day = st.slider("Transaction Day", 1, 31, 15)
        month = st.slider("Transaction Month", 1, 12, 6)

    submitted = st.form_submit_button("🚀 Check for Fraud")

# --- Prediction
if submitted:
    if merchant and category and cc_num:
        distance = haversine(lat, long, merch_lat, merch_long)

        input_data = pd.DataFrame(
            [[merchant, category, amt, distance, hour, day, month, gender, cc_num]],
            columns=[
                "merchant",
                "category",
                "amt",
                "distance",
                "hour",
                "day",
                "month",
                "gender",
                "cc_num",
            ],
        )

        categorical_col = ["merchant", "category", "gender"]
        for col in categorical_col:
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except ValueError:
                input_data[col] = -1

        input_data["cc_num"] = input_data["cc_num"].apply(lambda x: hash(x) % (10**2))

        prediction = model.predict(input_data)[0]
        result = (
            "🛑 **Fraudulent Transaction Detected!** ⚠️"
            if prediction == 1
            else "✅ **Legitimate Transaction**"
        )

        result_color = "#FF4B4B" if prediction == 1 else "#4CAF50"

        st.markdown(
            f"<h2 style='text-align: center; color: {result_color};'>{result}</h2>",
            unsafe_allow_html=True,
        )
    else:
        st.error("⚠️ Please fill all required fields.")


# --- Footer
st.markdown(
    """
    <hr style='border: 0.5px solid #DDDDDD;'>
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
        Made with ❤️ by <b>Dhanraj Sharma</b>
    </div>
    """,
    unsafe_allow_html=True,
)
