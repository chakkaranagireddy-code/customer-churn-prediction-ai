import streamlit as st
import pandas as pd
import joblib


# Load trained model
model = joblib.load("churn_model.joblib")


# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction AI",
    page_icon="🤖",
    layout="centered"
)


# Title
st.title("🤖 Customer Churn Prediction AI")

st.write(
    "Enter customer information to predict the probability of customer churn."
)


# Customer inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=120,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=500.0,
    value=80.0
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["Fiber", "DSL"]
)


# Prediction button
if st.button("Predict Churn"):

    customer = pd.DataFrame({
        "age": [age],
        "tenure": [tenure],
        "monthly_charges": [monthly_charges],
        "contract": [contract],
        "internet_service": [internet_service]
    })

    # Prediction
    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0][1]

    churn_percentage = probability * 100


    # Display result
    st.subheader("Prediction Result")

    st.write(
        f"Churn Probability: **{churn_percentage:.2f}%**"
    )

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")
