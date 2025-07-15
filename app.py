import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
import pickle

# Load model and encoders
model = tf.keras.models.load_model('model.keras')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('onehot_encoder.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

# Page setup
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("Customer Churn Prediction")
st.markdown("""
This tool predicts the probability of customer churn based on demographic and banking data.  
Provide the customer details below to get an instant prediction.
""")

# Input form
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox("Geography", onehot_encoder.categories_[0])
        gender = st.selectbox("Gender", label_encoder.classes_)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        credit_score = st.number_input("Credit Score", value=600)
        tenure = st.slider("Tenure (Years)", 0, 10, value=5)

    with col2:
        balance = st.number_input("Account Balance", value=50000.0)
        estimated_salary = st.number_input("Estimated Salary", value=60000.0)
        num_of_products = st.slider("Number of Products", 1, 4, value=1)
        credit_card = st.radio("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x else "No")
        is_active_member = st.radio("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x else "No")

    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare input
    input_data = {
        'CreditScore': [credit_score],
        'Gender': [label_encoder.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [credit_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    }

    input_df = pd.DataFrame(input_data)

    geo_encoded = onehot_encoder.transform([[geography]]).toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))

    full_input = pd.concat([input_df.reset_index(drop=True), geo_df], axis=1)
    scaled_input = scaler.transform(full_input)

    # Prediction
    prediction = model.predict(scaled_input)
    churn_probability = prediction[0][0]

    # Output
    st.subheader("Prediction Result")
    col1, col2 = st.columns(2)
    col1.metric(label="Churn Probability", value=f"{churn_probability*100:.2f} %")

    if churn_probability > 0.5:
        col2.error("⚠️ Likely to Churn")
        st.warning("The customer has a high probability of churning. Consider retention strategies.")
    else:
        col2.success("✅ Likely to Stay")
        st.success("The customer is expected to remain loyal.")

