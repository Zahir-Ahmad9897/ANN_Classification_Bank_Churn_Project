
import pickle
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# -------------------------
# Configuration & Constants
# -------------------------
st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

# Paths to the model and artifacts
# Using relative paths assuming these files are in the same directory
MODEL_PATH = 'Regression_model.h5'
LABEL_ENCODER_GENDER_PATH = 'label_encoder_gender.pkl'
ONE_HOT_ENCODER_GEO_PATH = 'one_hot_encoder.pkl'
SCALER_PATH = 'scaler.pkl'

# -------------------------
# Load Resources
# -------------------------
@st.cache_resource
def load_resources():
    try:
        # Load Model
        # Note: If the model file is named differently, update MODEL_PATH
        model = load_model(MODEL_PATH)
        
        # Load Encoders and Scaler
        with open(LABEL_ENCODER_GENDER_PATH, 'rb') as f:
            le_gender = pickle.load(f)
            
        with open(ONE_HOT_ENCODER_GEO_PATH, 'rb') as f:
            ohe_geo = pickle.load(f)
            
        with open(SCALER_PATH, 'rb') as f:
            scaler_obj = pickle.load(f)
            
        return model, le_gender, ohe_geo, scaler_obj
        
    except FileNotFoundError as e:
        st.error(f"Error loading resources: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.stop()

model, label_encoder_gender, one_hot_encoder_geo, scaler = load_resources()

# -------------------------
# User Interface
# -------------------------
st.title("🏦 Bank Customer Churn Prediction")
st.markdown("Enter customer details below to predict the likelihood of churn.")

with st.form("churn_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
        geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
        gender = st.selectbox("Gender", ['Male', 'Female'])
        age = st.slider("Age", 18, 100, 40)
        tenure = st.slider("Tenure (Years)", 0, 10, 3)

    with col2:
        balance = st.number_input("Balance", value=60000.0, step=1000.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_cr_card = st.selectbox("Has Credit Card?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active_member = st.selectbox("Is Active Member?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        estimated_salary = st.number_input("Estimated Salary", value=50000.0, step=1000.0)

    # Submit button
    submitted = st.form_submit_button("Predict Churn Status")

# -------------------------
# Prediction Logic
# -------------------------
if submitted:
    # 1. Create DataFrame from input
    input_data = {
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    }
    input_df = pd.DataFrame(input_data)
    
    try:
        # 2. Preprocessing
        # Gender Encoding
        # Note: If input gender not in classes, handle gracefully or ensure selectbox constraints
        input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])
        
        # Geography Encoding
        geo_encoded = one_hot_encoder_geo.transform(input_df[['Geography']]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))
        
        # Drop original Geography column and concatenate encoded ones
        input_df = input_df.drop('Geography', axis=1)
        input_df = pd.concat([input_df, geo_encoded_df], axis=1)
        
        # Scaling
        # Ensure the columns order matches what the scaler expects
        # (This is implicitly handled if the training DataFrame had the same column structure)
        input_scaled = scaler.transform(input_df)
        
        # 3. Prediction
        prediction = model.predict(input_scaled)
        prob = prediction[0][0]
        
        # 4. Result Display
        st.markdown("---")
        st.subheader("Prediction Result")
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.metric(label="Churn Probability", value=f"{prob:.2%}")
        
        with col_res2:
            if prob > 0.5:
                st.error("⚠️ **High Risk**: This customer is likely to churn.")
            else:
                st.success("✅ **Low Risk**: This customer is likely to stay.")
                
        # Optional: Add a progress bar for visual impact
        st.progress(float(prob))
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
