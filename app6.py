# Save this code as 'app.py'

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Ensure these are imported as they were saved

# --- Load Pre-trained Models and Preprocessors ---
try:
    # Load binomial model (e.g., Random Forest as it performed well)
    binomial_model = joblib.load('random_forest_binomial_model.joblib')
    # Load multinomial model
    multinomial_model = joblib.load('random_forest_multinomial_model.joblib')

    # Load preprocessors
    scaler_bin = joblib.load('scaler_binomial.joblib')
    encoder_bin = joblib.load('encoder_binomial.joblib')
    scaler_multi = joblib.load('scaler_multinomial.joblib')
    encoder_multi = joblib.load('encoder_multinomial.joblib')

    # Load feature lists and encoded names
    numerical_cols = joblib.load('numerical_features.joblib')
    categorical_cols = joblib.load('categorical_features.joblib')
    encoded_feature_names_bin = joblib.load('encoded_feature_names_bin.joblib')
    encoded_feature_names_multi = joblib.load('encoded_feature_names_multi.joblib')
    all_processed_cols_bin = joblib.load('all_processed_cols_bin.joblib')
    all_processed_cols_multi = joblib.load('all_processed_cols_multi.joblib')
    unique_multi_labels = joblib.load('unique_multi_labels.joblib')

    st.success("Models and preprocessors loaded successfully!")

except FileNotFoundError:
    st.error("Error: Model or preprocessor files not found. Please ensure 'random_forest_binomial_model.joblib', etc., are in the same directory.")
    st.stop() # Stop the app if files are missing
except Exception as e:
    st.error(f"An error occurred loading files: {e}")
    st.stop()

# --- Streamlit App Layout ---
st.set_page_config(page_title="Network Intrusion Detection", layout="wide")

st.title("Network Intrusion Detection System")
st.markdown("Enter the network traffic features below to predict if it's a normal connection or an attack.")

# --- Input Fields for Features ---
st.header("Input Network Connection Features")

# Create two columns for numerical and categorical inputs for better layout
col1, col2 = st.columns(2)

input_data = {}

with col1:
    st.subheader("Numerical Features")
    for col in numerical_cols:
        # Use appropriate input widgets based on feature ranges and types
        # For simplicity, using number_input, but you might use sliders etc.
        input_data[col] = st.number_input(f"{col}", value=0.0, step=0.1, key=f"num_{col}")

with col2:
    st.subheader("Categorical Features")
    for col in categorical_cols:
        # Get unique categories from the encoder's categories_ attribute
        # This is a bit complex for a generic example; you'd typically know your categories
        if col == 'protocol_type':
            options = ['tcp', 'udp', 'icmp'] # Example options for protocol_type
        elif col == 'service':
            options = ['http', 'smtp', 'ftp', 'other_service'] # Example options, use full list from original data
        elif col == 'flag':
            options = ['SF', 'S0', 'REJ', 'other_flag'] # Example options
        elif col in ['logged_in', 'is_host_login', 'is_guest_login']:
             options = [0, 1] # Binary options
        else:
            options = ['unknown'] # Fallback or infer from encoder.categories_ if available

        input_data[col] = st.selectbox(f"{col}", options=options, key=f"cat_{col}")


# --- Prediction Button ---
if st.button("Predict Attack Type"):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # --- Preprocessing Input Data (Crucial step for deployment) ---
    st.subheader("Preprocessing Input Data...")

    # Separate numerical and categorical parts of input_df
    input_numerical = input_df[numerical_cols]
    input_categorical = input_df[categorical_cols]

    # Apply scaling
    input_numerical_scaled = scaler_bin.transform(input_numerical) # Use binomial scaler for consistency
    input_numerical_df_scaled = pd.DataFrame(input_numerical_scaled, columns=numerical_cols, index=input_df.index)

    # Apply one-hot encoding
    input_categorical_encoded = encoder_bin.transform(input_categorical) # Use binomial encoder for consistency
    input_categorical_df_encoded = pd.DataFrame(input_categorical_encoded, columns=encoded_feature_names_bin, index=input_df.index)

    # Concatenate processed features
    processed_input = pd.concat([input_numerical_df_scaled, input_categorical_df_encoded], axis=1)

    # Ensure processed_input has the same columns and order as the training data
    # This is critical. Missing columns or extra columns will cause issues.
    # Align columns to the training features (all_processed_cols_bin)
    missing_cols = set(all_processed_cols_bin) - set(processed_input.columns)
    for c in missing_cols:
        processed_input[c] = 0 # Add missing columns with default value (e.g., 0 for one-hot encoded features)
    processed_input = processed_input[all_processed_cols_bin] # Ensure correct order

    st.success("Input data preprocessed!")

    # --- Make Predictions ---
    st.subheader("Prediction Results:")

    # Binomial Prediction
    bin_pred = binomial_model.predict(processed_input)[0]
    bin_prob = binomial_model.predict_proba(processed_input)[0]
    bin_prob_normal = bin_prob[0] if binomial_model.classes_[0] == 'normal' else bin_prob[1]
    bin_prob_attack = bin_prob[1] if binomial_model.classes_[1] == 'attack' else bin_prob[0]

    st.write(f"**Binomial Classification (Normal vs. Attack):**")
    if bin_pred == 'normal':
        st.success(f"Predicted: **NORMAL** connection")
        st.write(f"Probability of Normal: {bin_prob_normal:.2f}")
        st.write(f"Probability of Attack: {bin_prob_attack:.2f}")
    else:
        st.error(f"Predicted: **ATTACK**")
        st.write(f"Probability of Normal: {bin_prob_normal:.2f}")
        st.write(f"Probability of Attack: {bin_prob_attack:.2f}")


    # Multinomial Prediction (if attack is detected, or always run)
    st.write(f"**Multinomial Classification (Specific Attack Type):**")
    # For multinomial, ensure the processed_input matches multinomial model's expected columns
    # Re-align columns to the multinomial model's training features (all_processed_cols_multi)
    processed_input_multi = processed_input.copy() # Start with the same processed input
    missing_cols_multi = set(all_processed_cols_multi) - set(processed_input_multi.columns)
    for c in missing_cols_multi:
        processed_input_multi[c] = 0
    processed_input_multi = processed_input_multi[all_processed_cols_multi]


    multi_pred_idx = multinomial_model.predict(processed_input_multi)[0]
    # Convert numeric prediction back to label using unique_multi_labels
    multi_pred_label = multi_pred_idx # If model directly predicts labels, use it
    if hasattr(multinomial_model, 'classes_'): # If model has .classes_ attribute
        multi_pred_label = multinomial_model.classes_[multi_pred_idx] # Map index to class label

    multi_prob = multinomial_model.predict_proba(processed_input_multi)[0]
    multi_prob_df = pd.DataFrame({'Attack Type': unique_multi_labels, 'Probability': multi_prob}).sort_values(by='Probability', ascending=False)


    st.info(f"Predicted specific type: **{multi_pred_label}**")
    st.write("Top 3 Probabilities for Attack Types:")
    st.table(multi_prob_df.head(3))


st.markdown("---")
st.caption("Developed for Network Intrusion Detection System using Machine Learning")