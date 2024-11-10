import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('ensemble_model.joblib')

# Set up the UI
st.title("Machine Learning Application")
st.subheader("Choose File:")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)

    # Button to make predictions
    if st.button("Submit"):
        # Make predictions
        predictions = model.predict(data)

        # Map predictions to "good"/"bad"
        result_labels = ["good" if pred == 1 else "bad" for pred in predictions]
        
        # Display prediction results
        st.subheader("Prediction Results:")
        data['Prediction'] = result_labels
        st.write(data)  # This will display the prediction results
