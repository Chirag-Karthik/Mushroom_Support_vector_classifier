import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('SVM.joblib')

# Load the scaler that was used for numerical features (assuming you saved it)
# If you didn't save the scaler, you might need to retrain it on the training data or
# use the scaler that was fit on the full dataset before splitting.
# For this example, let's assume the scaler was fit on the original df before splitting.
# In a real deployment, it's best to save and load the fitted scaler.

# Define a function to preprocess user input
def preprocess_input(data):
    # Assuming 'stalk_height' and 'cap_diameter' are the numerical features
    numerical_cols = ['stalk_height', 'cap_diameter']
    # Assuming all other columns are categorical and were label encoded

    # Create a DataFrame from the input data
    input_df = pd.DataFrame([data])

    # Apply the same scaling to numerical features as done during training
    # This requires having the fitted scaler available.
    # For demonstration, let's assume we refit the scaler on the entire original df here
    # In production, you'd load the saved, fitted scaler
    scaler = StandardScaler()
    # Fit the scaler on the original data and then transform the input data
    # This is a simplification; ideally, save and load the fitted scaler
    # For this to work, you would need access to the original dataframe 'df' or its scaled version
    # As a workaround for this example, we'll just scale the input directly
    # using a scaler fitted on a dummy dataset with similar range or assuming the original scaler's parameters are known
    # A better approach is to save the fitted scaler during training

    # Let's assume we saved the scaler and load it here
    # For this example, we'll simulate applying scaling based on the mean and std of the training data if we had them
    # As a simple example, let's just pass the input data without scaling for now.
    # **Important:** In a real application, you MUST apply the same preprocessing steps (scaling, encoding)
    # that were applied to the training data to the new input data.

    # Since the model was trained on label-encoded data, we need to label encode the input categorical features
    # This requires mapping the input string values to the numerical labels used during training.
    # This mapping should also be saved during training and loaded here.

    # For simplicity in this example, let's assume the input data is already in the correct numerical format
    # based on the label encoding and scaling applied during training.
    # **In a real application, you need to implement the label encoding and scaling here.**

    return input_df

# Create the Streamlit app
st.title("Mushroom Classification App")

st.write("Enter the features of the mushroom to predict if it is edible or poisonous.")

# Create input fields for features
# You need to create input fields for all 24 features your model was trained on.
# The type of input field depends on the feature (e.g., text input for categorical, number input for numerical)

# Example input fields (replace with all your features and appropriate input types)
cap_shape = st.text_input("Cap Shape")
cap_surface = st.text_input("Cap Surface")
cap_color = st.text_input("Cap Color")
bruises = st.text_input("Bruises")
odor = st.text_input("Odor")
gill_attachment = st.text_input("Gill Attachment")
gill_spacing = st.text_input("Gill Spacing")
gill_size = st.text_input("Gill Size")
gill_color = st.text_input("Gill Color")
stalk_shape = st.text_input("Stalk Shape")
stalk_root = st.text_input("Stalk Root")
stalk_surface_above_ring = st.text_input("Stalk Surface Above Ring")
stalk_surface_below_ring = st.text_input("Stalk Surface Below Ring")
stalk_color_above_ring = st.text_input("Stalk Color Above Ring")
stalk_color_below_ring = st.text_input("Stalk Color Below Ring")
veil_type = st.text_input("Veil Type")
veil_color = st.text_input("Veil Color")
ring_number = st.text_input("Ring Number")
ring_type = st.text_input("Ring Type")
spore_print_color = st.text_input("Spore Print Color")
population = st.text_input("Population")
habitat = st.text_input("Habitat")
stalk_height = st.number_input("Stalk Height")
cap_diameter = st.number_input("Cap Diameter")


# Create a dictionary with input data (replace with all your features)
input_data = {
    'cap_shape': cap_shape,
    'cap_surface': cap_surface,
    'cap_color': cap_color,
    'bruises': bruises,
    'odor': odor,
    'gill_attachment': gill_attachment,
    'gill_spacing': gill_spacing,
    'gill_size': gill_size,
    'gill_color': gill_color,
    'stalk_shape': stalk_shape,
    'stalk_root': stalk_root,
    'stalk_surface_above_ring': stalk_surface_above_ring,
    'stalk_surface_below_ring': stalk_surface_below_ring,
    'stalk_color_above_ring': stalk_color_above_ring,
    'stalk_color_below_ring': stalk_color_below_ring,
    'veil_type': veil_type,
    'veil_color': veil_color,
    'ring_number': ring_number,
    'ring_type': ring_type,
    'spore_print_color': spore_print_color,
    'population': population,
    'habitat': habitat,
    'stalk_height': stalk_height,
    'cap_diameter': cap_diameter,
}


# Create a predict button
if st.button("Predict"):
    # Preprocess the input data
    # **Important:** You need to implement the preprocessing steps (label encoding and scaling) here
    # to match the training data format.
    processed_data = preprocess_input(input_data)

    # Make a prediction
    prediction = model.predict(processed_data)

    # Display the prediction
    if prediction[0] == 0:
        st.success("The mushroom is predicted to be edible.")
    else:
        st.error("The mushroom is predicted to be poisonous.")
