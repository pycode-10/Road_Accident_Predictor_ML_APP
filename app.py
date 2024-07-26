import numpy as np
import pandas as pd
import pickle
import streamlit as st
import os

# Function to load all the saved models

def load_models():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    models = {}
    
    # Dictionary to map model names to their files
    
    model_files = {
        "Linear Regression": "model_1.pkl",
        "Decision Tree": "model_2.pkl",
        "Random Forest": "model_3.pkl",
        "KNN": "model_4.pkl",
        "SVM": "model_5.pkl",
        "Gradient Boosting": "model_6.pkl"
    }
    
    for name, filename in model_files.items():
        pickle_file_path = os.path.join(current_dir, filename)
        with open(pickle_file_path, "rb") as pickle_in:
            models[name] = pickle.load(pickle_in)
    
    return models

# Load all models as models 
models = load_models()

def welcome():
    return "Welcome All!!"

def predictor(model, speed, weather, time, drunk, distraction, reckless):
    # Converting input values into numeric data
    
    speed = float(speed)
    weather = float(weather)
    time = float(time)
    drunk = float(drunk)
    distraction = float(distraction)
    reckless = float(reckless)
    
    # Makeing Prediction
    prediction = model.predict([[speed, weather, time, drunk, distraction, reckless]])
    return prediction

#Main function that describes the webpage layout
def main():
    st.set_page_config(page_title="Road Accident Prediction", layout="centered", initial_sidebar_state="expanded")
    
    st.markdown("""
    <style>
    .main {
        background-color: #282B2D; /* Dark background color */
        color: white; /* Text color */
    }
    .stButton>button {
        background-color: #e74c3c; /* Red button color */
        color: white;
    }
    .stButton>button:hover {
        background-color: #c0392b; /* Darker red on hover */
        color: white;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#e74c3c, #c0392b); /* Gradient sidebar background */
        color: white; /* Sidebar text color */
    }
    .stTitle {
        background-color: #e74c3c; /* Red title background */
        color: white; /* Title text color */
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Road Accident Prediction")
    
    st.markdown("""
    <div class="stTitle">
    <h2>Road Accident Predictor ML App</h2>
    </div>
    <p></p>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("User Input Parameters")
    
    # Input fields
    speed = st.sidebar.radio("Overspeeding", ("No", "Yes"))
    weather = st.sidebar.selectbox("Weather", ["Clear", "Rainy", "Foggy"])
    time = st.sidebar.radio("Time of Day", ('00:01-06:00', '06:01-12:00', '12:01-18:00', '18:01-00:00'))
    drunk = st.sidebar.radio("Drunk/Sober", ("Sober", "Drunk"))
    distraction = st.sidebar.selectbox("Distraction", ["No", "Mobile", "Music", "Other"])
    reckless = st.sidebar.radio("Reckless/Aggressive", ("No", "Yes"))
    
    # Mapping dictionaries
    speed_dict = {"No": 0, "Yes": 1}
    time_dict = {'00:01-06:00': 1, '06:01-12:00': 2, '12:01-18:00': 3, '18:01-00:00': 4}
    weather_dict = {"Clear": 1, "Rainy": 2, "Foggy": 3}
    drunk_dict = {"Sober": 0, "Drunk": 1}
    distraction_dict = {"No": 3, "Mobile": 1, "Music": 2, "Other": 4}
    reckless_dict = {"No": 0, "Yes": 1}
    
    # Convert inputs to numerical values
    speed = speed_dict[speed]
    time = time_dict[time]
    weather = weather_dict[weather]
    drunk = drunk_dict[drunk]
    distraction = distraction_dict[distraction]
    reckless = reckless_dict[reckless]
    
    # Selecting the Model
    model_names = ["Linear Regression", "Decision Tree", "Random Forest", "KNN", "SVM", "Gradient Boosting"]
    model_choice = st.sidebar.selectbox("Choose a model", model_names)
    
    result = ""
    if st.button("Predict", key="predict_button"):
        result = predictor(models[model_choice], speed, weather, time, drunk, distraction, reckless)
        st.success('The probability of the vehicle being involved in an accident is {}'.format(result))
        
    if st.button("About", key="about_button"):
        st.markdown("""
        <div style="background-color:tomato;padding:10px;border-radius:10px">
        <p style="color:white;">Machine Learning Application that predicts the probability of a vehicle being involved in an accident, based on the parameters entered by the user</p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("Disclaimer", key="disclaimer_button"):
        st.markdown("""
        <div style="background-color:tomato;padding:10px;border-radius:10px">
        <p style="color:white;text-align:center;">The predicted probability of vehicle accident provided by this model is a theoretical estimate based on the data it was trained on and the parameters entered by the user. This prediction is intended for informational purposes only and should not be considered a definitive assessment of risk.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
