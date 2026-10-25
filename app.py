import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Fitness Calorie Predictor",
    page_icon="🏃",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "best_fitness_model.pkl"
    )

    with open(model_path, "rb") as file:
        model = joblib.load(model_path)

    return model

model = load_model()

# Main heading
st.title("🏃 Fitness Calorie Burn Prediction System")

st.write(
    "Estimate the calories burned during an exercise session "
    "using your personal details and workout information."
)

st.divider()

# Input section
st.header("Enter Your Information")

with st.form("fitness_form"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["female", "male"]
        )

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=100,
            value=22
        )

        height = st.number_input(
            "Height (cm)",
            min_value=50.0,
            max_value=250.0,
            value=165.0
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=10.0,
            max_value=300.0,
            value=60.0
        )

    with col2:
        duration = st.number_input(
            "Workout Duration (minutes)",
            min_value=1.0,
            max_value=300.0,
            value=60.0
        )

        heart_rate = st.number_input(
            "Heart Rate (BPM)",
            min_value=30.0,
            max_value=250.0,
            value=140.0
        )

        body_temp = st.number_input(
            "Body Temperature (°C)",
            min_value=30.0,
            max_value=45.0,
            value=38.0
        )

    submitted = st.form_submit_button(
        "Predict Calories Burned",
        use_container_width=True
    )

# Prediction
if submitted:

    input_data = pd.DataFrame({
        "Gender": [gender],
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp]
    })

    prediction = model.predict(input_data)[0]

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        label="Estimated Calories Burned",
        value=f"{prediction:.2f} calories"
    )

    st.info(
        "This result is a machine-learning estimate, "
        "not a medical measurement."
    )

# Footer
st.divider()

st.caption(
    "Fitness Calorie Burn Prediction System | "
    "Machine Learning Regression Project"
)