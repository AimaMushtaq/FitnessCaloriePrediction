import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor


# -------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------

st.set_page_config(
    page_title="Fitness Calorie Predictor",
    page_icon="🏃",
    layout="wide"
)


# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------

@st.cache_resource
def train_model():

    # Load dataset
    data = pd.read_csv("calories.csv")

    # Remove extra spaces from column names
    data.columns = data.columns.str.strip()

    # Input features
    X = data[
        [
            "Gender",
            "Age",
            "Height",
            "Weight",
            "Duration",
            "Heart_Rate",
            "Body_Temp"
        ]
    ]

    # Output
    y = data["Calories"]

    # Numerical columns
    numeric_columns = [
        "Age",
        "Height",
        "Weight",
        "Duration",
        "Heart_Rate",
        "Body_Temp"
    ]

    # Categorical column
    categorical_columns = ["Gender"]

    # Numerical preprocessing
    numeric_transformer = Pipeline([
        ("missing", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_transformer = Pipeline([
        ("missing", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ("numbers", numeric_transformer, numeric_columns),
        ("categories", categorical_transformer, categorical_columns)
    ])

    # Regression algorithm
    regressor = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )

    # Complete pipeline
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("model", regressor)
    ])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    return model


# Train/load cached model
model = train_model()


# -------------------------------------------------
# MAIN PAGE
# -------------------------------------------------

st.title("🏃 Fitness Calorie Burn Prediction System")

st.write(
    "Estimate the calories burned during an exercise session "
    "using your personal details and workout information."
)

st.divider()

st.header("Enter Your Information")


# -------------------------------------------------
# INPUT FORM
# -------------------------------------------------

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


# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

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


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.caption(
    "Fitness Calorie Burn Prediction System | "
    "Machine Learning Regression Project"
)
