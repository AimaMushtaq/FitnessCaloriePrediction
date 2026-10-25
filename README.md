# Fitness Calorie Burn Prediction System

## Project Overview

The Fitness Calorie Burn Prediction System estimates the number of calories burned during an exercise session using machine learning regression algorithms.

## Input Features

The application uses the following features:

- Gender
- Age
- Height
- Weight
- Workout Duration
- Heart Rate
- Body Temperature

## Target Variable

The target variable is Calories, which represents the estimated calories burned during exercise.

## Machine Learning Methodology

Five regression algorithms are evaluated using 10-fold cross-validation. Their performance is compared using Root Mean Squared Error (RMSE) and the R² score. The model with the lowest mean RMSE is selected and trained on the complete dataset.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit

## Running the Application

Install the required packages:

    pip install -r requirements.txt

Start the application:

    streamlit run app.py

## Disclaimer

The predicted calorie value is an estimate produced by a machine learning model and should not be treated as a medical measurement.