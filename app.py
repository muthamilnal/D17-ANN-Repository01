# Stream lit application
# Create an input field, a Predict button, and an area to display the model's prediction.
# With the help of <Streamlit> without know HTML and CSS, with the help of python we can easily build

from pathlib import Path
import joblib
import numpy as np
import streamlit as st
from tensorflow import keras

# ARTIFACT_DIR = Path(__file__).parent / "D17_artifacts1"
# MODEL_PATH = ARTIFACT_DIR / "sine_regression_model.keras"
# SCALER_PATH = ARTIFACT_DIR / "input_scaler.joblib"

ARTIFACT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ARTIFACT_DIR / "sine_regression_model.keras"
SCALER_PATH = ARTIFACT_DIR / "input_scaler.joblib"

st.set_page_config(page_title="GKTrainings/MUTHAMIL ANN Prediction Demo", page_icon="🧠")
st.title("Deep Learning Model Prediction")
st.write("Enter a numeric input and ask the saved ANN to predict a value.")
st.caption("Educational synthetic-data example — not for real-world decisions.")

if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    st.error("Model/scaler files not found. Run the Day 18 training notebook first.")
    st.stop()

@st.cache_resource      # catch means load only one time, it can server any request. response time will be improved
def load_artifacts():
    loaded_model = keras.models.load_model(MODEL_PATH)  # These details will be cateced.(Model and Scaler)
    loaded_scaler = joblib.load(SCALER_PATH)
    return loaded_model, loaded_scaler

model, scaler = load_artifacts()

input_value = st.number_input(          # Providing input in the URL (like input python)
    "Input value",                      # Label
    min_value=0.1,
    max_value=250.0,
    value=100.0,
    step=2.0,
)

if st.button("Predict"):
    raw_input = np.array([[input_value]], dtype=np.float32)
    scaled_input = scaler.transform(raw_input)
    prediction = model.predict(scaled_input, verbose=0)
    predicted_value = float(prediction[0, 0])

    st.success(f"Predicted value: {predicted_value:.2f}")
    st.write(f"Input supplied: {input_value:.2f}")


# How to run straem lit application.


