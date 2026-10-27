# Model and Scller available in D17_artifacts1, using this we are going to do prediction.
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow import keras

ARTIFACT_DIR = Path(__file__).parent / "D17_artifacts1"
MODEL_PATH = ARTIFACT_DIR / "sine_regression_model.keras"
SCALER_PATH = ARTIFACT_DIR / "input_scaler.joblib"

if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    raise RuntimeError(
        "Model/scaler files not found. Run the training notebook first."
    )

model = keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = FastAPI(                                                      # this is my FastAPI object.
    title="Day 18 ANN Prediction API",                              # this is doc strinc, these are optional 
    description="Beginner demo API for a saved regression model",
    version="1.0.0",
)

class PredictionRequest(BaseModel):
    input_value: float = Field(
        ..., gt=0, le=250,
        description="Numeric input for the demo model (must be > 0 and <= 250)"
    )

class PredictionResponse(BaseModel):
    input_value: float
    predicted_value: float

@app.get("/")                                                   # associated with root("/") funciton
def root():
    return {"message": "ANN prediction API is running"}

@app.get("/health")                                             # this end point is associated with health function.
def health():                                                   # get method means just we are providing
    return {"status": "ok", "model_loaded": True}               # to check the status, model is properly loaded or not

@app.post("/predict", response_model=PredictionResponse)        # this end point is associated with health predict.
def predict(request: PredictionRequest):
    try:
        raw_input = np.array([[request.input_value]], dtype=np.float32)
        scaled_input = scaler.transform(raw_input)
        result = model.predict(scaled_input, verbose=0)
        return {
            "input_value": request.input_value,
            "predicted_value": float(result[0, 0]),
        }
    except Exception as exc:                                        # compulsary it will return in dictionary 
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


# print("API file:", Path(__file__).resolve())
# print("Artifact directory:", ARTIFACT_DIR.resolve())
# print("Model exists:", MODEL_PATH.exists())
# print("Scaler exists:", SCALER_PATH.exists())


# # How to run FastAPI application
# # D:\Tamil\Dl\uvicorn api:app --reload                ==>(app is pydantic object name)
#http://127.0.0.1:8000/docs          ==> get method
#http://127.0.0.1:8000/health        ==> get method
#http://127.0.0.1:8000/health        ==> post method, we cant see the result in the browser, we have to use swagerui

# the advangage of FastAPI is we can get the documentation here
# For testing purpose only we are using browser, In real time for testing purpose we have to write anoter python file.
# I am providing request through a python file, not from the browser    

# / End point is nothing but a funciton. (/ means end point)
# / prediction
# / health
# if you want to provide your function to the externel world, we have to create as a end point(API end point)
# to create end point
#  @.app(ojcecat), http method we have to provide, 
#  @app.http method we have to provide (get method ==> just extracting, and 
#                                                     post method ==> means when ever we are providing value in the body of the request, is called as postmethod )

# when you mention /health in the url, the associated helath funciton is going to call
# EX in Brower:=>  http:localhost/8000/health


 
# # the advangage of FastAPI is we can get the documentation here
# # For testing purpose only we are using browser, In real time for testing purpose we have to write anoter python file.
# # I am providing request through a python file, not from the browser 

