from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

app = FastAPI(title="Housing Price Prediction API")

# Load model and scaler
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

class HouseFeatures(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    parking: int
    mainroad: str # 'yes' or 'no'
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    prefarea: str
    furnishingstatus: str # 'furnished', 'semi-furnished', 'unfurnished'

@app.post("/predict")
def predict(features: HouseFeatures):
    #convert binary features to 0/1
    binary_map = {'yes': 1, 'no': 0}

    input_data = np.array([
        features.area,
        features.bathrooms,
        features.stories,
        binary_map[features.airconditioning],
        features.parking,
        features.bedrooms,
        binary_map[features.prefarea],
        binary_map[features.mainroad],
        binary_map[features.guestroom],
        1 if features.furnishingstatus == 'furnished' else 0,
        binary_map[features.basement],
        binary_map[features.hotwaterheating],
        1 if features.furnishingstatus == 'semi-furnished' else 0,
        1 if features.furnishingstatus == 'unfurnished' else 0
    ]).reshape(1, -1)

    #Scale features
    scaled_data = scaler.transform(input_data)

    #make prediction
    prediction = model.predict(scaled_data)

    return {"predicted_price": prediction[0]}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Housing Price Prediction API. Use the /predict endpoint to get predictions."}