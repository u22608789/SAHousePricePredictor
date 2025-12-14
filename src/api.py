from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import numpy as np

app = FastAPI(title="SA House Price Predictor API")

# Load model
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "model.joblib")

try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class PropertyInput(BaseModel):
    Bedrooms: float
    Bathrooms: float
    Erf_Size: float
    Type_of_Property: str

    class Config:
        schema_extra = {
            "example": {
                "Bedrooms": 3,
                "Bathrooms": 2,
                "Erf_Size": 500,
                "Type_of_Property": "House"
            }
        }

@app.get("/")
def read_root():
    return {"message": "Welcome to the SA House Price Predictor API"}

@app.post("/predict")
def predict_price(prop: PropertyInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Create DataFrame from input
    input_data = pd.DataFrame([{
        "Bedrooms": prop.Bedrooms,
        "Bathrooms": prop.Bathrooms,
        "Erf Size": prop.Erf_Size,
        "Type of Property": prop.Type_of_Property
    }])

    # Preprocess (pipeline handles this)
    try:
        prediction = model.predict(input_data)
        predicted_price = float(prediction[0])
        return {
            "predicted_price": predicted_price,
            "currency": "ZAR"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
