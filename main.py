from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pickle
import pandas as pd
import numpy as np
from enum import Enum
import uvicorn

class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"

class FamilyHistoryEnum(str, Enum):
    yes = "yes"
    no = "no"

class FAVCEnum(str, Enum):
    yes = "yes"
    no = "no"

class CAECEnum(str, Enum):
    no = "no"
    Sometimes = "Sometimes"
    Frequently = "Frequently"
    Always = "Always"

class SmokeEnum(str, Enum):
    yes = "yes"
    no = "no"

class SCCEnum(str, Enum):
    yes = "yes"
    no = "no"

class CALCEnum(str, Enum):
    no = "no"
    Sometimes = "Sometimes"
    Frequently = "Frequently"
    Always = "Always"

class MTRANSEnum(str, Enum):
    Automobile = "Automobile"
    Bike = "Bike"
    Motorbike = "Motorbike"
    Public_Transportation = "Public_Transportation"
    Walking = "Walking"

class ObesityInput(BaseModel):
    Gender: GenderEnum = Field(..., description="Jenis kelamin: Male atau Female")
    Age: float = Field(..., ge=0, le=120, description="Usia dalam tahun")
    Height: float = Field(..., ge=0.5, le=3.0, description="Tinggi badan dalam meter")
    Weight: float = Field(..., ge=10, le=500, description="Berat badan dalam kilogram")
    family_history_with_overweight: FamilyHistoryEnum = Field(..., description="Riwayat keluarga obesitas")
    FAVC: FAVCEnum = Field(..., description="Sering konsumsi makanan berkalori tinggi")
    FCVC: float = Field(..., ge=1, le=3, description="Frekuensi konsumsi sayuran (1-3)")
    NCP: float = Field(..., ge=1, le=10, description="Jumlah makanan utama per hari")
    CAEC: CAECEnum = Field(..., description="Frekuensi makan di antara waktu makan")
    SMOKE: SmokeEnum = Field(..., description="Status merokok")
    CH2O: float = Field(..., ge=1, le=3, description="Asupan air harian (1-3)")
    SCC: SCCEnum = Field(..., description="Memantau asupan kalori")
    FAF: float = Field(..., ge=0, le=3, description="Frekuensi aktivitas fisik (0-3)")
    TUE: float = Field(..., ge=0, le=3, description="Waktu penggunaan teknologi (0-3)")
    CALC: CALCEnum = Field(..., description="Frekuensi konsumsi alkohol")
    MTRANS: MTRANSEnum = Field(..., description="Moda transportasi utama")

    class Config:
        schema_extra = {
            "example": {
                "Gender": "Male",
                "Age": 25.0,
                "Height": 1.75,
                "Weight": 70.0,
                "family_history_with_overweight": "yes",
                "FAVC": "yes",
                "FCVC": 2.0,
                "NCP": 3.0,
                "CAEC": "Sometimes",
                "SMOKE": "no",
                "CH2O": 2.0,
                "SCC": "no",
                "FAF": 1.0,
                "TUE": 2.0,
                "CALC": "no",
                "MTRANS": "Public_Transportation"
            }
        }

class PredictionResponse(BaseModel):
    prediction: str
    probability: dict
    bmi: float
    risk_level: str

app = FastAPI(
    title="Obesity Classification API",
    description="API untuk klasifikasi tingkat obesitas berdasarkan faktor gaya hidup dan karakteristik fisik",
    version="1.0.0"
)

model = None
def load_model():
    global model
    try:
        with open('C:/COOLYEAH/SEM 4/Model Deployment/UAS/best_obesity_model.pkl', 'rb') as file:
            model= pickle.load(file)

        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: Model file not found!")
        model = None
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None


def calculate_bmi(height: float, weight: float) -> float:
    return weight / (height ** 2)

def get_risk_level(prediction: str) -> str:
    risk_mapping = {
        "Insufficient_Weight": "Rendah - Berat badan kurang",
        "Normal_Weight": "Normal - Berat badan ideal",
        "Overweight_Level_I": "Sedang - Kelebihan berat badan tingkat 1",
        "Overweight_Level_II": "Sedang - Kelebihan berat badan tingkat 2",
        "Obesity_Type_I": "Tinggi - Obesitas tipe 1",
        "Obesity_Type_II": "Sangat Tinggi - Obesitas tipe 2",
        "Obesity_Type_III": "Ekstrem - Obesitas tipe 3"
    }
    return risk_mapping.get(prediction, "Unknown")

def preprocess_input(input_data: ObesityInput) -> pd.DataFrame:
    data_dict = {
        'Gender': input_data.Gender.value,
        'Age': input_data.Age,
        'Height': input_data.Height,
        'Weight': input_data.Weight,
        'family_history_with_overweight': input_data.family_history_with_overweight.value,
        'FAVC': input_data.FAVC.value,
        'FCVC': input_data.FCVC,
        'NCP': input_data.NCP,
        'CAEC': input_data.CAEC.value,
        'SMOKE': input_data.SMOKE.value,
        'CH2O': input_data.CH2O,
        'SCC': input_data.SCC.value,
        'FAF': input_data.FAF,
        'TUE': input_data.TUE,
        'CALC': input_data.CALC.value,
        'MTRANS': input_data.MTRANS.value
    }

    df = pd.DataFrame([data_dict])
    return df

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    return {
        "message": "Obesity Classification API",
        "description": "API untuk prediksi tingkat obesitas",
        "endpoints": {
            "/predict": "POST - Prediksi tingkat obesitas",
            "/health": "GET - Status kesehatan API",
            "/docs": "GET - Dokumentasi API"
        }
    }

@app.get("/health")
async def health_check():
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "message": "API is running properly"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_obesity(input_data: ObesityInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    try:
        processed_data = preprocess_input(input_data)  
        
        prediction = model.predict(processed_data)[0]
        prediction_proba = model.predict_proba(processed_data)[0]
        
        classes = model.classes_
        probability_dict = {
            class_name: float(prob) for class_name, prob in zip(classes, prediction_proba)
        }

        bmi = calculate_bmi(input_data.Height, input_data.Weight)
        risk_level = get_risk_level(prediction)

        return PredictionResponse(
            prediction=prediction,
            probability=probability_dict,
            bmi=round(bmi, 2),
            risk_level=risk_level
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


@app.post("/predict-batch")
async def predict_batch(input_data_list: list[ObesityInput]):
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded. Please check if 'best_obesity_model.pkl' exists."
        )
    
    try:
        results = []
        for input_data in input_data_list:
            processed_data = preprocess_input(input_data)
            
            prediction = model.predict(processed_data)[0]
            prediction_proba = model.predict_proba(processed_data)[0]
            
            if hasattr(model, 'classes_'):
                classes = model.classes_
            else:
                classes = [
                    'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
                    'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
                ]
            
            probability_dict = {
                class_name: float(prob) for class_name, prob in zip(classes, prediction_proba)
            }

            bmi = calculate_bmi(input_data.Height, input_data.Weight)
            
            risk_level = get_risk_level(prediction)
            
            results.append(PredictionResponse(
                prediction=prediction,
                probability=probability_dict,
                bmi=round(bmi, 2),
                risk_level=risk_level
            ))
        
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during batch prediction: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )