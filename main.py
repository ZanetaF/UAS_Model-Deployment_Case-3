from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import pickle
import pandas as pd
import numpy as np
from enum import Enum
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    @field_validator('Height')
    @classmethod
    def validate_height(cls, v):
        if v <= 0:
            raise ValueError('Height must be greater than 0')
        return v
    
    @field_validator('Weight')
    @classmethod
    def validate_weight(cls, v):
        if v <= 0:
            raise ValueError('Weight must be greater than 0')
        return v

    model_config = {
        "json_schema_extra": {
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
    }

class BMIInput(BaseModel):
    height: float = Field(..., gt=0, description="Tinggi badan dalam meter")
    weight: float = Field(..., gt=0, description="Berat badan dalam kilogram")

class BMIInfo(BaseModel):
    value: float = Field(..., description="Nilai BMI")
    category: str = Field(..., description="Kategori BMI")
    description: str = Field(..., description="Deskripsi kategori BMI")

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Prediksi klasifikasi obesitas")
    probability: Dict[str, float] = Field(..., description="Probabilitas untuk setiap kelas")
    bmi: BMIInfo = Field(..., description="Informasi BMI lengkap")
    risk_level: str = Field(..., description="Tingkat risiko kesehatan")
    recommendation: str = Field(..., description="Rekomendasi kesehatan")

model = None

def load_model():
    global model
    try:
        model_path = 'best_obesity_model.pkl'
        
        with open(model_path, 'rb') as file:
            model = pickle.load(file) 
        
        logger.info("Model loaded successfully!")
        return True
    except FileNotFoundError:
        logger.error(f"Error: Model file not found at {model_path}")
        model = None
        return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Obesity Classification API...")
    success = load_model()
    if success:
        logger.info("API started successfully with model loaded")
    else:
        logger.warning("API started but model failed to load")
    
    yield
    
    logger.info("Shutting down Obesity Classification API...")

app = FastAPI(
    title="Obesity Classification API",
    description="API untuk klasifikasi tingkat obesitas berdasarkan faktor gaya hidup dan karakteristik fisik",
    version="1.0.0",
    lifespan=lifespan
)

def calculate_bmi(height: float, weight: float) -> Dict[str, Any]:
    try:
        bmi_value = weight / (height ** 2)
        
        if bmi_value < 18.5:
            category = "Underweight"
            description = "Berat badan kurang dari normal"
        elif 18.5 <= bmi_value < 25.0:
            category = "Normal"
            description = "Berat badan normal dan sehat"
        elif 25.0 <= bmi_value < 30.0:
            category = "Overweight"
            description = "Kelebihan berat badan"
        elif 30.0 <= bmi_value < 35.0:
            category = "Obese Class I"
            description = "Obesitas tingkat 1"
        elif 35.0 <= bmi_value < 40.0:
            category = "Obese Class II"
            description = "Obesitas tingkat 2"
        else:
            category = "Obese Class III"
            description = "Obesitas tingkat 3 (ekstrem)"
        
        return {
            "value": round(bmi_value, 2),
            "category": category,
            "description": description
        }
    
    except Exception as e:
        logger.error(f"Error calculating BMI: {e}")
        return {
            "value": 0.0,
            "category": "Error",
            "description": "Tidak dapat menghitung BMI"
        }

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

def get_health_recommendation(prediction: str, bmi_info: Dict[str, Any]) -> str:
    recommendations = {
        "Insufficient_Weight": "Konsultasi dengan ahli gizi untuk program penambahan berat badan yang sehat. Fokus pada makanan bergizi tinggi dan olahraga ringan.",
        "Normal_Weight": "Pertahankan pola hidup sehat dengan diet seimbang dan olahraga teratur. Lanjutkan kebiasaan baik Anda!",
        "Overweight_Level_I": "Mulai program penurunan berat badan dengan diet seimbang dan olahraga rutin. Kurangi makanan tinggi kalori.",
        "Overweight_Level_II": "Konsultasi dengan dokter atau ahli gizi. Diperlukan program penurunan berat badan yang terstruktur.",
        "Obesity_Type_I": "Konsultasi medis diperlukan. Program penurunan berat badan intensif dengan pengawasan profesional.",
        "Obesity_Type_II": "Konsultasi medis segera. Risiko tinggi komplikasi kesehatan. Diperlukan intervensi medis.",
        "Obesity_Type_III": "Konsultasi medis darurat. Risiko sangat tinggi. Mungkin diperlukan intervensi bedah atau terapi intensif."
    }
    
    base_recommendation = recommendations.get(prediction, "Konsultasi dengan tenaga medis profesional.")
    
    if bmi_info["value"] > 30:
        base_recommendation += " BMI Anda menunjukkan risiko tinggi untuk penyakit kardiovaskular, diabetes, dan masalah kesehatan lainnya."
    
    return base_recommendation

def preprocess_input(input_data: ObesityInput) -> pd.DataFrame:
    try:
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
        logger.info(f"Preprocessed data shape: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Error preprocessing input: {e}")
        raise

@app.get("/root")
async def root():
    return {
        "message": "Obesity Classification API",
        "description": "API untuk prediksi tingkat obesitas dengan analisis BMI yang diperbaiki",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Prediksi tingkat obesitas",
            "/predict-batch": "POST - Prediksi batch multiple data",
            "/health": "GET - Status kesehatan API",
            "/bmi": "POST - Hitung BMI saja",
            "/docs": "GET - Dokumentasi API interaktif"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "message": "API is running properly",
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.post("/bmi")
async def calculate_bmi_only(bmi_input: BMIInput):
    """Endpoint khusus untuk menghitung BMI saja"""
    try:
        bmi_info = calculate_bmi(bmi_input.height, bmi_input.weight)
        return {
            "bmi": bmi_info,
            "message": "BMI berhasil dihitung"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error calculating BMI: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_obesity(input_data: ObesityInput):
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded. Please check if the model file exists and restart the API."
        )
    
    try:
        logger.info(f"Processing prediction for: Height={input_data.Height}m, Weight={input_data.Weight}kg")
        
        processed_data = preprocess_input(input_data)  
        
        prediction = model.predict(processed_data)[0]
        prediction_proba = model.predict_proba(processed_data)[0]
        
        classes = model.classes_ if hasattr(model, 'classes_') else [
            'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
            'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
        ]
        
        probability_dict = {
            class_name: round(float(prob), 4) for class_name, prob in zip(classes, prediction_proba)
        }

        bmi_info = calculate_bmi(input_data.Height, input_data.Weight)
        
        risk_level = get_risk_level(prediction)
        
        recommendation = get_health_recommendation(prediction, bmi_info)
        
        logger.info(f"Prediction successful: {prediction}, BMI: {bmi_info['value']}")
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability_dict,
            bmi=BMIInfo(**bmi_info),
            risk_level=risk_level,
            recommendation=recommendation
        )

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(input_data_list: list[ObesityInput]):
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded. Please check if the model file exists."
        )
    
    try:
        results = []
        logger.info(f"Processing batch prediction for {len(input_data_list)} samples")
        
        for i, input_data in enumerate(input_data_list):
            logger.info(f"Processing sample {i+1}/{len(input_data_list)}")
            
            processed_data = preprocess_input(input_data)
            
            prediction = model.predict(processed_data)[0]
            prediction_proba = model.predict_proba(processed_data)[0]
            
            classes = model.classes_ if hasattr(model, 'classes_') else [
                'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
                'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
            ]
            
            probability_dict = {
                class_name: round(float(prob), 4) for class_name, prob in zip(classes, prediction_proba)
            }

            bmi_info = calculate_bmi(input_data.Height, input_data.Weight)
            risk_level = get_risk_level(prediction)
            recommendation = get_health_recommendation(prediction, bmi_info)
            
            results.append(PredictionResponse(
                prediction=prediction,
                probability=probability_dict,
                bmi=BMIInfo(**bmi_info),
                risk_level=risk_level,
                recommendation=recommendation
            ))
        
        logger.info(f"Batch prediction completed successfully for {len(results)} samples")
        return {"predictions": results, "total_processed": len(results)}
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during batch prediction: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
