from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import time
import logging
import os
from typing import List
from backend.models import (
    CustomerData, 
    PredictionResponse, 
    BatchPredictionRequest, 
    BatchPredictionResponse, 
    HealthResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model artifacts
model = None
feature_names = None
model_metadata = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model artifacts on startup
    global model, feature_names, model_metadata
    try:
        logger.info("Loading model artifacts...")
        model_path = "backend/churn_model.pkl"
        features_path = "backend/feature_names.pkl"
        metadata_path = "backend/model_metadata.pkl"

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            feature_names = joblib.load(features_path)
            model_metadata = joblib.load(metadata_path)
            logger.info("✅ Model loaded successfully.")
        else:
            logger.error("❌ Model file not found!")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
    
    yield
    
    # Clean up on shutdown
    model = None

app = FastAPI(
    title="Churn Prediction API",
    description="Production-ready API for predicting customer churn using LightGBM.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Log latency warning if > 50ms
    if process_time > 0.05:
        logger.warning(f"⚠️ High latency: {process_time:.4f}s for {request.url.path}")
        
    response.headers["X-Process-Time"] = str(process_time)
    return response

def get_risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    return "Low"

@app.get("/", tags=["Root"])
async def root():
    return FileResponse('frontend/index.html')

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_accuracy=model_metadata.get("cv_accuracy", 0.0),
        features=len(feature_names) if feature_names else 0,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        # Use mode='json' to convert Enums to values
        data_dict = customer.model_dump(mode='json')
        
        # Map Pydantic fields to Dataset feature names
        # Dataset has specific spacing (some double spaces)
        mapping = {
            "Call_Failure": "Call  Failure",
            "Complains": "Complains",
            "Subscription_Length": "Subscription  Length",
            "Charge_Amount": "Charge  Amount",
            "Seconds_of_Use": "Seconds of Use",
            "Frequency_of_use": "Frequency of use",
            "Frequency_of_SMS": "Frequency of SMS",
            "Distinct_Called_Numbers": "Distinct Called Numbers",
            "Age_Group": "Age Group",
            "Tariff_Plan": "Tariff Plan",
            "Status": "Status",
            "Age": "Age",
            "Customer_Value": "Customer Value"
        }
        
        mapped_data = {mapping.get(k, k): v for k, v in data_dict.items()}
        input_data = pd.DataFrame([mapped_data])
        
        # Ensure correct feature order
        if feature_names:
            # Check if all features are present
            missing_cols = set(feature_names) - set(input_data.columns)
            if missing_cols:
                logger.error(f"Missing columns: {missing_cols}")
                # Fallback or error? 
                # The mapping should cover it.
            
            input_data = input_data[feature_names]
            
        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        return PredictionResponse(
            churn_prediction=int(prediction),
            churn_probability=float(probability),
            risk_level=get_risk_level(probability),
            confidence=float(probability if prediction == 1 else 1 - probability)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    try:
        customers = request.customers
        
        # Mapping (same as above)
        mapping = {
            "Call_Failure": "Call  Failure",
            "Complains": "Complains",
            "Subscription_Length": "Subscription  Length",
            "Charge_Amount": "Charge  Amount",
            "Seconds_of_Use": "Seconds of Use",
            "Frequency_of_use": "Frequency of use",
            "Frequency_of_SMS": "Frequency of SMS",
            "Distinct_Called_Numbers": "Distinct Called Numbers",
            "Age_Group": "Age Group",
            "Tariff_Plan": "Tariff Plan",
            "Status": "Status",
            "Age": "Age",
            "Customer_Value": "Customer Value"
        }
        
        batch_data = []
        for c in customers:
            data_dict = c.model_dump(mode='json')
            mapped_data = {mapping.get(k, k): v for k, v in data_dict.items()}
            batch_data.append(mapped_data)
            
        input_df = pd.DataFrame(batch_data)
        
        if feature_names:
            input_df = input_df[feature_names]
            
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[:, 1]
        
        response_list = []
        high_risk_count = 0
        
        for pred, prob in zip(predictions, probabilities):
            risk = get_risk_level(prob)
            if risk == "High":
                high_risk_count += 1
                
            response_list.append(PredictionResponse(
                churn_prediction=int(pred),
                churn_probability=float(prob),
                risk_level=risk,
                confidence=float(prob if pred == 1 else 1 - prob)
            ))
            
        processing_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=response_list,
            total_customers=len(customers),
            high_risk_count=high_risk_count,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info", tags=["Model"])
async def model_info():
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not available")
        
    return {
        "model_type": model_metadata.get("model_type", "Unknown"),
        "dataset": "UCI Iranian Churn Dataset (#563)",
        "feature_count": len(feature_names) if feature_names else 0,
        "feature_names": feature_names,
        "metrics": {
            "cv_accuracy": model_metadata.get("cv_accuracy"),
            "cv_std": model_metadata.get("cv_std"),
            "test_accuracy": model_metadata.get("test_accuracy")
        },
        "paper_reference": "https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset"
    }
