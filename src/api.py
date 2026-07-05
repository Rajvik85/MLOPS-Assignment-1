"""
FastAPI application for serving predictions.
Exposes /health, /predict (with input validation), and /metrics (Prometheus) endpoints.
"""

import os
import logging
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from prometheus_fastapi_instrumentator import Instrumentator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api")

# Define model paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "best_model.joblib")

# Global reference to the trained model pipeline
model_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifecycle manager to handle resource setup and teardown.
    """
    global model_pipeline
    logger.info("Initializing API and loading model pipeline...")

    if not os.path.exists(MODEL_PATH):
        logger.warning(
            f"Model file not found at: {MODEL_PATH}. Prediction service might fail until trained."
        )
    else:
        try:
            model_pipeline = joblib.load(MODEL_PATH)
            logger.info("Model pipeline successfully loaded from disk.")
        except Exception as e:
            logger.error(f"Failed to load model pipeline: {str(e)}")
            raise e

    yield
    logger.info("Shutting down API...")


# Initialize FastAPI application
app = FastAPI(
    title="Heart Disease Prediction API",
    description="A production-ready FastAPI service to predict heart disease risk.",
    version="1.0.0",
    lifespan=lifespan,
)


# Input data validation schema
class PatientData(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 63.0,
                "sex": 1.0,
                "cp": 3.0,
                "trestbps": 145.0,
                "chol": 233.0,
                "fbs": 1.0,
                "restecg": 0.0,
                "thalach": 150.0,
                "exang": 0.0,
                "oldpeak": 2.3,
                "slope": 2.0,
                "ca": 0.0,
                "thal": 3.0,
            }
        }
    )

    age: float = Field(..., description="Age in years")
    sex: float = Field(..., description="Sex (1.0 = male, 0.0 = female)")
    cp: float = Field(..., description="Chest pain type (1.0, 2.0, 3.0, 4.0)")
    trestbps: float = Field(
        ..., description="Resting blood pressure in mm Hg on admission"
    )
    chol: float = Field(..., description="Serum cholesterol in mg/dl")
    fbs: float = Field(
        ...,
        description="Fasting blood sugar > 120 mg/dl (1.0 = true, 0.0 = false)",
    )
    restecg: float = Field(
        ...,
        description="Resting electrocardiographic results (0.0, 1.0, 2.0)",
    )
    thalach: float = Field(..., description="Maximum heart rate achieved")
    exang: float = Field(
        ..., description="Exercise induced angina (1.0 = yes, 0.0 = no)"
    )
    oldpeak: float = Field(
        ...,
        description="ST depression induced by exercise relative to rest",
    )
    slope: float = Field(
        ...,
        description="The slope of the peak exercise ST segment (1.0, 2.0, 3.0)",
    )
    ca: float = Field(
        ...,
        description="Number of major vessels (0.0 - 3.0) colored by fluoroscopy",
    )
    thal: float = Field(
        ...,
        description="Thalassemia (3.0 = normal, 6.0 = fixed defect, 7.0 = reversible defect)",
    )


# Output prediction schema
class PredictionOutput(BaseModel):
    prediction: int = Field(
        ..., description="Binary target (1 = heart disease present, 0 = absent)"
    )
    confidence: float = Field(
        ..., description="Probability of heart disease presence (0.0 to 1.0)"
    )
    model_version: str = Field(
        "1.0.0", description="API model version serving the response"
    )


@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint for Kubernetes liveness/readiness probes.
    """
    global model_pipeline
    if model_pipeline is None:
        if os.path.exists(MODEL_PATH):
            # Try reloading if file now exists
            try:
                model_pipeline = joblib.load(MODEL_PATH)
                return {"status": "healthy", "model_loaded": True}
            except Exception:
                pass
        return {
            "status": "degraded",
            "model_loaded": False,
            "detail": "Model pipeline artifact is missing",
        }
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(data: PatientData):
    """
    Scores patient data to determine risk of heart disease.
    """
    if model_pipeline is None:
        logger.error("Prediction requested but model is not loaded.")
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Please run model training first.",
        )

    try:
        # Convert Pydantic object to Pandas DataFrame for pipeline input
        request_payload = data.model_dump()
        input_dict = {k: [v] for k, v in request_payload.items()}
        input_df = pd.DataFrame(input_dict)

        # Log input request
        logger.info(f"Received prediction request: {request_payload}")

        # Run inference using the pre-fitted full pipeline
        prediction = int(model_pipeline.predict(input_df)[0])
        probabilities = model_pipeline.predict_proba(input_df)[0]
        confidence = float(probabilities[1])

        # Log output result
        logger.info(
            f"Prediction result: prediction={prediction}, confidence={confidence:.4f}"
        )

        return PredictionOutput(prediction=prediction, confidence=confidence)

    except Exception as e:
        logger.error(f"Inference pipeline execution failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error executing prediction: {str(e)}"
        )


# Instrument the FastAPI app to collect and expose Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)
