"""
FastAPI application for serving predictions.
Exposes /health, /predict (with input validation), and /metrics (Prometheus) endpoints.
"""

import os
import logging
import random
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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

SAMPLE_PATIENTS = [
    {
        "age": 63.0,
        "sex": 1.0,
        "cp": 1.0,
        "trestbps": 145.0,
        "chol": 233.0,
        "fbs": 1.0,
        "restecg": 2.0,
        "thalach": 150.0,
        "exang": 0.0,
        "oldpeak": 2.3,
        "slope": 3.0,
        "ca": 0.0,
        "thal": 6.0,
    },
    {
        "age": 67.0,
        "sex": 1.0,
        "cp": 4.0,
        "trestbps": 160.0,
        "chol": 286.0,
        "fbs": 0.0,
        "restecg": 2.0,
        "thalach": 108.0,
        "exang": 1.0,
        "oldpeak": 1.5,
        "slope": 2.0,
        "ca": 3.0,
        "thal": 3.0,
    },
    {
        "age": 41.0,
        "sex": 0.0,
        "cp": 2.0,
        "trestbps": 130.0,
        "chol": 204.0,
        "fbs": 0.0,
        "restecg": 2.0,
        "thalach": 172.0,
        "exang": 0.0,
        "oldpeak": 1.4,
        "slope": 1.0,
        "ca": 0.0,
        "thal": 3.0,
    },
    {
        "age": 60.0,
        "sex": 1.0,
        "cp": 4.0,
        "trestbps": 130.0,
        "chol": 206.0,
        "fbs": 0.0,
        "restecg": 2.0,
        "thalach": 132.0,
        "exang": 1.0,
        "oldpeak": 2.4,
        "slope": 2.0,
        "ca": 2.0,
        "thal": 7.0,
    },
    {
        "age": 50.0,
        "sex": 0.0,
        "cp": 3.0,
        "trestbps": 120.0,
        "chol": 219.0,
        "fbs": 0.0,
        "restecg": 0.0,
        "thalach": 158.0,
        "exang": 0.0,
        "oldpeak": 1.6,
        "slope": 2.0,
        "ca": 0.0,
        "thal": 3.0,
    },
    {
        "age": 65.0,
        "sex": 0.0,
        "cp": 4.0,
        "trestbps": 150.0,
        "chol": 225.0,
        "fbs": 0.0,
        "restecg": 2.0,
        "thalach": 114.0,
        "exang": 0.0,
        "oldpeak": 1.0,
        "slope": 2.0,
        "ca": 3.0,
        "thal": 7.0,
    },
    {
        "age": 44.0,
        "sex": 1.0,
        "cp": 2.0,
        "trestbps": 130.0,
        "chol": 219.0,
        "fbs": 0.0,
        "restecg": 2.0,
        "thalach": 188.0,
        "exang": 0.0,
        "oldpeak": 0.0,
        "slope": 1.0,
        "ca": 0.0,
        "thal": 3.0,
    },
    {
        "age": 62.0,
        "sex": 0.0,
        "cp": 4.0,
        "trestbps": 160.0,
        "chol": 164.0,
        "fbs": 0.0,
        "restecg": 2.0,
        "thalach": 145.0,
        "exang": 0.0,
        "oldpeak": 6.2,
        "slope": 3.0,
        "ca": 3.0,
        "thal": 7.0,
    },
]


def ensure_model_compatibility(pipeline):
    """
    Repair known cross-version sklearn model attributes after joblib loading.
    """
    classifier = getattr(pipeline, "named_steps", {}).get("classifier")
    if classifier.__class__.__name__ == "LogisticRegression" and not hasattr(
        classifier, "multi_class"
    ):
        classifier.multi_class = "auto"
    return pipeline


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
            model_pipeline = ensure_model_compatibility(joblib.load(MODEL_PATH))
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
                model_pipeline = ensure_model_compatibility(joblib.load(MODEL_PATH))
                return {"status": "healthy", "model_loaded": True}
            except Exception:
                pass
        return {
            "status": "degraded",
            "model_loaded": False,
            "detail": "Model pipeline artifact is missing",
        }
    return {"status": "healthy", "model_loaded": True}


@app.get("/sample", response_model=PatientData, tags=["User Interface"])
def random_sample():
    """
    Returns a random sample patient record for the browser prediction form.
    """
    return random.choice(SAMPLE_PATIENTS).copy()


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


@app.get("/", response_class=HTMLResponse, tags=["User Interface"])
def prediction_form():
    """
    Browser-based form for manual heart disease prediction checks.
    """
    return """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Heart Disease Risk Prediction</title>
      <style>
        :root {
          color-scheme: light;
          --bg: #f6f8fb;
          --panel: #ffffff;
          --text: #1f2937;
          --muted: #5f6b7a;
          --line: #d9e1ec;
          --primary: #0f766e;
          --primary-dark: #115e59;
          --danger: #b42318;
          --ok: #047857;
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          font-family: Arial, Helvetica, sans-serif;
          background: var(--bg);
          color: var(--text);
        }
        main {
          max-width: 1120px;
          margin: 0 auto;
          padding: 28px 20px 40px;
        }
        header {
          display: flex;
          justify-content: space-between;
          gap: 20px;
          align-items: flex-start;
          border-bottom: 1px solid var(--line);
          padding-bottom: 18px;
          margin-bottom: 22px;
        }
        h1 {
          margin: 0 0 8px;
          font-size: 30px;
          line-height: 1.2;
        }
        p {
          margin: 0;
          color: var(--muted);
          line-height: 1.5;
        }
        .status {
          min-width: 180px;
          background: var(--panel);
          border: 1px solid var(--line);
          padding: 12px 14px;
          font-size: 14px;
        }
        .layout {
          display: grid;
          grid-template-columns: 1.3fr 0.7fr;
          gap: 20px;
        }
        section {
          background: var(--panel);
          border: 1px solid var(--line);
          padding: 18px;
        }
        h2 {
          margin: 0 0 14px;
          font-size: 18px;
        }
        form {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 14px;
        }
        label {
          display: grid;
          gap: 6px;
          font-size: 13px;
          font-weight: 700;
        }
        input {
          width: 100%;
          min-height: 40px;
          border: 1px solid var(--line);
          padding: 8px 10px;
          font-size: 15px;
        }
        input:focus {
          border-color: var(--primary);
          outline: 2px solid rgba(15, 118, 110, 0.16);
        }
        .actions {
          grid-column: 1 / -1;
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
          margin-top: 4px;
        }
        button {
          border: 0;
          background: var(--primary);
          color: white;
          min-height: 42px;
          padding: 0 18px;
          font-size: 15px;
          font-weight: 700;
          cursor: pointer;
        }
        button.secondary {
          background: #334155;
        }
        button:hover { background: var(--primary-dark); }
        button.secondary:hover { background: #1f2937; }
        .result {
          border: 1px solid var(--line);
          padding: 16px;
          min-height: 132px;
          background: #fbfdff;
        }
        .result strong {
          display: block;
          font-size: 22px;
          margin-bottom: 8px;
        }
        .sample-panel {
          grid-column: 1 / -1;
          border: 1px solid var(--line);
          background: #fbfdff;
          padding: 14px;
          margin-top: 4px;
        }
        .sample-panel h3 {
          margin: 0 0 8px;
          font-size: 15px;
        }
        .risk-high { color: var(--danger); }
        .risk-low { color: var(--ok); }
        code {
          display: block;
          white-space: pre-wrap;
          background: #0f172a;
          color: #e2e8f0;
          padding: 12px;
          margin-top: 14px;
          font-size: 13px;
          overflow-x: auto;
        }
        .field-help {
          color: var(--muted);
          font-size: 12px;
          font-weight: 400;
        }
        @media (max-width: 860px) {
          header, .layout, form {
            grid-template-columns: 1fr;
            display: grid;
          }
          .status {
            min-width: 0;
          }
        }
      </style>
    </head>
    <body>
      <main>
        <header>
          <div>
            <h1>Heart Disease Risk Prediction</h1>
            <p>Enter patient health values and submit them to the deployed model API.</p>
          </div>
          <div class="status" id="healthStatus">Checking API health...</div>
        </header>

        <div class="layout">
          <section>
            <h2>Patient Inputs</h2>
            <form id="predictionForm">
              <label>Age
                <input name="age" type="number" step="0.1" value="63" required />
              </label>
              <label>Sex
                <input name="sex" type="number" step="1" value="1" required />
                <span class="field-help">1 = male, 0 = female</span>
              </label>
              <label>Chest Pain Type
                <input name="cp" type="number" step="1" value="3" required />
                <span class="field-help">Typical UCI values: 1, 2, 3, 4</span>
              </label>
              <label>Resting BP
                <input name="trestbps" type="number" step="0.1" value="145" required />
              </label>
              <label>Cholesterol
                <input name="chol" type="number" step="0.1" value="233" required />
              </label>
              <label>Fasting Blood Sugar
                <input name="fbs" type="number" step="1" value="1" required />
                <span class="field-help">1 = true, 0 = false</span>
              </label>
              <label>Rest ECG
                <input name="restecg" type="number" step="1" value="0" required />
              </label>
              <label>Max Heart Rate
                <input name="thalach" type="number" step="0.1" value="150" required />
              </label>
              <label>Exercise Angina
                <input name="exang" type="number" step="1" value="0" required />
                <span class="field-help">1 = yes, 0 = no</span>
              </label>
              <label>Oldpeak
                <input name="oldpeak" type="number" step="0.1" value="2.3" required />
              </label>
              <label>Slope
                <input name="slope" type="number" step="1" value="2" required />
              </label>
              <label>Major Vessels
                <input name="ca" type="number" step="1" value="0" required />
              </label>
              <label>Thal
                <input name="thal" type="number" step="1" value="3" required />
                <span class="field-help">3 = normal, 6 = fixed defect, 7 = reversible defect</span>
              </label>
              <div class="actions">
                <button type="submit">Predict Risk</button>
                <button type="button" class="secondary" id="loadSample">Load Random Sample</button>
              </div>
              <div class="sample-panel">
                <h3>Loaded Sample Record</h3>
                <p>The current form values below are the payload that will be sent to the API.</p>
                <code id="sampleRecord">No sample loaded yet.</code>
              </div>
            </form>
          </section>

          <section>
            <h2>Prediction Result</h2>
            <div class="result" id="resultBox">
              Submit the form to call the live `/predict` API endpoint.
            </div>
          </section>
        </div>
      </main>

      <script>
        function populateForm(payload) {
          for (const [key, value] of Object.entries(payload)) {
            document.querySelector(`[name="${key}"]`).value = value;
          }
          document.getElementById("sampleRecord").textContent =
            JSON.stringify(payload, null, 2);
        }

        async function loadRandomSample() {
          const resultBox = document.getElementById("resultBox");
          try {
            const response = await fetch("/sample", { cache: "no-store" });
            const data = await response.json();
            populateForm(data);
            resultBox.textContent =
              "Random sample loaded. Click Predict Risk to score it.";
          } catch (error) {
            resultBox.innerHTML =
              `<strong class="risk-high">Unable to load sample</strong><p>${error}</p>`;
          }
        }

        function formToPayload(form) {
          const payload = {};
          new FormData(form).forEach((value, key) => {
            payload[key] = Number(value);
          });
          return payload;
        }

        function updateSampleRecordFromForm() {
          const form = document.getElementById("predictionForm");
          const payload = formToPayload(form);
          document.getElementById("sampleRecord").textContent =
            JSON.stringify(payload, null, 2);
        }

        async function checkHealth() {
          const status = document.getElementById("healthStatus");
          try {
            const response = await fetch("/health");
            const data = await response.json();
            status.textContent = data.model_loaded
              ? "API healthy. Model loaded."
              : "API degraded. Model not loaded.";
          } catch (error) {
            status.textContent = "API health check failed.";
          }
        }

        document.getElementById("loadSample").addEventListener("click", loadRandomSample);
        document.getElementById("predictionForm").addEventListener(
          "input",
          updateSampleRecordFromForm
        );

        document.getElementById("predictionForm").addEventListener("submit", async (event) => {
          event.preventDefault();
          const resultBox = document.getElementById("resultBox");
          const payload = formToPayload(event.target);
          resultBox.textContent = "Calling prediction API...";

          try {
            const response = await fetch("/predict", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload)
            });
            const data = await response.json();

            if (!response.ok) {
              const errorJson = JSON.stringify(data, null, 2);
              resultBox.innerHTML =
                `<strong class="risk-high">Request failed</strong><code>${errorJson}</code>`;
              return;
            }

            const isHighRisk = data.prediction === 1;
            const label = isHighRisk ? "Heart disease risk detected" : "No heart disease risk detected";
            const className = isHighRisk ? "risk-high" : "risk-low";
            resultBox.innerHTML = `
              <strong class="${className}">${label}</strong>
              <p>Prediction class: ${data.prediction}</p>
              <p>Confidence for disease class: ${(data.confidence * 100).toFixed(2)}%</p>
              <p>Model version: ${data.model_version}</p>
              <code>${JSON.stringify(data, null, 2)}</code>
            `;
          } catch (error) {
            resultBox.innerHTML = `<strong class="risk-high">Unable to call API</strong><p>${error}</p>`;
          }
        });

        loadRandomSample();
        checkHealth();
      </script>
    </body>
    </html>
    """
