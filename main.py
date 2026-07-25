from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

from src.api.schema import CustomerData
from src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(title="Telco Churn Prediction API")

# Initialize pipeline lazily to avoid loading during test collection if any
pipeline = None

@app.on_event("startup")
def startup_event():
    global pipeline
    pipeline = PredictionPipeline()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": pipeline is not None}

@app.post("/predict")
async def predict_churn(data: CustomerData):
    if pipeline is None:
        return {"error": "Pipeline not loaded"}
    
    # Convert input to DataFrame
    df = pd.DataFrame([data.model_dump()])
    
    # Predict
    result = pipeline.predict(df)
    return result
