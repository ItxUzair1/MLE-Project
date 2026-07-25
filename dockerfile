FROM python:3.11-slim

WORKDIR /app

# Copy only what is needed — artifacts/ is excluded via .dockerignore
# The app loads the model, preprocessor, and SHAP background from DAGsHub at runtime
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# DAGsHub credentials are injected at runtime via environment variables.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]