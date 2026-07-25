FROM python:3.11-slim

WORKDIR /app

# Copy only what is needed — artifacts/ is excluded via .dockerignore
# The app loads the model, preprocessor, and SHAP background from DAGsHub at runtime
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# DAGsHub credentials are injected at runtime via environment variables.
# Pass them with: docker run -e DAGSHUB_USER_TOKEN=... -e DAGSHUB_REPO_OWNER=... -e DAGSHUB_REPO_NAME=...
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]