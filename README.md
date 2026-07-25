# Telco Customer Churn Prediction 🚀

An end-to-end Machine Learning pipeline designed to predict customer churn in the telecommunications sector. This project covers the entire ML lifecycle, from data ingestion and strict data validation, to model training, hyperparameter tuning, experiment tracking, model explainability, and deployment via a RESTful API.

---

## 🏗️ Architecture & Pipeline Stages

The project is structured into a robust modular pipeline:

1. **Data Ingestion**: Extracts raw data from CSV and securely loads it into a local SQLite database for persistence, simulating an enterprise data warehouse environment.
2. **Data Validation**: Leverages **Great Expectations** to ensure data integrity. Validates schema, handles missing values, and checks for statistical anomalies before any data touches the training logic.
3. **Data Transformation**: Cleans text anomalies, handles missing numerical features, and encodes categorical variables. Outputs a reusable serialized `ColumnTransformer` (preprocessor).
4. **Model Training & Tuning**: 
   - Trains multiple baseline models (Logistic Regression, Random Forest, XGBoost, SVM, etc.).
   - Utilizes **Optuna** for Bayesian hyperparameter tuning on the best-performing model to maximize the F1-Score.
   - Logs all experiments, hyperparameters, and metrics to an **MLflow** tracking server.
5. **Model Evaluation**: Generates final classification metrics and registers the production-ready model artifact into the MLflow Model Registry.
6. **Model Explainability**: Implements **SHAP (SHapley Additive exPlanations)** to provide local interpretability. For every prediction, the API returns the top 3 driving features (e.g., *why* the customer is likely to churn), utilizing a robust explainer cascade (`TreeExplainer` → `LinearExplainer` → `KernelExplainer`).
7. **Model Serving**: Exposes a real-time prediction endpoint via **FastAPI**.

---

## 🛠️ Technology Stack

### **Machine Learning & Data Processing**
* **Python 3.11**
* **Pandas / NumPy**: Data manipulation and numerical operations
* **Scikit-learn / XGBoost**: Predictive modeling algorithms
* **Optuna**: Automated hyperparameter optimization
* **SHAP**: Machine learning interpretability

### **MLOps & Engineering**
* **MLflow**: Experiment tracking, model versioning, and model registry
* **Great Expectations**: Strict data validation profiling
* **SQLite / SQLAlchemy**: Database management for raw data ingestion
* **Pytest**: Unit and integration testing

### **Deployment & CI/CD**
* **FastAPI & Uvicorn**: High-performance REST API development and serving
* **Docker**: Application containerization
* **GitHub Actions**: Continuous Integration pipeline (Automated testing and linting)
* **Ruff**: Lightning-fast Python linter
* **uv**: Ultra-fast Python package installer and resolver

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.11+ installed. We recommend using `uv` for lightning-fast dependency management.

```bash
# Install uv (if not installed)
pip install uv
```

### 2. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/ItxUzair1/MLE-Project.git
cd MLE-Project

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### 3. Run the Training Pipeline
To execute the end-to-end training pipeline (Ingestion ➡️ Validation ➡️ Transformation ➡️ Training ➡️ Evaluation):

```bash
python -m src.pipeline.run_pipeline
```
*Note: This will automatically log the experiment to MLflow and save the best model and preprocessor artifacts into the `artifacts/` directory.*

### 4. View MLflow UI
To visualize your experiments, compare model metrics, and view the registered models:

```bash
mlflow ui
```
Navigate to `http://127.0.0.1:5000` in your browser.

### 5. Start the Prediction API
Launch the FastAPI server to serve predictions:

```bash
uvicorn main:app --reload
```
The API will be available at `http://127.0.0.1:8000`. You can test the endpoints interactively via the built-in Swagger UI at `http://127.0.0.1:8000/docs`.

---

## 🐳 Docker Deployment

To containerize and run the API using Docker:

```bash
# Build the image
docker build -t telco-churn-api .

# Run the container
docker run -p 8000:8000 telco-churn-api
```

---

## 🧪 Testing

The project uses `pytest` for rigorous unit and integration testing. The test suite is automatically executed on every push to the `main` branch via GitHub Actions.

To run tests locally:
```bash
pytest -v
```