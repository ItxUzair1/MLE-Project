import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def dummy_data():
    data = {
        "customerID": [f"ID-{i:04d}" for i in range(110)],
        "gender": ["Male", "Female"] * 55,
        "SeniorCitizen": [0, 1] * 55,
        "Partner": ["Yes", "No"] * 55,
        "Dependents": ["No", "Yes"] * 55,
        "tenure": np.random.randint(0, 72, 110),
        "PhoneService": ["Yes", "No"] * 55,
        "MultipleLines": ["No phone service", "Yes", "No"] * 36 + ["Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "No"] * 36 + ["DSL", "Fiber optic"],
        "OnlineSecurity": ["No internet service", "Yes", "No"] * 36 + ["Yes", "No"],
        "OnlineBackup": ["No internet service", "Yes", "No"] * 36 + ["Yes", "No"],
        "DeviceProtection": ["No internet service", "Yes", "No"] * 36 + ["Yes", "No"],
        "TechSupport": ["No internet service", "Yes", "No"] * 36 + ["Yes", "No"],
        "StreamingTV": ["No internet service", "Yes", "No"] * 36 + ["Yes", "No"],
        "StreamingMovies": ["No internet service", "Yes", "No"] * 36 + ["Yes", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"] * 36 + ["Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No"] * 55,
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"] * 27 + ["Electronic check", "Mailed check"],
        "MonthlyCharges": np.random.uniform(10, 150, 110),
        "TotalCharges": ["100.5", " ", "200.0", "300.5"] * 27 + ["100.5", "200.0"],
        "Churn": ["Yes", "No"] * 55
    }
    df = pd.DataFrame(data)
    return df
