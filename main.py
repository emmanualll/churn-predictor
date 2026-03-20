from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

#initilizing the fastapi app
app = FastAPI(title="Customer Retention API", version="1.0")

#loading the production assests into memory outside the app function
try:
    rf_model = joblib.load('models/random_forest_churn_v1.joblib')
    scaler = joblib.load('models/standard_scaler_v1.joblib')
    expected_columns = joblib.load('models/expected_columns_v1.joblib')
    print("System Boot -  Machine Learning assets loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Startup Failed. Could not load models: {e}")

#Creating a pydantic schema so that there are no false datatypes
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    PaymentMethod: str
    PaperlessBilling: str
    OnlineSecurity: str
    TechSupport: str
    DeviceProtection: str
    OnlineBackup: str
    StreamingTV: str
    StreamingMovies: str
    PhoneService: str
    MultipleLines: str
    Dependents: str
    Partner: str
    SeniorCitizen: int
    gender: str

@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        df_new = pd.DataFrame([customer.model_dump()])

        #isolate feature types
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        cat_cols = [col for col in df_new.columns if col not in num_cols]
        
        #Ensuring everything aligns strucutralllt to the blueprint
        df_encoded = pd.get_dummies(df_new, columns=cat_cols, drop_first=True)
        df_production = df_encoded.reindex(columns=expected_columns, fill_value=0)
        
        #apply prefitted scaling
        df_production[num_cols] = scaler.transform(df_production[num_cols])
        
        churn_probability = rf_model.predict_proba(df_production)[0, 1]
        
        is_high_risk = bool(churn_probability >= 0.65)
        
        return {
            "status": "success",
            "prediction": {
                "churn_probability": round(float(churn_probability), 3),
                "risk_classification": "HIGH RISK" if is_high_risk else "SAFE",
                "trigger_retention_protocol": is_high_risk
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Pipeline Error: {str(e)}")