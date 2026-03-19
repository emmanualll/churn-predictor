import pandas as pd
import joblib

def predict_churn_risk():
    print("1 - Loading production assets into the memory...")
    rf_model = joblib.load('models/random_forest_churn_v1.joblib')
    scaler = joblib.load('models/standard_scaler_v1.joblib')
    expected_columns = joblib.load('models/expected_columns_v1.joblib')

    print("2 - Receiving raw JSON payload from the frontend...")
    raw_api_payload = {
        'tenure': 2,
        'MonthlyCharges': 85.50,
        'TotalCharges': 171.00,
        'Contract': 'Month-to-month',
        'InternetService': 'Fiber optic',
        'PaymentMethod': 'Electronic check',
        'PaperlessBilling': 'Yes',
        'OnlineSecurity': 'No',
        'TechSupport': 'No',
        'DeviceProtection': 'No',
        'OnlineBackup': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'No',
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'Dependents': 'No',
        'Partner': 'No',
        'SeniorCitizen': 0,
        'gender': 'Female'
    }

    df_new = pd.DataFrame([raw_api_payload])

    print("3 - Executing automated data pipeline...")
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = [col for col in df_new.columns if col not in num_cols]

    df_encoded = pd.get_dummies(df_new, columns=cat_cols, drop_first=True)
    df_production = df_encoded.reindex(columns=expected_columns, fill_value=0)
    df_production[num_cols] = scaler.transform(df_production[num_cols])

    print("4 - Generating algorithmic risk assessment...")
    churn_probability = rf_model.predict_proba(df_production)[0, 1]

    print("----------------------")
    print("      INFERENCE       ")
    print("----------------------")
    print(f"Customer Churn Risk: {churn_probability * 100:.1f}%")

    if churn_probability >= 0.65:
        print("System Decision:     [HIGH RISK] -> Trigger Retention Protocol")
    else:
        print("System Decision:     [SAFE]      -> No Action Required")
    print("==================================================\n")

if __name__ == "__main__":
    predict_churn_risk()