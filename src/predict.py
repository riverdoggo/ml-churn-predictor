
import joblib
import pandas as pd

model = joblib.load("model/churn_model.pkl")

def predict(data):
    df = pd.DataFrame([data])
    p = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "prediction":int(p),
        "probability":float(prob)
    }
