from fastapi import FastAPI
import pandas as pd
import joblib
import time

app = FastAPI()

model = joblib.load("model/churn_model.pkl")
features = joblib.load("model/features.pkl")

@app.get("/")
def home():
    return {"msg":"churn api running"}

@app.post("/predict")
def predict(data: dict):

    start = time.time()

    df = pd.DataFrame([data])

    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features]

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    latency = time.time() - start

    return {
        "prediction": int(pred),
        "probability": float(prob),
        "latency": latency
    }
