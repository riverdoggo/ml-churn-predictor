# Customer Churn Prediction API

A simple end-to-end machine learning project that predicts whether a customer is likely to churn based on service and account information.

The project trains several ML models, selects the best one using evaluation metrics, and exposes the model through a FastAPI REST API for predictions.

The goal of this project was to understand the full ML workflow from data preprocessing to model deployment.

---

# Project Architecture

```
Client Request
      ↓
   FastAPI API
      ↓
Feature Alignment
      ↓
   ML Model
      ↓
Prediction Response
```

---

# Features

- Data preprocessing pipeline
- Feature encoding using pandas
- Multiple ML models trained and compared
- Logistic Regression
- Random Forest
- XGBoost
- Evaluation using ROC-AUC and classification metrics
- Model persistence using joblib
- FastAPI prediction service
- Automatic API documentation via Swagger
- Basic latency tracking for inference

---

# Dataset

The dataset used is the **Telco Customer Churn dataset**.

Download it here:

https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv

Place it inside:

```
data/churn.csv
```

The dataset contains information such as:

- tenure
- monthly charges
- internet service
- contract type
- payment method
- churn label

Target variable:

```
Churn
```

---

# Repository Structure

```
ml-churn-predictor

api/
    main.py                FastAPI server

src/
    preprocess.py          data cleaning + encoding
    train.py               model training + evaluation
    predict.py             prediction helper

data/
    churn.csv              dataset

model/
    churn_model.pkl        trained model
    features.pkl           feature schema

notebooks/
    eda.ipynb              exploratory analysis (optional)

requirements.txt
Dockerfile
README.md
```

---

# Setup

Clone the project or download the repository.

Create virtual environment:

```bash
python -m venv venv
```

Activate it.

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Train the Model

Run:

```bash
python src/train.py
```

This will:

- load the dataset
- preprocess and encode features
- train multiple ML models
- evaluate them
- save the best model

Saved files:

```
model/churn_model.pkl
model/features.pkl
```

---

# Run the API

Start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000
```

Interactive API documentation:

```
http://127.0.0.1:8000/docs
```

---

# Example Prediction

POST request to:

```
/predict
```

Example input:

```json
{
 "tenure": 12,
 "MonthlyCharges": 70,
 "TotalCharges": 900
}
```

Example response:

```json
{
 "prediction": 0,
 "probability": 0.13,
 "latency": 0.018
}
```

Interpretation:

- **prediction**
  - `0` → customer likely stays
  - `1` → customer likely churns

- **probability**
  - estimated churn probability

- **latency**
  - time taken for inference

---

# Models Used

The project compares several models:

- Logistic Regression
- Random Forest
- XGBoost

The best performing model based on ROC-AUC is saved and used by the API.

---

# Docker (Optional)

Build container:

```bash
docker build -t churn-api .
```

Run container:

```bash
docker run -p 8000:8000 churn-api
```

---

# Learning Goals

This project was built to practice:

- machine learning model training
- evaluation metrics
- feature engineering
- deploying ML models with FastAPI
- basic ML system design
