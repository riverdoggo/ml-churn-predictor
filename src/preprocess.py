
import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    if "customerID" in df.columns:
        df = df.drop("customerID",axis=1)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
    df = df.dropna()

    df["Churn"] = df["Churn"].map({"Yes":1,"No":0})

    df = pd.get_dummies(df)

    return df
