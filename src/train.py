
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from preprocess import load_data

df = load_data("data/churn.csv")

X = df.drop("Churn",axis=1)
y = df["Churn"]

joblib.dump(X.columns.tolist(), "model/features.pkl")

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

models = {
"logistic":LogisticRegression(max_iter=2000),
"rf":RandomForestClassifier(),
"xgb":XGBClassifier()
}

best_model=None
best_score=0

for name,m in models.items():
    m.fit(X_train,y_train)

    preds=m.predict(X_test)
    prob=m.predict_proba(X_test)[:,1]

    score=roc_auc_score(y_test,prob)

    print(name)
    print(classification_report(y_test,preds))
    print("roc_auc:",score)

    if score>best_score:
        best_score=score
        best_model=m

joblib.dump(best_model,"model/churn_model.pkl")

print("saved model")
