import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import joblib

def train(train_scaled):
    mlflow.set_experiment("Streamlit-Pipeline")

    x_train = train_scaled.drop("Credit_Score", axis = 1)
    y_train = train_scaled["Credit_Score"]

    with mlflow.start_run() as run:
        model = RandomForestClassifier(max_depth = None, max_features = "sqrt", min_samples_leaf = 1, n_estimators = 200)
        
        model.fit(x_train, y_train)
        mlflow.log_param("max_depth", None)
        mlflow.log_param("max_features", "sqrt")
        mlflow.log_param("min_samples_leaf", 1)
        mlflow.log_param("n_estimators", 200)
        mlflow.sklearn.log_model(sk_model = model,
                                 artifact_path = "model")
        joblib.dump(model, "model.pkl", compress=3)
        
        return run.info.run_id
    
if __name__ == "__main__":
    train()