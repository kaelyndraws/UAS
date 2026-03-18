import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR

def evaluate(test_scaled, run_id):
    x_test = test_scaled.drop("Credit_Score", axis = 1)
    y_test = test_scaled["Credit_Score"]
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    predictions = model.predict(x_test)
    predictions = predictions.flatten()
    y_test = y_test.values
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average = "weighted")
    recall = recall_score(y_test, predictions, average = "weighted")
    f1 = f1_score(y_test, predictions, average = "weighted")

    with mlflow.start_run(run_id = run_id):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

    print(f"Evaluation completed | Accuracy = {accuracy:.3f}")

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    evaluate()