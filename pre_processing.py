import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess():
    os.makedirs("artifacts", exist_ok=True)

    df = pd.read_csv("ingested/data_B_cleaned.csv")
    x = df.drop("Credit_Score", axis=1)
    y = df["Credit_Score"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()

    scaler_train = pd.DataFrame(
        scaler.fit_transform(x_train),
        columns=x_train.columns
    )

    scaler_test = pd.DataFrame(
        scaler.transform(x_test),
        columns=x_test.columns
    )

    joblib.dump(scaler, "artifacts/preprocessor.pkl")

    train_scaled = pd.concat(
        [scaler_train.reset_index(drop=True), y_train.reset_index(drop=True)],
        axis=1
    )

    test_scaled = pd.concat(
        [scaler_test.reset_index(drop=True), y_test.reset_index(drop=True)],
        axis=1
    )

    return train_scaled, test_scaled

if __name__ == "__main__":
    preprocess()
