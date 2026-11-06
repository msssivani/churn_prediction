
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

from mlflow.tracking import MlflowClient


def configure_mlflow(config):
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or config["mlflow"].get("tracking_uri")
    )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    print(
        f"[INFO] MLflow Tracking URI: {mlflow.get_tracking_uri()}"
    )


def clean_dataframe(df, config):
    df = df.copy()

    target_column = config["data"]["target_column"]
    id_column = config["data"]["id_column"]

    if id_column in df.columns:
        df = df.drop(columns=[id_column])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    if target_column in df.columns:
        df = df.drop(columns=[target_column])

    return df


def get_production_model_version(client, model_name):
    print("\n--- MLflow Model Registry Information ---")

    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    if not versions:
        print("[ERROR] No registered model versions found.")
        return None

    production_model = None

    for mv in versions:

        print(f"\nVersion: {mv.version}")
        print(f"Stage: {mv.current_stage}")
        print(f"Source: {mv.source}")
        print(f"Run ID: {mv.run_id}")

        if mv.run_id:
            run = client.get_run(mv.run_id)

            print(
                f"Run Artifact URI: {run.info.artifact_uri}"
            )

        if mv.current_stage == "Production":
            production_model = mv

    print("-----------------------------------------\n")

    if production_model is None:
        print("[ERROR] No Production model found.")
        return None

    return production_model


def load_production_model(model_name, production_model):
    # Load the exact Production version rather than relying
    # on a mutable Production stage URI.

    model_uri = (
        f"models:/{model_name}/{production_model.version}"
    )

    print(
        f"[INFO] Loading Production Version "
        f"{production_model.version}"
    )

    print(f"[INFO] Model URI: {model_uri}")

    try:
        model = mlflow.sklearn.load_model(model_uri)

        print(
            "[SUCCESS] Production model loaded successfully."
        )

        return model

    except Exception as exc:

        print(
            f"[ERROR] Could not load Production model: {exc}"
        )

        print(
            f"[ERROR] Registered model source: "
            f"{production_model.source}"
        )

        print(
            "[ERROR] Check the model artifact storage and "
            "MLflow tracking configuration."
        )

        return None


def run_inference_demo():
    print(
        "[INFO] Starting ML Pipeline Production Inference..."
    )

    # STEP 1: Load configuration.
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)

    model_name = config["registry"]["model_name"]

    client = MlflowClient()

    # STEP 2: Identify Production model version.
    production_model = get_production_model_version(
        client, model_name
    )

    if production_model is None:
        return False

    # STEP 3: Load Production model.
    model = load_production_model(
        model_name, production_model
    )

    if model is None:
        return False

    # STEP 4: Check required files.
    preprocessor_path = config["paths"]["preprocessor"]
    raw_data_path = config["data"]["raw_path"]

    if not os.path.exists(preprocessor_path):
        print(
            f"[ERROR] Preprocessor not found: "
            f"{preprocessor_path}"
        )
        return False

    if not os.path.exists(raw_data_path):
        print(
            f"[ERROR] Raw dataset not found: "
            f"{raw_data_path}"
        )
        return False

    # STEP 5: Load preprocessor and raw dataset.
    print("[INFO] Loading preprocessor and raw dataset...")

    preprocessor = joblib.load(preprocessor_path)
    df = pd.read_csv(raw_data_path)

    # STEP 6: Select five random records.
    sample = df.sample(
        n=min(5, len(df)),
        random_state=7
    )

    X_sample = clean_dataframe(sample, config)

    # STEP 7: Apply preprocessing.
    X_sample = preprocessor.transform(X_sample)

    # STEP 8: Generate predictions.
    predictions = model.predict(X_sample)

    probabilities = model.predict_proba(X_sample)[:, 1]

    # STEP 9: Prepare output.
    result = sample.copy()

    result["Predicted_Churn"] = predictions

    result["Churn_Probability"] = np.round(
        probabilities, 4
    )

    print(
        "\n--- ML Pipeline Production Inference Results ---"
    )

    print(
        result[
            ["Predicted_Churn", "Churn_Probability"]
        ].to_string(index=False)
    )

    print("-----------------------------------------\n")

    # STEP 10: Save predictions.
    output_path = config["paths"]["inference_output"]

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    result.to_csv(output_path, index=False)

    print(
        f"[SUCCESS] Predictions saved to {output_path}"
    )

    return True


if __name__ == "__main__":
    passed = run_inference_demo()
    sys.exit(0 if passed else 1)
