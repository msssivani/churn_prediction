
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import yaml
import os


# --------------------------------------------------
# STEP 1: Load Configuration
# --------------------------------------------------

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


# --------------------------------------------------
# STEP 2: Configure MLflow
# --------------------------------------------------

tracking_uri = (
    os.getenv("MLFLOW_TRACKING_URI")
    or config["mlflow"].get("tracking_uri")
)

if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient()

model_name = config["registry"]["model_name"]
metric_to_optimize = config["registry"].get(
    "metric_to_optimize", "recall"
)

print("\n[INFO] Starting Automated Model Lifecycle Manager...")
print(f"[INFO] MLflow Tracking URI: {mlflow.get_tracking_uri()}")


# --------------------------------------------------
# STEP 3: Check Whether a Model Artifact Is Accessible
# --------------------------------------------------

def validate_model_artifact(model_name, version):

    model_uri = f"models:/{model_name}/{version}"

    print(f"\n[INFO] Validating model artifact: {model_uri}")

    try:
        mlflow.sklearn.load_model(model_uri)

        print(
            f"[SUCCESS] Version {version} artifact is accessible."
        )

        return True

    except Exception as error:

        print(
            f"[ERROR] Version {version} artifact validation failed: {error}"
        )

        return False


# --------------------------------------------------
# STEP 4: Get All Model Versions
# --------------------------------------------------

try:
    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

except Exception as error:

    print(f"[ERROR] Could not retrieve model versions: {error}")
    raise

if not versions:
    print("[WARNING] No registered model versions found.")
    raise SystemExit(0)

print(f"[INFO] Found {len(versions)} model versions.")


# --------------------------------------------------
# STEP 5: Move New Models to Staging
# --------------------------------------------------

for version in versions:

    if version.current_stage == "None":

        try:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Staging"
            )

            print(
                f"[INFO] Version {version.version} moved to Staging."
            )

        except Exception as error:

            print(
                f"[ERROR] Could not move Version {version.version} "
                f"to Staging: {error}"
            )


# Refresh model versions after stage transitions
versions = client.search_model_versions(
    f"name='{model_name}'"
)


# --------------------------------------------------
# STEP 6: Validate Staging Models and Find Best Model
# --------------------------------------------------

staging_models = [
    version
    for version in versions
    if version.current_stage == "Staging"
]

valid_staging_models = []

for version in staging_models:

    print(f"\n[INFO] Checking Version {version.version}")

    print(f"[INFO] Model source: {version.source}")

    try:
        run = client.get_run(version.run_id)
        print(f"[INFO] Run Artifact URI: {run.info.artifact_uri}")

        score = run.data.metrics.get(
            metric_to_optimize
        )

        if score is None:
            print(
                f"[WARNING] Version {version.version} "
                f"does not contain metric '{metric_to_optimize}'."
            )
            continue

        if validate_model_artifact(
            model_name,
            version.version
        ):

            valid_staging_models.append(
                (version, score)
            )

        else:

            print(
                f"[WARNING] Version {version.version} rejected "
                "because its artifact is inaccessible."
            )

    except Exception as error:

        print(
            f"[ERROR] Could not check Version {version.version}: {error}"
        )


# --------------------------------------------------
# STEP 7: Select Best Valid Staging Model
# --------------------------------------------------

if not valid_staging_models:

    print(
        "\n[WARNING] No valid Staging models available."
    )

    raise SystemExit(0)


best_model, best_score = max(
    valid_staging_models,
    key=lambda item: item[1]
)

print(
    f"\n[SUCCESS] Best valid Staging model: "
    f"Version {best_model.version}"
)

print(
    f"[INFO] Best {metric_to_optimize}: {best_score:.4f}"
)


# --------------------------------------------------
# STEP 8: Validate Existing Production Model
# --------------------------------------------------

production_models = [
    version
    for version in versions
    if version.current_stage == "Production"
]

production_model = (
    max(production_models, key=lambda item: int(item.version))
    if production_models
    else None
)

move_to_production = False

if production_model is None:

    print("[INFO] No existing Production model found.")

    move_to_production = True

else:

    print(
        f"\n[INFO] Existing Production Version: "
        f"{production_model.version}"
    )

    production_is_valid = validate_model_artifact(
        model_name,
        production_model.version
    )

    if not production_is_valid:

        print(
            "[WARNING] Existing Production model is inaccessible."
        )

        print(
            "[INFO] A valid Staging model will replace it."
        )

        move_to_production = True

    else:

        production_run = client.get_run(
            production_model.run_id
        )

        production_score = production_run.data.metrics.get(
            metric_to_optimize
        )

        if production_score is None:

            print(
                f"[WARNING] Production model does not contain "
                f"'{metric_to_optimize}'."
            )

            move_to_production = True

        else:

            print(
                f"[INFO] Production {metric_to_optimize}: "
                f"{production_score:.4f}"
            )

            # Promote if the new model is better.
            # Also promote when scores are equal.
            move_to_production = (
                best_score >= production_score
            )


# --------------------------------------------------
# STEP 9: Promote or Archive Models
# --------------------------------------------------

if move_to_production:

    print(
        f"\n[INFO] Promoting Version {best_model.version} "
        "to Production..."
    )

    # Archive existing Production models
    for version in production_models:

        try:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived"
            )

            print(
                f"[INFO] Previous Production Version "
                f"{version.version} archived."
            )

        except Exception as error:

            print(
                f"[ERROR] Could not archive Production "
                f"Version {version.version}: {error}"
            )

    # Promote best Staging model
    client.transition_model_version_stage(
        name=model_name,
        version=best_model.version,
        stage="Production"
    )

    print(
        f"[SUCCESS] Version {best_model.version} "
        "promoted to Production."
    )

else:

    print(
        "\n[INFO] Existing Production model is better."
    )

    print(
        f"[INFO] Version {best_model.version} "
        "will not replace Production."
    )


# --------------------------------------------------
# STEP 10: Archive Remaining Staging Models
# --------------------------------------------------

versions = client.search_model_versions(
    f"name='{model_name}'"
)

for version in versions:

    if version.current_stage == "Staging":

        try:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived"
            )

            print(
                f"[INFO] Version {version.version} moved to Archived."
            )

        except Exception as error:

            print(
                f"[ERROR] Could not archive Version "
                f"{version.version}: {error}"
            )


# --------------------------------------------------
# STEP 11: Execution Complete
# --------------------------------------------------

print("\n[SUCCESS] Automated Model Lifecycle execution complete!")
