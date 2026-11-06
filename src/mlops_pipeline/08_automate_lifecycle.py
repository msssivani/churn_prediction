
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import yaml
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIFECYCLE_REPORT_PATH = PROJECT_ROOT / "reports" / "lifecycle_report.json"


def configure_mlflow(config):
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or config["mlflow"].get("tracking_uri")
    )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    print(f"[INFO] MLflow Tracking URI: {mlflow.get_tracking_uri()}")


def write_lifecycle_report(report):
    LIFECYCLE_REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with LIFECYCLE_REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        f.write("\n")


def validate_model_artifact(model_name, version):
    """
    Verify that the registered model version can actually be loaded.
    A model must pass this check before Production promotion.
    """

    model_uri = f"models:/{model_name}/{version}"

    print(f"[INFO] Validating model artifact: {model_uri}")

    try:
        model = mlflow.sklearn.load_model(model_uri)

        if model is None:
            print("[ERROR] Loaded model is empty.")
            return False

        print(
            f"[SUCCESS] Version {version} artifact is accessible."
        )
        return True

    except Exception as exc:
        print(
            f"[ERROR] Version {version} artifact validation failed: {exc}"
        )
        return False


def automate_model_lifecycle():
    print("[INFO] Starting Automated Model Lifecycle Manager...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)

    client = MlflowClient()

    model_name = config["registry"]["model_name"]
    metric_to_optimize = config["registry"]["decision_metric"]

    try:

        # STEP 1: Retrieve registered model versions.
        all_versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        if not all_versions:
            print("[ERROR] No registered model versions found.")
            return False

        print(f"[INFO] Found {len(all_versions)} model versions.")

        # STEP 2: Move newly registered models to Staging.
        new_models = [
            mv for mv in all_versions
            if mv.current_stage == "None"
        ]

        for mv in new_models:
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Staging",
                archive_existing_versions=False,
            )

            print(f"[INFO] Version {mv.version} moved to Staging.")

        # Refresh version information.
        all_versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        # STEP 3: Identify Staging and Production models.
        staging_models = [
            mv for mv in all_versions
            if mv.current_stage == "Staging"
        ]

        production_models = [
            mv for mv in all_versions
            if mv.current_stage == "Production"
        ]

        production_model = (
            production_models[0] if production_models else None
        )

        production_before = (
            int(production_model.version)
            if production_model else None
        )

        if not staging_models:
            print("[INFO] No models available in Staging.")

            write_lifecycle_report({
                "status": "completed",
                "model_name": model_name,
                "candidate_version": None,
                "candidate_metric": None,
                "decision_metric": metric_to_optimize,
                "production_version_before": production_before,
                "production_version_after": production_before,
                "promoted": False,
                "reason": "No models were available in Staging.",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })

            return True

        # STEP 4: Validate artifacts and evaluate Staging models.
        valid_candidates = []

        for mv in staging_models:

            print(f"\n[INFO] Checking Version {mv.version}")

            print(f"[INFO] Model source: {mv.source}")

            if mv.run_id:
                run = client.get_run(mv.run_id)

                print(
                    f"[INFO] Run artifact URI: {run.info.artifact_uri}"
                )

                score = run.data.metrics.get(metric_to_optimize)

            else:
                print("[WARNING] Model has no associated Run ID.")
                continue

            if score is None:
                print(
                    f"[WARNING] Metric '{metric_to_optimize}' not found."
                )
                continue

            # IMPORTANT: Validate artifact before considering promotion.
            if not validate_model_artifact(
                model_name, mv.version
            ):
                print(
                    f"[WARNING] Version {mv.version} rejected "
                    "because its artifact is inaccessible."
                )
                continue

            valid_candidates.append((mv, float(score)))

        # STEP 5: Select the best valid candidate.
        if not valid_candidates:
            print(
                "[ERROR] No Staging model has a valid accessible artifact."
            )

            write_lifecycle_report({
                "status": "failed",
                "model_name": model_name,
                "candidate_version": None,
                "candidate_metric": None,
                "decision_metric": metric_to_optimize,
                "production_version_before": production_before,
                "production_version_after": production_before,
                "promoted": False,
                "reason": "No Staging model passed artifact validation.",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })

            return False

        best_model, best_score = max(
            valid_candidates,
            key=lambda item: item[1]
        )

        print(
            f"\n[SUCCESS] Best valid Staging model: "
            f"Version {best_model.version}"
        )

        print(
            f"[INFO] Best {metric_to_optimize}: {best_score:.4f}"
        )

        # STEP 6: Compare against current Production model.
        if production_model is None:

            move_to_production = True

            print(
                "[INFO] No Production model exists. "
                "Candidate will be promoted."
            )

        else:

            production_run = client.get_run(
                production_model.run_id
            )

            production_score = production_run.data.metrics.get(
                metric_to_optimize, 0.0
            )

            print(
                f"[INFO] Production Version "
                f"{production_model.version} | "
                f"{metric_to_optimize}={production_score:.4f}"
            )

            move_to_production = best_score > production_score

        # STEP 7: Promote or archive the candidate.
        if move_to_production:

            # Validate once more immediately before promotion.
            if not validate_model_artifact(
                model_name, best_model.version
            ):
                print(
                    "[ERROR] Final artifact validation failed. "
                    "Promotion cancelled."
                )
                return False

            client.transition_model_version_stage(
                name=model_name,
                version=best_model.version,
                stage="Production",
                archive_existing_versions=False,
            )

            if production_model:
                client.transition_model_version_stage(
                    name=model_name,
                    version=production_model.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )

            print(
                f"[SUCCESS] Version {best_model.version} "
                "is now Production."
            )

        else:

            client.transition_model_version_stage(
                name=model_name,
                version=best_model.version,
                stage="Archived",
                archive_existing_versions=False,
            )

            print(
                f"[INFO] Version {best_model.version} archived. "
                "It did not improve Production."
            )

        # STEP 8: Archive remaining Staging versions.
        final_versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        for mv in final_versions:
            if mv.current_stage == "Staging":

                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )

                print(
                    f"[INFO] Version {mv.version} moved to Archived."
                )

        # STEP 9: Generate lifecycle report.
        final_versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        production_after = [
            mv for mv in final_versions
            if mv.current_stage == "Production"
        ]

        production_version_after = (
            int(production_after[0].version)
            if production_after else None
        )

        write_lifecycle_report({
            "status": "completed",
            "model_name": model_name,
            "candidate_version": int(best_model.version),
            "candidate_metric": best_score,
            "decision_metric": metric_to_optimize,
            "production_version_before": production_before,
            "production_version_after": production_version_after,
            "promoted": move_to_production,
            "reason": (
                "Candidate improved the configured production metric."
                if move_to_production
                else "Candidate did not improve the configured production metric."
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })

        print(
            "[SUCCESS] Automated Model Lifecycle execution complete!"
        )

        return True

    except Exception as exc:

        print(
            f"[ERROR] Failed to automate model lifecycle: {exc}"
        )

        write_lifecycle_report({
            "status": "failed",
            "model_name": model_name,
            "promoted": False,
            "error": str(exc),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })

        return False


if __name__ == "__main__":
    passed = automate_model_lifecycle()
    sys.exit(0 if passed else 1)
