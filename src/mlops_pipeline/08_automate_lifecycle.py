import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIFECYCLE_REPORT_PATH = PROJECT_ROOT / "reports" / "lifecycle_report.json"


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def write_lifecycle_report(report):
    LIFECYCLE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LIFECYCLE_REPORT_PATH.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=4)
        report_file.write("\n")


def automate_model_lifecycle():
    print("[INFO] Starting Automated Model Lifecycle Manager...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    client = MlflowClient()
    model_name = config["registry"]["model_name"]
    metric_to_optimize = config["registry"]["decision_metric"]

    try:
        # 1. Find new models and move them to Staging.
        all_versions = client.search_model_versions(f"name='{model_name}'")
        new_models = [mv for mv in all_versions if mv.current_stage == "None"]

        for mv in new_models:
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Staging",
                archive_existing_versions=False,
            )
            print(f"[INFO] Version {mv.version} moved to Staging.")

        # 2. Find the Staging models.
        all_versions = client.search_model_versions(f"name='{model_name}'")
        models_staging = [mv for mv in all_versions if mv.current_stage == "Staging"]

        if not models_staging:
            print("[INFO] No models in Staging. Nothing to promote.")
            production_models = [
                mv for mv in all_versions if mv.current_stage == "Production"
            ]
            production_version = (
                int(production_models[0].version) if production_models else None
            )
            write_lifecycle_report(
                {
                    "status": "completed",
                    "model_name": model_name,
                    "candidate_version": None,
                    "candidate_metric": None,
                    "decision_metric": metric_to_optimize,
                    "production_version_before": production_version,
                    "production_version_after": production_version,
                    "promoted": False,
                    "reason": "No models were available in Staging.",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            return True

        # 3. Select the best Staging model using the configured metric.
        best_model = None
        best_score = -1.0

        for mv in models_staging:
            run = client.get_run(mv.run_id)
            score = run.data.metrics.get(metric_to_optimize, 0.0)
            print(
                f"[INFO] Staging Version {mv.version} | "
                f"{metric_to_optimize}={score:.4f}"
            )

            if score > best_score:
                best_score = score
                best_model = mv

        # 4. Find the current Production model.
        production_models = [
            mv for mv in all_versions if mv.current_stage == "Production"
        ]
        production_model = production_models[0] if production_models else None

        if production_model is None:
            move_to_production = True
            print("[INFO] No Production model exists. Staging model will become Production.")
        else:
            production_run = client.get_run(production_model.run_id)
            production_score = production_run.data.metrics.get(metric_to_optimize, 0.0)

            print(
                f"[INFO] Production Version {production_model.version} | "
                f"{metric_to_optimize}={production_score:.4f}"
            )

            move_to_production = best_score > production_score

            if move_to_production:
                print("[SUCCESS] Staging model is better than Production.")
            else:
                print("[INFO] Production model remains unchanged.")

        # 5. Promote the better model or archive it.
        if move_to_production:
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

            print(f"[SUCCESS] Version {best_model.version} is now Production.")
        else:
            client.transition_model_version_stage(
                name=model_name,
                version=best_model.version,
                stage="Archived",
                archive_existing_versions=False,
            )

            print(f"[INFO] Version {best_model.version} archived because it did not improve.")

        # 6. Archive any other models left in Staging.
        final_versions = client.search_model_versions(f"name='{model_name}'")
        for mv in final_versions:
            if mv.current_stage == "Staging":
                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )

        final_versions = client.search_model_versions(f"name='{model_name}'")
        production_after = [
            mv for mv in final_versions if mv.current_stage == "Production"
        ]
        production_version_after = (
            int(production_after[0].version) if production_after else None
        )
        write_lifecycle_report(
            {
                "status": "completed",
                "model_name": model_name,
                "candidate_version": int(best_model.version),
                "candidate_metric": best_score,
                "decision_metric": metric_to_optimize,
                "production_version_before": (
                    int(production_model.version) if production_model else None
                ),
                "production_version_after": production_version_after,
                "promoted": move_to_production,
                "reason": (
                    "Candidate improved the configured production metric."
                    if move_to_production
                    else "Candidate did not improve the configured production metric."
                ),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        print("[SUCCESS] Automated Model Lifecycle execution complete!")
        return True

    except Exception as exc:
        print(f"[ERROR] Failed to automate model lifecycle: {exc}")
        write_lifecycle_report(
            {
                "status": "failed",
                "promoted": False,
                "error": str(exc),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return False


if __name__ == "__main__":
    passed = automate_model_lifecycle()
    sys.exit(0 if passed else 1)
