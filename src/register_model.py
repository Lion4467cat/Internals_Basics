import json
import os
import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "shieldops-resolution-hours"
REGISTERED_NAME = "shieldops-resolution-hours-predictor"
RESULTS_DIR = "results"


def main():
    print("=== ShieldOps: Model Registration ===")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    # Get the best run by MAE from the tuning parent run
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'tuning-shieldops'",
        order_by=["metrics.best_mae ASC"],
    )

    best_run = runs[0]
    run_id = best_run.info.run_id
    best_mae = best_run.data.metrics.get("best_mae", 0.0)

    print(f"Best run ID: {run_id}")
    print(f"Best MAE: {best_mae:.4f}")

    # Register the model
    model_uri = f"runs:/{run_id}/best_model"
    mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_NAME)

    print(f"Registered version: {mv.version}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "registered_model_name": REGISTERED_NAME,
        "version": int(mv.version),
        "run_id": run_id,
        "source_metric": "mae",
        "source_metric_value": round(best_mae, 4),
    }
    with open(f"{RESULTS_DIR}/step4_s6.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Model registered as version {mv.version}")
    print(f"Results saved to results/step4_s6.json")


if __name__ == "__main__":
    main()
