import json
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

DATA_PATH = "data/training_data.csv"
RESULTS_DIR = "results"
EXPERIMENT_NAME = "shieldops-resolution-hours"


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)
    return mae, rmse, r2


def main():
    print("=== ShieldOps: Experiment Tracking & Model Comparison ===")

    df = pd.read_csv(DATA_PATH)
    X = df[["severity_level", "alerts_count", "analyst_experience", "is_automated"]]
    y = df["resolution_hours"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    # --- SVR ---
    with mlflow.start_run(run_name="SVR"):
        mlflow.set_tag("experiment_type", "baseline_comparison")
        model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
        model.fit(X_train, y_train)
        mae, rmse, r2 = evaluate(model, X_test, y_test)

        mlflow.log_param("kernel", "rbf")
        mlflow.log_param("C", 1.0)
        mlflow.log_param("epsilon", 0.1)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, name="model")

        print(f"SVR     → MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
        results.append(
            {
                "name": "SVR",
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "r2": round(r2, 4),
            }
        )

    # --- GradientBoosting ---
    with mlflow.start_run(run_name="GradientBoosting"):
        mlflow.set_tag("experiment_type", "baseline_comparison")
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        )
        model.fit(X_train, y_train)
        mae, rmse, r2 = evaluate(model, X_test, y_test)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, name="model")

        print(f"GradBoost → MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
        results.append(
            {
                "name": "GradientBoosting",
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "r2": round(r2, 4),
            }
        )

    # --- Best model ---
    best = min(results, key=lambda x: x["mae"])

    import joblib

    best_model_obj = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    best_model_obj.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model_obj, "models/best_model.pkl")
    print("✅ Best model saved to models/best_model.pkl")

    # --- Save results ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "experiment_name": EXPERIMENT_NAME,
        "models": results,
        "best_model": best["name"],
        "best_metric_name": "mae",
        "best_metric_value": best["mae"],
    }
    with open(f"{RESULTS_DIR}/step1_s1.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Best model: {best['name']} with MAE: {best['mae']}")
    print(f"Results saved to results/step1_s1.json")


if __name__ == "__main__":
    main()
