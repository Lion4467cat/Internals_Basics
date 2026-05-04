import json
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, ParameterSampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

DATA_PATH = "data/training_data.csv"
RESULTS_DIR = "results"
EXPERIMENT_NAME = "shieldops-resolution-hours"


def main():
    print("=== ShieldOps: Hyperparameter Tuning ===")

    df = pd.read_csv(DATA_PATH)
    X = df[["severity_level", "alerts_count", "analyst_experience", "is_automated"]]
    y = df["resolution_hours"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5],
    }

    # Random search - sample 6 combinations
    sampler = list(ParameterSampler(param_grid, n_iter=6, random_state=42))
    total_trials = len(sampler)

    best_mae = float("inf")
    best_cv_mae = float("inf")
    best_params = {}
    best_model = None
    trial_results = []

    with mlflow.start_run(run_name="tuning-shieldops") as parent_run:
        mlflow.set_tag("search_type", "random")
        mlflow.set_tag("n_folds", 3)
        mlflow.set_tag("total_trials", total_trials)

        for i, params in enumerate(sampler):
            with mlflow.start_run(run_name=f"trial-{i + 1}", nested=True):
                model = GradientBoostingRegressor(
                    n_estimators=params["n_estimators"],
                    learning_rate=params["learning_rate"],
                    max_depth=params["max_depth"],
                    random_state=42,
                )

                # 3-fold cross validation
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=3, scoring="neg_mean_absolute_error"
                )
                cv_mae = -cv_scores.mean()

                # Train on full train set and evaluate on test
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mae = mean_absolute_error(y_test, preds)
                rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                r2 = r2_score(y_test, preds)

                # Log to MLflow
                mlflow.log_param("n_estimators", params["n_estimators"])
                mlflow.log_param("learning_rate", params["learning_rate"])
                mlflow.log_param("max_depth", params["max_depth"])
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("cv_mae", cv_mae)

                print(
                    f"Trial {i + 1}: params={params} → MAE: {mae:.3f}, CV_MAE: {cv_mae:.3f}"
                )

                if cv_mae < best_cv_mae:
                    best_cv_mae = cv_mae
                    best_mae = mae
                    best_params = params
                    best_model = model

        # Log best to parent
        mlflow.log_param("best_n_estimators", best_params["n_estimators"])
        mlflow.log_param("best_learning_rate", best_params["learning_rate"])
        mlflow.log_param("best_max_depth", best_params["max_depth"])
        mlflow.log_metric("best_mae", best_mae)
        mlflow.log_metric("best_cv_mae", best_cv_mae)
        mlflow.sklearn.log_model(best_model, "best_model")
        mlflow.set_tag("best_run_mae", round(best_mae, 4))
        mlflow.log_param("best_model_type", "GradientBoostingRegressor")

    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    print(f"\n✅ Best model saved to models/best_model.pkl")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "search_type": "random",
        "n_folds": 3,
        "total_trials": total_trials,
        "best_params": best_params,
        "best_mae": round(best_mae, 4),
        "best_cv_mae": round(best_cv_mae, 4),
        "parent_run_name": "tuning-shieldops",
    }
    with open(f"{RESULTS_DIR}/step2_s2.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Best params: {best_params}")
    print(f"✅ Results saved to results/step2_s2.json")


if __name__ == "__main__":
    main()
