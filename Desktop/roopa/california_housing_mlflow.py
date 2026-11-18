from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
TRACKING_DB = PROJECT_DIR / "mlflow.db"
ARTIFACT_ROOT = PROJECT_DIR / "mlruns" / "california-housing-price-prediction"
EXPERIMENT_NAME = "california-housing-price-prediction"
TEST_SIZE = 0.2
RANDOM_STATE = 42
COMPARISON_GROUP = "ridge-scaler-grid"
REGISTERED_MODEL_NAME = "california-housing-price-predictor"
CHAMPION_ALIAS = "champion"

RUN_CONFIGS = [
    {
        "run_name": "standard-scaler-ridge-alpha-0.1",
        "scaler_name": "standard",
        "scaler": StandardScaler(),
        "alpha": 0.1,
    },
    {
        "run_name": "minmax-scaler-ridge-alpha-0.1",
        "scaler_name": "minmax",
        "scaler": MinMaxScaler(),
        "alpha": 0.1,
    },
    {
        "run_name": "standard-scaler-ridge-alpha-10",
        "scaler_name": "standard",
        "scaler": StandardScaler(),
        "alpha": 10.0,
    },
    {
        "run_name": "minmax-scaler-ridge-alpha-10",
        "scaler_name": "minmax",
        "scaler": MinMaxScaler(),
        "alpha": 10.0,
    },
]


def get_experiment_id():
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=ARTIFACT_ROOT.as_uri()
        )
    return experiment.experiment_id


def main():
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
    mlflow.set_experiment(experiment_id=get_experiment_id())

    housing = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        housing.data, housing.target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = []
    for config in RUN_CONFIGS:
        run_name = config["run_name"]
        model = Pipeline(
            [
                ("scaler", config["scaler"]),
                ("regressor", Ridge(alpha=config["alpha"])),
            ]
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = {
            "mae": mean_absolute_error(y_test, predictions),
            "r2": r2_score(y_test, predictions),
        }

        with mlflow.start_run(
            run_name=run_name,
            tags={"comparison_group": COMPARISON_GROUP},
        ) as run:
            mlflow.set_tag("model_type", "Ridge")
            mlflow.set_tag("dataset", "California Housing")
            mlflow.log_params(
                {
                    "model": "Ridge",
                    "preprocessing": config["scaler_name"],
                    "alpha": config["alpha"],
                    "test_size": TEST_SIZE,
                    "random_state": RANDOM_STATE,
                    "training_rows": len(X_train),
                    "test_rows": len(X_test),
                }
            )
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            signature = infer_signature(X_test, predictions)
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=X_test.iloc[:5],
            )

        results.append(
            {"run_name": run_name, "run_id": run.info.run_id, "metrics": metrics}
        )
        print(
            f"{run_name}: MAE={metrics['mae']:.3f}, R2={metrics['r2']:.3f}"
        )

    best_result = max(
        results,
        key=lambda result: (
            result["metrics"]["r2"],
            -result["metrics"]["mae"],
        ),
    )
    best_name = best_result["run_name"]
    best_run_id = best_result["run_id"]
    best_metrics = best_result["metrics"]

    client = MlflowClient()
    client.set_tag(best_run_id, "selection_metric", "r2_desc_mae_asc")
    client.set_tag(best_run_id, "selected_for_registry", "true")
    model_version = mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/model",
        name=REGISTERED_MODEL_NAME,
        tags={
            "comparison_group": COMPARISON_GROUP,
            "selection_metric": "r2_desc_mae_asc",
            "source_run_id": best_run_id,
        },
    )
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS, model_version.version
    )

    print(
        f"Best model: {best_name} "
        f"(R2={best_metrics['r2']:.3f}, MAE={best_metrics['mae']:.3f})"
    )
    print(
        f"Registered {REGISTERED_MODEL_NAME} version {model_version.version} "
        f"with alias '{CHAMPION_ALIAS}'."
    )
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()
