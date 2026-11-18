import shutil
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient


PROJECT_DIR = Path(__file__).resolve().parent
TRACKING_DB = PROJECT_DIR / "mlflow.db"
REGISTERED_MODEL_NAME = "california-housing-price-predictor"
MODEL_ALIAS = "champion"
MODEL_DIR = PROJECT_DIR / "model"


def main():
    if not TRACKING_DB.is_file():
        raise FileNotFoundError(
            f"MLflow tracking database was not found: {TRACKING_DB}"
        )
    if MODEL_DIR.exists():
        raise FileExistsError(
            f"{MODEL_DIR} already exists. Remove or move it before exporting again."
        )

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
    model_version = MlflowClient().get_model_version_by_alias(
        REGISTERED_MODEL_NAME, MODEL_ALIAS
    )

    with tempfile.TemporaryDirectory(
        prefix="housing-model-", dir=PROJECT_DIR
    ) as temp_dir:
        downloaded_model = mlflow.artifacts.download_artifacts(
            artifact_uri=model_version.source,
            dst_path=temp_dir,
        )
        shutil.copytree(downloaded_model, MODEL_DIR)

    print(
        f"Exported {REGISTERED_MODEL_NAME}@{MODEL_ALIAS} "
        f"(version {model_version.version}) to {MODEL_DIR}."
    )


if __name__ == "__main__":
    main()
