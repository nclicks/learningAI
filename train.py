import argparse
import platform
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
import sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

DEFAULT_MODEL_PATH = Path(__file__).parent / "app" / "iris_model.joblib"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT_NAME = "iris-flower-classifier"
REGISTERED_MODEL_NAME = "iris-flower-classifier"
RANDOM_STATE = 42
TEST_SIZE = 0.2
RUN_CONFIGS = (
    {"n_estimators": 50, "max_depth": None},
    {"n_estimators": 100, "max_depth": None},
    {"n_estimators": 200, "max_depth": None},
    {"n_estimators": 100, "max_depth": 3},
)


@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    accuracy: float
    macro_f1: float
    model: RandomForestClassifier


@dataclass(frozen=True)
class TrainingResult:
    best_run_id: str
    best_accuracy: float
    best_macro_f1: float
    registered_model_version: str
    model_path: Path


def train_model(model_path: Path = DEFAULT_MODEL_PATH) -> RandomForestClassifier:
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(iris.data, iris.target)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


def run_experiments(
    model_path: Path = DEFAULT_MODEL_PATH,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> TrainingResult:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=iris.target,
    )
    completed_runs: list[ExperimentRun] = []

    for config in RUN_CONFIGS:
        with mlflow.start_run() as run:
            model = RandomForestClassifier(
                **config,
                random_state=RANDOM_STATE,
            )
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            accuracy = float(accuracy_score(y_test, predictions))
            macro_f1 = float(f1_score(y_test, predictions, average="macro"))

            mlflow.log_params(
                {
                    **config,
                    "random_state": RANDOM_STATE,
                    "test_size": TEST_SIZE,
                    "stratified": True,
                    "dataset": "scikit-learn iris",
                    "python_version": platform.python_version(),
                    "sklearn_version": sklearn.__version__,
                    "mlflow_version": mlflow.__version__,
                }
            )
            mlflow.log_metrics({"accuracy": accuracy, "macro_f1": macro_f1})
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=infer_signature(x_test, predictions),
                input_example=x_test[:1],
                serialization_format="cloudpickle",
            )
            completed_runs.append(
                ExperimentRun(
                    run_id=run.info.run_id,
                    accuracy=accuracy,
                    macro_f1=macro_f1,
                    model=model,
                )
            )

    best = max(
        completed_runs,
        key=lambda result: (result.accuracy, result.macro_f1),
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.model, model_path)

    model_uri = f"runs:/{best.run_id}/model"
    registered_model = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    MlflowClient().set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "champion",
        registered_model.version,
    )

    return TrainingResult(
        best_run_id=best.run_id,
        best_accuracy=best.accuracy,
        best_macro_f1=best.macro_f1,
        registered_model_version=registered_model.version,
        model_path=model_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reproducible Iris classifier experiments with MLflow."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Where to save the best model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help=f"MLflow tracking URI (default: {DEFAULT_TRACKING_URI})",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help=f"MLflow experiment name (default: {DEFAULT_EXPERIMENT_NAME})",
    )
    args = parser.parse_args()

    result = run_experiments(
        model_path=args.model_path,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )
    print(f"Best run: {result.best_run_id}")
    print(f"Accuracy: {result.best_accuracy:.4f}")
    print(f"Macro F1: {result.best_macro_f1:.4f}")
    print(
        f"Registered {REGISTERED_MODEL_NAME} version "
        f"{result.registered_model_version} as @champion"
    )
    print(f"Saved deployment model to {result.model_path}")


if __name__ == "__main__":
    main()
