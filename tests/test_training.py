import joblib
from mlflow import MlflowClient
from sklearn.datasets import load_iris

from train import REGISTERED_MODEL_NAME, RUN_CONFIGS, run_experiments


def test_runs_experiments_and_registers_best_model(tmp_path):
    model_path = tmp_path / "best_model.joblib"
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    result = run_experiments(
        model_path=model_path,
        tracking_uri=tracking_uri,
        experiment_name="test-iris-classifier",
    )

    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name("test-iris-classifier")
    runs = client.search_runs([experiment.experiment_id], max_results=10)
    champion = client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, "champion"
    )
    model = joblib.load(model_path)

    assert len(runs) == len(RUN_CONFIGS)
    assert all("accuracy" in run.data.metrics for run in runs)
    assert all("macro_f1" in run.data.metrics for run in runs)
    assert all(run.data.params["random_state"] == "42" for run in runs)
    assert max(run.data.metrics["accuracy"] for run in runs) == result.best_accuracy
    assert champion.version == result.registered_model_version
    assert champion.run_id == result.best_run_id
    assert client.list_artifacts(result.best_run_id, "model")
    assert model.predict(load_iris().data[:3]).tolist() == [0, 0, 0]
