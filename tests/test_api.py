import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from train import train_model


def test_health(tmp_path):
    model_path = tmp_path / "iris_model.joblib"
    train_model(model_path)
    app = create_app(model_path)

    with TestClient(app) as client:
        health = client.get("/health")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("features", "class_id", "species"),
    [
        ([5.1, 3.5, 1.4, 0.2], 0, "setosa"),
        ([7.0, 3.2, 4.7, 1.4], 1, "versicolor"),
        ([6.3, 3.3, 6.0, 2.5], 2, "virginica"),
    ],
)
def test_prediction_for_each_species(tmp_path, features, class_id, species):
    model_path = tmp_path / "iris_model.joblib"
    train_model(model_path)
    app = create_app(model_path)

    with TestClient(app) as client:
        prediction = client.post("/predict", json={"features": features})

    assert prediction.status_code == 200
    assert prediction.json() == {"class_id": class_id, "species": species}


def test_prediction_requires_four_features(tmp_path):
    model_path = tmp_path / "iris_model.joblib"
    train_model(model_path)
    app = create_app(model_path)

    with TestClient(app) as client:
        response = client.post("/predict", json={"features": [5.1, 3.5, 1.4]})

    assert response.status_code == 422
