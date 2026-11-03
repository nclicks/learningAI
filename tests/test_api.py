from fastapi.testclient import TestClient

from app.main import create_app
from train import train_model


def test_health_and_prediction(tmp_path):
    model_path = tmp_path / "iris_model.joblib"
    train_model(model_path)
    app = create_app(model_path)

    with TestClient(app) as client:
        health = client.get("/health")
        prediction = client.post(
            "/predict", json={"features": [5.1, 3.5, 1.4, 0.2]}
        )

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert prediction.status_code == 200
    assert prediction.json() == {"class_id": 0, "species": "setosa"}


def test_prediction_requires_four_features(tmp_path):
    model_path = tmp_path / "iris_model.joblib"
    train_model(model_path)
    app = create_app(model_path)

    with TestClient(app) as client:
        response = client.post("/predict", json={"features": [5.1, 3.5, 1.4]})

    assert response.status_code == 422
