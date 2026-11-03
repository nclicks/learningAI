# Iris Flower Classifier API

A small Iris flower classifier served with FastAPI. Training experiments are
tracked with MLflow, and the best validation model is registered as the
`champion` alias.

## Project files

- `app/main.py` — FastAPI health check and prediction endpoints.
- `train.py` — runs reproducible Random Forest experiments, logs parameters,
  metrics, and models to MLflow, registers the best run, and saves the model
  used by the API.
- `tests/` — API and MLflow training tests.
- `requirements.txt` — Python dependencies.
- `Dockerfile` — builds the application image and trains/registers a model.

## Requirements

- Python 3 and Docker Desktop with the Docker engine running.

## Install dependencies

In PowerShell:

```powershell
Set-Location 'C:\Users\nClick\Documents\github\iris-flower-classifier-api'
python -m pip install -r requirements.txt
```

## Reproduce training

Training uses scikit-learn's Iris dataset, a stratified 80/20 train/test split,
and random seed `42`. It runs four Random Forest configurations and ranks them
by validation accuracy, breaking ties with macro F1.

```powershell
python train.py `
  --tracking-uri sqlite:///mlflow.db `
  --experiment-name iris-flower-classifier
```

Each run records its parameters, accuracy, macro F1, and model artifact. The
best run is registered as `iris-flower-classifier`; its `champion` alias is
updated and its model is saved to `app/iris_model.joblib` for the API.
Re-running the command repeats the same split and configurations, adds runs to
the experiment, and updates the alias to the new best run.

View the recorded experiments in the MLflow UI:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1
```

Open `http://127.0.0.1:5000` in a browser. The SQLite database and `mlartifacts`
directory are local training outputs and should not be committed.

## Run tests

The Docker image contains Python 3.12 and avoids native dependency issues that
can occur with some Windows Python installations:

```powershell
docker build -t iris-flower-classifier-api .
docker run --rm -v 'C:\Users\nClick\Documents\github\iris-flower-classifier-api\tests:/app/tests:ro' iris-flower-classifier-api python -m pytest -v /app/tests
```

## Build and run the API

The Docker build trains the model and registers it in the image's local MLflow
store. Start the API container:

```powershell
docker run --name iris-flower-classifier-api -d -p 8000:8000 iris-flower-classifier-api
```

Health check:

```powershell
Invoke-RestMethod -Uri 'http://localhost:8000/health'
```

## Try all three species

Each prediction input contains sepal length, sepal width, petal length, and
petal width in centimeters:

```powershell
# setosa
Invoke-RestMethod -Uri 'http://localhost:8000/predict' `
  -Method Post `
  -ContentType 'application/json' `
  -Body '{"features":[5.1,3.5,1.4,0.2]}'

# versicolor
Invoke-RestMethod -Uri 'http://localhost:8000/predict' `
  -Method Post `
  -ContentType 'application/json' `
  -Body '{"features":[7.0,3.2,4.7,1.4]}'

# virginica
Invoke-RestMethod -Uri 'http://localhost:8000/predict' `
  -Method Post `
  -ContentType 'application/json' `
  -Body '{"features":[6.3,3.3,6.0,2.5]}'
```

The expected class IDs are `0` (`setosa`), `1` (`versicolor`), and `2`
(`virginica`).

Stop the container when finished:

```powershell
docker stop iris-flower-classifier-api
```
