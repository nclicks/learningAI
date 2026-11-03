# Iris Classifier API

A small Iris flower classifier served with FastAPI. The model is trained from
scikit-learn's Iris dataset and included in the Docker image.

## Project files

- `app/main.py` — FastAPI health check and prediction endpoints.
- `train.py` — trains the Random Forest model and saves it to
  `app/iris_model.joblib`.
- `tests/test_api.py` — API tests for health, prediction, and invalid input.
- `requirements.txt` — Python dependencies.
- `Dockerfile` — builds the application image and trains the model.

## Requirements

- Docker Desktop with the Docker engine running.
- Python 3 for local dependency installation or training.

## Install dependencies

Run these commands in PowerShell:

```powershell
Set-Location 'C:\Users\nClick\Documents\github\mlops-week1'
python -m pip install -r requirements.txt
```

## Train the model locally

The Docker image trains the model during its build. To train it on your machine
instead:

```powershell
python train.py
```

## Run tests

To run tests inside the Docker image's Python 3.12 environment:

```powershell
docker run --rm -v 'C:\Users\nClick\Documents\github\mlops-week1\tests:/app/tests:ro' iris-classifier-api python -m pytest -v /app/tests
```

The two API tests passed in this environment. Running the tests directly under
Python 3.15 on Windows was blocked by a Windows policy preventing SciPy's native
DLL from loading.

## Build and run with Docker

Build the image:

```powershell
docker build -t iris-classifier-api .
```

Start the API container:

```powershell
docker run --name iris-classifier-api-week1 -d -p 8000:8000 iris-classifier-api
```

## Check the API

Health check:

```powershell
Invoke-RestMethod -Uri 'http://localhost:8000/health'
```

Prediction (four features in centimeters: sepal length, sepal width, petal
length, and petal width):

```powershell
Invoke-RestMethod -Uri 'http://localhost:8000/predict' `
  -Method Post `
  -ContentType 'application/json' `
  -Body '{"features":[5.1,3.5,1.4,0.2]}'
```

The expected example prediction is class `0`, species `setosa`.

Confirm the container is running:

```powershell
docker ps --filter 'name=^/iris-classifier-api-week1$'
```

Stop the container:

```powershell
docker stop iris-classifier-api-week1
```
