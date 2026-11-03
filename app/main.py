from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).with_name("iris_model.joblib")
SPECIES = ("setosa", "versicolor", "virginica")


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=4, max_length=4)


class PredictionResponse(BaseModel):
    class_id: int
    species: str


def create_app(model_path: Path = MODEL_PATH) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not model_path.is_file():
            raise RuntimeError(
                f"Model not found at {model_path}. Run `python train.py` first."
            )
        app.state.model = joblib.load(model_path)
        yield

    app = FastAPI(title="Iris Classifier API", lifespan=lifespan)

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        class_id = int(request.app.state.model.predict([payload.features])[0])
        return PredictionResponse(class_id=class_id, species=SPECIES[class_id])

    return app


app = create_app()
