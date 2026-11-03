import argparse
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

DEFAULT_MODEL_PATH = Path(__file__).parent / "app" / "iris_model.joblib"


def train_model(model_path: Path = DEFAULT_MODEL_PATH) -> RandomForestClassifier:
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(iris.data, iris.target)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Iris classifier.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Where to save the model (default: {DEFAULT_MODEL_PATH})",
    )
    args = parser.parse_args()

    train_model(args.model_path)
    print(f"Saved trained model to {args.model_path}")


if __name__ == "__main__":
    main()
