import argparse
import logging
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("évaluation du modèle")

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))
    return {"rmse": rmse, "r2": r2}


def save_metrics(metrics, metrics_dir="metrics"):
    os.makedirs(metrics_dir, exist_ok=True)
    output_path = os.path.join(metrics_dir, "score.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("evaluation du modèle finie avec les scores %s et enregistrer dans %s", metrics,metrics_dir)


def main():
    parser = argparse.ArgumentParser(description="Évaluation des performances du modèle")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed_data",
        help="Chemin du répertoire contenant les données de tests",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="metrics",
        help="Chemin du répertoire d'enregistrement des métriques",
    )
    args = parser.parse_args()

    model = joblib.load('models/best_model.joblib')

    # Charger les données de test
    X_test_path = os.path.join(args.data_dir, "X_test_scaled.csv")
    y_test_path = os.path.join(args.data_dir, "y_test.csv")
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    # Évaluer le modèle et enregistrer les métriques
    scores = evaluate_model(model, X_test, y_test)
    save_metrics(scores, args.metrics_dir)


if __name__ == "__main__":
    main()
