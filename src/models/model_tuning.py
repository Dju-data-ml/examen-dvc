import argparse
import logging
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("optimisation du modèle")

def optimize_hyperparameters(data_dir="data/processed_data"):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).iloc[:, 0]

    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, 30],
        "min_samples_split": [2, 5, 10],
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_


def save_best_params(best_params):
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "best_params.pkl")

    with open(output_path, "wb") as f:
        pickle.dump(best_params, f)
    logger.info("optimisation du modèle RF finie avec les paramètres : %s", best_params)

def main():
    parser = argparse.ArgumentParser(
        description="Optimisation des hyperparamètres d'un RF Regressor"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed_data",
        help="Chemin vers le dossier contenant les données d'entraînement",
    )
    args = parser.parse_args()

    best_params = optimize_hyperparameters(args.data_dir)
    save_best_params(best_params)

if __name__ == "__main__":
    main()
