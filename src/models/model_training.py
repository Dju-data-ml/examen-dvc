import os
import logging
import argparse
import pickle
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("entrainement du modèle")


def train_model(data_dir: str = 'data/processed_data') -> RandomForestRegressor:
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()

    with open('models/best_params.pkl', 'rb') as f:
        best_params = pickle.load(f)

    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train)
    return model


def save_model(model: RandomForestRegressor, output_dir: str = 'models/best_model.joblib') -> None:
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    joblib.dump(model, output_dir)
    logger.info("modèle entrainé et enregistré dans %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Entraînement du modèle RFRegressor")
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed_data',
        help="Chemin vers le dossier contenant les données d'entrainement"
    )
    args = parser.parse_args()

    model = train_model(data_dir=args.data_dir)
    save_model(model)


if __name__ == '__main__':
    main()
