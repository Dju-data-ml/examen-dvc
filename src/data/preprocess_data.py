import argparse
import logging
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("preprocessing")

def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled


def save_scaled_data(X_train_scaled: pd.DataFrame, X_test_scaled: pd.DataFrame, data_dir: str):
    X_train_scaled.to_csv(os.path.join(data_dir, "X_train_scaled.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(data_dir, "X_test_scaled.csv"), index=False)
    logger.info("jeu de données normalisé et enregistré dans dans %s", data_dir)

def main():
    parser = argparse.ArgumentParser(description="Normalise les données X_train et X_test.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed_data",
        help="Dossier contenant X_train.csv et X_test.csv",
    )
    args = parser.parse_args()

    X_train = pd.read_csv(os.path.join(args.data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(args.data_dir, "X_test.csv"))

    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    save_scaled_data(X_train_scaled, X_test_scaled, args.data_dir)


if __name__ == "__main__":
    main()
