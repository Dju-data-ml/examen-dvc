import argparse
import logging
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("préparation donnée")


def split_data(raw_path: str):
    df = pd.read_csv(raw_path)
    X = df.drop(columns=["silica_concentrate","date"])
    y = df["silica_concentrate"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def save_data(X_train, X_test, y_train, y_test, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    logger.info("jeu de données de train et test enregistré dans %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Prepare les données en séparant les variables X,y et les jeux train et tests. Sauvegarde les csv")
    parser.add_argument(
        "--source",
        type=str,
        default="data/raw_data",
        help="Dossier contenant raw.csv",
    )
    parser.add_argument(
        "--destination",
        type=str,
        default="data/processed_data",
        help="Dossier de destination de X_train.csv et X_test.csv",
    )
    args = parser.parse_args()

    raw_file = os.path.join(args.source, "raw.csv")

    X_train, X_test, y_train, y_test = split_data(raw_file)
    save_data(X_train, X_test, y_train, y_test, args.destination)


if __name__ == "__main__":
    main()
