import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

# =========================
# Logging Setup
# =========================
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    log_file_path = os.path.join(log_dir, 'model_building.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# =========================
# Load the params
# =========================

def load_params(params_path: str) -> dict:
    """load the params from yaml file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters received from %s', params_path)
        return params
    except FileNotFoundError as e:
        logger.error('File Not Found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML Error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured while loading the params: %s', e)
        raise


# =========================
# Load Data
# =========================
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.info(f'Data loaded from {file_path}, shape: {df.shape}')
        return df

    except pd.errors.ParserError as e:
        logger.error(f'CSV parsing error: {e}')
        raise

    except FileNotFoundError:
        logger.error(f'File not found: {file_path}')
        raise

    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        raise


# =========================
# Train Model
# =========================
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Mismatch in X_train and y_train length")

        # ✅ Direct parameters (no YAML)
        # n_estimators = 100
        # random_state = 42
        params = load_params(params_path='params.yaml')
        n_estimators = params['model_building']['n_estimators']
        random_state = params['model_building']['random_state']

        logger.info(f"Training RandomForest with n_estimators={n_estimators}")

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

        clf.fit(X_train, y_train)

        logger.info("Model training completed")
        return clf

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


# =========================
# Save Model
# =========================
def save_model(model, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(model, file)

        logger.info(f"Model saved at {file_path}")

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


# =========================
# Main Pipeline
# =========================
def main():
    try:
        logger.info("Model building pipeline started")

        train_data = load_data('./data/processed/train_tfidf.csv')

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train)

        save_model(clf, 'models/model.pkl')

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Error: {e}")


# =========================
# Entry Point
# =========================
if __name__ == '__main__':
    main()