import os
import numpy as np
import pandas as pd
import pickle
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# =========================
# Logging Setup
# =========================
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    log_file_path = os.path.join(log_dir, 'model_evaluation.log')
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
# Load Model
# =========================
def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)

        logger.info(f"Model loaded from {file_path}")
        return model

    except FileNotFoundError:
        logger.error(f"Model file not found: {file_path}")
        raise

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


# =========================
# Load Data
# =========================
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded from {file_path}, shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


# =========================
# Evaluate Model
# =========================
def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba)
        }

        logger.info("Model evaluation completed")
        return metrics

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


# =========================
# Save Metrics
# =========================
def save_metrics(metrics: dict, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Metrics saved at {file_path}")

    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise


# =========================
# Main Pipeline
# =========================
def main():
    try:
        logger.info("Evaluation pipeline started")

        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)

        save_metrics(metrics, 'reports/metrics.json')

        logger.info("Evaluation pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Error: {e}")


# =========================
# Entry Point
# =========================
if __name__ == '__main__':
    main()