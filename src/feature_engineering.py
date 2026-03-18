import pandas as pd
import os
import logging
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# Logging Setup
# =========================
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

# Avoid duplicate handlers
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
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
# Load Data
# =========================
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from csv file"""
    try:
        df = pd.read_csv(file_path)
        df.fillna("", inplace=True)

        logger.info(f"Data loaded successfully: {file_path}")
        return df

    except pd.errors.EmptyDataError as e:
        logger.error(f"CSV file is empty: {e}")
        raise

    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise


# =========================
# Apply TF-IDF
# =========================
def apply_tfidf(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    max_features: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply TF-IDF vectorization"""

    try:
        required_cols = ['text', 'target']

        # Validate columns
        for col in required_cols:
            if col not in train_data.columns:
                raise ValueError(f"Missing column in train data: {col}")
            if col not in test_data.columns:
                raise ValueError(f"Missing column in test data: {col}")

        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data['text'].values
        y_train = train_data['target'].values

        X_test = test_data['text'].values
        y_test = test_data['target'].values

        # Fit and transform
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # Convert to DataFrame
        train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=feature_names)
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray(), columns=feature_names)
        test_df['label'] = y_test

        logger.info("TF-IDF applied successfully")

        return train_df, test_df

    except Exception as e:
        logger.error(f"Error during TF-IDF transformation: {e}")
        raise


# =========================
# Save Data
# =========================
def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save dataframe to CSV"""

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)

        logger.info(f"Data saved to {file_path}")

    except Exception as e:
        logger.error(f"Error while saving data: {e}")
        raise


# =========================
# Main Pipeline
# =========================
def main():
    try:
        logger.info("Pipeline started")

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = apply_tfidf(
            train_data,
            test_data,
            max_features=50
        )

        save_data(train_df, './data/processed/train_tfidf.csv')
        save_data(test_df, './data/processed/test_tfidf.csv')

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Error: {e}")


# =========================
# Entry Point
# =========================
if __name__ == '__main__':
    main()