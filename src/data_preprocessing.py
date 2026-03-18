import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

nltk.download('stopwords')
nltk.download('punkt')

# Ensure the 'logs' directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger =logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
     """

     Transform the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming
     """

     ps = PorterStemmer()
     # Convert to lowercase
     text = text.lower()
     # tokenize the text
     text = nltk.word_tokenize(text)
     # Remove non-alphanumeric tokens
     text = [word for word in text if word.isalnum()]
     # remove stopwords and punctuations
     text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
     # stem the word
     text = [ps.stem(word) for word in text]
     # Join the token back into the single string
     return " ".join(text)

def preprocess_df(df,text_column='text', target_column='target'):
    """
    Preprocess the DataFrame by encoding the largest column, removing duplicates, and transforming the text column
    """
    try:
        logger.debug('starting preprocessing for DataFrame')
        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encode')

        # Remove duplicate rows
        df = df.drop_duplicates(keep='last')
        logger.debug('Duplicate columns dropped')

        # Apply text transformation ti the specified text column
        df.loc[:,text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    except KeyError as e:
         logger.error('Column not found')
         raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)

def main(text_column='text',target_column='target'):
    """
    Main function to find the raw data , preprocess it and save the processed data
    """
    try:
        # Fetch data from dataframe
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('data loaded properly')

        # Transforming the data inside data/processed
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Store the data inside data/processed

        data_path = os.path.join('./data','interim')
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path,'train_processed.csv'))
        test_processed_data.to_csv(os.path.join(data_path,'test_processed.csv'))

        logger.debug('processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found')
    except pd.errors.EmptyDataError as e:
        logger.error('Empty Data: %s', e)
    except Exception as e:
        logger.error('Error during data preprocessing: %s', e)
        print(f'Error: {e}')

if __name__ == '__main__':
    main()