import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split
import yaml
from typing import Tuple

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize the lemmatizer and stopwords once
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data from a given file path.
    """
    return pd.read_csv(file_path)


def preprocess_text(text: str) -> str:
    """
    Preprocess text by removing special characters, tokenizing, removing stopwords, and lemmatizing.
    """
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stopwords and lemmatize the tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Join tokens back into a single string
    return " ".join(tokens)


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the data by applying text preprocessing and encoding sentiment labels.
    """
    # Handle missing values in 'description'
    df["description"] = df["description"].fillna("")

    # Apply text preprocessing to 'description'
    df["cleaned_description"] = df["description"].apply(preprocess_text)

    # Create sentiment labels based on 'points'
    df["sentiment"] = df["points"].apply(lambda x: 1 if x >= 90 else 0)

    # Keep only relevant columns
    df = df[["cleaned_description", "sentiment"]]

    return df


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_description"],
        df["sentiment"],
        test_size=test_size,
        random_state=random_state,
    )
    train_df = pd.DataFrame({"description": X_train, "sentiment": y_train})
    test_df = pd.DataFrame({"description": X_test, "sentiment": y_test})
    return train_df, test_df


def save_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame, train_path: str, test_path: str
) -> None:
    """
    Save the processed train and test data to CSV files.
    """
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)


if __name__ == "__main__":
    # Load parameters from the YAML file
    params = yaml.safe_load(open("params.yaml"))["preprocessing"]
    raw_data_path = params["raw_data_path"]
    train_data_path = params["train_data_path"]
    test_data_path = params["test_data_path"]

    # Load the raw data
    df = load_data(raw_data_path)

    # Process the data
    processed_df = process_data(df)

    # Split the data into train and test sets
    train_df, test_df = split_data(processed_df)

    # Save the processed data
    save_data(train_df, test_df, train_data_path, test_data_path)

    print("Data preprocessing completed and saved.")
