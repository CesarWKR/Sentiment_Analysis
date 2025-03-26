import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from db_connection import get_db_url # Uncomment this line if you add it to extraPaths in a json file
from src.database.db_connection import get_db_url
from sqlalchemy import create_engine

# Download stopwords if not already available
nltk.download("stopwords")
nltk.download("punkt")

def clean_text(text: str) -> str:
    """
    Cleans a given text by removing special characters, URLs, and stopwords.

    :param text: Raw text from Reddit post
    :return: Cleaned text
    """
    if not isinstance(text, str) or text.strip() == "": # Check if text is empty
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word not in stop_words]

    return " ".join(filtered_words)

def fetch_and_clean_data():
    """
    Fetches raw data from the database, cleans the text, and returns a processed DataFrame.
    
    :return: Cleaned DataFrame
    """
    db_url = get_db_url()
    engine = create_engine(db_url)

    query = "SELECT * FROM reddit_posts"
    df = pd.read_sql(query, engine)

    print(f"🔍 Retrieved {len(df)} posts from the database.")

    # Remove duplicates and null values
    df.drop_duplicates(subset=["id"], keep="first", inplace=True)
    df.dropna(subset=["text"], inplace=True)

    # Clean text column
    df["clean_text"] = df["text"].apply(clean_text)

    print(f"✅ Cleaned {len(df)} posts.")

    return df

if __name__ == "__main__":
    cleaned_data = fetch_and_clean_data()
    print(cleaned_data.head())  # Display first 5 cleaned posts
