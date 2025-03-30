import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os
import sys
from sqlalchemy import create_engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import importlib
import src.database.store_data
import src.database.db_connection

from data_augmentation import apply_data_augmentation
from src.database.db_connection import get_db_url
from src.database.store_data import store_data

# Reload modules to ensure the latest changes are applied
importlib.reload(src.database.store_data)
importlib.reload(src.database.db_connection)

# Download NLTK data files
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """Clean text by converting to lowercase, removing special characters, and stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    words = text.split() # Tokenize text
    words = [word for word in words if word not in stop_words] # Remove stopwords
    return " ".join(words) # Join words back into a sentence

def load_data_from_db():
    """Carga los datos de reddit_posts desde la base de datos."""
    db_url = get_db_url()
    engine = create_engine(db_url)
    query = "SELECT id, text FROM reddit_posts"  # Query to fetch data from reddit_posts table
    
    try:
        df = pd.read_sql(query, con=engine)
        if df.empty:
            print("⚠️ There are not data in the table 'reddit_posts'.")
            return None
        df["label"] = "neutral"  # Add a default label for all posts
        return df
    except Exception as e:
        print(f"❌ Error loading data from database: {e}")
        return None

def clean_data(df):
    """Cleans and augments dataset."""
    augmented_rows = []
    
    for _, row in df.iterrows(): # Iterate over DataFrame rows
        original_text = row["text"]
        cleaned_text = clean_text(original_text)

        augmented_texts = apply_data_augmentation(cleaned_text)

        for text in augmented_texts:
            augmented_rows.append({"text": text, "label": row["label"]}) # Add augmented row

    return pd.DataFrame(augmented_rows)

if __name__ == "__main__":
    # df = pd.read_csv("new_posts.csv")    # Read data from CSV file
    df = load_data_from_db()  # Load data from database

    if df is not None:
        clean_df = clean_data(df)
        store_data(clean_df, table_name="cleaned_data")  # Store cleaned data in the database
        print("✅ Cleaned data stored in the database successfully!")
    else:
        print("❌ It could not clean and storage data.")


