import os
import re
import sys
import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from src.database.db_connection import connect_to_db
from src.preprocessing.metrics import metrics
from src.preprocessing.data_augmentation import apply_data_augmentation
import logging


# Download NLTK data files
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))  # English stopwords set

# Kafka settings
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")  # Default localhost
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "reddit_posts")  # Topic where raw data is published
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "cleaner_consumer_group")  # Group ID for Kafka consumer


def clean_text(text):
    """Clean text by converting to lowercase, removing special characters, and stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces

    # Remove emojis
    text = re.sub(r'[\U0001F600-\U0001F64F]', '', text, flags=re.UNICODE)  # Removes most common emojis
    text = re.sub(r'[\U0001F300-\U0001F5FF]', '', text, flags=re.UNICODE)  # Removes symbols & pictographs
    text = re.sub(r'[\U0001F680-\U0001F6FF]', '', text, flags=re.UNICODE)  # Removes transport & map symbols
    text = re.sub(r'[\U0001F1E0-\U0001F1FF]', '', text, flags=re.UNICODE)  # Removes flags (iOS)
    text = re.sub(r'[\u2600-\u26FF\u2700-\u27BF]', '', text, flags=re.UNICODE) # Remove Miscellaneous Symbols and Dingbats
    text = re.sub(r'(\u00a9|\u00ae|[\u25A0-\u25FF]|\u263a|\u2605|\u2606|\u2668|\u2665|\u2660|\u2764|\u2744|\u2B50|\u2B55)', '', text, flags=re.UNICODE) # Remove more symbols

    words = text.split() # Tokenize text
    words = [word.lower() for word in words if word.lower() not in stop_words] # Remove stopwords
    return " ".join(words) # Join words back into a sentence


def is_valid_text(text):
    """Check if text is meaningful and not placeholder content."""
    if not isinstance(text, str):
        return False
    text = text.strip().lower()
    invalid_patterns = [
        r"^\[?no text content\]?$",
        r"^\[?removed\]?$",
        r"^\[?deleted\]?$",
        r"^text\b",
        r"text content",
        r"lorem ipsum",
        r"^content$",
    ]
    # If the text is too short or contains invalid patterns, consider it invalid
    if len(text.split()) < 3:
        return False
    for pattern in invalid_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    return True


def clean_generated_text(text, prompt):
    """
    Clean generated text by:
    - Deleting the prompt if it appears at the start
    - Removing links
    - Filtering out empty or invalid lines
    """
    if not isinstance(text, str):
        return ""

    # Delete the prompt if it appears at the start of the text
    text = text.replace(prompt, "").strip()
    text = re.sub(r"^"+re.escape(prompt)+r"\s*", "", text, flags=re.IGNORECASE)

    # Detect and remove URLs
    text = re.sub(r"http\S+|www\.\S+|\S+\.(com|net|org)\S*", "", text)

    # Filter out empty or invalid lines
    if len(text.strip()) == 0 or re.fullmatch(r"[\W_]+", text):
        return ""

    return text.strip()


def process_and_store_data():
    """Reads data from relabeled_data, cleans it, applies data augmentation, and stores in cleaned_data."""
    from src.database.store_data import store_data
    engine = connect_to_db()  # Connect to the database
    df = pd.read_sql("SELECT * FROM relabeled_data", engine)
    all_processed_data = []
    
    for _, row in df.iterrows():  # Iterate through each row in the DataFrame
        text = row["text"]
        label = row["label"]

        if not text:
            metrics.empty_text_count += 1
            continue  # Skip empty text
        
        # Apply text cleaning
        cleaned_text = clean_text(text)
        if not is_valid_text(cleaned_text):
            metrics.invalid_text_count += 1  # Increment invalid text count
            continue  # Skip if the text is not valid

        # Apply data augmentation
        augmented_data = apply_data_augmentation(cleaned_text)

        for aug_text, aug_type in augmented_data: # Unpack the tuple into text and augmentation type
                all_processed_data.append({  # Create a dictionary for each processed data entry
                    "text": aug_text,
                    "label": label,
                    "is_augmented": aug_type is not None,
                    "augmentation_type": aug_type
                })

    if all_processed_data: # Check if there are any processed data to store
        df_cleaned = pd.DataFrame(all_processed_data) # Create DataFrame from all processed data
        store_data(df_cleaned, table_name="cleaned_data") # Store all processed data in DB


def process_kafka_messages(messages):
    """
    Cleans and validates Kafka messages. Returns a list of dictionaries ready to store and send to another topic.
    """
    cleaned_data = []

    for message in messages:
        try:
            msg_value = message.value if hasattr(message, 'value') else message  # Supports KafkaMessage or dict
            
            if not isinstance(msg_value, dict):
                logging.warning(f"⚠️ Unexpected message format: {msg_value}")
                continue
            
            text = msg_value.get("text", "")
            label = msg_value.get("label", None)

            if not text:
                metrics.empty_text_count += 1
                continue

            cleaned_text = clean_text(text)
            if not is_valid_text(cleaned_text):
                metrics.invalid_text_count += 1
                continue

            # We do not apply augmentation here because this is for the real-time flow
            cleaned_data.append({
                "text": cleaned_text,
                "label": label,
                "is_augmented": False,
                "augmentation_type": None
            })

        except Exception as e:
            print(f"❌ Error processing message: {e}| message: {message}")
            continue

    return cleaned_data if cleaned_data else None