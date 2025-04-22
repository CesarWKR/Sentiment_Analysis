import os
import re
import sys
import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from kafka import KafkaConsumer
from sqlalchemy import create_engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import importlib
import src.database.store_data
import src.database.db_connection

from src.preprocessing.data_augmentation import apply_data_augmentation
from src.database.db_connection import connect_to_db
from src.database.store_data import store_data

# Reload modules to ensure the latest changes are applied
importlib.reload(src.database.store_data)
importlib.reload(src.database.db_connection)

# Download NLTK data files
nltk.download("stopwords")
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
    words = text.split() # Tokenize text
    words = [word.lower() for word in words if word.lower() not in stop_words] # Remove stopwords
    return " ".join(words) # Join words back into a sentence

def process_and_store_data(batch_messages):
    """Processes and stores a single message received from Kafka."""
    all_processed_data = []
    error_count = 0  # Initialize error count
    
    for message in batch_messages:
        try:
            # data = json.loads(message.value.decode("utf-8"))  # Decode JSON message
            # data = message.value  # No need decode if already in dict format
            data = message.value if hasattr(message, "value") else message # Check if message is already in dict format

            post_id = data.get("id")
            text = data.get("text", "")
            label = data.get("label", "Neutral")  # Default label to "Neutral" if not provided

            if not text:
                # print(f"⚠️ Skipping empty text for post ID {post_id}")
                # return None
                continue  # Skip empty text
        
            # Apply text cleaning
            cleaned_text = clean_text(text)
            
            # Apply data augmentation
            augmented_texts = apply_data_augmentation(cleaned_text)

            # Create a DataFrame with the cleaned and augmented texts
            processed_data = [{"text": txt, "label": label} for txt in augmented_texts] # Create a list of dictionaries for each augmented text and provide them the corresponding label

            all_processed_data.extend(processed_data)  # Add to the list of all processed data

            # Convert to DataFrame and store in DB
            # df_cleaned = pd.DataFrame(processed_data) # Create DataFrame from processed data
            
        except Exception as e:
            # print(f"❌ Error processing message: {e}")
            error_count += 1 # Increment error count


    if all_processed_data: # Check if there are any processed data to store
        df_cleaned = pd.DataFrame(all_processed_data) # Create DataFrame from all processed data
        store_data(df_cleaned, table_name="cleaned_data") # Store all processed data in DB
        # print(f"✅ Successfully processed and stored {len(all_processed_data)} entries in batch")  # Log the number of entries processed
    
    
    if error_count > 0: # Check if there were any errors during processing
        print(f"❌ Skipped {error_count} messages due to processing errors") # Log the number of errors
    elif not all_processed_data:
        print("⚠️ No valid data to store in this batch")
        
    

def consume_messages(batch_size=1000):
    """Consumes messages from Kafka topic and processes them."""
    consumer = KafkaConsumer(
        KAFKA_TOPIC,   # Fetch messages from raw data topic
        bootstrap_servers=KAFKA_BROKER,  # Kafka broker address
        group_id=KAFKA_GROUP_ID,  # Consumer group to prevent duplicate processing
        auto_offset_reset="earliest",   # Start from the earliest message
        enable_auto_commit=True,    # Automatically commit offsets
        value_deserializer=lambda x: json.loads(x.decode("utf-8"))  # Decode JSON messages deserializer  
    )

    print(f"🔄 Listening to Kafka topic: {KAFKA_TOPIC}")

    batch = []
    for message in consumer:
        batch.append(message) # Collect messages in a batch
        if len(batch) >= batch_size: # Check if batch size is reached
            process_and_store_data(batch) # Process the batch of messages
            # process_and_store_data(message)
            batch.clear() # Clear the batch after processing

if __name__ == "__main__":
    consume_messages()  # Start consuming messages


