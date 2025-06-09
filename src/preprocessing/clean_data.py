import os
import re
import sys
import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from kafka import KafkaConsumer
import src.preprocessing.metrics as metrics
from src.preprocessing.data_augmentation import apply_data_augmentation


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


def process_and_store_data(batch_messages):
    """Processes and stores a single message received from Kafka."""
    from src.database.store_data import store_data
    all_processed_data = []
    error_count = 0  # Initialize error count
    
    for message in batch_messages:
        try:
            data = message.value if hasattr(message, "value") else message # Check if message is already in dict format
            post_id = data.get("id")
            text = data.get("text", "")
            label = data.get("label", "Neutral")  # Default label to "Neutral" if not provided

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

            processed_data = []
            for aug_text, aug_type in augmented_data: # Unpack the tuple into text and augmentation type
                processed_data.append({
                    "text": aug_text,
                    "label": label,
                    "is_augmented": aug_type is not None,
                    "augmentation_type": aug_type
                })
            all_processed_data.extend(processed_data)  # Add to the list of all processed data

        except Exception as e:
            print(f"❌ Error processing message: {e}")
            error_count += 1 # Increment error count
    
    if all_processed_data: # Check if there are any processed data to store
        df_cleaned = pd.DataFrame(all_processed_data) # Create DataFrame from all processed data
        store_data(df_cleaned, table_name="cleaned_data") # Store all processed data in DB
    
    count_errors = 0
    if error_count > 0: # Check if there were any errors during processing
        count_errors += error_count # Increment error count
        if count_errors >= 1000: # If error count exceeds 1000, log a warning
            print(f"❌ Warning: {count_errors} messages failed to process in this batch due to processing errors") # Log the warning
        
    elif not all_processed_data:
        pass

 
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
    total_invalid_texts = 0
    total_stored_texts = 0
    try:
        for message in consumer:
            batch.append(message) # Collect messages in a batch
            if len(batch) >= batch_size: # Check if batch size is reached
                invalid_texts = process_and_store_data(batch) # Process the batch of messages
                total_invalid_texts += invalid_texts
                total_stored_texts += batch_size - invalid_texts
                batch.clear() # Clear the batch after processing
    except KeyboardInterrupt:
        print("🛑 Stopped consuming manually.")
    finally:
        if batch:
            invalid_texts = process_and_store_data(batch)
            total_invalid_texts += invalid_texts
            total_stored_texts += len(batch) - invalid_texts
        print(f"📉 Total discarded invalid texts: {total_invalid_texts}")
        if total_stored_texts == 0:
            print("⚠️ No valid data was stored during this session.")
        else:
            print(f"✅ Total valid entries stored: {total_stored_texts}")


if __name__ == "__main__":
    consume_messages()  # Start consuming messages


