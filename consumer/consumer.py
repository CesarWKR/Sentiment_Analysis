from kafka import KafkaConsumer, KafkaProducer
import json
import os
import logging
from src.database.db_connection import connect_to_db
from src.database.store_data import store_data
from src.preprocessing.clean_data import clean_text
from src.preprocessing.clean_data import process_and_store_data  # Import the function to clean and store data
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# # Kafka settings from docker-compose
# KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
# KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "reddit_posts")
# GROUP_ID = "consumer_group"


logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)


KAFKA_BROKER = "localhost:9092"  # Default localhost
# KAFKA_BROKER = "kafka:9092"  # Docker container name and port
KAFKA_TOPIC = "reddit_posts"
TOPIC_OUTPUT = "cleaned_data"
GROUP_ID = "reddit_consumer_group"

consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    group_id=GROUP_ID,
    auto_offset_reset="earliest", # Start reading at the earliest message
    enable_auto_commit=True, # Commit offsets automatically
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
)


producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

def consume_messages():
    logging.info(f"🔄 Listening to Kafka topic: {KAFKA_TOPIC}")
    total_consumed = 0
    for message in consumer:
        try:
            raw_data = message.value # Get the raw data from the message
            store_data(raw_data, table_name="reddit_posts")  # Store raw data in the database
            cleaned_data = process_and_store_data([message])  # Call cleaning function with the list message value


            if cleaned_data:  # Check if cleaned_data is not empty
                producer.send(TOPIC_OUTPUT, cleaned_data)  # Send cleaned data to output topic
        except Exception as e:
            logging.error(f"❌ Error processing message: {e}")
        
        total_consumed += 1
        if total_consumed % 10000 == 0:  # Log every 10000 messages
            logging.info(f"🔄 Total consumed mesages: {total_consumed}")    

if __name__ == "__main__":
    consume_messages()
    print("All messages consumed and processed.")
