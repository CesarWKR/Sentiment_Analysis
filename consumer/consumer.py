from kafka import KafkaConsumer, KafkaProducer
import json
import os
import logging
from src.database.store_data import store_data
from src.preprocessing.clean_data import process_kafka_messages
from src.preprocessing.clean_data import process_and_store_data  # Import the function to clean and store data
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# # Kafka settings from docker-compose
# KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
# KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "reddit_posts")
# GROUP_ID = "reddit_consumer_group"


logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Kafka settings
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "reddit_posts")
TOPIC_OUTPUT = os.getenv("TOPIC_OUTPUT", "cleaned_data")
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "reddit_consumer_group")

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

def consume_messages(batch_size=1000):
    logging.info(f"🔄 Listening to Kafka topic: {KAFKA_TOPIC}")
    batch = []
    total_consumed = 0
    try:
        for message in consumer:
            batch.append(message)
            if len(batch) >= batch_size:
                process_batch(batch)
                # total_consumed += batch_size
                total_consumed += len(batch)
                batch.clear()

                if total_consumed % 10000 == 0:
                    logging.info(f"🔄 Messages consumed so far: {total_consumed}") 

    except KeyboardInterrupt:
        logging.info("🛑 Stopped consuming manually.")
    finally:
        if batch:
            process_batch(batch)
            logging.info("✅ Remaining batch processed.")   


def process_batch(batch):  
    """Process a batch of messages from Kafka."""
    raw_records = []
    for message in batch:
        try:
            raw_data = message.value 
            store_data(raw_data, table_name="reddit_posts")  # Store raw data in the database
            raw_records.append(message)  # Store the raw record for further processing

        except Exception as e:
            logging.error(f"❌ Error storing raw data: {e}")

    try:
        cleaned_data = process_kafka_messages(raw_records)  # Apply cleaning + DA and store in cleaned_data

        # Resend the cleaned result to Kafka
        # if cleaned_data is not None:
        if cleaned_data:  # Check if there is any cleaned data to send
            for record in cleaned_data:
                try: 
                    producer.send(TOPIC_OUTPUT, record)  # Send cleaned data to the output topic
                except Exception as e:
                    logging.error(f"❌ Error sending message to {TOPIC_OUTPUT} in Kafka: {e}")
        logging.info(f"✅ Sent {len(cleaned_data)} cleaned messages to {TOPIC_OUTPUT}")

    except Exception as e:
        logging.error(f"❌ Error in processing batch: {e}")

if __name__ == "__main__":
    consume_messages()
    logging.info("✅ All messages consumed and processed.")
