from kafka import KafkaProducer
import json
import time
from src.api.fetch_reddit import fetch_reddit_posts  

# KAFKA_BROKER = "kafka:9092"  # Docker container name and port
KAFKA_BROKER = "localhost:9092"  # Default localhost
KAFKA_TOPIC = "reddit_posts"

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

def produce_messages(max_messages=2000):
    """ Change max_messages depending on the number of posts you want to send, must be equal to TOTAL_LIMIT in fetch_reddit.py """
    sent = 0
    while sent < max_messages:
        reddit_data = fetch_reddit_posts() # Fetch posts from Reddit API
        print(f"🔄 Fetched {len(reddit_data)} posts from Reddit")

        for post in reddit_data:  # Iterate over each post
            if sent >= max_messages: # Check if the limit is reached
                break
            producer.send(KAFKA_TOPIC, post)  # Send post to Kafka topic
            print(f"📤 Sent {sent + 1}: {post.get('id', 'No ID')}")   
            sent += 1
            
        time.sleep(1)  # Sleep for 1 second to avoid overwhelming the producer

    print(f"✅ Total messages sent: {sent}")

if __name__ == "__main__":
    produce_messages()