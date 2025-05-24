import os
import json
from kafka import KafkaConsumer
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
#from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declarative_base, sessionmaker
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# from src.api.fetch_reddit import fetch_reddit_posts
from src.database.db_connection import connect_to_db
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

# Load environment variables
load_dotenv()

# Kafka settings
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")  # Default localhost
KAFKA_TOPIC_RAW = os.getenv("KAFKA_TOPIC", "reddit_posts") # Topic for raw Reddit posts
KAFKA_TOPIC_CLEANED = os.getenv("KAFKA_TOPIC_CLEANED", "cleaned_data") # Topic for cleaned data
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "reddit_consumer_group")  # Select a group ID for the consumer

# Define the database model
Base = declarative_base() 

class RedditPost(Base): 
    __tablename__ = "reddit_posts"

    id = Column(String, primary_key=True)  # Reddit post ID
    # title = Column(String(255), nullable=False)
    title = Column(Text, nullable=False)  # Change to Text for longer titles
    score = Column(Integer, nullable=False)
    url = Column(Text, nullable=True)
    num_comments = Column(Integer, nullable=False)
    created_utc = Column(DateTime, nullable=False)
    text = Column(Text, nullable=True)
    label = Column(String(50), nullable=False)  # Label for sentiment analysis or classification

class CleanedData(Base):
    __tablename__ = "cleaned_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    label = Column(String(50), nullable=False)

SUBREDDIT_LABELS = {
    # Negative sentiment
    "depression": "Negative",
    "anxiety": "Negative",
    "mentalhealth": "Negative",
    "bipolarreddit": "Negative",
    "SuicideWatch": "Negative",
    "ptsd": "Negative",
    "selfharm": "Negative",
    "breakups": "Negative",
    "relationships": "Negative",
    "dating_advice": "Negative",
    "sad": "Negative",
    "lonely": "Negative",

    # Positive sentiment
    "GetMotivated": "Positive",
    "DecidingToBeBetter": "Positive",
    "selfimprovement": "Positive",
    "Happy": "Positive",
    "humansbeingbros": "Positive",
    "MadeMeSmile": "Positive",
    "UpliftingNews": "Positive",
    "KindVoice": "Positive",
    "WholesomeMemes": "Positive",
    "funny": "Positive",
    "memes": "Positive",
    "aww": "Positive",
    "Eyebleach": "Positive",
    "UnexpectedlyWholesome": "Positive",

    # Neutral sentiment
    "Psychology": "Neutral",
    "Emotions": "Neutral",
    "CasualConversation": "Neutral",
    "offmychest": "Neutral",
    "TrueOffMyChest": "Neutral",
    "confession": "Neutral",
    "Vent": "Neutral",
    "Advice": "Neutral",
    "LifeProTips": "Neutral",
    "NoStupidQuestions": "Neutral",
    "AskReddit": "Neutral",
    "AskWomen": "Neutral",
    "AskMen": "Neutral",
    "ChangeMyView": "Neutral",
    "TrueAskReddit": "Neutral",
    "TooAfraidToAsk": "Neutral",
    "unpopularopinion": "Neutral",
    "TodayILearned": "Neutral",
    "philosophy": "Neutral",
    "science": "Neutral"
}

vader_analyzer = SentimentIntensityAnalyzer()

def get_combined_label(text, subreddit=None):
    """
    Returns a combined label based on the subreddit (if provided)
    and automatic sentiment analysis with VADER.

    Priority: uses the subreddit if it is in the predefined list,
    if not, uses VADER to infer the label.
    """
    # Assign label based on subreddit name if provided
    if subreddit:
        subreddit_label = SUBREDDIT_LABELS.get(subreddit.lower()) # Get the label from the subreddit name
        if subreddit_label:
            return subreddit_label

    # If subreddit is not in the list, use VADER for sentiment analysis
    vader_score = vader_analyzer.polarity_scores(text or "")
    compound = vader_score["compound"] # Get the compound score from VADER

    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def store_reddit_posts(data: pd.DataFrame):
    """
    Stores Reddit post data in the 'reddit_posts' table.

    :param data: DataFrame or dict containing Reddit post data.
    """
    engine = connect_to_db()  # Connect to the database
    Base.metadata.create_all(engine)    # Create table if it doesn't exist
    Session = sessionmaker(bind=engine)  # Create a new session
    session = Session()  # Create a session to interact with the database

    try:
        if isinstance(data, pd.DataFrame): # Check if data is a DataFrame
            for _, row in data.iterrows(): # Iterate over each row in the DataFrame
                if not row.get("id") or not row.get("text"):  # Check if 'id' and 'text' are present
                    print("⚠️ Invalid data in row: missing 'id' or 'text'. Skipping...")
                    continue

                post = RedditPost(
                    id=row["id"],
                    title=row["title"],   
                    score=row["score"],
                    url=row["url"],
                    num_comments=row["num_comments"],
                    created_utc=pd.to_datetime(row["created_utc"], unit="s"),
                    text=row["text"],
                    label=get_combined_label(row.get("text", ""), row.get("subreddit")) # Use the get_combined_label function to determine the label
                )
                session.merge(post)  # Merge the post into the session to avoid duplicates

        elif isinstance(data, dict):   # Check if data is a dictionary
            if not data.get("id") or not data.get("text"):  
                print("⚠️ Invalid data: missing 'id' or 'text'. Skipping...")
                return False

            post = RedditPost(
                id=data["id"],
                title=data["title"],    
                score=data["score"],
                url=data["url"],
                num_comments=data["num_comments"],
                created_utc=pd.to_datetime(data["created_utc"], unit="s"),
                text=data["text"],
                label=get_combined_label(data.get("text", ""), data.get("subreddit"))
            )
            session.merge(post)

        session.commit()   # Commit the session to save changes
        return True

    except Exception as e:
        session.rollback()
        print(f"❌ Error storing data in reddit_posts: {e}")
        return False


    finally: # Ensure the session is closed after processing
        session.close()


def store_cleaned_data(data: pd.DataFrame):
    """
    Stores cleaned data in the 'cleaned_data' table.

    :param data: DataFrame or dict containing cleaned text and labels.
    """
    engine = connect_to_db()
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        if isinstance(data, pd.DataFrame):
            for _, row in data.iterrows():
                if not row.get("text") or not row.get("label"): # Check if 'text' and 'label' are present
                    print("⚠️ Invalid cleaned data: missing 'text' or 'label'. Skipping...")
                    continue

                cleaned_entry = CleanedData(
                    text=row["text"],
                    label=row["label"]
                )
                session.add(cleaned_entry) # Add the cleaned entry to the session

        elif isinstance(data, dict): # Check if data is a dictionary
            if not data.get("text") or not data.get("label"):
                # print("⚠️ Invalid cleaned data: missing 'text' or 'label'. Skipping...")
                return False

            cleaned_entry = CleanedData(
                text=data["text"],
                label=data["label"]
            )
            session.add(cleaned_entry) 

        else:
            print("⚠️ Data format not supported for cleaned_data.")
            return False

        session.commit() # Commit the session to save changes
        return True

    except Exception as e: 
        session.rollback()
        print(f"❌ Error storing data in cleaned_data: {e}")
        return False

    finally:
        session.close()


def store_data(data: pd.DataFrame, table_name="cleaned_data"):
    """
    Wrapper function that routes data to the correct storage function
    based on the table_name.

    :param data: DataFrame or dict to store.
    :param table_name: Name of the target table ('reddit_posts' or 'cleaned_data').
    """
    if table_name == "reddit_posts":
        return store_reddit_posts(data)
    elif table_name == "cleaned_data":
        return store_cleaned_data(data)
    else:
        print(f"⚠️ Unknown table name: {table_name}")
        return False


def consume_from_kafka():
    """
    It consumes Kafka messages to store data in the corresponding tables.
    """
    consumer = KafkaConsumer(
        KAFKA_TOPIC_RAW, KAFKA_TOPIC_CLEANED,  
        bootstrap_servers=KAFKA_BROKER,
        group_id=KAFKA_GROUP_ID,  # Consumer group ID  
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),  # Deserialize JSON messages
        auto_offset_reset="earliest",
        enable_auto_commit=True   # Automatically commit offsets
    )
    print("🔄 Waiting for Kafka messages...")

    total_messages = 0
    stored = 0
    failed = 0

    for message in consumer: # Iterate over messages from Kafka
        total_messages += 1 # Increment total message count
        topic = message.topic # Get the topic of the message
        # data = message.value # Get the message value
        data = message.value if hasattr(message, "value") else message # Check if the message is already in dict format

        if topic == KAFKA_TOPIC_RAW: # Check if the topic is reddit_posts
            success = store_data(data, table_name="reddit_posts")
        elif topic == KAFKA_TOPIC_CLEANED: # Check if the topic is cleaned_data
            success = store_data(data, table_name="cleaned_data")
        else:
            success = False # Invalid topic

        if success:        # Check if data was stored successfully
            stored += 1    # Increment stored count
        else:              # If data storage failed
            failed += 1    # Increment failed count

        # Displays summary every 10000 messages to avoid overloading the terminal
        if total_messages % 10000 == 0: # Print summary every 10000 messages
            print(f"📊 Processed: {total_messages}, stored: {stored}, failed: {failed}")

    # Final summary after processing all messages
    print(f"✅ Finished. Total messages: {total_messages}, stored: {stored}, failed: {failed}")



def save_validation_data(val_texts, val_labels, table_name="validation_data", if_exists="replace"):  # Save validation data (20% of the fetched data) to the database to be used in the testing process
    """
    Stores validation data into the database.

    Args:
        val_texts (list): List of validation texts.
        val_labels (list): List of corresponding numeric labels.
        table_name (str): Name of the table to store the data.
        if_exists (str): What to do if the table already exists: 'fail', 'replace', 'append'.

    Returns:
        bool: True if data was saved successfully, False otherwise.
    """

    if not val_texts or not val_labels: # Check if validation texts and labels are provided
        print("⚠️ Validation data is empty. Nothing was stored.")
        return False

    if len(val_texts) != len(val_labels): # Check if the lengths of texts and labels match
        print("❌ Mismatch in length between texts and labels.")
        return False
     
    try:
        # Map numeric labels to text labels
        label_inverse_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        val_labels_str = [label_inverse_mapping[label] for label in val_labels]

        # Create a DataFrame for validation data
        val_df = pd.DataFrame({
            "text": val_texts, 
            "label": val_labels, # Numerical labels
            'label_name': val_labels_str  # Add a new column for label names
            
        }) # Create a DataFrame with text and label columns converted to string

        # db_url = get_db_url()
        engine = connect_to_db()  # Create a database engine

        # Save validation data to the database 
        val_df.to_sql(table_name, engine, if_exists=if_exists, index=False)

        print(f"✅ Storaged {len(val_df)} validation records in the database for the '{table_name}' table.")
        return True

    except SQLAlchemyError as e: # Handle SQLAlchemy errors
        print(f"❌ Error storing validation data: {e}")
        return False



def store_results_in_db(engine, results_to_store): # Store validation results in the database
    """Stores validation results in the database.
        :param engine: SQLAlchemy database engine.
        :param results_to_store: List of (text, predicted_label) tuples.
    """
    with engine.begin() as conn:
        # Ensure the validation_results table exists
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS validation_results (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                predicted_label VARCHAR(50) NOT NULL
            )
        """))

        # Insert predictions into the database
        for text_val, prediction in results_to_store: # Iterate over the results to store
            conn.execute(text("""
                INSERT INTO validation_results (text, predicted_label) 
                VALUES (:text, :pred)
            """), {"text": text_val, "pred": prediction})

    print("✅ Validation results stored in database.")


if __name__ == "__main__":
    consume_from_kafka()
