import logging
import threading
import time
import pandas as pd
import os
import glob
from kafka import KafkaProducer
import json
import shutil
from src.api.fetch_reddit import fetch_reddit_posts, subreddits, posts_per_subreddit, global_sent, TOTAL_LIMIT
from src.utils.model_utils import update_latest_model, get_latest_model_path
from src.training.fine_tune_bert import train_model
from consumer.consumer import consume_messages
from src.database.db_connection import connect_to_db
from src.database.store_data import store_data, store_results_in_db, save_validation_data
from dotenv import load_dotenv
from tqdm import tqdm
from sqlalchemy import text as sql_text

# Load environment variables
load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

LABELS = ["Negative", "Neutral", "Positive"]



""" Function to get the latest model path is now defined in model_utils.py
# This function is used to find the most recent BERT model directory."""
# def get_latest_model_path():
#     """Find the most recent BERT model directory."""
#     model_dirs = sorted(glob.glob("bert_sentiment_model_*"), reverse=True)
#     return model_dirs[0] if model_dirs else None


def start_kafka_consumer():
    """Start the Kafka consumer to process and store messages."""
    logging.info("🔄 Starting Kafka consumer for data cleaning & storage...")
    consume_messages()


def cleaned_data_exists():
    """Check if cleaned data exists in the database."""
    try:
        engine = connect_to_db()
        df = pd.read_sql("SELECT COUNT(*) as count FROM cleaned_data", engine) 
        return df["count"].iloc[0] > 0 # Check if count is greater than 0
    except Exception as e:
        logging.warning(f"⚠️ Could not check for cleaned data: {e}")
        return False


def main():
    """Main function to execute the Reddit sentiment analysis pipeline."""
    logging.info("🚀 Starting Reddit Sentiment Analysis Pipeline...")

    # Step 1: Start Kafka consumer in a separate thread
    consumer_thread = threading.Thread(target=start_kafka_consumer, daemon=True)  # Daemon thread will exit when the main program exits
    consumer_thread.start()


    # Step 2: Fetch Reddit posts and send to Kafka directly
    logging.info("📥 Fetching Reddit posts and streaming to Kafka...")
    progress_bar = tqdm(total=TOTAL_LIMIT, desc="📬 Sending posts to Kafka", ncols=100)


    for subreddit in subreddits:
        # before = global_sent
        df, num_sent = fetch_reddit_posts(subreddit, desired_count=posts_per_subreddit) 
        # after = global_sent
        progress_bar.update(num_sent)  # Update the progress bar with the number of posts sent

        # if df is not None and not df.empty: 
        #         logging.info(f"🗃️ Storing raw data from subreddit '{subreddit}' into the database...")
        #         store_data(df, table_name="reddit_posts")  # Store the raw data in the database"
        # else:
        #         logging.warning(f"⚠️ No data fetched from subreddit '{subreddit}'. Skipping DB storage.")

    
    progress_bar.close()
    
    logging.info(f"✅ All posts fetched and sent to Kafka.")

    # Wait a bit for consumer to finish processing
    consumer_thread.join(timeout=30)  # Wait for the consumer thread to finish processing



    # Step 4: Summary of data stored in database
    try:
        engine = connect_to_db()
        with engine.connect() as conn:
            reddit_count = conn.execute(sql_text("SELECT COUNT(*) FROM reddit_posts")).scalar()
            cleaned_count = conn.execute(sql_text("SELECT COUNT(*) FROM cleaned_data")).scalar()

        logging.info("📦 Summary of data stored in the database:")
        logging.info(f"🔸 Posts in 'reddit_posts': {reddit_count}")
        logging.info(f"🔹 Posts in 'cleaned_data': {cleaned_count}")
    except Exception as e:
        logging.error(f"❌ Error while fetching summary from database: {e}")



    # Step 5: Fine-tune BERT model
    latest_model_path = get_latest_model_path()

    if latest_model_path:
        logging.info(f"✅ Found existing model: {latest_model_path}. Skipping training.")
    elif cleaned_data_exists(): # Check if cleaned data exists in the database
        logging.info("🧠 No existing model found. Fine-tuning RoBERTa model...")
        train_model(data_source="both") # Choose the data source for training (raw, cleaned, or both)
        update_latest_model()
        logging.info("✅ RoBERTa model trained and latest model updated.")
    else:
        logging.warning("⚠️ No data available for training. Skipping model training.")



    # Step 6: Test model with sample texts
    from src.evaluation.test_model import ( # Import the test_model functions from the evaluation module
    predict_sentiment as test_model,
    run_validation_and_store_results,
    evaluate_model,
    manual_prediction,
    )
   

    logging.info("📝 Testing model...")
    test_samples = [
        "I love this new update! The features are amazing.",
        "This is the worst experience I've had. So disappointing!",
        "I'm not sure how I feel about this. It's okay, I guess."
    ]

    print("\n📊 Model Predictions:")
    print("=" * 50)
    # results_to_store = []

    for text in test_samples:
        predicted_index = test_model(text) # Call the test_model function to get the predicted index
        #sentiment = test_model(text)
        sentiment = LABELS[predicted_index]  # Get the sentiment label from the index
        # results_to_store.append((text, sentiment))

        print(f"📝 Text: \"{text}\"")
        print(f"🔍 Prediction: {sentiment.upper()}") # Show the label instead the index
        print("-" * 50) 
    

    engine = connect_to_db()
    print("📡 Connected to DB:", engine.url)


    # Step 6.1: Fetch validation_data from the database and make validation predictions and store results in validation_results table 
    logging.info("🔍 Running full validation on validation_data set...")
    run_validation_and_store_results() # Call the function to run validation and store results in the database


    # Step 6.2: Fetch validation results from the database and plot confusion matrix 
    logging.info("📈 Evaluating model with validation_results...")
    evaluate_model() # Call the function to evaluate the model and plot confusion matrix


    # Step 6.3: Manual prediction for user input
    logging.info("🤖 Manual Sentiment Prediction Mode (type 'exit' to quit)")
    manual_prediction()  # Call the function to allow manual predictions



    logging.info("🎉 Pipeline execution complete!")

if __name__ == "__main__":
    main()