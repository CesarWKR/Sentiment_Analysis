import logging
from src.api.fetch_reddit import fetch_reddit_posts
from src.database.db_connection import connect_to_db
from src.database.store_data import store_data
from src.preprocessing.clean_data import clean_text
from src.training.fine_tune_bert import train_model
from src.evaluation.test_model import predict_sentiment as test_model

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    """Main function to execute the Reddit sentiment analysis pipeline."""
    logging.info("🚀 Starting Reddit Sentiment Analysis Pipeline...")

    # Step 1: Fetch Reddit data
    logging.info("📥 Fetching Reddit data...")
    subreddit_name = "depression"  # Change this to any subreddit of interest
    raw_data = fetch_reddit_posts(subreddit_name, limit=2000)
    logging.info(f"✅ Retrieved {len(raw_data)} posts.")


    # Step 2: Store data in the database
    logging.info("🗄️ Storing data in the database...")
    try: # Attempt to connect to the database and store data
        conn = connect_to_db()
        store_data(raw_data, conn)
        conn.close()
        logging.info("✅ Data stored successfully.")
    except Exception as e:
        logging.error(f"❌ Error storing data: {e}")
        return # Exit if data storage fails


    # Step 3: Clean data
    if raw_data: # Check if cleaned data is available
        logging.info("🧼 Cleaning text data...")
        raw_data["clean_text"] = raw_data["text"].apply(clean_text)
        logging.info("✅ Data cleaned.")
    else:
        logging.warning("⚠️ No data available to clean. Skipping this step.") # Handle empty data
    

    # Step 4: Fine-tune BERT model
    logging.info("🧠 Fine-tuning BERT model...")
    train_model()
    logging.info("✅ BERT model trained.")

    # Step 5: Test model with sample texts
    logging.info("📝 Testing model...")
    test_samples = [
        "I love this new update! The features are amazing.",
        "This is the worst experience I've had. So disappointing!",
        "I'm not sure how I feel about this. It's okay, I guess."
    ]

    LABELS = ["Negative", "Neutral", "Positive"]

    print("\n📊 Model Predictions:")
    print("=" * 50)
    for text in test_samples:
        predicted_index = test_model(text)
        #sentiment = test_model(text)
        sentiment = LABELS[predicted_index]  # Get the sentiment label from the index
        print(f"📝 Text: \"{text}\"")
        print(f"🔍 Prediction: {sentiment.upper()}") # Show the label instead the index
        print("-" * 50) 

    logging.info("🎉 Pipeline execution complete!")

if __name__ == "__main__":
    main()