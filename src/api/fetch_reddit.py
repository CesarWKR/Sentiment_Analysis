import logging
import praw
import os
import pandas as pd
import json
import random
from kafka import KafkaProducer
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Reddit API credentials
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Kafka settings
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")  # Default localhost
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "reddit_posts")

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')  # Convert to JSON
)

# List of subreddits (balanced mix of topics)
subreddits = [
    # Mental health & support
    "depression", "anxiety", "mentalhealth", "bipolarreddit", "SuicideWatch",

    # Positive & motivational
    "GetMotivated", "DecidingToBeBetter", "selfimprovement", "Happy",

    # Psychology & emotions
    "Psychology", "Emotions", "CasualConversation", "offmychest",

    # Stories & uplifting content
    "TrueOffMyChest", "humansbeingbros", "MadeMeSmile", "UpliftingNews",

    # Advice & general discussion
    "Advice", "LifeProTips", "NoStupidQuestions", "AskReddit"
]

# List of categories to fetch posts from
categories = ["hot", "new", "top", "rising"]


# Total limit of posts to fetch
TOTAL_LIMIT = 2000
# posts_per_subreddit = TOTAL_LIMIT // len(subreddits)  # Distribute posts evenly
posts_per_subreddit = min(100, TOTAL_LIMIT // len(subreddits))  # No more than 100 posts per subreddit


# Global counters
global_total = 0
global_sent = 0
global_failed = 0


 
def assign_label(subreddit_name: str, mode: str = "sentiment") -> str:  # Labeling mode: "subreddit" or "sentiment"
    """
    Assign a label to the post based on the subreddit name or mode.

    Parameters:
        subreddit_name (str): The name of the subreddit.
        mode (str): The mode to assign the label. Options are "sentiment" or "subreddit".
                    Default is "sentiment".
    Returns:
        str: The assigned label.
    """

    # Validate mode
    if mode not in {"sentiment", "subreddit"}:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'sentiment' or 'subreddit'.")
    
    if mode == "sentiment":
        sentiment_labels = {
            "depression": "Negative",
            "anxiety": "Negative",
            "mentalhealth": "Negative",
            "bipolarreddit": "Negative",
            "SuicideWatch": "Negative",

            "GetMotivated": "Positive",
            "DecidingToBeBetter": "Positive",
            "selfimprovement": "Positive",
            "Happy": "Positive",
            "humansbeingbros": "Positive",
            "MadeMeSmile": "Positive",
            "UpliftingNews": "Positive",

            "Psychology": "Neutral",
            "Emotions": "Neutral",
            "CasualConversation": "Neutral",
            "offmychest": "Neutral",
            "TrueOffMyChest": "Neutral",
            "Advice": "Neutral",
            "LifeProTips": "Neutral",
            "NoStupidQuestions": "Neutral",
            "AskReddit": "Neutral"
        }

        return sentiment_labels.get(subreddit_name, "Neutral") # Default to "Neutral" if subreddit not found in the mapping
    
    # Default: label is just the subreddit name
    return subreddit_name


def fetch_reddit_posts(subreddit_name, desired_count, mode="sentiment"):
    """ Fetch posts from a given subreddit in a random category. """
    global global_total, global_sent, global_failed

    local_total = 0
    local_sent = 0
    local_failed = 0
    
    collected_posts = [] # List to store collected posts
    subreddit = reddit.subreddit(subreddit_name)
    category = random.choice(categories)  # Pick a random category
    
    # Select the category dynamically
    if category == "hot":
        posts = subreddit.hot(limit=100)
    elif category == "new":
        posts = subreddit.new(limit=100)
    elif category == "top":
        posts = subreddit.top(limit=100)
    elif category == "rising":
        posts = subreddit.rising(limit=100)

    for post in posts:
        if local_total >= desired_count:
            break  # Stop if we reach the desired count

        local_total += 1 # Increment total posts count
        # text_content = post.selftext if post.selftext.strip() else "[No text content]"  # Fallback if no text content
        text_content = post.selftext.strip() if post.selftext and post.selftext.strip() else "[No text content]"

        post_data = {
            "title": post.title,
            "score": post.score,
            "id": post.id,
            "url": post.url,
            "num_comments": post.num_comments,
            "created_utc": post.created_utc,
            # "text": post.selftext
            "text": text_content, # Use selftext if available, else fallback to a message
            # "subreddit": subreddit_name,
            "subreddit": post.subreddit.display_name,
            "category": category,
            "label": assign_label(post.subreddit.display_name, mode=mode)  # Label the post with sentiment or subreddit name depending on the mode you want
        }

        try:
            producer.send(KAFKA_TOPIC, post_data).get(timeout=10) # Send post data to Kafka topic and wait to be sent
            # record_metadata = future.get(timeout=10)  # Wait for the message to be sent
            # future.get(timeout=10)  # Wait for the message to be sent
            local_sent += 1  # Increment sent posts count
        except Exception as e:
            print(f"❌ Error sending post {post.id} to Kafka: {e}")
            local_failed += 1

        collected_posts.append(post_data)  # Append the post data to the collected posts list
    
    producer.flush()  # Ensure all messages are sent before proceeding only once per subreddit

    # Update global counters
    global_total += local_total
    global_sent += local_sent
    global_failed += local_failed

    # Print local summary
    print(f"📊 [{subreddit_name}] Category: {category} | Total: {local_total} | Sent: {local_sent} | Failed: {local_failed}")

    # return collected_posts  # Return the list of collected posts
    return pd.DataFrame(collected_posts), local_sent  # Return as DataFrame and the number of posts sent
    
    
if __name__ == "__main__":
    # for subreddit in subreddits:
    #     fetch_reddit_posts(subreddit, desired_count=posts_per_subreddit)
    all_sent = 0
    progress_bar = tqdm(total=TOTAL_LIMIT, desc="📥 Sending posts to Kafka", ncols=100)

    for subreddit in subreddits:
        # before = global_sent # Store the current number of sent posts
        df, num_sent = fetch_reddit_posts(subreddit, desired_count=posts_per_subreddit)
        # after = global_sent # Update the progress bar based on the number of posts sent to Kafka
        # delta = after - before
        # all_sent += delta
        # progress_bar.update(df[0].shape[0])  # Update the progress bar with the number of posts sent
        if df is not None and not df.empty:
            progress_bar.update(num_sent)  # Update the progress bar with the number of posts sent
        
        else:
            logging.warning(f"⚠️ No data fetched from subreddit '{subreddit}'. Skipping DB storage.")

    progress_bar.close()
    
    
    # Final global summary
    print("\n📦 Reddit Fetch Summary:")
    print(f"🔹 Total posts fetched: {global_total}")
    print(f"✅ Total posts sent to Kafka: {global_sent}")
    print(f"❌ Total posts failed to send: {global_failed}")
    print("\n🎉 Finished sending posts to Kafka!")
    
    
