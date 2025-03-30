import praw
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Reddit API credentials
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

def fetch_reddit_posts(subreddit_name: str, limit: int = 5):
    """
    Fetches top posts from a given subreddit.
    
    :param subreddit_name: Name of the subreddit
    :param limit: Number of posts to retrieve (default: 200)
    :return: DataFrame containing post data
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    
    for post in subreddit.hot(limit=limit):
        text_content = post.selftext if post.selftext.strip() else "[No text content]"  # Fallback if no text content
        posts.append({
            "title": post.title,
            "score": post.score,
            "id": post.id,
            "url": post.url,
            "num_comments": post.num_comments,
            "created_utc": post.created_utc,
            # "text": post.selftext
            "text": text_content # Use selftext if available, else fallback to a message
        })
    
    return pd.DataFrame(posts)

if __name__ == "__main__":
    subreddit_name = "depression"  # Change this to the desired subreddit
    data = fetch_reddit_posts(subreddit_name, limit=5)
    print(data.head())  # Display first 5 rows
    
    # Save to CSV file
    data.to_csv(f"{subreddit_name}_posts.csv", index=False)
    print(f"Data saved to {subreddit_name}_posts.csv")
