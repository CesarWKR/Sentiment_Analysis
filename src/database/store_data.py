import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from db_connection import get_db_url

# Define the database model
Base = declarative_base()

class RedditPost(Base):
    __tablename__ = "reddit_posts"

    id = Column(String, primary_key=True)  # Reddit post ID
    title = Column(String(255), nullable=False)
    score = Column(Integer, nullable=False)
    url = Column(String(255), nullable=True)
    num_comments = Column(Integer, nullable=False)
    created_utc = Column(DateTime, nullable=False)
    text = Column(Text, nullable=True)

def store_reddit_posts(data: pd.DataFrame):
    """
    Stores Reddit posts in the database.

    :param data: DataFrame containing Reddit post data.
    """
    db_url = get_db_url()
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)  # Create table if it doesn't exist

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        for _, row in data.iterrows():
            post = RedditPost(
                id=row["id"],
                title=row["title"],
                score=row["score"],
                url=row["url"],
                num_comments=row["num_comments"],
                created_utc=pd.to_datetime(row["created_utc"], unit="s"), # Convert to datetime
                text=row["text"]
            )
            session.merge(post)  # Insert or update if exists
        session.commit() # Save changes
        print(f"✅ Successfully stored {len(data)} posts.")
    except Exception as e: 
        session.rollback() 
        print(f"❌ Error storing posts: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    from src.api.fetch_reddit import fetch_reddit_posts

    subreddit_name = "news"  # Change as needed
    reddit_data = fetch_reddit_posts(subreddit_name, limit=200)
    store_reddit_posts(reddit_data)
