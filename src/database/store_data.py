import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
#from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.api.fetch_reddit import fetch_reddit_posts
from src.database.db_connection import get_db_url


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

class CleanedData(Base):
    __tablename__ = "cleaned_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    label = Column(String(50), nullable=False)

def store_data(data: pd.DataFrame, table_name="cleaned_data"):
    """
    Stores data in the database (Reddit posts or cleaned dataset).

    :param data: DataFrame to store.
    :param table_name: Name of the database table.
    """
    db_url = get_db_url()
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)  # Create table if it doesn't exist

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        if table_name == "reddit_posts":
            for _, row in data.iterrows():
                post = RedditPost(
                    id=row["id"],
                    title=row["title"],
                    score=row["score"],
                    url=row["url"],
                    num_comments=row["num_comments"],
                    created_utc=pd.to_datetime(row["created_utc"], unit="s"),
                    text=row["text"]
                )
                session.merge(post)  
        elif table_name == "cleaned_data":
            for _, row in data.iterrows():
                cleaned_entry = CleanedData(
                    text=row["text"],
                    label=row["label"]
                )
                session.add(cleaned_entry)

        session.commit()
        print(f"✅ Successfully stored {len(data)} records in {table_name}.")
    except Exception as e: 
        session.rollback()
        print(f"❌ Error storing data: {e}")
    finally:
        session.close()

if __name__ == "__main__":

    subreddit_name = "depression"  
    reddit_data = fetch_reddit_posts(subreddit_name, limit=5)
    store_data(reddit_data, table_name="reddit_posts")
