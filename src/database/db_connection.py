import os
import psycopg2  # For PostgreSQL
import mysql.connector  # For MySQL
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Load environment variables
load_dotenv()


# Database settings
DB_TYPE = os.environ["DB_TYPE"] # "postgres" or "mysql"
DB_HOST = os.environ["DB_HOST"] # Default is "localhost"
DB_PORT = os.environ["DB_PORT"] # Default for PostgreSQL, put 3306 for MySQL
DB_NAME = os.environ["DB_NAME"] 
DB_USER = os.environ["DB_USER"]  
DB_PASSWORD = os.environ["DB_PASSWORD"]  # If it doesn't exist in .env, it will raise an KeyError

print(f"DB_TYPE: {DB_TYPE}")  # Debugging line to check the DB_TYPE


def get_db_url() -> str:
    """
    Constructs a database URL for SQLAlchemy based on the environment variables.
    
    :return: Database URL string
    """
    if DB_TYPE == "postgres":
        return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    elif DB_TYPE == "mysql":
        return f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    else:
        raise ValueError("❌ Unsupported database type. Use 'postgres' or 'mysql'.")


def connect_to_db() -> Engine:
    """
    Establishes a connection to the database (PostgreSQL or MySQL).
    
    :return: Database connection object
    """
    if getattr(connect_to_db, "_connected_once", False): # Check if already connected
        return connect_to_db._engine  # Return the existing engine if already connected
    
    try:
        db_url = get_db_url()
        engine = create_engine(db_url)

        connect_to_db._engine = engine # Store the engine in the function's attribute for reuse
        connect_to_db._connected_once = True    # Set the flag to True to indicate that the connection has been established

        print(f"✅ Connected to the {DB_TYPE.upper()} database successfully with SQLAlchemy!")
        return engine # Return the SQLAlchemy engine object for further use

    except Exception as e:
        print(f"❌ Error creating SQLAlchemy engine: {e}")
        return None


if __name__ == "__main__":
    engine = connect_to_db()
    if engine:
        try:
            with engine.connect() as conn:
                conn.execute("SELECT 1")  # Test the connection
                print("✅ Connection test successful!")
        except Exception as e:
            print(f"❌ Error testing connection: {e}")
