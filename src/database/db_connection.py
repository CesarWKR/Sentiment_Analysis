import os
import psycopg2  # For PostgreSQL
import mysql.connector  # For MySQL
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Database settings
DB_TYPE = os.environ["DB_TYPE"] # "postgres" or "mysql"
DB_HOST = os.environ["DB_HOST"] # Default is "localhost"
DB_PORT = os.environ["DB_PORT"] # Default for PostgreSQL, put 3306 for MySQL
DB_NAME = os.environ["DB_NAME"] 
DB_USER = os.environ["DB_USER"]  
DB_PASSWORD = os.environ["DB_PASSWORD"]  # If it doesn't exist in .env, it will raise an KeyError

print(f"DB_TYPE: {DB_TYPE}")  # Depuración
def get_db_url():
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


def connect_to_db():
    """
    Establishes a connection to the database (PostgreSQL or MySQL).
    
    :return: Database connection object
    """
    try:
        if DB_TYPE == "postgres":
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            print("✅ Connected to PostgreSQL successfully!")
        
        elif DB_TYPE == "mysql":
            conn = mysql.connector.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            print("✅ Connected to MySQL successfully!")
        
        else:
            raise ValueError("❌ Unsupported database type. Use 'postgres' or 'mysql'.")
        
        return conn

    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

if __name__ == "__main__":
    conn = connect_to_db()
    if conn:
        conn.close()
        print("✅ Database connection closed.")