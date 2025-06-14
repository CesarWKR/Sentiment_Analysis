import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.database.db_connection import connect_to_db
from src.database.store_data import store_data
from tqdm import tqdm

# Configuration
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"  # Pre-trained model for sentiment analysis
LABELS = ["Negative", "Neutral", "Positive"]
SOURCE_TABLE = "reddit_posts"  # Source table containing the data to be relabeled
TARGET_TABLE = "relabeled_data"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  # Optimize GPU performance

# Charge the model and tokenizer which is already trained and will be used for inference and prediction to relabel mislabeled data
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)  # Move model to the appropriate device (GPU or CPU)
model.eval()

engine = connect_to_db()  # Connect to the database

def predict_label(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,     # Limit to 256 tokens
        padding="max_length"  # Ensure fixed size
    ).to(device)  # Move inputs to the same device as the model

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        return LABELS[predicted_class_id]

def simple_cleaning(text):
    text = str(text).lower()
    return text.replace('\n', ' ').replace('\r', ' ').strip()  # Remove newlines and extra spaces

def hybrid_relabeling():
    with engine.connect() as conn:
        df = pd.read_sql(f"SELECT * FROM {SOURCE_TABLE};", conn)

    if TEXT_COLUMN not in df or LABEL_COLUMN not in df:
        raise ValueError(f"Columns '{TEXT_COLUMN}' or '{LABEL_COLUMN}' are missing in table {SOURCE_TABLE}")

    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(simple_cleaning)

    predicted_labels = []
    for text in tqdm(df[TEXT_COLUMN], desc="🔁 Re-labeling"):
        try:
            predicted = predict_label(text)
        except Exception as e:
            predicted = "neutral"  # Default label in case of error
        predicted_labels.append(predicted)

    df["predicted_label"] = predicted_labels

    # Re-labeling logic
    df["final_label"] = df.apply(
        lambda row: row["predicted_label"] if row["label"] != row["predicted_label"] else row["label"],
        axis=1
    )

    # Create a new DataFrame to store the relabeled data
    df_relabeled = df[[TEXT_COLUMN, "final_label"]].rename(columns={"final_label": "label"})

    # Store the relabeled data
    success = store_data(df_relabeled, table_name=TARGET_TABLE)
    if success:
        print(f"[✅] Table '{TARGET_TABLE}' stored with {len(df_relabeled)} relabeled records.")
    else:
        print(f"[❌] Failed to store table '{TARGET_TABLE}'.")


if __name__ == "__main__":
    hybrid_relabeling()