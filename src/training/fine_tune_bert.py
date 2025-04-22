import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F
import sys
import os
import shutil  
import datetime
# from torch.cuda.amp import autocast, GradScaler # For mixed precision training (only with GPU)
from torch.amp import autocast, GradScaler # For mixed precision training (works with CPU and GPU)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.database.db_connection import connect_to_db
from src.database.store_data import save_validation_data
from sqlalchemy import create_engine

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  # Optimize GPU performance

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class RedditDataset(Dataset):
    """Custom dataset class for Reddit posts."""
    def __init__(self, texts, labels, tokenizer, max_length=512):  # Initialize the dataset, use max_length equal to the test file
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): # Return the number of samples
        return len(self.texts)

    def __getitem__(self, idx): # Return a sample at the specified index
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer( # Tokenize the text
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0), # Remove batch dimension
            "attention_mask": encoding["attention_mask"].squeeze(0), 
            "label": torch.tensor(label, dtype=torch.long) # Convert label to tensor
        }

def load_data():
    """Fetches cleaned data from the database."""
    # db_url = get_db_url()   # Get database URL from environment variables
    engine = connect_to_db()  # Create a database engine

    # Load reddit_posts (original data)
    query_original = "SELECT text, label FROM reddit_posts"
    df_orig = pd.read_sql(query_original, engine) # Load original data from the database
    # df_orig = df_orig.rename(columns={"sentiment": "label"}) # Rename sentiment column to label


    # Load cleaned_data (augmented data)
    query_augmented = "SELECT text, label FROM cleaned_data"
    df_aug = pd.read_sql(query_augmented, engine)

    # Map labels to integers
    label_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
    df_orig["label"] = df_orig["label"].map(label_mapping)
    df_aug["label"] = df_aug["label"].map(label_mapping)

    # Drop NaN values
    df_orig.dropna(inplace=True)
    df_aug.dropna(inplace=True)

    # Get min size
    min_size = min(len(df_orig), len(df_aug))

    # Sample to match size
    df_orig_sampled = df_orig.sample(n=min_size, random_state=42) # Take reddit_posts
    df_aug_sampled = df_aug.sample(n=min_size, random_state=42) # Take cleaned_data


    # Merge & shuffle
    df_combined = pd.concat([df_orig_sampled, df_aug_sampled], ignore_index=True) # Combine the two DataFrames
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle the combined DataFrame

    print(f"✅ Loaded {len(df_combined)} records (50% original, 50% augmented).")
    return df_combined


def train_model():
    """Fine-tunes BERT for sentiment analysis."""
    df = load_data()
     
    # Splitting data into train and test
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )


    # Save validation data (20% of data) to the database
    save_validation_data(val_texts.tolist(), val_labels.tolist())


    train_dataset = RedditDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
    val_dataset = RedditDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)

    batch_size = 16  # Reduce if hitting memory limit
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True) # Shuffle for training
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True) # No shuffle for validation

    # Load pre-trained BERT model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)  # 3 labels: Negative, Neutral, Positive
    model.to(device)

    # optimizer = AdamW(model.parameters(), lr=2e-5)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # AdamW optimizer with weight decay

    # Learning rate scheduler
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 3)

    scaler = GradScaler("cuda")  # For mixed precision training

    # Training loop
    epochs = 3
    accumulation_steps = 2  # Simulate larger batch sizes by accumulating gradients
    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, leave=True) # Progress bar
        optimizer.zero_grad() # Reset gradients

        for i, batch in enumerate(loop):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast(device_type="cuda"):  # Mixed precision
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader): # Update weights every accumulation_steps
                scaler.step(optimizer) # Update weights
                scaler.update() # Update scaler
                optimizer.zero_grad() # Reset gradients
                scheduler.step() # Update learning rate

            loop.set_description(f"Epoch {epoch+1}") # Update progress bar description
            loop.set_postfix(loss=loss.item()) # Update progress bar postfix



        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)  # No need for softmax
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")

        torch.cuda.empty_cache()  # Free memory

    # Save model with timestamp
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # model_path = f"bert_sentiment_model_{timestamp}"
    # model.save_pretrained(model_path)
    # tokenizer.save_pretrained(model_path)
    # print(f"✅ Model saved successfully at {model_path}!")


    # Save model in fixed directory (no timestamp)
    model_path = "bert_sentiment_model"

    # Delete existing directory if exists
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    # Save model and tokenizer
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"✅ Model saved successfully at {model_path}!")


if __name__ == "__main__":
    train_model()
