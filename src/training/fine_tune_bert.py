import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.database.db_connection import get_db_url
from sqlalchemy import create_engine

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

torch.cuda.empty_cache() # Clear GPU memory

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
    db_url = get_db_url()
    engine = create_engine(db_url)

    # query = "SELECT text, sentiment FROM reddit_posts"  # Query to fetch original data from reddit_posts table
    query = "SELECT text, label FROM cleaned_data"  # Query to fetch cleaned data from cleaned_data table
    df = pd.read_sql(query, engine)

    return df

def train_model():
    """Fine-tunes BERT for sentiment analysis."""
    df = load_data()
    
    label_mapping = {"negative": 0, "neutral": 1, "positive": 2}  # Map labels to integers
    df["label"] = df["label"].map(label_mapping)   # Convert labels to integers

    # Splitting data into train and test
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    train_dataset = RedditDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
    val_dataset = RedditDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Load pre-trained BERT model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)  # 3 labels: negative, neutral, positive
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Training loop
    epochs = 3
    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, leave=True) # Progress bar
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad() # Reset gradients
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss  # Loss tensor value
            loss.backward()  
            optimizer.step()  # Update model parameters

            loop.set_description(f"Epoch {epoch+1}") # Update progress bar
            loop.set_postfix(loss=loss.item())  # Convert loss tensor to float

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
                print(outputs.logits)  # Logits tensor value
                predictions = torch.argmax(F.softmax(outputs.logits, dim=1), dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")

    # Save model
    model.save_pretrained("bert_sentiment_model")
    tokenizer.save_pretrained("bert_sentiment_model")
    print("✅ Model saved successfully!")

if __name__ == "__main__":
    train_model()
