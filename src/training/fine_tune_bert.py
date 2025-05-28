from sklearn.utils import compute_class_weight
from sklearn.metrics import classification_report
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
import os
import shutil  
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
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
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base") # Load RoBERTa tokenizer

class RedditDataset(Dataset):
    """Custom dataset class for Reddit posts."""
    def __init__(self, texts, labels, tokenizer, max_length=256):  # Initialize the dataset, use max_length equal to the test file
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
            "label": torch.tensor(label, dtype=torch.long), # Convert label to tensor
            "text": text  # Include text for error analysis
        }
    

class FocalLoss(torch.nn.Module):   # Custom Focal Loss class
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=None, gamma=1.0, reduction='mean', device=None):  # Initialize Focal Loss parameters
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.alpha = None  # No alpha specified, will be set in forward pass
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        elif isinstance(alpha, (int, float)):
            raise ValueError("alpha must be a list, ndarray or tensor with one value per class")
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha.float()
        else:
            raise TypeError("Type of alpha not supported")
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha.to(self.device)

    def forward(self, inputs, targets):  # Forward pass for Focal Loss
        """
        inputs: logits (shape [batch_size, num_classes])
        targets: real labels (shape [batch_size])
        """

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction="none")  # Compute CrossEntropy loss
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)

            alpha_t = self.alpha[targets]  # shape: [batch_size]
            focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss  # Compute Focal Loss
        else:
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':  # Reduce loss
            return focal_loss.mean()
        elif self.reduction == 'sum':  # Sum loss
            return focal_loss.sum()
        else:
            return focal_loss


def get_class_weights(labels, num_classes):  # Function to compute class weights
    """Computes class weights for imbalanced datasets."""
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)] # Compute weights for each class
    return torch.tensor(weights, dtype=torch.float)


def load_data(data_source="cleaned"):
    """Fetches cleaned data from the database."""
    # db_url = get_db_url()   # Get database URL from environment variables
    engine = connect_to_db()  # Create a database engine

    # Load reddit_posts (original data)
    query_original = "SELECT text, label FROM reddit_posts"
    df_orig = pd.read_sql(query_original, engine) # Load original data from the database

    # Load cleaned_data (cleaned data)
    query_clenaed = "SELECT text, label FROM cleaned_data"
    df_cleaned = pd.read_sql(query_clenaed, engine)

    # Map labels to integers
    label_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
    df_orig["label"] = df_orig["label"].map(label_mapping)
    df_cleaned["label"] = df_cleaned["label"].map(label_mapping)

    # Drop NaN values
    df_orig.dropna(inplace=True)
    df_cleaned.dropna(inplace=True)

    # Depending on the source
    if data_source == "raw":
        df = df_orig # Use original data
        print(f"✅ Loaded {len(df)} raw records from reddit_posts.")
    elif data_source == "cleaned":
        df = df_cleaned # Use cleaned data
        print(f"✅ Loaded {len(df)} cleaned records from cleaned_data.")
    elif data_source == "both":
        # Match sizes
        min_size = min(len(df_orig), len(df_cleaned))
        df_orig_sampled = df_orig.sample(n=min_size, random_state=42)
        df_cleaned_sampled = df_cleaned.sample(n=min_size, random_state=42) # Sample to match size
        df = pd.concat([df_orig_sampled, df_cleaned_sampled], ignore_index=True) # Combine the two DataFrames
        df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle the combined DataFrame
        print(f"✅ Loaded {len(df)} combined records (50% raw, 50% cleaned).")
    else:
        raise ValueError(f"Invalid data_source '{data_source}'. Must be 'raw', 'cleaned', or 'both'.")

    return df


def analyze_data_distribution(df):  # Function to analyze data distribution
    """Analyzes data distribution by class and generates visual statistics."""

    # Count the number of records per class
    label_counts = df["label"].value_counts()
    print("🔢 Count of records per class:")
    print(label_counts)

    # Calculate the percentage of each class
    percentages = label_counts / len(df) * 100
    print("\n📊 Percentage of each class:")
    print(percentages)

    #   Plot the distribution of labels
    plt.figure(figsize=(8, 5))
    label_counts.plot(kind="bar", color=["red", "blue", "green"])
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Distribution of labels in the dataset")
    plt.xticks(rotation=0)
    plt.show()


def freeze_roberta_layers(model, num_layers_to_freeze=8):
    for i in range(num_layers_to_freeze): # Freeze the first layers
        for param in model.roberta.encoder.layer[i].parameters():  # Freeze the parameters of the first layers
            param.requires_grad = False  # Set requires_grad to False freeze selected layers
    print(f"🧊 Frozen first {num_layers_to_freeze} RoBERTa layers.")
    return num_layers_to_freeze


def unfreeze_roberta_layers(model, layer_index: int):
    """
    Unfreeze the last ones num_layers_to_unfreeze layers from the Roberta encoder.
    """
    # for i in range(11, 11 - num_layers_to_unfreeze, -1): # Unfreeze the last layers
    for param in model.roberta.encoder.layer[layer_index].parameters(): 
        param.requires_grad = True
    print(f"🔓 Unfrozen RoBERTa layer {layer_index}.") 


def get_focal_alpha(labels, num_classes):
    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=labels)
    alpha = class_weights / class_weights.sum()
    # print(f"⚖️  Alpha for Focal Loss: {alpha}")
    return alpha


def print_classification_report(labels, predictions, label_map={0: "Negative", 1: "Neutral", 2: "Positive"}):
    print("Classification Report:\n")
    print(classification_report(labels, predictions, target_names=label_map.values()))


def show_misclassified_examples(texts, labels, predictions, label_map={0: "Negative", 1: "Neutral", 2: "Positive"}, max_errors=3):
    print(f"\n🔍 Wrongly Classified Examples (showing up to {max_errors}):")
    errors_shown = 0
    for text, true_label, pred_label in zip(texts, labels, predictions):
        if true_label != pred_label:
            print(f"\n🟥 Text: {text}")
            print(f"   Real: {label_map[true_label]}, Predicted: {label_map[pred_label]}")
            errors_shown += 1
            if errors_shown >= max_errors:
                break


def train_model(data_source="cleaned"):
    """Fine-tunes BERT for sentiment analysis."""

    df = load_data(data_source) # Load data from the database
    analyze_data_distribution(df) # Analyze data distribution


    # Splitting data into train and test
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Save validation data (20% of data) to the database
    # save_validation_data(val_texts.tolist(), val_labels.tolist())

    train_dataset = RedditDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
    val_dataset = RedditDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)

    batch_size = 32  # Reduce if hitting memory limit
    accumulation_steps = 1  # Simulate larger batch sizes by accumulating gradients
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True) # Shuffle for training
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True) # No shuffle for validation

    num_classes = 3  # Number of classes
    

    config = RobertaConfig.from_pretrained(
        "roberta-base",
        num_labels=3,
        hidden_dropout_prob=0.3,  
        attention_probs_dropout_prob=0.3  
    )
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", config=config)

    model.to(device)  # Move model to GPU if available

    total_roberta_layers = 12  # Total number of RoBERTa layers
    num_layers_to_freeze = freeze_roberta_layers(model, num_layers_to_freeze=8)  # Freeze the first 8 layers of RoBERTa

    # ======= Loss Function: You can switch between FocalLoss and CrossEntropy =======
    alpha = get_focal_alpha(train_labels, num_classes=3)
    class_weights = get_class_weights(train_labels, num_classes)
    cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))  # CrossEntropyLoss with class weights
    focal_loss = FocalLoss(alpha=alpha, gamma=2.0, device=device)

     # Group parameters for optimizer
    no_decay = ['bias', 'LayerNorm.weight']  # Parameters that should not decay
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5) # Optimizer with grouped parameters

    num_epochs = 6  # Number of epochs for training
    total_steps = len(train_loader) * num_epochs // accumulation_steps  # Total training steps
    num_warmup_steps = int(0.1 * total_steps)  # Warmup steps for learning rate scheduler
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
    # scaler = GradScaler("cuda")  # For mixed precision training
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # Save model in fixed directory (no timestamp)
    model_path = "bert_sentiment_model"

    # ======= Early Stopping Settings =======
    best_val_f1 = 0.0
    patience = 1  # Number of epochs to wait for improvement
    patience_counter = 0

    train_losses, val_losses = [], []  # To track losses for plotting

    best_model_state_dict = None  # To save the best model state dict
    best_val_loss = float("inf")

    final_texts_for_analysis = []
    final_true_labels_for_analysis = []
    final_preds_for_analysis = []

    use_focal = True  # Set to True to use Focal Loss, False for CrossEntropy

    # Training loop
    for epoch in range(num_epochs):
        print(f"\n🔁 Epoch {epoch+1}/{num_epochs}")

        if use_focal:  # Use Focal Loss from the first epoch or if use_focal is False use CrossEntropyLoss
            loss_fn = focal_loss  
            print("📌 Using Focal Loss")
            print(f"⚖️ Alpha for Focal Loss: {alpha}")
        else:
            loss_fn = cross_entropy_loss # CrossEntropyLoss when use_focal is False
            print("📌 Using CrossEntropy Loss")
            print(f"⚖️ Class Weights: {class_weights}")

        if epoch < num_layers_to_freeze:
            layer_to_unfreeze = num_layers_to_freeze - 1 - epoch
            unfreeze_roberta_layers(model, layer_index=layer_to_unfreeze)  # Unfreeze the last layers one by one

        # ======= Training =======
        model.train()
        total_loss = 0.0
        # loop = tqdm(train_loader, leave=True) # Progress bar
        optimizer.zero_grad() # Reset gradients
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True) # Progress bar

        for i, batch in (loop):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):  # Mixed precision
                outputs = model(input_ids, attention_mask=attention_mask)
                # loss = outputs.loss / accumulation_steps   # Normalize CrossEntropyLoss.
                loss = loss_fn(outputs.logits, labels) / accumulation_steps # Normalize FocalLoss.

            scaler.scale(loss).backward()   # Backpropagation
            total_loss += loss.item()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader): # Update weights every accumulation_steps
                scaler.step(optimizer) # Update weights
                scaler.update() # Update scaler
                optimizer.zero_grad() # Reset gradients
                lr_scheduler.step() # Update learning rate

            loop.set_postfix(loss=loss.item()) # Update progress bar postfix

        avg_train_loss = total_loss / len(train_loader)  # Average training loss

        # ======= Validation =======
        model.eval()
        all_preds = []
        all_labels = []
        val_total_loss = 0.0
        texts_for_analysis = []      # Texts for error analysis
        true_labels_for_analysis = []  # True labels
        preds_for_analysis = []        # Predictions

        with torch.no_grad():
            for batch in val_loader:
            #for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    # loss = criterion(logits, labels)
                    loss = loss_fn(logits, labels)
                    val_total_loss += loss.item()
                    preds = torch.argmax(outputs.logits, dim=1)  # No need for softmax

                texts_for_analysis.extend(batch["text"])  # Texts for error analysis
                true_labels_for_analysis.extend(labels.cpu().numpy()) # True labels
                preds_for_analysis.extend(preds.cpu().numpy()) # Predictions

                all_labels.extend(labels.cpu().numpy())  # Store true labels
                all_preds.extend(preds.cpu().numpy())  # Store predictions

            # analyze_errors(texts_for_analysis, true_labels_for_analysis, preds_for_analysis)   # This print every epoch, so it is commented out
            final_texts_for_analysis = texts_for_analysis
            final_true_labels_for_analysis = true_labels_for_analysis
            final_preds_for_analysis = preds_for_analysis

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        avg_val_loss = val_total_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        train_losses.append(avg_train_loss)
        print(f"✅ Validation Accuracy: {val_acc:.4f} | F1 Score (macro): {val_f1:.4f} | 🧪 Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        # print(classification_report(all_labels, all_preds, digits=3))
        print_classification_report(all_labels, all_preds)  # print classification report every epoch

        # ======= Early Stopping and save the model======
        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and avg_val_loss < best_val_loss):
            best_val_f1 = val_f1
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state_dict = model.state_dict()  # Save model in memory
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

        torch.cuda.empty_cache()  # Free memory

    print("\n📌 Training completed. Showing error analysis on final validation set:")
    show_misclassified_examples(final_texts_for_analysis, final_true_labels_for_analysis, final_preds_for_analysis, max_errors=5)

    # === Save best model at the end of training ===
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)  # Load best model state dict
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        model.save_pretrained(model_path)  # Save model to disk
        tokenizer.save_pretrained(model_path)  # Save tokenizer to disk
        print(f"\n📦 Final best model saved to '{model_path}'")

    # Plot training vs validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss (Overfitting Diagnosis)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  
    plt.show()

    return model, val_loader  # Returning model and validation loader for evaluation

if __name__ == "__main__":
    train_model()
