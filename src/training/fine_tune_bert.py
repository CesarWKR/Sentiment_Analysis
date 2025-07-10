from sklearn.utils import compute_class_weight
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaLayer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
from sqlalchemy import inspect
import os
import shutil  
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
# from torch.cuda.amp import autocast, GradScaler # For mixed precision training (only with GPU)
from torch.amp import autocast, GradScaler # For mixed precision training (works with CPU and GPU)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.database.db_connection import connect_to_db
from src.database.store_data import store_generic_table, save_validation_data
from src.preprocessing.data_augmentation import balance_dataset


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  # Optimize GPU performance

# Load RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base") # Load RoBERTa tokenizer

engine = connect_to_db()  # Create a database engine

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
           self.alpha = torch.tensor(alpha, dtype=torch.float)  # Convert list or ndarray to tensor
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
    """
    Loads data from the database based on the specified source,
    applies label mapping, drops NaNs, and stores balanced and synthetic versions
    in corresponding dynamic tables (e.g., balanced_cleaned, synthetic_raw).
    """

    df_orig = pd.read_sql("SELECT text, label FROM reddit_posts", engine) # Load reddit_posts (original data)
    df_relabeled = pd.read_sql("SELECT text, label FROM relabeled_data", engine) # Load relabeled_data (relabeled data)
    df_cleaned = pd.read_sql("SELECT text, label FROM cleaned_data", engine) # Load cleaned_data (cleaned data)
    

    # Map labels to integers
    label_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
    df_orig["label"] = df_orig["label"].map(label_mapping)
    df_relabeled["label"] = df_relabeled["label"].map(label_mapping)
    df_cleaned["label"] = df_cleaned["label"].map(label_mapping)

    # Drop NaN values
    df_orig.dropna(inplace=True)
    df_cleaned.dropna(inplace=True)

    # Select data based on source
    if data_source == "raw":
        df = df_orig # Use original data
        print(f"✅ Loaded {len(df)} raw records from reddit_posts.")
    elif data_source == "relabeled":
        df = df_relabeled # Use relabeled data
        print(f"✅ Loaded {len(df)} relabeled records from relabeled_data.")
    elif data_source == "cleaned":
        df = df_cleaned # Use cleaned data
        print(f"✅ Loaded {len(df)} cleaned records from cleaned_data.")
    elif data_source == "combined":   # Combine relabeled and cleaned data
        # Match sizes
        min_size = min(len(df_relabeled), len(df_cleaned))
        df_relabeled_sampled = df_relabeled.sample(n=min_size, random_state=42)
        df_cleaned_sampled = df_cleaned.sample(n=min_size, random_state=42) # Sample to match size
        df = pd.concat([df_relabeled_sampled, df_cleaned_sampled], ignore_index=True) # Combine the two DataFrames
        df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle the combined DataFrame
        print(f"✅ Loaded {len(df)} combined records (50% relabeled, 50% cleaned).")
    else:
        raise ValueError(f"Invalid data_source '{data_source}'. Must be 'raw', 'relabeled', 'cleaned', or 'combined'.")

    # Store the loaded data in a dynamic table
    df_balanced, df_synthetic = balance_dataset(df)

    balanced_table = f"balanced_{data_source}"
    synthetic_table = f"synthetic_{data_source}"

    store_generic_table(df_balanced, balanced_table)
    if not df_synthetic.empty: # If synthetic data is generated, store it
        store_generic_table(df_synthetic, synthetic_table)
        print(f"📦 Stored synthetic data in table '{synthetic_table}'")
    
    print(f"📦 Stored balanced data in table '{balanced_table}'")

    return df_balanced, df_synthetic, df  # Return balanced, synthetic, and original data


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


def get_focal_alpha(labels, num_classes, neutral_boost=1.5):
    """ Computes the alpha parameter for Focal Loss based on class distribution.
    This is used to balance the loss function for imbalanced datasets.
    Add a especial boost to the Neutral class (1) to make it more important.
    
    Args:
        labels (list or np.array): List or array of labels.
        num_classes (int): Number of classes in the dataset.
        neutral_boost (float): Boost factor for the Neutral class (1).

    Returns:
        np.array: Normalized alpha values for each class."""
    
    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=labels)
    alpha = np.array(class_weights, dtype=np.float32)
    if len(alpha) > 1:  # Apply boost to Neutral class
        alpha[1] *= neutral_boost
        alpha = alpha / alpha.sum()  # Normalize after boosting

    return alpha


def print_classification_report(labels, predictions, label_map={0: "Negative", 1: "Neutral", 2: "Positive"}):
    print("Classification Report:\n")
    print(classification_report(labels, predictions, target_names=label_map.values()))


def table_exists(table_name: str) -> bool:
    """
    Checks whether a table exists in the connected database.
    """
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def load_synthetic_table_from_db(table_name="synthetic_combined"):
    """
    Loads a pre-generated synthetic dataset directly from the database without additional processing.
    """

    if not table_exists(table_name):
        raise ValueError(f"❌ Table '{table_name}' does not exist in the database. "
                         f"Please generate synthetic data before training.")

    print(f"📥 Loading synthetic data from table '{table_name}'...")
    df_synthetic = pd.read_sql(f"SELECT text, label FROM {table_name}", engine)
    
    # Only apply label mapping if values are still in text
    if df_synthetic["label"].dtype == object:
        label_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
        df_synthetic["label"] = df_synthetic["label"].map(label_mapping)
        
    df_synthetic.dropna(inplace=True)
    
    print(f"✅ Loaded {len(df_synthetic)} synthetic records from '{table_name}'.")
    return df_synthetic


def load_balanced_table_from_db(table_name="balanced_combined"):
    """
    Loads a pre-balanced dataset directly from the database without additional processing.
    """

    if not table_exists(table_name):
        raise ValueError(f"❌ Table '{table_name}' does not exist in the database. "
                         f"Please run the pipeline or balance the dataset before training.")

    print(f"📥 Loading pre-balanced data from table '{table_name}'...")
    df_balanced = pd.read_sql(f"SELECT text, label FROM {table_name}", engine)
    
    # Only apply label mapping if values are still in text
    if df_balanced["label"].dtype == object:
        label_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
        df_balanced["label"] = df_balanced["label"].map(label_mapping)
    
    df_balanced.dropna(inplace=True)
    
    print(f"✅ Loaded {len(df_balanced)} records from '{table_name}'.")
    return df_balanced


def train_model(data_source="combined", dataset_type="balanced", use_prebalanced=True):
    """Fine-tunes RoBERTa for sentiment analysis with flexible dataset selection."""

    df_balanced, df_synthetic, df = None, None, None  # Initialize DataFrames

    if use_prebalanced: 
        if dataset_type == "balanced":
            df_balanced = load_balanced_table_from_db(f"balanced_{data_source}")
            df = df_balanced  # Use the balanced dataset as the main dataset
        elif dataset_type == "synthetic":
            df_synthetic = load_synthetic_table_from_db(f"synthetic_{data_source}")
            df = df_synthetic  # Use the synthetic dataset as the main dataset
        elif dataset_type == "unbalanced":
            df = pd.read_sql(f"SELECT text, label FROM {data_source}", engine)    
    else:
        df_balanced, df_synthetic, df = load_data(data_source)


    if dataset_type == "unbalanced":
        selected_df = df
        print("📂 Using unbalanced dataset.")
    elif dataset_type == "balanced":
        selected_df = df_balanced
        print("⚖️ Using balanced dataset.")
    elif dataset_type == "synthetic":
        selected_df = df_synthetic
        print("🧪 Using only synthetic dataset.")
    else:
        raise ValueError(f"Invalid dataset_type '{dataset_type}'. Choose from 'unbalanced', 'balanced', or 'synthetic'.")
    
    
    # Split the dataset into training and validation sets (80% train, 20% validation)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        selected_df["text"], selected_df["label"],
        test_size=0.2, random_state=42, stratify=selected_df["label"]
    )

    # Save validation data (20% of data) to the database
    save_validation_data(val_texts.tolist(), val_labels.tolist())
    
    train_dataset = RedditDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
    val_dataset = RedditDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)

    batch_size = 24  # Reduce if hitting memory limit
    accumulation_steps = 3  # Simulate larger batch sizes by accumulating gradients
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True) # Shuffle for training
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True) # No shuffle for validation
    num_classes = 3  # Number of classes
    
    config = RobertaConfig.from_pretrained(
        "roberta-base",
        num_labels=3,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1 
    )
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", config=config)
    model.to(device)  # Move model to GPU if available

    # ======= Loss Function: You can switch between FocalLoss and CrossEntropy =======
    alpha = get_focal_alpha(train_labels, num_classes=num_classes, neutral_boost=1.5)  # Compute alpha for Focal Loss
    class_weights = get_class_weights(train_labels, num_classes=num_classes)
    cross_entropy_loss = nn.CrossEntropyLoss(weight=class_weights.to(device))  # CrossEntropyLoss with class weights
    focal_loss = FocalLoss(alpha=alpha, gamma=2.0, device=device)

    # Group parameters for optimizer
    no_decay = {'bias', 'LayerNorm.weight'}  # Parameters that should not decay
    optimizer_grouped_parameters = [
        {
            #"params": [param for name, param in model.named_parameters() if not any(nd in name for nd in no_decay) and param.requires_grad],
            "params": [],
            "weight_decay": 0.01,   # Apply weight decay to all parameters except bias and LayerNorm.weight
        },
        {
            #"params": [param for name, param in model.named_parameters() if any(nd in name for nd in no_decay) and param.requires_grad],
            "params": [],
            "weight_decay": 0.0,   # Exclude bias and LayerNorm.weight from weight decay
        },
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay): # Check if the parameter is in no_decay
            optimizer_grouped_parameters[1]["params"].append(param)  # Without decay
        else:
            optimizer_grouped_parameters[0]["params"].append(param)  # With decay

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5) # Optimizer with grouped parameters

    # ======= Learning Rate Scheduler =======
    # OneCycleLR is a learning rate scheduler that adjusts the learning rate dynamically during training
    num_epochs = 15  # Number of epochs for training
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=2e-5,
        steps_per_epoch=len(train_loader) // accumulation_steps,
        epochs=num_epochs,
        pct_start=0.1,  # 10% warm-up
        anneal_strategy='cos',
        div_factor=25.0,  # Initial learning rate divided by this factor
        final_div_factor=1e4,  # Final learning rate divided by this factor
    )
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # ======= SWA Settings =======
    from torch.optim.swa_utils import AveragedModel, SWALR, update_bn  # Import SWA utilities
    # SWA is used to improve generalization by averaging weights over multiple epochs
    swa_start = num_epochs - 4
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-6)

    # Save model in fixed directory (no timestamp)
    model_path = "Roberta_sentiment_model"

    # ======= Early Stopping Settings =======
    best_val_f1 = 0.0
    patience = 2  # Number of epochs to wait for improvement
    patience_counter = 0
    train_losses, val_losses = [], []  # To track losses for plotting
    best_model_state_dict = None  # To save the best model state dict
    best_val_loss = float("inf")
    use_focal = True  # Set to True to use Focal Loss, False for CrossEntropy

    layers_to_freeze = 6  
    unfreeze_start_epoch = 2
    unfreeze_every_n_epochs = 2  # Unfreeze every 2 epochs after the first one
    freeze_roberta_layers(model, num_layers_to_freeze=layers_to_freeze)  # Freeze the first 6 layers of 12 layers of RoBERTa

    num_roberta_layers = len(model.roberta.encoder.layer)
    layers_to_unfreeze_order = list(reversed(range(num_roberta_layers - layers_to_freeze, num_roberta_layers)))  # [12, 11, 10, 9, 8, 7] for RoBERTa base

    layers_to_unfreeze_per_epoch = 2  # Num of layers to Unfreeze per epoch
    current_unfreeze_idx = 0  # Index of the next layer to unfreeze


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

        # ======= Unfreeze the last layers gradually =======
        if (epoch + 1) >= unfreeze_start_epoch and (epoch + 1 - unfreeze_start_epoch) % unfreeze_every_n_epochs == 0:# Sum +1 because epoch starts in 0, and execute the condicional every n epochs
            for _ in range(layers_to_unfreeze_per_epoch):
                if current_unfreeze_idx < len(layers_to_unfreeze_order):
                    layer_idx = layers_to_unfreeze_order[current_unfreeze_idx]
                    unfreeze_roberta_layers(model, layer_idx)
                    current_unfreeze_idx += 1
                    for group in optimizer.param_groups:
                        group['lr'] *= 0.8  # Reduce learning rate by 20% when unfreezing a layer

        if epoch > num_epochs - 4:
            model.config.hidden_dropout_prob = 0.05
            model.config.attention_probs_dropout_prob = 0.05

        # ======= Training =======
        model.train()
        total_loss = 0.0
        optimizer.zero_grad() # Reset gradients
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True) # Progress bar

        for i, batch in (loop):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):  # Mixed precision
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels) / accumulation_steps # Normalize FocalLoss and CrossEntropyLoss.

            scaler.scale(loss).backward()   # Backpropagation
            total_loss += loss.item()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader): # Update weights every accumulation_steps
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping to prevent exploding gradients
                scaler.step(optimizer) # Update weights
                scaler.update() # Update scaler
                optimizer.zero_grad() # Reset gradients

            loop.set_postfix(loss=loss.item()) # Update progress bar postfix

        avg_train_loss = total_loss / len(train_loader)  # Average training loss
        train_losses.append(avg_train_loss)

        # ======= Validation =======
        model.eval()
        all_preds = []
        all_labels = []
        val_total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs.logits, labels)
                    val_total_loss += loss.item()
                    preds = torch.argmax(outputs.logits, dim=1)  # No need for softmax

                all_labels.extend(labels.cpu().numpy())  # Store true labels
                all_preds.extend(preds.cpu().numpy())  # Store predictions

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="weighted") # Use weighted F1 score
        avg_val_loss = val_total_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"✅ Validation Accuracy: {val_acc:.4f} | F1 Score (weighted): {val_f1:.4f} | 🧪 Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print_classification_report(all_labels, all_preds)  # print classification report every epoch

        # ======= SWA Update =======
        # Update SWA model parameters after swa_start epochs
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            lr_scheduler.step() # Update learning rate scheduler for normal training and coincide with OneCycleLR

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

    # === Save best model at the end of training ===
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
    
    print("Updating BatchNorm for SWA model...")
    update_bn(train_loader, swa_model)  

    if os.path.exists(model_path): 
        shutil.rmtree(model_path)
    swa_model.module.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"\n📦 Final best SWA model saved to '{model_path}'")


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
