import os
import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix, multilabel_confusion_matrix, balanced_accuracy_score, matthews_corrcoef
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.database.db_connection import connect_to_db
from src.database.store_data import store_results_in_db
from sqlalchemy import text as sql_text

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Sentiment labels
LABELS = ["Negative", "Neutral", "Positive"]

# Model path (ensure it exists)
model_path = "Roberta_sentiment_model"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model path '{model_path}' not found. Train the model first!")

tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()  # Set model to evaluation mode

# Load validation data from database connection
engine = connect_to_db()  # Connect to the database

""" Fetch validation data from the database use this to fetch 20% of the data for validation or instead you can use the 
    validation_data set which is already in the database
"""

# Fetch validation data from the 'validation_data' table
with engine.connect() as conn:
    result = conn.execute(sql_text("SELECT id, text FROM validation_data"))
    val_data = result.fetchall()  # Fetch all validation data from the database
    val_ids, val_texts = zip(*val_data)
    print(f"🔢 Loaded {len(val_texts)} validation samples from 'validation_data'.")


def predict_sentiment(text):
    """
    Predicts the sentiment of a given text using the fine-tuned BERT model.

    :param text: The input text for sentiment analysis.
    :return: The predicted sentiment label index.
    """
    if not text.strip():
        return None  # Handle empty input gracefully

    encoding = tokenizer(
        text, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():  # Disable gradient calculation
        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = F.softmax(outputs.logits, dim=1)
        predicted_class = probabilities.argmax(dim=1).item()  # Get the predicted class index 
        # predicted_class = probabilities.argmax(dim=1).detach().cpu().numpy()[0]  # Get the predicted class index 

    return predicted_class  # Return the index instead of the label


def normalize_text(text):
    return text.strip().lower().replace("’", "'").replace("“", '"').replace("”", '"')


def evaluate_model():
    """Evaluates the model using validation_results stored in the database."""

    with engine.connect() as conn: # Connect to the database
        result = conn.execute(sql_text("SELECT text, predicted_label FROM validation_results"))
        results = result.fetchall() # Fetch all validation results from the database
        res_ids, predicted_labels = zip(*results) # Unzip the results into ids and predicted labels

        result = conn.execute(sql_text("SELECT id, text, label FROM validation_data")) # Fetch validation data from the database
        validation_data = result.fetchall() # Fetch all validation data from the database
        val_ids, val_texts, val_labels = zip(*validation_data) # Unzip the validation data into ids, texts and labels

    if not results:
        print("⚠️ No validation results found in the database.")
        return

    texts, predicted_labels = zip(*results) # Unzip the results into texts and predicted labels
    true_label_map = {}
    
    for _, text, label in validation_data: # Iterate over the validation data to create a mapping of text to label
        try:
            label_int = int(label) # Convert label to integer
            # Check if the label is within the valid range of indices for LABELS
            if 0 <= label_int < len(LABELS):    # Ensure label is a valid index
                norm_text = normalize_text(text) # Normalize the text
                true_label_map[norm_text] = LABELS[label_int]
        except (ValueError, TypeError):
            print(f"⚠️ Skipping invalid label: {label}")

    filtered_true_labels = []
    filtered_pred_labels = []

    for text, pred_label in zip(texts, predicted_labels): # Iterate over the texts and predicted labels
        norm_text = normalize_text(text) # Normalize the text
        if norm_text in true_label_map: # Check if the text exists in the true label map
            filtered_true_labels.append(true_label_map[norm_text]) # Append the true label to the filtered list
            filtered_pred_labels.append(pred_label) # Append the predicted label to the filtered list
    
    if not filtered_true_labels: # Check if there are any matching texts in the validation data
        print("⚠️ No matching texts found in validation data.")
        return
    
    accuracy = accuracy_score(filtered_true_labels, filtered_pred_labels) 
    f1 = f1_score(filtered_true_labels, filtered_pred_labels, average="weighted") 

    print(f"\n✅ **Accuracy:** {accuracy:.4f}")
    print(f"✅ **F1-score:** {f1:.4f}")

    # Balanced Accuracy and Matthews Correlation Coefficient
    label_to_int = {label: i for i, label in enumerate(LABELS)}
    true_ints = [label_to_int[label] for label in filtered_true_labels]
    pred_ints = [label_to_int[label] for label in filtered_pred_labels]


    # Confusion Matrix
    global_conf_matrix = confusion_matrix(true_ints, pred_ints, labels=range(len(LABELS))) # Create confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(global_conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    # Multi-label confusion matrix for per-class metrics
    conf_matrix = multilabel_confusion_matrix(true_ints, pred_ints, labels=range(len(LABELS)))

    # Classification Report
    print("\n📊 **Per-Class Detailed Metrics Table:**")
    """ Generate a detailed metrics table for each class 
    tn: True Negatives
    fp: False Positives
    fn: False Negatives
    tp: True Positives
    mcc: Matthews Correlation Coefficient"""
    table_data = []

    for idx, (label, matrix) in enumerate(zip(LABELS, conf_matrix)):
        tn, fp, fn, tp = matrix.ravel()
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        balanced_acc = (recall + tn / (tn + fp)) / 2 if (tn + fp) > 0 else 0

        mcc = matthews_corrcoef([1 if t == idx else 0 for t in true_ints], [1 if p == idx else 0 for p in pred_ints])

        table_data.append({
            "Class": label,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1-Score": round(f1, 3),
            "Balanced Acc.": round(balanced_acc, 3),
            "MCC": round(mcc, 3),
            "Support": support
        })

    # Create a DataFrame for the metrics table
    df_metrics = pd.DataFrame(table_data)
    print(df_metrics.to_string(index=False))


def manual_prediction():
    """Allows the user to input text and get a sentiment prediction."""
    print("\n🔍 **Manual Sentiment Prediction Mode** (type 'exit' to quit)")
    while True:
        text = input("\n📝 Enter a text to analyze sentiment: ").strip()
        if text.lower() == "exit":
            print("👋 Exiting manual prediction mode.")
            break

        predicted_class = predict_sentiment(text)
        if predicted_class is None:
            print("⚠️ Please enter a valid text!")
            continue

        sentiment = LABELS[predicted_class]
        print(f"🔍 **Predicted Sentiment:** {sentiment} ({predicted_class})")


def run_validation_and_store_results():
    """Generates predictions for validation data and stores them in the database."""
    results_to_store = [] # List to store results for database insertion
    for idx, text in zip(val_ids, val_texts): # Iterate over the validation texts and ids
        pred_index = predict_sentiment(text) # Get the predicted index
        if pred_index is not None: # Check if prediction is valid
            label = LABELS[pred_index] # Get the label from the index (Negative, Neutral, Positive instead of 0, 1, 2)
            results_to_store.append((idx, text, label)) # Append the text and label to the results list
    store_results_in_db(engine, results_to_store) # Store results in the database

if __name__ == "__main__":
    run_validation_and_store_results()  # Run validation and store results in DB
    evaluate_model()  # Run model evaluation
    manual_prediction()  # Allow user to input texts
